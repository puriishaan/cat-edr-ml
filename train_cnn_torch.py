#!/usr/bin/env python3
"""
Train CatCNNTorch on ERA5 physics diagnostics + climate + satellite features.

Usage:
    python train_cnn_torch.py [--config configs/cnn.yaml] [--tune] [--folds N]
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import (
    build_raw_samples, fit_normalisation, save_norm, load_norm,
    make_folds, temporal_holdout, CatDataset,
    channel_names as _cnames,
)
from src.models.cnn_torch import build_model, DEFAULT_CFG
from src.models.losses import compute_loss, DEFAULT_LOSS_CFG

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MAX_EDR = 0.95


# ── Config helpers ─────────────────────────────────────────────────────────────

def load_config(path: str = "configs/cnn.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def flat_cfg(cfg: dict) -> dict:
    """Merge model + loss sub-dicts into a flat dict for easy passing."""
    out = {}
    out.update(cfg.get("model", {}))
    out.update(cfg.get("loss", {}))
    return out


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(y_true: np.ndarray, y_pred: np.ndarray,
            threshold: float = 0.35) -> dict:
    from sklearn.metrics import (
        mean_squared_error, average_precision_score, roc_auc_score
    )
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    log_rmse = float(np.sqrt(mean_squared_error(
        np.log1p(y_true * MAX_EDR), np.log1p(y_pred * MAX_EDR)
    )))
    corr = float(pd.Series(y_true).corr(pd.Series(y_pred)))
    mae  = float(np.abs(y_true - y_pred).mean())
    labels = (y_true >= threshold).astype(int)
    auprc = auroc = float("nan")
    if labels.sum() > 0 and labels.sum() < len(labels):
        auprc = float(average_precision_score(labels, y_pred))
        auroc = float(roc_auc_score(labels, y_pred))
    return dict(rmse=rmse, log_rmse=log_rmse, pearson=corr, mae=mae,
                auprc=auprc, auroc=auroc)


def objective_score(m: dict) -> float:
    return m["rmse"] - 0.1 * (m.get("pearson", 0) or 0)


# ── Data loaders ───────────────────────────────────────────────────────────────

def _loader(samples, norm, cnames, batch_size, augment=False, shuffle=False):
    ds = CatDataset(samples, norm, cnames, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def _to_dev(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device, loss_cfg: dict,
             threshold: float = 0.35) -> dict:
    model.eval()
    all_true, all_pred, total_loss = [], [], 0.0
    for batch in loader:
        batch = _to_dev(batch, device)
        out   = model(batch)
        loss, _ = compute_loss(out, batch, loss_cfg)
        total_loss += loss.item()
        all_true.extend(batch["y_max"].cpu().numpy().tolist())
        all_pred.extend(out["max_hat"].cpu().numpy().tolist())
    m = metrics(np.array(all_true), np.array(all_pred), threshold)
    m["loss"] = total_loss / max(len(loader), 1)
    return m


# ── Single-fold training ──────────────────────────────────────────────────────

def train_one(
    tr_samples, va_samples, norm, cnames,
    model_cfg: dict, loss_cfg: dict,
    train_cfg: dict, device: torch.device,
    save_path: str = None,
    threshold: float = 0.35,
) -> tuple[nn.Module, dict]:

    n_diag = tr_samples[0].diag.shape[0]
    model = build_model(model_cfg, n_diag).to(device)

    opt = torch.optim.AdamW(model.parameters(),
                             lr=train_cfg["lr"],
                             weight_decay=train_cfg["weight_decay"])

    total_steps = train_cfg["epochs"] * max(len(tr_samples) // train_cfg["batch_size"], 1)
    if train_cfg.get("scheduler") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    else:
        sched = None

    tr_loader = _loader(tr_samples, norm, cnames,
                        train_cfg["batch_size"], augment=True, shuffle=True)
    va_loader = _loader(va_samples, norm, cnames,
                        train_cfg["batch_size"] * 2)

    best_score = float("inf")
    best_state = None
    patience   = train_cfg.get("early_stop_patience", 15)
    wait       = 0

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tr_loader:
            batch = _to_dev(batch, device)
            opt.zero_grad()
            out = model(batch)
            loss, _ = compute_loss(out, batch, loss_cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched:
                sched.step()
            epoch_loss += loss.item()

        val_m = evaluate(model, va_loader, device, loss_cfg, threshold)
        score = objective_score(val_m)

        if score < best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log.info("Early stop at epoch %d", epoch)
                break

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "Ep %3d  loss=%.4f  val_rmse=%.4f  val_r=%.3f  val_auroc=%.3f",
                epoch, epoch_loss / len(tr_loader),
                val_m["rmse"], val_m["pearson"], val_m.get("auroc", float("nan"))
            )

    model.load_state_dict(best_state)
    val_m = evaluate(model, va_loader, device, loss_cfg, threshold)

    if save_path:
        torch.save(model.state_dict(), save_path)
        log.info("Saved model → %s", save_path)

    return model, val_m


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate(
    samples, cnames, cfg: dict, device: torch.device
) -> dict:
    set_seed(cfg["seed"])
    folds = make_folds(samples, n_folds=cfg["train"]["n_folds"], seed=cfg["seed"])
    model_cfg = flat_cfg(cfg)
    loss_cfg  = {k: cfg["loss"][k] for k in DEFAULT_LOSS_CFG if k in cfg["loss"]}
    threshold = cfg.get("moderate_threshold", 0.35)

    fold_results = []
    all_true, all_pred, all_eids = [], [], []

    for k, (tr, va) in enumerate(folds):
        log.info("── Fold %d/%d ──", k + 1, len(folds))
        norm = fit_normalisation(tr, cnames)
        model, val_m = train_one(
            tr, va, norm, cnames,
            model_cfg, loss_cfg, cfg["train"], device,
            threshold=threshold,
        )
        fold_results.append(val_m)
        log.info("Fold %d val: RMSE=%.4f  r=%.3f  AUROC=%.3f",
                 k + 1, val_m["rmse"], val_m["pearson"], val_m.get("auroc", float("nan")))

        # Collect OOF
        va_loader = _loader(va, norm, cnames, cfg["train"]["batch_size"] * 2)
        model.eval()
        with torch.no_grad():
            for batch in va_loader:
                batch = _to_dev(batch, device)
                out = model(batch)
                all_true.extend(batch["y_max"].cpu().numpy())
                all_pred.extend(out["max_hat"].cpu().numpy())
                all_eids.extend(batch["eid"].cpu().numpy())

    oof_m = metrics(np.array(all_true), np.array(all_pred), threshold)
    log.info("OOF metrics: RMSE=%.4f  r=%.3f  AUPRC=%.3f  AUROC=%.3f",
             oof_m["rmse"], oof_m["pearson"],
             oof_m.get("auprc", float("nan")), oof_m.get("auroc", float("nan")))

    return dict(fold_results=fold_results, oof=oof_m,
                oof_true=np.array(all_true),
                oof_pred=np.array(all_pred),
                oof_eids=np.array(all_eids))


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def suggest_from_space(trial, space: dict, base_cfg: dict) -> dict:
    cfg = dict(base_cfg)
    for name, spec in space.items():
        t = spec["type"]
        if t == "int":
            cfg[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "float":
            cfg[name] = trial.suggest_float(name, spec["low"], spec["high"],
                                             log=spec.get("log", False))
        elif t == "categorical":
            cfg[name] = trial.suggest_categorical(name, spec["choices"])
    return cfg


def tune(samples, cnames, cfg: dict, device: torch.device) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    space = cfg.get("tune", {}).get("space", {})
    base_model_cfg = {**DEFAULT_CFG, **cfg.get("model", {})}

    def objective(trial):
        trial_model_cfg = suggest_from_space(trial, space, base_model_cfg)
        trial_cfg = {**cfg, "model": trial_model_cfg}
        set_seed(cfg["seed"])
        folds = make_folds(samples, n_folds=3, seed=cfg["seed"])
        scores = []
        for k, (tr, va) in enumerate(folds):
            norm = fit_normalisation(tr, cnames)
            loss_cfg = {k2: cfg["loss"][k2] for k2 in DEFAULT_LOSS_CFG if k2 in cfg["loss"]}
            quick_cfg = {**cfg["train"], "epochs": min(cfg["train"]["epochs"], 30), "early_stop_patience": 8}
            _, val_m = train_one(tr, va, norm, cnames,
                                  trial_model_cfg, loss_cfg, quick_cfg, device)
            scores.append(objective_score(val_m))
            trial.report(np.mean(scores), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    tune_cfg = cfg.get("tune", {})
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    study.optimize(objective,
                   n_trials=tune_cfg.get("n_trials", 20),
                   timeout=tune_cfg.get("timeout", 3600),
                   show_progress_bar=False)

    best = study.best_params
    log.info("Best params: %s  (score=%.4f)", best, study.best_value)
    with open(str(MODELS_DIR / "cat_cnn_torch_best_params.json"), "w") as f:
        json.dump(best, f, indent=2)
    return best


# ── Main ──────────────────────────────────────────────────────────────────────

def _fmt(m: dict) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cnn.yaml")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--folds", type=int, default=None)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    if args.folds:
        cfg["train"]["n_folds"] = args.folds
    device = pick_device()
    log.info("Device: %s", device)

    set_seed(cfg["seed"])
    plevels = cfg.get("primary_levels", [225, 250, 300])
    cnames  = _cnames(plevels)

    log.info("Building dataset...")
    samples = build_raw_samples(
        primary_levels=plevels,
        grid_size=cfg.get("grid_size", 24),
    )
    log.info("Total samples: %d", len(samples))

    # Hold out test set
    train_samples, test_samples = temporal_holdout(samples, cfg.get("test_frac", 0.15))
    log.info("Train: %d  Test: %d", len(train_samples), len(test_samples))

    # Optional tuning
    if args.tune:
        log.info("Running hyperparameter tuning...")
        best_params = tune(train_samples, cnames, cfg, device)
        cfg["model"].update(best_params)

    # Cross-validation on train set
    log.info("Running %d-fold cross-validation...", cfg["train"]["n_folds"])
    cv_results = cross_validate(train_samples, cnames, cfg, device)

    log.info("CV OOF: %s", _fmt(cv_results["oof"]))

    # Final model: all train data
    log.info("Training final model on all training data...")
    norm = fit_normalisation(train_samples, cnames)
    model_cfg = flat_cfg(cfg)
    loss_cfg  = {k: cfg["loss"][k] for k in DEFAULT_LOSS_CFG if k in cfg["loss"]}
    model, _ = train_one(
        train_samples, test_samples, norm, cnames,
        model_cfg, loss_cfg, cfg["train"], device,
        save_path=str(MODELS_DIR / "cat_cnn_torch.pt"),
        threshold=cfg.get("moderate_threshold", 0.35),
    )

    save_norm(norm, str(MODELS_DIR / "cat_cnn_torch_norm.npz"))

    # Test set evaluation
    test_loader = _loader(test_samples, norm, cnames, cfg["train"]["batch_size"] * 2)
    test_m = evaluate(model, test_loader, device, loss_cfg, cfg.get("moderate_threshold", 0.35))
    log.info("TEST: %s", _fmt(test_m))

    # Collect all predictions
    all_rows = []
    for split_name, split_samples in [("cv_oof", train_samples), ("test", test_samples)]:
        loader = _loader(split_samples, norm, cnames, cfg["train"]["batch_size"] * 2)
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = _to_dev(batch, device)
                out = model(batch)
                for i in range(len(batch["y_max"])):
                    all_rows.append({
                        "event_id":  int(batch["eid"][i]),
                        "split":     split_name,
                        "y_max_true": float(batch["y_max"][i]) * MAX_EDR,
                        "y_max_pred": float(out["max_hat"][i]),
                        "y_mean_true": float(batch["y_mean"][i]) * MAX_EDR,
                        "y_mean_pred": float(out["mean_hat"][i]),
                    })

    # Use OOF predictions for train split (more honest)
    eid2pred = {int(cv_results["oof_eids"][i]): float(cv_results["oof_pred"][i])
                for i in range(len(cv_results["oof_eids"]))}
    eval_df = pd.DataFrame(all_rows)
    # Mark train rows with OOF pred
    train_eids = {s.event_id for s in train_samples}
    eval_df.loc[eval_df["event_id"].isin(train_eids), "split"] = "train_oof"

    eval_path = MODELS_DIR / "cat_cnn_torch_eval.csv"
    eval_df.to_csv(eval_path, index=False)
    log.info("Eval saved → %s", eval_path)

    # Save CV fold metrics
    fold_df = pd.DataFrame(cv_results["fold_results"])
    fold_df.index.name = "fold"
    fold_df.to_csv(str(MODELS_DIR / "cat_cnn_torch_folds.csv"))
    log.info("Fold metrics saved.")

    return model, eval_df, cv_results


if __name__ == "__main__":
    main()
