#!/usr/bin/env python3
"""
Train the physics-informed CAT CNN (PyTorch) + optional Optuna design-space search.

Pipeline
--------
  1. Build all per-event samples (diagnostics + climate + time + satellite + labels).
  2. Hold out the most-recent `test_frac` of events (temporal generalisation test).
  3. Plain run     : train on a train/val split of the remainder, evaluate on the hold-out.
     `--tune`      : Optuna over the design space (grouped/stratified CV), then train the
                     best config on the full remainder and evaluate on the hold-out.

Targets the scalar `max_edr` via the model's soft-aggregated heatmap; metrics are reported
in both EDR units and log space (the latter directly comparable to the XGBoost baseline).

Usage
-----
  python train_cnn_torch.py                      # train default config, eval on hold-out
  python train_cnn_torch.py --tune --trials 60   # search, then train best
  python train_cnn_torch.py --limit 30 --epochs 8  # quick wiring/overfit smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    CatDataset, SAT_FEATURES, build_raw_samples, fit_normalisation,
    make_folds, save_norm, temporal_holdout,
)
from src.models.cnn_torch import build_model
from src.models.losses import compute_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
CONFIG = Path("configs/cnn.yaml")


# ─── Config helpers ───────────────────────────────────────────────────────────

def load_config(path=CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def flat_cfg(cfg: dict) -> dict:
    """Merge model + loss + train sections into one flat dict the modules read."""
    out = {}
    for sec in ("model", "loss", "train"):
        out.update(cfg.get(sec, {}))
    return out


def suggest_from_space(trial, space: dict) -> dict:
    """Sample a flat cfg override from the yaml `tune.space` description."""
    out = {}
    for key, spec in space.items():
        if not isinstance(spec, dict):
            out[key] = spec                                  # fixed value
        elif "categorical" in spec:
            out[key] = trial.suggest_categorical(key, spec["categorical"])
        elif "float" in spec:
            lo, hi = spec["float"]
            out[key] = trial.suggest_float(key, lo, hi, log=spec.get("log", False))
        elif "int" in spec:
            lo, hi = spec["int"]
            out[key] = trial.suggest_int(key, lo, hi)
    return out


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── Metrics ──────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred, thr) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    err = y_pred - y_true
    out = {
        "rmse_edr": float(np.sqrt(np.mean(err ** 2))),
        "mae_edr": float(np.mean(np.abs(err))),
        "rmse_log": float(np.sqrt(np.mean((np.log1p(np.clip(y_pred, 0, None)) - np.log1p(y_true)) ** 2))),
        "pearson": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan"),
    }
    exceed = (y_true >= thr).astype(int)
    if 0 < exceed.sum() < len(exceed):
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            out["auprc"] = float(average_precision_score(exceed, y_pred))
            out["auroc"] = float(roc_auc_score(exceed, y_pred))
        except Exception:
            out["auprc"] = out["auroc"] = float("nan")
    else:
        out["auprc"] = out["auroc"] = float("nan")
    return out


def objective_score(m: dict) -> float:
    """Lower is better: log-RMSE + (1 - AUPRC) (AUPRC dropped if undefined)."""
    s = m["rmse_log"]
    if not np.isnan(m["auprc"]):
        s += (1.0 - m["auprc"])
    return s


# ─── Train / eval one split ───────────────────────────────────────────────────

def _loader(rs, idx, stats, batch, shuffle, augment=False, seed=0):
    ds = CatDataset(rs, idx, stats, augment=augment, seed=seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)


def _to_dev(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    yt, yp = [], []
    for batch in loader:
        out = model(_to_dev(batch, device))
        yp.append(out["max_hat"].cpu().numpy().ravel())
        yt.append(batch["y_max"].numpy().ravel())
    return np.concatenate(yt), np.concatenate(yp)


def train_one(cfg, rs, train_idx, val_idx, device, epochs, patience, thr, verbose=False):
    stats = fit_normalisation(rs, train_idx)
    batch = int(cfg.get("batch_size", 16))
    tr = _loader(rs, train_idx, stats, batch, True, augment=cfg.get("augment", False), seed=cfg.get("seed", 0))
    va = _loader(rs, val_idx, stats, batch, False)

    model = build_model(cfg, rs.diag.shape[1], rs.climate.shape[1], rs.time.shape[1],
                        len(SAT_FEATURES)).to(device)
    opt_name = cfg.get("optimizer", "adamw")
    Opt = torch.optim.AdamW if opt_name == "adamw" else torch.optim.Adam
    opt = Opt(model.parameters(), lr=float(cfg.get("lr", 1e-3)),
              weight_decay=float(cfg.get("weight_decay", 1e-4)))

    best, best_state, since = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for batch in tr:
            b = _to_dev(batch, device)
            opt.zero_grad()
            loss, _ = compute_loss(model(b), b, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        yt, yp = evaluate(model, va, device)
        m = metrics(yt, yp, thr)
        score = objective_score(m)
        if score < best - 1e-5:
            best, best_state, since = score, {k: v.detach().cpu().clone()
                                              for k, v in model.state_dict().items()}, 0
        else:
            since += 1
        if verbose and (ep % 10 == 0 or ep == 1):
            log.info("  ep %3d | val log-RMSE %.4f  RMSE %.4f EDR  AUPRC %.3f  (best %.4f)",
                     ep, m["rmse_log"], m["rmse_edr"], m["auprc"], best)
        if since >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    yt, yp = evaluate(model, va, device)
    return model, stats, metrics(yt, yp, thr)


def cross_validate(cfg, rs, trainval_idx, device, n_folds, epochs, thr) -> float:
    scores = []
    for fi, (tr, va) in enumerate(make_folds(rs, trainval_idx, n_folds=n_folds, seed=cfg.get("seed", 42))):
        _, _, m = train_one(cfg, rs, tr, va, device, epochs, patience=max(10, epochs // 3), thr=thr)
        scores.append(objective_score(m))
        log.info("    fold %d/%d | log-RMSE %.4f AUPRC %.3f → score %.4f",
                 fi + 1, n_folds, m["rmse_log"], m["auprc"], scores[-1])
    return float(np.mean(scores))


# ─── Optuna search ────────────────────────────────────────────────────────────

def tune(cfg, rs, trainval_idx, device, thr, n_trials):
    import optuna

    base = flat_cfg(cfg)
    tcfg = cfg["tune"]
    space = tcfg["space"]

    def objective(trial):
        c = {**base, **suggest_from_space(trial, space), "seed": cfg["seed"]}
        return cross_validate(c, rs, trainval_idx, device, tcfg["n_folds"], tcfg["epochs"], thr)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)
    log.info("Best CV score %.4f with params:\n%s", study.best_value,
             json.dumps(study.best_params, indent=2))
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "cat_cnn_torch_best_params.json").write_text(json.dumps(study.best_params, indent=2))
    return {**base, **study.best_params, "seed": cfg["seed"]}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--tune", action="store_true", help="run Optuna design-space search first")
    ap.add_argument("--trials", type=int, default=None, help="override tune.n_trials")
    ap.add_argument("--epochs", type=int, default=None, help="override final-train epochs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="use only first N events (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = pick_device(args.device)
    thr = cfg["moderate_threshold"]
    log.info("Device: %s", device)

    rs = build_raw_samples(grid_size=cfg["grid_size"], primary_levels=cfg["primary_levels"])
    if args.limit:
        keep = np.arange(min(args.limit, len(rs)))
        rs.diag, rs.climate, rs.time = rs.diag[keep], rs.climate[keep], rs.time[keep]
        rs.sat, rs.sat_mask, rs.phys = rs.sat[keep], rs.sat_mask[keep], rs.phys[keep]
        rs.y_max, rs.y_mean = rs.y_max[keep], rs.y_mean[keep]
        rs.eids, rs.bins, rs.times = rs.eids[keep], rs.bins[keep], rs.times[keep]

    trainval_idx, test_idx = temporal_holdout(rs, frac=cfg["test_frac"])
    log.info("Split: %d train/val  |  %d held-out test", len(trainval_idx), len(test_idx))

    final_cfg = flat_cfg(cfg)
    final_cfg["seed"] = cfg["seed"]
    if args.tune:
        n_trials = args.trials or cfg["tune"]["n_trials"]
        log.info("Optuna search: %d trials × %d-fold CV ...", n_trials, cfg["tune"]["n_folds"])
        final_cfg = tune(cfg, rs, trainval_idx, device, thr, n_trials)

    # carry data-shape settings into the saved cfg so inference rebuilds inputs identically
    final_cfg["grid_size"] = cfg["grid_size"]
    final_cfg["primary_levels"] = cfg["primary_levels"]

    # final model: carve a val split out of train/val for early stopping
    epochs = args.epochs or cfg["train"]["epochs"]
    (tr, va) = make_folds(rs, trainval_idx, n_folds=5, seed=cfg["seed"])[0]
    log.info("Training final model (%d epochs) ...", epochs)
    model, stats, val_m = train_one(final_cfg, rs, tr, va, device, epochs,
                                    cfg["train"]["patience"], thr, verbose=True)
    log.info("Validation | %s", _fmt(val_m))

    # held-out test
    test_loader = _loader(rs, test_idx, stats, 32, False)
    yt, yp = evaluate(model, test_loader, device)
    test_m = metrics(yt, yp, thr)
    log.info("HELD-OUT TEST | %s", _fmt(test_m))

    # ── save artefacts ────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "cfg": final_cfg,
        "dims": dict(in_channels=int(rs.diag.shape[1]), climate_dim=int(rs.climate.shape[1]),
                     time_dim=int(rs.time.shape[1]), sat_dim=len(SAT_FEATURES)),
        "names": list(rs.names),
        "metrics": {"val": val_m, "test": test_m},
    }, MODELS_DIR / "cat_cnn_torch.pt")
    save_norm(stats)

    # full-dataset predictions for inspection
    all_idx = np.arange(len(rs))
    yt_a, yp_a = evaluate(model, _loader(rs, all_idx, stats, 32, False), device)
    import pandas as pd
    split = np.array(["test" if i in set(test_idx) else "trainval" for i in all_idx])
    pd.DataFrame({
        "event_id": rs.eids, "edr_bin": rs.bins, "split": split,
        "true_max_edr": yt_a, "pred_max_edr": yp_a,
    }).to_csv(MODELS_DIR / "cat_cnn_torch_eval.csv", index=False)
    log.info("Saved → models/cat_cnn_torch.pt, cat_cnn_torch_norm.npz, cat_cnn_torch_eval.csv")


def _fmt(m: dict) -> str:
    return (f"RMSE {m['rmse_edr']:.4f} EDR  MAE {m['mae_edr']:.4f}  log-RMSE {m['rmse_log']:.4f}  "
            f"Pearson {m['pearson']:.3f}  AUPRC {m['auprc']:.3f}  AUROC {m['auroc']:.3f}")


if __name__ == "__main__":
    main()
