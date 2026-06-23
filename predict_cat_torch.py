#!/usr/bin/env python3
"""
Run inference with trained CatCNNTorch and save prediction heatmaps.

Usage:
    python predict_cat_torch.py [--event EVENT_ID] [--all] [--config configs/cnn.yaml]
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import (
    build_raw_samples, load_norm, CatDataset,
    channel_names as _cnames, satellite_features, _climate_vector,
    cyclic_time, load_climate_table,
)
from src.features.diagnostics import DEFAULT_PRIMARY_LEVELS, load_or_compute
from src.models.cnn_torch import build_model, DEFAULT_CFG
from src.models.losses import DEFAULT_LOSS_CFG

MODELS_DIR  = Path("models")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MAX_EDR = 0.95


def severity(edr: float) -> str:
    if edr < 0.15:
        return "smooth"
    if edr < 0.35:
        return "light"
    if edr < 0.55:
        return "moderate"
    return "severe"


def _load_model(cfg: dict, cnames: list[str], device: torch.device):
    model_path = MODELS_DIR / "cat_cnn_torch.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model_cfg = {**DEFAULT_CFG, **cfg.get("model", {})}
    n_diag = len(cnames)
    model = build_model(model_cfg, n_diag).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()
    return model


def _norm_diag(diag: np.ndarray, norm, cnames: list[str]) -> np.ndarray:
    from src.data.dataset import _apply_log
    diag = _apply_log(diag, cnames).astype(np.float32)
    diag = (diag - norm.diag_mu[:, None, None]) / norm.diag_sig[:, None, None]
    return diag


def _assemble(sample, norm, cnames, device):
    diag_n = _norm_diag(sample.diag, norm, cnames)
    sat = sample.sat.copy()
    if sample.sat_mask[0] > 0:
        sat = (sat - norm.sat_mu) / norm.sat_sig
    phys = (sample.phys - norm.phys_mu) / norm.phys_sig

    return {
        "diag":     torch.from_numpy(diag_n).unsqueeze(0).to(device),
        "climate":  torch.from_numpy(sample.climate).unsqueeze(0).to(device),
        "time":     torch.from_numpy(sample.time).unsqueeze(0).to(device),
        "sat":      torch.from_numpy(sat).unsqueeze(0).to(device),
        "sat_mask": torch.from_numpy(sample.sat_mask).unsqueeze(0).to(device),
        "phys":     torch.from_numpy(phys).unsqueeze(0).to(device),
        "y_max":    torch.tensor([sample.y_max]),
        "y_mean":   torch.tensor([sample.y_mean]),
        "eid":      torch.tensor([sample.event_id]),
    }


@torch.no_grad()
def predict_event(sample, model, norm, cnames, device) -> dict:
    batch = _assemble(sample, norm, cnames, device)
    out   = model(batch)
    field    = out["field"][0, 0].cpu().numpy()    # (H, W)
    max_hat  = float(out["max_hat"][0])
    mean_hat = float(out["mean_hat"][0])
    return dict(field=field, max_hat=max_hat, mean_hat=mean_hat,
                y_max=float(sample.y_max) * MAX_EDR, event_id=sample.event_id)


def _plot(result: dict, cfg: dict) -> Path:
    eid      = result["event_id"]
    field    = result["field"]
    max_hat  = result["max_hat"]
    y_max    = result["y_max"]
    sev_true = severity(y_max)
    sev_pred = severity(max_hat)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        f"Event {eid:04d} — True EDR: {y_max:.3f} ({sev_true}) | "
        f"Predicted: {max_hat:.3f} ({sev_pred})",
        fontsize=11, fontweight="bold",
    )

    cmap = plt.cm.YlOrRd
    norm_cmap = mcolors.Normalize(vmin=0, vmax=max(field.max(), 0.5))

    # Heatmap
    ax = axes[0]
    im = ax.imshow(field, cmap=cmap, norm=norm_cmap, origin="upper", aspect="auto")
    ax.set_title("Predicted turbulence intensity field")
    ax.set_xlabel("Longitude index")
    ax.set_ylabel("Latitude index")
    plt.colorbar(im, ax=ax, label="Intensity")

    # Histogram of field values
    ax2 = axes[1]
    ax2.hist(field.ravel(), bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax2.axvline(max_hat, color="red", linestyle="--", label=f"max_hat={max_hat:.3f}")
    ax2.axvline(y_max,   color="green", linestyle="--", label=f"true={y_max:.3f}")
    ax2.set_title("Field value distribution")
    ax2.set_xlabel("Intensity")
    ax2.set_ylabel("Count")
    ax2.legend()

    plt.tight_layout()
    out_path = FIGURES_DIR / f"pred_event_{eid:04d}.png"
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def predict_all(samples, model, norm, cnames, device, cfg) -> pd.DataFrame:
    rows = []
    for s in samples:
        try:
            r = predict_event(s, model, norm, cnames, device)
            path = _plot(r, cfg)
            rows.append({
                "event_id":  r["event_id"],
                "y_max":     r["y_max"],
                "max_hat":   r["max_hat"],
                "mean_hat":  r["mean_hat"],
                "severity_true": severity(r["y_max"]),
                "severity_pred": severity(r["max_hat"]),
                "figure":    str(path),
            })
            log.info("Event %04d  true=%.3f  pred=%.3f  fig=%s",
                     r["event_id"], r["y_max"], r["max_hat"], path.name)
        except Exception as exc:
            log.warning("Event %04d failed: %s", s.event_id, exc)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cnn.yaml")
    parser.add_argument("--event", type=int, default=None)
    parser.add_argument("--all",  action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = torch.device("cpu")
    plevels = cfg.get("primary_levels", DEFAULT_PRIMARY_LEVELS)
    cnames  = _cnames(plevels)

    norm  = load_norm(str(MODELS_DIR / "cat_cnn_torch_norm.npz"))
    model = _load_model(cfg, cnames, device)

    log.info("Building sample list...")
    samples = build_raw_samples(primary_levels=plevels, grid_size=cfg.get("grid_size", 24))

    if args.event is not None:
        matches = [s for s in samples if s.event_id == args.event]
        if not matches:
            log.error("Event %d not found", args.event)
            return
        r = predict_event(matches[0], model, norm, cnames, device)
        p = _plot(r, cfg)
        log.info("Saved → %s", p)
    elif args.all or True:   # default: predict all
        df = predict_all(samples, model, norm, cnames, device, cfg)
        out_csv = MODELS_DIR / "cat_cnn_torch_predictions.csv"
        df.to_csv(str(out_csv), index=False)
        log.info("Predictions saved → %s", out_csv)
        log.info("Figures saved → %s/", FIGURES_DIR)


if __name__ == "__main__":
    main()
