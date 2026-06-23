#!/usr/bin/env python3
"""
Predict CAT EDR for an event with the physics-informed PyTorch CNN.

Emits the scalar max-EDR estimate (+ severity) and writes the predicted EDR heatmap
(the spatial product) to results/figures/pred_event_XXXX.png.

Usage
-----
  python predict_cat_torch.py --event-id 42
  python predict_cat_torch.py --all
  python predict_cat_torch.py data/era5/event_0007.nc --time 2017-08-12T18:00

Requires a trained model: run `python train_cnn_torch.py` first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import (
    SAT_FEATURES, _climate_vector, cyclic_time, load_climate_table,
    load_norm, satellite_features,
)
from src.features.diagnostics import compute_event_diagnostics, load_or_compute
from src.models.cnn_torch import build_model

MODELS_DIR = Path("models")
ERA5_DIR = Path("data/era5")
EVENTS_CSV = Path("events.csv")
FIG_DIR = Path("results/figures")
CKPT = MODELS_DIR / "cat_cnn_torch.pt"
MAX_EDR = 0.95


def severity(edr: float) -> str:
    if edr >= 0.40:
        return "SEVERE"
    if edr >= 0.20:
        return "MODERATE"
    if edr >= 0.10:
        return "LIGHT"
    return "SMOOTH"


# ─── Model + normalisation ────────────────────────────────────────────────────

def _load_model(device):
    if not CKPT.exists():
        raise FileNotFoundError(f"{CKPT} not found — run train_cnn_torch.py first.")
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    d = ckpt["dims"]
    model = build_model(ckpt["cfg"], d["in_channels"], d["climate_dim"], d["time_dim"], d["sat_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def _norm_diag(X, stats):
    log_mask = np.asarray(stats["log_mask"])
    ix = np.where(log_mask)[0]
    X = X.copy()
    X[ix] = np.sign(X[ix]) * np.log1p(np.abs(X[ix]))
    return (X - stats["diag_mu"][0]) / stats["diag_sig"][0]


def _assemble(event_id, ts, stats, cfg, grid_size, primary_levels, era5_path=None):
    """Build a single-sample batch (N=1) of normalised tensors for the model."""
    if era5_path is not None:
        import xarray as xr
        X, _ = compute_event_diagnostics(xr.open_dataset(era5_path), primary_levels, grid_size)
    else:
        X, _ = load_or_compute(event_id, ERA5_DIR, "data/diagnostics", primary_levels, grid_size)
    if X is None:
        raise FileNotFoundError(f"No ERA5/diagnostics for event {event_id}")

    diag = _norm_diag(X, stats).astype(np.float32)
    clim = (_climate_vector(ts, load_climate_table()) - stats["clim_mu"][0]) / stats["clim_sig"][0]
    tvec = cyclic_time(ts)
    sat, mask = satellite_features(event_id) if event_id is not None else (np.zeros(len(SAT_FEATURES), np.float32), 0.0)
    sat = ((sat - stats["sat_mu"][0]) / stats["sat_sig"][0]) * mask

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()[None]
    return {
        "diag": t(diag), "climate": t(clim.astype(np.float32)), "time": t(tvec),
        "sat": t(sat.astype(np.float32)), "sat_mask": torch.tensor([[mask]]).float(),
    }


# ─── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_event(event_id: int, ts: pd.Timestamp | None = None, plot: bool = True,
                  device=None) -> dict:
    device = device or torch.device("cpu")
    model, ckpt = _load_model(device)
    stats = load_norm()
    cfg = ckpt["cfg"]
    grid_size = int(cfg.get("grid_size", 24))
    primary_levels = cfg.get("primary_levels", [225, 250, 300])

    if ts is None:
        ev = pd.read_csv(EVENTS_CSV)
        row = ev[ev["event_id"] == event_id]
        ts = pd.to_datetime(row.iloc[0]["start_utc"], utc=True).tz_localize(None) if len(row) \
            else pd.Timestamp("2015-01-01")

    batch = _assemble(event_id, ts, stats, cfg, grid_size=grid_size, primary_levels=primary_levels)
    out = model({k: v.to(device) for k, v in batch.items()})
    edr = float(out["max_hat"].item())
    field = out["field"][0, 0].cpu().numpy()

    print(f"Event          : {event_id}")
    print(f"Estimated EDR  : {edr:.3f} m^2/3 s^-1   (mean {float(out['mean_hat'].item()):.3f})")
    print(f"Severity       : {severity(edr)}")

    if plot:
        _plot(field, event_id, edr)
    return {"event_id": event_id, "max_edr": edr, "severity": severity(edr), "field": field}


def _plot(field, event_id, edr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(field, origin="lower", cmap="turbo", vmin=0, vmax=max(0.42, field.max()))
    ax.set_title(f"Event {event_id} — predicted EDR heatmap (max≈{edr:.3f})")
    fig.colorbar(im, ax=ax, label="EDR  m$^{2/3}$s$^{-1}$")
    out = FIG_DIR / f"pred_event_{event_id:04d}.png"
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print(f"Heatmap        : {out}")


def predict_all(device=None) -> pd.DataFrame:
    device = device or torch.device("cpu")
    ev = pd.read_csv(EVENTS_CSV)
    rows = []
    for eid in ev["event_id"].astype(int):
        try:
            r = predict_event(eid, plot=False, device=device)
            rows.append({"event_id": eid, "pred_max_edr": r["max_edr"], "severity": r["severity"]})
        except FileNotFoundError:
            continue
    df = pd.DataFrame(rows)
    out = MODELS_DIR / "cat_cnn_torch_all_predictions.csv"
    df.to_csv(out, index=False)
    print(f"\n{len(df)} predictions saved → {out}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nc_file", nargs="?", help="ERA5 NetCDF path (uses --time for conditioning)")
    ap.add_argument("--event-id", type=int)
    ap.add_argument("--time", help="ISO timestamp for climate/time conditioning (nc_file mode)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if args.all:
        predict_all()
    elif args.event_id is not None:
        predict_event(args.event_id, plot=not args.no_plot)
    elif args.nc_file:
        ts = pd.Timestamp(args.time) if args.time else pd.Timestamp("2015-01-01")
        model_dev = torch.device("cpu")
        model, ckpt = _load_model(model_dev)
        stats = load_norm()
        eid = int(Path(args.nc_file).stem.split("_")[-1]) if "event_" in args.nc_file else None
        batch = _assemble(eid, ts, stats, ckpt["cfg"], int(ckpt["cfg"].get("grid_size", 24)),
                          ckpt["cfg"].get("primary_levels", [225, 250, 300]), era5_path=args.nc_file)
        with torch.no_grad():
            out = model(batch)
        edr = float(out["max_hat"].item())
        print(f"File           : {args.nc_file}")
        print(f"Estimated EDR  : {edr:.3f}   Severity: {severity(edr)}")
        if not args.no_plot and eid is not None:
            _plot(out["field"][0, 0].numpy(), eid, edr)
    else:
        ap.print_help(); sys.exit(1)


if __name__ == "__main__":
    main()
