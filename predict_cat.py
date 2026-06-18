#!/usr/bin/env python3
"""
Predict CAT turbulence intensity for an ERA5 event file.

Usage
-----
  # Single file
  python predict_cat.py data/era5/event_0000.nc

  # By event ID (looks up data/era5/event_XXXX.nc)
  python predict_cat.py --event-id 42

  # Run on all events and save summary
  python predict_cat.py --all

Output: turbulence intensity in [0, 1] and estimated EDR in m^(2/3)s^(-1).

Severity mapping (FAA EDR thresholds):
  0.00–0.10  → SMOOTH
  0.10–0.21  → LIGHT
  0.21–0.42  → MODERATE
  0.42–1.00  → SEVERE
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).parent))
from src.models.cnn_cat import CatCNN

# ─── Constants (must match train_cnn.py) ──────────────────────────────────────

ERA5_VARS  = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "geopotential",
    "vertical_velocity",
    "specific_humidity",
]
USE_LEVELS  = [225, 250, 300]
GRID_SIZE   = 24
N_ERA5_CH   = len(ERA5_VARS) * len(USE_LEVELS)
MAX_EDR     = 0.95

MODELS_DIR  = Path("models")
ERA5_DIR    = Path("data/era5")
EVENTS_CSV  = Path("events.csv")


# ─── Severity label ───────────────────────────────────────────────────────────

def severity(intensity: float) -> str:
    edr = intensity * MAX_EDR
    if edr >= 0.40:
        return "SEVERE"
    if edr >= 0.20:
        return "MODERATE"
    if edr >= 0.10:
        return "LIGHT"
    return "SMOOTH"


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_era5_file(nc_path: str | Path) -> np.ndarray:
    ds = xr.open_dataset(nc_path)
    avail = ds.level.values.tolist()
    channels = []
    for var in ERA5_VARS:
        for lv in USE_LEVELS:
            lv_use = min(avail, key=lambda x: abs(x - lv))
            field = ds[var].sel(level=lv_use).mean(dim="time").values.astype(np.float32)
            H, W = field.shape
            channels.append(zoom(field, (GRID_SIZE / H, GRID_SIZE / W), order=1))
    return np.stack(channels, axis=0)  # (C, GRID_SIZE, GRID_SIZE)


def _load_norm():
    norm_path = MODELS_DIR / "cat_cnn_norm.npz"
    if not norm_path.exists():
        raise FileNotFoundError(
            f"Normalisation file not found at {norm_path}. "
            "Run train_cnn.py first."
        )
    return np.load(norm_path)


def _load_model() -> CatCNN:
    model_path = MODELS_DIR / "cat_cnn.npz"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. "
            "Run train_cnn.py first."
        )
    model = CatCNN(in_channels=N_ERA5_CH)
    model.load(str(model_path))
    return model


# ─── Inference ────────────────────────────────────────────────────────────────

def predict_file(nc_path: str | Path) -> float:
    """
    Predict turbulence intensity for a single ERA5 NetCDF file.
    Returns intensity ∈ [0, 1].
    """
    nc_path = Path(nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(nc_path)

    x = _load_era5_file(nc_path)[None]  # (1, C, H, W)

    norm  = _load_norm()
    mu    = norm["era5_mu"]
    sig   = norm["era5_sig"]
    x_n   = (x - mu) / (sig + 1e-8)

    model     = _load_model()
    intensity = float(model.predict(x_n)[0])

    edr_est = intensity * MAX_EDR
    sev     = severity(intensity)

    print(f"File           : {nc_path.name}")
    print(f"Intensity      : {intensity:.4f}  (0 = calm, 1 = max observed)")
    print(f"Estimated EDR  : {edr_est:.3f} m²³s⁻¹")
    print(f"Severity       : {sev}")

    return intensity


def predict_event(event_id: int) -> float:
    nc_path = ERA5_DIR / f"event_{event_id:04d}.nc"
    return predict_file(nc_path)


def predict_all() -> pd.DataFrame:
    """Run inference on every ERA5 event file and return a summary DataFrame."""
    norm  = _load_norm()
    mu    = norm["era5_mu"]
    sig   = norm["era5_sig"]
    model = _load_model()

    events_df = pd.read_csv(EVENTS_CSV) if EVENTS_CSV.exists() else None

    rows = []
    for nc_path in sorted(ERA5_DIR.glob("event_*.nc")):
        eid = int(nc_path.stem.split("_")[1])
        try:
            x   = _load_era5_file(nc_path)[None]
            x_n = (x - mu) / (sig + 1e-8)
            intensity = float(model.predict(x_n)[0])
            edr_est   = intensity * MAX_EDR
        except Exception as e:
            print(f"  [skip] event {eid:04d}: {e}")
            continue

        row = {"event_id": eid, "intensity": intensity, "edr_est": edr_est,
               "severity": severity(intensity)}
        if events_df is not None and eid in events_df["event_id"].values:
            meta = events_df[events_df["event_id"] == eid].iloc[0]
            row["true_max_edr"] = float(meta["max_edr"])
            row["edr_bin"]      = meta["edr_bin"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("event_id").reset_index(drop=True)
    out = MODELS_DIR / "cat_cnn_all_predictions.csv"
    df.to_csv(out, index=False)
    print(f"\nPredictions for {len(df)} events saved → {out}")
    print(df[["event_id", "edr_bin", "true_max_edr", "intensity", "edr_est", "severity"]].to_string(index=False))
    return df


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nc_file", nargs="?", help="Path to ERA5 NetCDF file")
    ap.add_argument("--event-id", type=int, help="Event ID (uses data/era5/event_XXXX.nc)")
    ap.add_argument("--all", action="store_true", help="Run on all ERA5 event files")
    args = ap.parse_args()

    if args.all:
        predict_all()
    elif args.event_id is not None:
        predict_event(args.event_id)
    elif args.nc_file:
        predict_file(args.nc_file)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
