#!/usr/bin/env python3
"""
Train CatCNN on ERA5 + satellite data to predict CAT turbulence intensity (0–1).

Input features
--------------
  ERA5   : 6 atmospheric variables at 3 jet-stream pressure levels
            (225 hPa, 250 hPa, 300 hPa) → 18 channels × 24×24 spatial grid
            (each event grid is resized to 24×24 with bilinear interpolation)
  Satellite (optional, zero-padded when missing):
            tb_min, tb_max, tb_mean, tb_std from GOES-16/18 ABI Band 13
            — aggregated to event-mean (4 scalars, concatenated after GAP)

Target
------
  max_edr normalised to [0, 1] by dividing by MAX_EDR = 0.95 m^(2/3)s^(-1)
  (the maximum observed EDR in the 150-event dataset).

Output
------
  models/cat_cnn.npz       — trained weights
  models/cat_cnn_norm.npz  — per-channel normalisation (mean, std)
  models/cat_cnn_eval.csv  — per-event predictions vs ground truth
"""

import warnings
warnings.filterwarnings("ignore")
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import zoom

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from src.models.cnn_cat import CatCNN

# ─── Constants ────────────────────────────────────────────────────────────────

EVENTS_CSV = Path("events.csv")
ERA5_DIR   = Path("data/era5")
SAT_DIR    = Path("data/satellite")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

GRID_SIZE  = 24                         # all ERA5 grids resized to GRID_SIZE × GRID_SIZE
ERA5_VARS  = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "geopotential",
    "vertical_velocity",
    "specific_humidity",
]
USE_LEVELS = [225, 250, 300]            # hPa — jet-stream / upper-troposphere
N_ERA5_CH  = len(ERA5_VARS) * len(USE_LEVELS)   # 18 channels
SAT_DIM    = 4                                   # tb_min, tb_max, tb_mean, tb_std

MAX_EDR    = 0.95                       # dataset-wide max EDR for normalisation

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_era5(event_id: int) -> np.ndarray | None:
    """
    Load ERA5 for one event → (N_ERA5_CH, GRID_SIZE, GRID_SIZE) float32.
    Time-averaged, then bilinearly resized to GRID_SIZE × GRID_SIZE.
    Returns None if the file is missing or unreadable.
    """
    path = ERA5_DIR / f"event_{event_id:04d}.nc"
    if not path.exists():
        return None
    try:
        ds = xr.open_dataset(path)
        avail = ds.level.values.tolist()

        channels = []
        for var in ERA5_VARS:
            for lv in USE_LEVELS:
                # Find nearest available level
                lv_use = min(avail, key=lambda x: abs(x - lv))
                field = ds[var].sel(level=lv_use).mean(dim="time").values.astype(np.float32)
                # Resize to GRID_SIZE × GRID_SIZE
                H, W = field.shape
                zy = GRID_SIZE / H
                zx = GRID_SIZE / W
                field_r = zoom(field, (zy, zx), order=1)  # bilinear
                channels.append(field_r)

        return np.stack(channels, axis=0)  # (C, GRID_SIZE, GRID_SIZE)
    except Exception as exc:
        log.warning("Event %04d ERA5 load failed: %s", event_id, exc)
        return None


def load_satellite(event_id: int) -> np.ndarray:
    """
    Load satellite brightness temperature statistics for one event.
    Returns (4,) float32 array [tb_min, tb_max, tb_mean, tb_std],
    or zeros if satellite data is unavailable.
    """
    path = SAT_DIR / f"event_{event_id:04d}.parquet"
    if not path.exists():
        return np.zeros(SAT_DIM, dtype=np.float32)
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return np.zeros(SAT_DIM, dtype=np.float32)
        return np.array(
            [df["tb_min"].mean(), df["tb_max"].mean(),
             df["tb_mean"].mean(), df["tb_std"].mean()],
            dtype=np.float32,
        )
    except Exception:
        return np.zeros(SAT_DIM, dtype=np.float32)


def build_dataset():
    """
    Build (X_era5, X_sat, y, event_ids) from all events.
    X_era5 : (N, C, H, W)  ERA5 spatial multi-channel fields
    X_sat  : (N, 4)         satellite TB stats (zeros where missing)
    y      : (N, 1)         normalised max_edr ∈ [0, 1]
    """
    events = pd.read_csv(EVENTS_CSV)
    X_era5_list, X_sat_list, y_list, eids = [], [], [], []

    for _, row in events.iterrows():
        eid = int(row["event_id"])
        x_era5 = load_era5(eid)
        if x_era5 is None:
            log.debug("Skipping event %04d — no ERA5 data", eid)
            continue
        x_sat = load_satellite(eid)
        y_val = float(row["max_edr"]) / MAX_EDR
        y_val = min(y_val, 1.0)  # clip to [0, 1] in case of extreme values

        X_era5_list.append(x_era5)
        X_sat_list.append(x_sat)
        y_list.append(y_val)
        eids.append(eid)

    X_era5 = np.stack(X_era5_list).astype(np.float32)   # (N, C, H, W)
    X_sat  = np.stack(X_sat_list).astype(np.float32)    # (N, 4)
    y      = np.array(y_list, dtype=np.float32)[:, None] # (N, 1)

    log.info(
        "Dataset built: %d events  |  ERA5 shape %s  |  y ∈ [%.3f, %.3f]",
        len(eids), X_era5.shape, float(y.min()), float(y.max()),
    )
    edr_bins = pd.read_csv(EVENTS_CSV).set_index("event_id")["edr_bin"]
    bin_counts = {b: sum(edr_bins.get(e, "?") == b for e in eids)
                  for b in ["smooth", "light", "moderate", "severe"]}
    log.info("Bin breakdown: %s", bin_counts)
    return X_era5, X_sat, y, eids


def normalise(X_train, X_val=None, X_test=None):
    """Per-channel z-normalisation fitted on training set only."""
    mu  = X_train.mean(axis=(0, 2, 3), keepdims=True)
    sig = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    out = [(X_train - mu) / sig]
    for X in (X_val, X_test):
        if X is not None:
            out.append((X - mu) / sig)
    out += [mu, sig]
    return out


def normalise_sat(X_sat_train, X_sat_val=None):
    """Normalise satellite TB features (ignore zero rows — missing data)."""
    mask = (X_sat_train != 0).any(axis=1)
    if mask.sum() == 0:
        return (X_sat_train, X_sat_val, np.zeros((1, 4)), np.ones((1, 4)))
    mu  = X_sat_train[mask].mean(axis=0, keepdims=True)
    sig = X_sat_train[mask].std(axis=0, keepdims=True) + 1e-8
    def _norm(X):
        if X is None:
            return None
        present = (X != 0).any(axis=1)
        Xn = X.copy()
        Xn[present] = (Xn[present] - mu) / sig
        return Xn
    return _norm(X_sat_train), _norm(X_sat_val), mu, sig


# ─── Training ─────────────────────────────────────────────────────────────────

def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def train():
    np.random.seed(42)

    log.info("Loading dataset...")
    X_era5, X_sat, y, eids = build_dataset()
    N = len(eids)

    # Train / val split (80 / 20), stratified by edr_bin
    events_df = pd.read_csv(EVENTS_CSV).set_index("event_id")
    bins = np.array([events_df.at[e, "edr_bin"] for e in eids])
    val_mask = np.zeros(N, dtype=bool)
    for b in np.unique(bins):
        idx_b = np.where(bins == b)[0]
        n_val_b = max(1, len(idx_b) // 5)
        np.random.shuffle(idx_b)
        val_mask[idx_b[:n_val_b]] = True

    train_mask = ~val_mask
    X_tr, X_va = X_era5[train_mask], X_era5[val_mask]
    S_tr, S_va = X_sat[train_mask],  X_sat[val_mask]
    y_tr, y_va = y[train_mask],      y[val_mask]

    log.info("Train: %d  |  Val: %d", train_mask.sum(), val_mask.sum())

    # Normalise
    X_tr_n, X_va_n, era5_mu, era5_sig = normalise(X_tr, X_va)
    S_tr_n, S_va_n, sat_mu, sat_sig   = normalise_sat(S_tr, S_va)

    # Build model
    model = CatCNN(in_channels=N_ERA5_CH)

    # Hyper-parameters
    EPOCHS       = 150
    LR           = 1e-3
    BATCH        = 16
    WEIGHT_DECAY = 1e-4
    m_state, v_state = {}, {}
    t = 0
    best_val = float("inf")
    best_path = str(MODELS_DIR / "cat_cnn_best.npz")

    log.info("Training %d epochs  |  LR=%.4f  |  batch=%d  |  wd=%.0e",
             EPOCHS, LR, BATCH, WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        perm = np.random.permutation(len(X_tr_n))
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, len(X_tr_n), BATCH):
            bi = perm[start : start + BATCH]
            Xb = X_tr_n[bi]
            yb = y_tr[bi]

            t += 1
            pred  = model.forward(Xb)
            epoch_loss += mse_loss(pred, yb)
            n_batches  += 1

            dloss = 2.0 * (pred - yb) / len(bi)
            model.backward(dloss)
            model.update(LR, m_state, v_state, t, weight_decay=WEIGHT_DECAY)

        # Validation
        val_pred = model.forward(X_va_n)
        val_loss = mse_loss(val_pred, y_va)
        val_rmse = np.sqrt(val_loss) * MAX_EDR  # back to EDR units

        if val_loss < best_val:
            best_val = val_loss
            model.save(best_path)

        if epoch % 15 == 0 or epoch in (1, 5):
            log.info(
                "Epoch %3d/%d  train=%.5f  val=%.5f  val_RMSE=%.3f EDR  best=%.5f",
                epoch, EPOCHS, epoch_loss / n_batches, val_loss, val_rmse, best_val,
            )

    log.info("Training complete. Best val MSE=%.5f  (RMSE≈%.3f EDR)",
             best_val, np.sqrt(best_val) * MAX_EDR)

    # Load best weights for final save
    model.load(best_path)

    # ── Save artefacts ────────────────────────────────────────────────────────
    model.save(str(MODELS_DIR / "cat_cnn.npz"))
    np.savez(
        str(MODELS_DIR / "cat_cnn_norm.npz"),
        era5_mu=era5_mu, era5_sig=era5_sig,
        sat_mu=sat_mu,   sat_sig=sat_sig,
    )

    # ── Evaluation on full dataset ────────────────────────────────────────────
    X_all_n = (X_era5 - era5_mu) / era5_sig
    all_pred = model.predict(X_all_n)

    eval_df = pd.DataFrame({
        "event_id":   eids,
        "split":      ["val" if val_mask[i] else "train" for i in range(N)],
        "edr_bin":    [bins[i] for i in range(N)],
        "true_max_edr":    y.ravel() * MAX_EDR,
        "pred_intensity":  all_pred,
        "pred_edr_est":    all_pred * MAX_EDR,
    })
    eval_path = MODELS_DIR / "cat_cnn_eval.csv"
    eval_df.to_csv(eval_path, index=False)
    log.info("Evaluation saved → %s", eval_path)

    val_rows = eval_df[eval_df["split"] == "val"]
    log.info(
        "Val set  |  n=%d  |  Pearson r=%.3f  |  MAE=%.3f EDR",
        len(val_rows),
        float(val_rows["true_max_edr"].corr(val_rows["pred_edr_est"])),
        float((val_rows["true_max_edr"] - val_rows["pred_edr_est"]).abs().mean()),
    )

    log.info("Models saved to %s/", MODELS_DIR)
    return model, era5_mu, era5_sig


if __name__ == "__main__":
    train()
