"""
CatDataset — assembles all per-event features for the CAT turbulence CNN.

Feature groups:
  diag     : physics diagnostics from ERA5  (C_diag, H, W)
  climate  : ONI + Nino3.4 + PDO + QBO     (4,)
  time     : sin/cos DOY + sin/cos hour     (4,)
  sat      : satellite TB stats             (5,)
  sat_mask : float 0/1 (satellite present)  (1,)
  phys     : Ri/VWS/TI1 at 250 hPa          (3,)

Targets:
  y_max  : max_edr normalised to [0,1]
  y_mean : mean_edr normalised
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.diagnostics import (
    DEFAULT_PRIMARY_LEVELS,
    load_or_compute,
    channel_index,
    channel_names,
)

log = logging.getLogger(__name__)

EVENTS_CSV = Path("events.csv")
ERA5_DIR   = Path("data/era5")
SAT_DIR    = Path("data/satellite")
DIAG_DIR   = Path("data/diagnostics")

CLIMATE_NAMES  = ["ONI", "Nino34", "PDO", "QBO"]
SAT_FEATURES   = ["tb_cold", "tb_mean", "tb_std", "tb_max", "tb_cooling"]
PHYS_CHANNELS  = ["Ri", "VWS", "TI1"]   # scalar summary per channel at 250
MAX_EDR        = 0.95


# ── Climate indices (NOAA PSL) ────────────────────────────────────────────────

_PSL_URLS = {
    "ONI":    "https://psl.noaa.gov/data/correlation/oni.data",
    "Nino34": "https://psl.noaa.gov/data/correlation/nina34.data",
    "PDO":    "https://psl.noaa.gov/data/correlation/pdo.data",
    "QBO":    "https://psl.noaa.gov/data/correlation/qbo.data",
}
_CLIMATE_CACHE = Path("data/climate_indices.csv")


def _parse_psl(text: str, name: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        year = int(parts[0])
        vals = parts[1:]
        for mo, v in enumerate(vals, 1):
            try:
                fv = float(v)
                if abs(fv) < 90:  # sentinel removal
                    rows.append({"year": year, "month": mo, name: fv})
            except ValueError:
                continue
    return pd.DataFrame(rows).set_index(["year", "month"])


def load_climate_table() -> pd.DataFrame:
    if _CLIMATE_CACHE.exists():
        return pd.read_csv(_CLIMATE_CACHE, index_col=["year", "month"])
    import requests
    dfs = []
    for name, url in _PSL_URLS.items():
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            dfs.append(_parse_psl(r.text, name))
        except Exception as exc:
            log.warning("Failed to download %s: %s — using zeros", name, exc)
            dfs.append(pd.DataFrame(columns=[name]))
    climate = dfs[0]
    for df in dfs[1:]:
        climate = climate.join(df, how="outer")
    climate = climate.fillna(0.0)
    _CLIMATE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    climate.to_csv(_CLIMATE_CACHE)
    return climate


def _climate_vector(dt: pd.Timestamp, table: pd.DataFrame) -> np.ndarray:
    key = (dt.year, dt.month)
    if key in table.index:
        return table.loc[key].values.astype(np.float32)
    return np.zeros(len(CLIMATE_NAMES), dtype=np.float32)


# ── Time features ─────────────────────────────────────────────────────────────

def cyclic_time(dt: pd.Timestamp) -> np.ndarray:
    doy = dt.day_of_year
    hour = dt.hour + dt.minute / 60.0
    return np.array([
        np.sin(2 * np.pi * doy  / 365.25),
        np.cos(2 * np.pi * doy  / 365.25),
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
    ], dtype=np.float32)


# ── Satellite features ────────────────────────────────────────────────────────

def satellite_features(event_id: int) -> tuple[np.ndarray, float]:
    """
    Returns (feat, mask) where feat is (5,) and mask is 1.0 if data present.
    Features: tb_cold (min), tb_mean, tb_std, tb_max, tb_cooling (max-min).
    """
    path = SAT_DIR / f"event_{event_id:04d}.parquet"
    if not path.exists():
        return np.zeros(5, dtype=np.float32), 0.0
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return np.zeros(5, dtype=np.float32), 0.0
        tb_cold   = float(df["tb_min"].min())
        tb_mean   = float(df["tb_mean"].mean())
        tb_std    = float(df["tb_std"].mean())
        tb_max    = float(df["tb_max"].max())
        tb_cooling = tb_max - tb_cold
        return np.array([tb_cold, tb_mean, tb_std, tb_max, tb_cooling], dtype=np.float32), 1.0
    except Exception:
        return np.zeros(5, dtype=np.float32), 0.0


# ── Raw sample container ──────────────────────────────────────────────────────

class RawSample:
    __slots__ = ("event_id", "diag", "climate", "time", "sat", "sat_mask",
                 "phys", "y_max", "y_mean", "edr_bin")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def build_raw_samples(
    primary_levels: list[int] = DEFAULT_PRIMARY_LEVELS,
    grid_size: int = 24,
) -> list[RawSample]:
    events = pd.read_csv(EVENTS_CSV)
    climate_table = load_climate_table()
    names = channel_names(primary_levels)
    phys_indices = [i for i, n in enumerate(names) if any(n.startswith(p + "@") for p in PHYS_CHANNELS)]

    samples = []
    for _, row in events.iterrows():
        eid = int(row["event_id"])
        result = load_or_compute(eid, str(ERA5_DIR), str(DIAG_DIR), primary_levels, grid_size)
        if result is None:
            continue
        diag, _ = result
        dt = pd.Timestamp(row["start_utc"]).tz_convert("UTC") if hasattr(pd.Timestamp(row["start_utc"]), "tz_convert") else pd.Timestamp(row["start_utc"])

        climate = _climate_vector(dt, climate_table)
        time_f  = cyclic_time(dt)
        sat_f, sat_mask = satellite_features(eid)

        # Physics scalars: spatial mean of selected channels
        phys = np.array([diag[i].mean() for i in phys_indices], dtype=np.float32)

        y_max  = min(float(row["max_edr"])  / MAX_EDR, 1.0)
        y_mean = min(float(row["mean_edr"]) / MAX_EDR, 1.0)

        samples.append(RawSample(
            event_id=eid,
            diag=diag,
            climate=climate,
            time=time_f,
            sat=sat_f,
            sat_mask=np.array([sat_mask], dtype=np.float32),
            phys=phys,
            y_max=np.float32(y_max),
            y_mean=np.float32(y_mean),
            edr_bin=str(row["edr_bin"]),
        ))
    log.info("Built %d raw samples", len(samples))
    return samples


# ── Normalisation ─────────────────────────────────────────────────────────────

def _log_mask(X: np.ndarray, channel_names_list: list[str]) -> np.ndarray:
    from src.features.diagnostics import LOG_COMPRESS
    out = X.copy()
    for i, n in enumerate(channel_names_list):
        base = n.split("@")[0]
        if base in LOG_COMPRESS:
            out[:, i] = np.sign(out[:, i]) * np.log1p(np.abs(out[:, i]))
    return out


def _apply_log(x: np.ndarray, channel_names_list: list[str]) -> np.ndarray:
    from src.features.diagnostics import LOG_COMPRESS
    out = x.copy()
    for i, n in enumerate(channel_names_list):
        base = n.split("@")[0]
        if base in LOG_COMPRESS:
            out[i] = np.sign(out[i]) * np.log1p(np.abs(out[i]))
    return out


class NormStats:
    def __init__(self, diag_mu, diag_sig, sat_mu, sat_sig, phys_mu, phys_sig):
        self.diag_mu  = diag_mu
        self.diag_sig = diag_sig
        self.sat_mu   = sat_mu
        self.sat_sig  = sat_sig
        self.phys_mu  = phys_mu
        self.phys_sig = phys_sig


def fit_normalisation(samples: list[RawSample], cnames: list[str]) -> NormStats:
    diags  = np.stack([s.diag for s in samples])  # (N,C,H,W)
    diags  = _log_mask(diags.reshape(len(samples), len(cnames), -1).mean(-1), cnames)
    # per-channel mean/std over spatial mean
    diag_mu  = diags.mean(0)
    diag_sig = diags.std(0).clip(min=1e-6)

    sats = np.stack([s.sat for s in samples if s.sat_mask[0] > 0])
    sat_mu  = sats.mean(0) if len(sats) > 0 else np.zeros(5, dtype=np.float32)
    sat_sig = (sats.std(0).clip(min=1e-6) if len(sats) > 0 else np.ones(5, dtype=np.float32))

    phys = np.stack([s.phys for s in samples])
    phys_mu  = phys.mean(0)
    phys_sig = phys.std(0).clip(min=1e-6)

    return NormStats(diag_mu, diag_sig, sat_mu, sat_sig, phys_mu, phys_sig)


def save_norm(norm: NormStats, path: str) -> None:
    np.savez(path,
             diag_mu=norm.diag_mu, diag_sig=norm.diag_sig,
             sat_mu=norm.sat_mu,   sat_sig=norm.sat_sig,
             phys_mu=norm.phys_mu, phys_sig=norm.phys_sig)


def load_norm(path: str) -> NormStats:
    d = np.load(path)
    return NormStats(d["diag_mu"], d["diag_sig"],
                     d["sat_mu"],  d["sat_sig"],
                     d["phys_mu"], d["phys_sig"])


# ── Train/val splits ──────────────────────────────────────────────────────────

def temporal_holdout(samples: list[RawSample], test_frac: float = 0.15):
    """Chronological holdout — last test_frac of events by event_id."""
    n = len(samples)
    split = int(n * (1 - test_frac))
    return samples[:split], samples[split:]


def make_folds(
    samples: list[RawSample],
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[list[RawSample], list[RawSample]]]:
    """Stratified k-fold by edr_bin, group-safe (each event in exactly one fold)."""
    rng = np.random.default_rng(seed)
    bins = [s.edr_bin for s in samples]
    unique_bins = sorted(set(bins))
    fold_indices: list[list[int]] = [[] for _ in range(n_folds)]
    for b in unique_bins:
        idx = [i for i, s in enumerate(samples) if s.edr_bin == b]
        idx = rng.permutation(idx).tolist()
        for k, i in enumerate(idx):
            fold_indices[k % n_folds].append(i)
    folds = []
    for k in range(n_folds):
        val_idx = set(fold_indices[k])
        tr  = [samples[i] for i in range(len(samples)) if i not in val_idx]
        val = [samples[i] for i in fold_indices[k]]
        folds.append((tr, val))
    return folds


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class CatDataset(Dataset):
    def __init__(
        self,
        samples: list[RawSample],
        norm: NormStats,
        cnames: list[str],
        augment: bool = False,
    ):
        self.samples = samples
        self.norm    = norm
        self.cnames  = cnames
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        n = self.norm

        diag = _apply_log(s.diag, self.cnames).astype(np.float32)  # (C,H,W)
        # per-channel normalise (use spatial mean stats, broadcast over H,W)
        c = diag.shape[0]
        diag = (diag - n.diag_mu[:, None, None]) / n.diag_sig[:, None, None]

        if self.augment and np.random.rand() < 0.5:
            diag = diag[:, :, ::-1].copy()

        sat = s.sat.copy()
        if s.sat_mask[0] > 0:
            sat = (sat - n.sat_mu) / n.sat_sig

        phys = (s.phys - n.phys_mu) / n.phys_sig

        return {
            "diag":     torch.from_numpy(diag),
            "climate":  torch.from_numpy(s.climate),
            "time":     torch.from_numpy(s.time),
            "sat":      torch.from_numpy(sat),
            "sat_mask": torch.from_numpy(s.sat_mask),
            "phys":     torch.from_numpy(phys),
            "y_max":    torch.tensor(s.y_max),
            "y_mean":   torch.tensor(s.y_mean),
            "eid":      torch.tensor(s.event_id),
        }
