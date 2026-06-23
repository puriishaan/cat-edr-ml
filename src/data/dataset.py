"""
Dataset assembly for the physics-informed CAT CNN.

Brings together, per event:
  • diagnostics   — (C, H, W) physics channels from src.features.diagnostics
  • climate       — (4,) ONI, Niño3.4, PDO, QBO at the event month (broadcast channels + FiLM)
  • time          — (4,) cyclic sin/cos of day-of-year and hour-of-day (FiLM conditioning)
  • satellite     — (5,) scalar TB summary + present/absent mask (optional MLP stream)
  • labels        — scalar max_edr / mean_edr from events.csv
  • phys (raw)    — Ri, VWS, TI1 fields, *unnormalised*, for the physics-penalty losses

Normalisation (per-channel z-score, heavy-tailed channels log-compressed first) is fitted on
the TRAIN split only and persisted, exactly like the existing NumPy pipeline.

Climate indices are fetched once from NOAA PSL, cached to data/climate_indices.csv, and the
loader is offline-safe: if neither network nor cache is available it returns zeros and warns,
so the rest of the pipeline still runs (the model just sees a null climate prior).
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

try:                       # torch is optional at import time (e.g. for the XGBoost baseline)
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except Exception:          # pragma: no cover
    _HAS_TORCH = False
    Dataset = object       # type: ignore

from src.features.diagnostics import (
    DEFAULT_PRIMARY_LEVELS,
    LOG_COMPRESS,
    load_or_compute,
)

log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
EVENTS_CSV      = Path("events.csv")
SAT_DIR         = Path("data/satellite")
CLIMATE_CACHE   = Path("data/climate_indices.csv")
NORM_PATH       = Path("models/cat_cnn_torch_norm.npz")

CLIMATE_NAMES   = ["ONI", "Nino34", "PDO", "QBO"]
_PSL = "https://psl.noaa.gov/data/correlation/{}.data"
_PSL_FILES = {"ONI": "oni", "Nino34": "nina34", "PDO": "pdo", "QBO": "qbo"}
_UA = "cat-edr-ml/1.0 (research)"

SAT_FEATURES    = ["tb_cold", "tb_mean", "tb_std", "tb_max", "tb_cooling"]
PHYS_CHANNELS   = ["Ri", "VWS", "TI1"]    # raw fields handed to the physics losses

MAX_EDR         = 0.95                     # observed dataset max (climatological bound)


# ─── Climate indices ──────────────────────────────────────────────────────────

def _parse_psl(text: str) -> pd.Series:
    """PSL '.data' format: header 'startyr endyr', then 'year v1..v12' rows."""
    recs = {}
    for line in text.splitlines():
        t = line.split()
        if len(t) != 13:
            continue
        try:
            yr = int(t[0]); vals = [float(x) for x in t[1:]]
        except ValueError:
            continue
        if not (1900 <= yr <= 2100):
            continue
        for mo, v in enumerate(vals, start=1):
            recs[pd.Timestamp(yr, mo, 1)] = np.nan if v <= -90 else v
    return pd.Series(recs).sort_index()


def load_climate_table(refresh: bool = False) -> pd.DataFrame:
    """Monthly climate-index table indexed by month-start Timestamp.

    Columns = CLIMATE_NAMES. Cached to data/climate_indices.csv. Offline-safe:
    returns whatever can be loaded; missing indices become all-NaN columns.
    """
    if CLIMATE_CACHE.exists() and not refresh:
        df = pd.read_csv(CLIMATE_CACHE, parse_dates=["date"]).set_index("date")
        for c in CLIMATE_NAMES:
            if c not in df.columns:
                df[c] = np.nan
        return df[CLIMATE_NAMES]

    series = {}
    for name, slug in _PSL_FILES.items():
        try:
            req = urllib.request.Request(_PSL.format(slug), headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                txt = r.read().decode("utf-8", "replace")
            s = _parse_psl(txt)
            if s.dropna().empty:
                raise ValueError("no data parsed")
            series[name] = s
            log.info("  climate: fetched %-7s %d months", name, s.notna().sum())
        except Exception as e:
            log.warning("  climate: SKIP %-7s (%s) — will use zeros", name, e)
            series[name] = pd.Series(dtype=float)

    df = pd.DataFrame(series).sort_index()
    if not df.empty:
        CLIMATE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.reset_index(names="date").to_csv(CLIMATE_CACHE, index=False)
        log.info("  climate: cached → %s", CLIMATE_CACHE)
    return df.reindex(columns=CLIMATE_NAMES)


def _climate_vector(ts: pd.Timestamp, table: pd.DataFrame) -> np.ndarray:
    """Index values at the event month (most-recent ≤ month, i.e. forward-filled)."""
    if table.empty:
        return np.zeros(len(CLIMATE_NAMES), dtype=np.float32)
    month = pd.Timestamp(ts.year, ts.month, 1)
    sub = table[table.index <= month]
    row = sub.iloc[-1] if len(sub) else table.iloc[0]
    return row.reindex(CLIMATE_NAMES).fillna(0.0).values.astype(np.float32)


# ─── Cyclic time ──────────────────────────────────────────────────────────────

def cyclic_time(ts: pd.Timestamp) -> np.ndarray:
    """(sin DOY, cos DOY, sin hour, cos hour) — keeps Dec-31/Jan-1 adjacent."""
    doy = ts.dayofyear + ts.hour / 24.0
    hod = ts.hour + ts.minute / 60.0
    a = 2 * np.pi * doy / 365.25
    b = 2 * np.pi * hod / 24.0
    return np.array([np.sin(a), np.cos(a), np.sin(b), np.cos(b)], dtype=np.float32)


# ─── Satellite scalar features ────────────────────────────────────────────────

def satellite_features(event_id: int) -> tuple[np.ndarray, float]:
    """(5,) TB summary features + present-mask (1.0 if real data, else 0.0)."""
    path = SAT_DIR / f"event_{event_id:04d}.parquet"
    zero = np.zeros(len(SAT_FEATURES), dtype=np.float32)
    if not path.exists():
        return zero, 0.0
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return zero, 0.0
        if "scan_time" in df.columns:
            df = df.sort_values("scan_time")
        cooling = float(df["tb_mean"].iloc[-1] - df["tb_mean"].iloc[0]) if len(df) > 1 else 0.0
        feat = np.array([
            float(df["tb_min"].min()),    # coldest cloud top
            float(df["tb_mean"].mean()),
            float(df["tb_std"].mean()),
            float(df["tb_max"].mean()),
            cooling,                       # Δ tb_mean across snapshots (negative ⇒ deepening)
        ], dtype=np.float32)
        return feat, 1.0
    except Exception:
        return zero, 0.0


# ─── Build all per-event arrays ───────────────────────────────────────────────

class RawSamples:
    """In-memory bundle of all per-event arrays (pre-normalisation)."""

    def __init__(self, diag, climate, time, sat, sat_mask, phys,
                 y_max, y_mean, eids, bins, times, names):
        self.diag = diag          # (N, C, H, W)
        self.climate = climate    # (N, 4)
        self.time = time          # (N, 4)
        self.sat = sat            # (N, 5)
        self.sat_mask = sat_mask  # (N,)
        self.phys = phys          # (N, len(PHYS_CHANNELS), H, W) raw
        self.y_max = y_max        # (N,)
        self.y_mean = y_mean      # (N,)
        self.eids = eids          # (N,)
        self.bins = bins          # (N,) str
        self.times = times        # (N,) pd.Timestamp
        self.names = names        # list[str] diagnostic channel names

    def __len__(self):
        return len(self.eids)


def build_raw_samples(
    events_csv: str | Path = EVENTS_CSV,
    era5_dir: str | Path = "data/era5",
    diag_dir: str | Path = "data/diagnostics",
    primary_levels=DEFAULT_PRIMARY_LEVELS,
    grid_size: int = 24,
) -> RawSamples:
    events = pd.read_csv(events_csv)
    climate_table = load_climate_table()

    diag_l, clim_l, time_l, sat_l, mask_l, phys_l = [], [], [], [], [], []
    ymax_l, ymean_l, eid_l, bin_l, time_idx = [], [], [], [], []
    names: list[str] = []

    # locate raw phys channels once names are known
    phys_idx = None

    for _, row in events.iterrows():
        eid = int(row["event_id"])
        X, ch_names = load_or_compute(eid, era5_dir, diag_dir, primary_levels, grid_size)
        if X is None:
            continue
        if not names:
            names = ch_names
            phys_idx = [_first_level_index(p, names) for p in PHYS_CHANNELS]

        ts = pd.to_datetime(row["start_utc"], utc=True).tz_convert("UTC").tz_localize(None)
        sat_feat, mask = satellite_features(eid)

        diag_l.append(X)
        phys_l.append(X[phys_idx].copy())           # raw (pre-norm) Ri/VWS/TI1
        clim_l.append(_climate_vector(ts, climate_table))
        time_l.append(cyclic_time(ts))
        sat_l.append(sat_feat)
        mask_l.append(mask)
        ymax_l.append(min(float(row["max_edr"]), MAX_EDR))
        ymean_l.append(float(row["mean_edr"]))
        eid_l.append(eid)
        bin_l.append(str(row["edr_bin"]))
        time_idx.append(ts)

    if not diag_l:
        raise RuntimeError(
            "No events with diagnostics found. Pull ERA5 (scripts/step3...) and run "
            "`python -m src.features.diagnostics` first."
        )

    rs = RawSamples(
        diag=np.stack(diag_l).astype(np.float32),
        climate=np.stack(clim_l).astype(np.float32),
        time=np.stack(time_l).astype(np.float32),
        sat=np.stack(sat_l).astype(np.float32),
        sat_mask=np.array(mask_l, dtype=np.float32),
        phys=np.stack(phys_l).astype(np.float32),
        y_max=np.array(ymax_l, dtype=np.float32),
        y_mean=np.array(ymean_l, dtype=np.float32),
        eids=np.array(eid_l, dtype=int),
        bins=np.array(bin_l),
        times=np.array(time_idx),
        names=names,
    )
    log.info("Built %d samples | diag %s | sat present: %d/%d",
             len(rs), rs.diag.shape, int(rs.sat_mask.sum()), len(rs))
    return rs


def _first_level_index(short: str, names: list[str]) -> int:
    for i, n in enumerate(names):
        if n.split("@")[0] == short:
            return i
    raise KeyError(f"diagnostic {short} not in channels {names[:5]}...")


# ─── Normalisation ────────────────────────────────────────────────────────────

def _log_mask(names: list[str]) -> np.ndarray:
    return np.array([n.split("@")[0] in LOG_COMPRESS for n in names], dtype=bool)


def _apply_log(diag: np.ndarray, log_mask: np.ndarray) -> np.ndarray:
    """Signed-log compress heavy-tailed channels in place-safe fashion."""
    out = diag.copy()
    idx = np.where(log_mask)[0]
    out[:, idx] = np.sign(out[:, idx]) * np.log1p(np.abs(out[:, idx]))
    return out


def fit_normalisation(rs: RawSamples, train_idx: np.ndarray) -> dict:
    """Fit per-channel diag stats + climate/sat stats on the training split only."""
    log_mask = _log_mask(rs.names)
    diag_tr = _apply_log(rs.diag[train_idx], log_mask)
    diag_mu = diag_tr.mean(axis=(0, 2, 3), keepdims=True)
    diag_sig = diag_tr.std(axis=(0, 2, 3), keepdims=True) + 1e-6

    clim_tr = rs.climate[train_idx]
    clim_mu = clim_tr.mean(axis=0, keepdims=True)
    clim_sig = clim_tr.std(axis=0, keepdims=True) + 1e-6

    present = rs.sat_mask[train_idx] > 0
    if present.sum() >= 2:
        sat_mu = rs.sat[train_idx][present].mean(axis=0, keepdims=True)
        sat_sig = rs.sat[train_idx][present].std(axis=0, keepdims=True) + 1e-6
    else:
        sat_mu = np.zeros((1, len(SAT_FEATURES)), dtype=np.float32)
        sat_sig = np.ones((1, len(SAT_FEATURES)), dtype=np.float32)

    return dict(
        diag_mu=diag_mu.astype(np.float32), diag_sig=diag_sig.astype(np.float32),
        clim_mu=clim_mu.astype(np.float32), clim_sig=clim_sig.astype(np.float32),
        sat_mu=sat_mu.astype(np.float32), sat_sig=sat_sig.astype(np.float32),
        log_mask=log_mask, names=np.array(rs.names),
    )


def save_norm(stats: dict, path: str | Path = NORM_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **stats)


def load_norm(path: str | Path = NORM_PATH) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


# ─── Splits (group-aware / temporal) ──────────────────────────────────────────

def temporal_holdout(rs: RawSamples, frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Most-recent `frac` of events (by start time) become the held-out test set."""
    order = np.argsort(rs.times)
    n_test = max(1, int(round(frac * len(order))))
    test = np.sort(order[-n_test:])
    trainval = np.sort(order[:-n_test])
    return trainval, test


def make_folds(rs: RawSamples, idx: np.ndarray, n_folds: int = 5, seed: int = 42):
    """Stratified-by-edr_bin K-fold over events. Each event is one sample, so by
    construction no event leaks across folds (the group-by-event guarantee). Cluster-
    level grouping is a future hook if events.csv gains a cluster id."""
    try:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return [(idx[tr], idx[va]) for tr, va in skf.split(idx, rs.bins[idx])]
    except Exception:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(idx)
        return [(np.setdiff1d(idx, va), va) for va in np.array_split(perm, n_folds)]


# ─── Torch Dataset ────────────────────────────────────────────────────────────

if _HAS_TORCH:

    class CatDataset(Dataset):
        """Normalised tensors for one split. `stats` come from fit_normalisation()."""

        def __init__(self, rs: RawSamples, indices: np.ndarray, stats: dict,
                     augment: bool = False, seed: int = 0):
            self.rs = rs
            self.idx = np.asarray(indices)
            self.stats = stats
            self.augment = augment
            self.rng = np.random.default_rng(seed)
            self._log_mask = np.asarray(stats["log_mask"])

        def __len__(self):
            return len(self.idx)

        def _norm_diag(self, X: np.ndarray) -> np.ndarray:
            X = X.copy()
            ix = np.where(self._log_mask)[0]
            X[ix] = np.sign(X[ix]) * np.log1p(np.abs(X[ix]))
            return (X - self.stats["diag_mu"][0]) / self.stats["diag_sig"][0]

        def __getitem__(self, i):
            j = self.idx[i]
            diag = self._norm_diag(self.rs.diag[j])
            climate = (self.rs.climate[j] - self.stats["clim_mu"][0]) / self.stats["clim_sig"][0]
            sat = (self.rs.sat[j] - self.stats["sat_mu"][0]) / self.stats["sat_sig"][0]
            sat = sat * self.rs.sat_mask[j]      # zero-out features when absent
            phys = self.rs.phys[j]               # raw, for physics losses

            if self.augment:
                diag, phys = self._augment(diag, phys)

            return {
                "diag": torch.from_numpy(np.ascontiguousarray(diag)).float(),
                "climate": torch.from_numpy(climate).float(),
                "time": torch.from_numpy(self.rs.time[j]).float(),
                "sat": torch.from_numpy(sat).float(),
                "sat_mask": torch.tensor([self.rs.sat_mask[j]]).float(),
                "phys": torch.from_numpy(np.ascontiguousarray(phys)).float(),
                "y_max": torch.tensor([self.rs.y_max[j]]).float(),
                "y_mean": torch.tensor([self.rs.y_mean[j]]).float(),
                "eid": int(self.rs.eids[j]),
            }

        def _augment(self, diag, phys):
            """Flips only (90° rotations would require rotating wind vectors; the
            diagnostics here are scalar invariants, so axis flips are safe)."""
            if self.rng.random() < 0.5:
                diag = diag[:, :, ::-1]; phys = phys[:, :, ::-1]
            if self.rng.random() < 0.5:
                diag = diag[:, ::-1, :]; phys = phys[:, ::-1, :]
            return diag, phys
