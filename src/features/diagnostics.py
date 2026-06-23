"""
CAT physics diagnostics from ERA5 — the input channels for the physics-informed CNN.

Rather than feeding the network raw (u, v, T, z, ω, q) and asking it to rediscover
shear and deformation, we hand it the established operational turbulence diagnostics
(the GTG family) computed per pressure level. The CNN then only has to learn how these
combine into observed EDR, not re-derive decades of dynamical meteorology.

Per primary pressure level we compute 12 channels:

    VWS     vertical wind shear        sqrt((du/dz)^2 + (dv/dz)^2)        [1/s]
    N2      Brunt–Vaisala freq^2       (g/theta) d(theta)/dz             [1/s^2]
    Ri      Richardson number          N2 / VWS^2     (KH when < 0.25)   [-]
    DEF     total deformation          sqrt(DST^2 + DSH^2)               [1/s]
    DIV     horizontal divergence      du/dx + dv/dy                     [1/s]
    TI1     Ellrod index 1             VWS * DEF                          [1/s^2]
    TI2     Ellrod index 2             VWS * (DEF - DIV)                  [1/s^2]
    VORT    relative vorticity         dv/dx - du/dy                     [1/s]
    FRONTO  Petterssen frontogenesis   d|grad theta|/dt (kinematic)      [K/m/s]
    WSPD    wind speed (jet proxy)     sqrt(u^2 + v^2)                    [m/s]
    OMEGA   vertical velocity          (raw ERA5 ω)                       [Pa/s]
    VADV    vorticity advection (NVA)  -(u dζ/dx + v dζ/dy)              [1/s^2]

Vertical derivatives use the geometric height z = geopotential / g0 between the level
above and below each primary level (central difference; one-sided at the edges).
Horizontal derivatives use metric grid spacing on the native ERA5 lat/lon grid; the
event boxes are a few degrees wide so a mean-latitude cos(lat) factor for dx is accurate.

Implemented in NumPy (robust, dependency-light, easy to review). MetPy could replace the
kinematic pieces 1:1; the formulas above are the operational definitions it would use.

CLI
---
    python -m src.features.diagnostics --events events.csv \
        --era5 data/era5 --out data/diagnostics --grid 24

Programmatic
------------
    from src.features.diagnostics import load_or_compute, CHANNEL_NAMES
    X, names = load_or_compute(event_id=0)        # (C, grid, grid), list[str]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import zoom

log = logging.getLogger(__name__)

# ─── Physical constants ───────────────────────────────────────────────────────
G0     = 9.80665        # m/s^2   standard gravity (geopotential → height)
R_EARTH = 6_371_000.0   # m       mean Earth radius
RD_CP  = 0.286          # Rd/cp   (≈ 2/7) for potential temperature
P0     = 1000.0         # hPa     reference pressure for theta

# ─── Channel definition ───────────────────────────────────────────────────────
# Per-level diagnostic short names, in fixed order. The full channel list is the
# cartesian product (level × diagnostic), built in compute_event_diagnostics().
DIAGNOSTICS = [
    "VWS", "N2", "Ri", "DEF", "DIV", "TI1", "TI2",
    "VORT", "FRONTO", "WSPD", "OMEGA", "VADV",
]

# Default primary levels (hPa) — jet-stream / upper-troposphere band. Diagnostics
# are computed at each; flanking available levels supply the vertical derivatives.
DEFAULT_PRIMARY_LEVELS = [225, 250, 300]

# Heavy-tailed channels that the dataset layer should log-compress before z-scoring.
LOG_COMPRESS = {"VWS", "DEF", "TI1", "TI2", "WSPD"}

ERA5_VARS = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "geopotential",
    "vertical_velocity",
    "specific_humidity",
]

# Built lazily once primary levels are known (so predict/train share ordering).
def channel_names(primary_levels=DEFAULT_PRIMARY_LEVELS) -> list[str]:
    return [f"{d}@{lv}" for lv in primary_levels for d in DIAGNOSTICS]


CHANNEL_NAMES = channel_names()


# ─── Horizontal / vertical gradient helpers ───────────────────────────────────

def _hgrad(f: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """∂f/∂x, ∂f/∂y in physical units (per metre) on a lat/lon grid.

    f    : (ny, nx)
    lat  : (ny,)  degrees (may descend, as in ERA5)
    lon  : (nx,)  degrees
    Uses metric coordinates; dx uses cos(mean latitude) (boxes are small).
    """
    latr = np.deg2rad(lat)
    lonr = np.deg2rad(lon)
    y = R_EARTH * latr                                  # (ny,)
    x = R_EARTH * np.cos(latr.mean()) * lonr            # (nx,)
    # np.gradient honours non-uniform / descending coordinate arrays and their sign.
    dfdy = np.gradient(f, y, axis=0, edge_order=1)
    dfdx = np.gradient(f, x, axis=1, edge_order=1)
    return dfdx, dfdy


def _brackets(p: float, avail: list[float]) -> tuple[float, float]:
    """Return (p_below, p_above) bracketing pressure level p for a vertical derivative.

    'below' = higher pressure (lower altitude), 'above' = lower pressure (higher altitude).
    Falls back to one-sided (returns p itself on the missing side)."""
    higher = [q for q in avail if q > p]   # higher pressure → below in altitude
    lower  = [q for q in avail if q < p]   # lower pressure  → above in altitude
    p_below = min(higher) if higher else p
    p_above = max(lower)  if lower else p
    if p_below == p_above:                 # p is an interior duplicate / single level
        # use nearest distinct neighbour on either side
        others = [q for q in avail if q != p]
        if others:
            nb = min(others, key=lambda q: abs(q - p))
            p_below, p_above = (max(p, nb), min(p, nb))
    return p_below, p_above


# ─── Core computation ─────────────────────────────────────────────────────────

def compute_event_diagnostics(
    ds: xr.Dataset,
    primary_levels=DEFAULT_PRIMARY_LEVELS,
    grid_size: int = 24,
) -> tuple[np.ndarray, list[str]]:
    """ERA5 event Dataset → (C, grid_size, grid_size) diagnostics + channel names.

    Fields are time-averaged over the event window, diagnostics are computed on the
    native grid, then each channel is bilinearly resized to grid_size × grid_size.
    """
    avail = [float(v) for v in np.atleast_1d(ds.level.values)]
    lat = ds.latitude.values.astype(np.float64)
    lon = ds.longitude.values.astype(np.float64)

    # Time-mean every variable, keep (level, y, x). Missing vars → zeros.
    def field(var: str, lv: float) -> np.ndarray:
        lv_use = min(avail, key=lambda q: abs(q - lv))
        da = ds[var].sel(level=lv_use)
        if "time" in da.dims:
            da = da.mean(dim="time")
        return da.values.astype(np.float64)

    channels: list[np.ndarray] = []
    names: list[str] = []

    for lv in primary_levels:
        p_below, p_above = _brackets(float(lv), avail)

        U = field("u_component_of_wind", lv)
        V = field("v_component_of_wind", lv)
        T = field("temperature", lv)
        OMEGA = field("vertical_velocity", lv)

        # ── horizontal kinematics ────────────────────────────────────────────
        ux, uy = _hgrad(U, lat, lon)
        vx, vy = _hgrad(V, lat, lon)
        DST = ux - vy                       # stretching deformation
        DSH = vx + uy                       # shearing deformation
        DEF = np.hypot(DST, DSH)            # total deformation
        DIV = ux + vy                       # divergence
        VORT = vx - uy                      # relative vorticity
        WSPD = np.hypot(U, V)

        # potential temperature & its gradient for frontogenesis
        theta = T * (P0 / float(lv)) ** RD_CP
        thx, thy = _hgrad(theta, lat, lon)
        gradmag = np.hypot(thx, thy) + 1e-12
        # Petterssen kinematic frontogenesis: rate of change of |∇θ| by deformation/divergence
        FRONTO = -(thx * thx * ux + thx * thy * (uy + vx) + thy * thy * vy) / gradmag

        # vorticity advection (negative-VA CAT proxy)
        zx, zy = _hgrad(VORT, lat, lon)
        VADV = -(U * zx + V * zy)

        # ── vertical derivatives (shear, stratification, Richardson) ─────────
        Ua, Ub = field("u_component_of_wind", p_above), field("u_component_of_wind", p_below)
        Va, Vb = field("v_component_of_wind", p_above), field("v_component_of_wind", p_below)
        Ta, Tb = field("temperature", p_above), field("temperature", p_below)
        Za = field("geopotential", p_above) / G0      # geometric height (m)
        Zb = field("geopotential", p_below) / G0
        dz = (Za - Zb)
        dz = np.where(np.abs(dz) < 1.0, np.sign(dz + 1e-9) * 1.0, dz)  # guard tiny/zero

        dudz = (Ua - Ub) / dz
        dvdz = (Va - Vb) / dz
        VWS = np.hypot(dudz, dvdz)

        theta_a = Ta * (P0 / p_above) ** RD_CP
        theta_b = Tb * (P0 / p_below) ** RD_CP
        theta_mid = 0.5 * (theta_a + theta_b) + 1e-6
        N2 = (G0 / theta_mid) * (theta_a - theta_b) / dz

        Ri = N2 / (VWS ** 2 + 1e-10)
        Ri = np.clip(Ri, -5.0, 50.0)        # tame the 1/shear^2 tail for a usable channel

        # Ellrod turbulence indices
        TI1 = VWS * DEF
        TI2 = VWS * (DEF - DIV)

        per_level = {
            "VWS": VWS, "N2": N2, "Ri": Ri, "DEF": DEF, "DIV": DIV,
            "TI1": TI1, "TI2": TI2, "VORT": VORT, "FRONTO": FRONTO,
            "WSPD": WSPD, "OMEGA": OMEGA, "VADV": VADV,
        }
        for d in DIAGNOSTICS:
            arr = np.nan_to_num(per_level[d], nan=0.0, posinf=0.0, neginf=0.0)
            channels.append(_resize(arr, grid_size))
            names.append(f"{d}@{lv}")

    X = np.stack(channels, axis=0).astype(np.float32)   # (C, grid, grid)
    return X, names


def _resize(field: np.ndarray, grid_size: int) -> np.ndarray:
    """Bilinear resize a 2-D field to grid_size × grid_size."""
    H, W = field.shape
    if (H, W) == (grid_size, grid_size):
        return field.astype(np.float32)
    return zoom(field, (grid_size / H, grid_size / W), order=1).astype(np.float32)


# ─── Cache layer ──────────────────────────────────────────────────────────────

def cache_path(event_id: int, out_dir: Path) -> Path:
    return out_dir / f"event_{event_id:04d}.npz"


def load_or_compute(
    event_id: int,
    era5_dir: str | Path = "data/era5",
    out_dir: str | Path = "data/diagnostics",
    primary_levels=DEFAULT_PRIMARY_LEVELS,
    grid_size: int = 24,
    recompute: bool = False,
) -> tuple[np.ndarray | None, list[str]]:
    """Return cached diagnostics for an event, computing + caching on first access.

    Returns (None, names) if the ERA5 file is missing/unreadable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_path(event_id, out_dir)

    if cp.exists() and not recompute:
        d = np.load(cp, allow_pickle=True)
        return d["data"].astype(np.float32), list(d["names"])

    nc = Path(era5_dir) / f"event_{event_id:04d}.nc"
    if not nc.exists():
        return None, channel_names(primary_levels)
    try:
        ds = xr.open_dataset(nc)
    except Exception as exc:  # pragma: no cover - I/O guard
        log.warning("Event %04d ERA5 unreadable: %s", event_id, exc)
        return None, channel_names(primary_levels)

    X, names = compute_event_diagnostics(ds, primary_levels, grid_size)
    ds.close()
    np.savez_compressed(cp, data=X, names=np.array(names))
    return X, names


def channel_index(name: str, names: list[str], level=None) -> int | None:
    """Index of a diagnostic by short name (e.g. 'Ri'); first matching level unless
    `level` is given. Used by the physics losses to pull Ri / VWS / TI1 fields."""
    key = name if level is None else f"{name}@{level}"
    for i, n in enumerate(names):
        if (level is None and n.split("@")[0] == name) or n == key:
            return i
    return None


# ─── CLI: precompute all events ───────────────────────────────────────────────

def main():
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Precompute ERA5 CAT diagnostics → cached npz")
    ap.add_argument("--events", default="events.csv")
    ap.add_argument("--era5", default="data/era5")
    ap.add_argument("--out", default="data/diagnostics")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--levels", type=int, nargs="+", default=DEFAULT_PRIMARY_LEVELS)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    events = pd.read_csv(args.events)
    ok = miss = 0
    for eid in events["event_id"].astype(int):
        X, _ = load_or_compute(eid, args.era5, args.out, args.levels, args.grid, args.recompute)
        if X is None:
            miss += 1
            continue
        ok += 1
        if ok % 25 == 0:
            log.info("  %d events done (last shape %s)", ok, X.shape)
    log.info("Diagnostics cached for %d events (%d missing ERA5) → %s/", ok, miss, args.out)


if __name__ == "__main__":
    main()
