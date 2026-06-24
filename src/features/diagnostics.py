"""
Physics-based diagnostic channels computed from ERA5 data.

Produces 12 diagnostics × N pressure levels = N*12 channels per event.
Default: 3 levels (225, 250, 300 hPa) → 36 channels.
"""

import logging
import os
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import zoom

log = logging.getLogger(__name__)

# Physical constants
G0     = 9.80665       # m s^-2
R_EARTH = 6371000.0   # m
RD_CP  = 0.2857        # Rd/Cp (dry adiabatic)
P0     = 100000.0      # Pa reference pressure

DIAGNOSTICS = [
    "VWS",      # vertical wind shear magnitude
    "N2",       # Brunt-Väisälä frequency squared
    "Ri",       # Richardson number
    "DEF",      # horizontal deformation
    "DIV",      # horizontal divergence
    "TI1",      # Ellrod turbulence index 1  (DEF × VWS)
    "TI2",      # Ellrod turbulence index 2  ((DEF + DIV) × VWS)
    "VORT",     # relative vorticity
    "FRONTO",   # frontogenesis function (simplified)
    "WSPD",     # wind speed magnitude
    "OMEGA",    # vertical velocity (Pa/s)
    "VADV",     # vorticity advection (proxy)
]

DEFAULT_PRIMARY_LEVELS = [225, 250, 300]

# Channels with heavy-tailed distributions — log-compress before normalising
LOG_COMPRESS = {"TI1", "TI2", "VADV", "FRONTO"}

ERA5_VARS = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "geopotential",
    "vertical_velocity",
    "specific_humidity",
]

CHANNEL_NAMES: list[str] = []  # populated by channel_names()


def channel_names(primary_levels: list[int] = DEFAULT_PRIMARY_LEVELS) -> list[str]:
    return [f"{d}@{lv}" for lv in primary_levels for d in DIAGNOSTICS]


def _hgrad(field: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    """
    Finite-difference horizontal gradients on a lat/lon grid.
    Returns (df_dy, df_dx) in units of field / metre.
    """
    dlat_rad = np.deg2rad(np.gradient(lat))         # (H,)
    dlon_rad = np.deg2rad(np.gradient(lon))         # (W,)
    # metric factors
    dy = R_EARTH * dlat_rad[:, None]                # (H, 1)  m per lat step
    cos_lat = np.cos(np.deg2rad(lat))[:, None]      # (H, 1)
    dx = R_EARTH * cos_lat * dlon_rad[None, :]      # (H, W)  m per lon step
    df_dy = np.gradient(field, axis=0) / dy
    df_dx = np.gradient(field, axis=1) / dx
    return df_dy, df_dx


def _brackets(levels: list[int], target: int):
    """Return indices of the two levels bracketing target for ∂/∂p."""
    levs = sorted(levels)
    idx = levs.index(target)
    if idx == 0:
        return 0, 1
    if idx == len(levs) - 1:
        return idx - 1, idx
    return idx - 1, idx + 1


def compute_event_diagnostics(
    ds: xr.Dataset,
    primary_levels: list[int] = DEFAULT_PRIMARY_LEVELS,
    grid_size: int = 24,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute all diagnostic channels for one ERA5 event.

    Parameters
    ----------
    ds : xarray Dataset with u, v, T, Z, w, q on pressure levels
    primary_levels : pressure levels (hPa) to compute diagnostics at
    grid_size : output spatial resolution (grid_size × grid_size)

    Returns
    -------
    X : (C, grid_size, grid_size)  float32 array
    names : list of C channel names
    """
    # Time-average
    ds_mean = ds.mean(dim="time")

    avail_lev = sorted(int(l) for l in ds.level.values)
    lat = ds.latitude.values.astype(np.float64)
    lon = ds.longitude.values.astype(np.float64)

    channels = []
    names = []

    for lv in primary_levels:
        lv_use = min(avail_lev, key=lambda x: abs(x - lv))

        # Raw fields at target level
        u  = ds_mean["u_component_of_wind"].sel(level=lv_use).values.astype(np.float64)
        v  = ds_mean["v_component_of_wind"].sel(level=lv_use).values.astype(np.float64)
        T  = ds_mean["temperature"].sel(level=lv_use).values.astype(np.float64)
        Z  = ds_mean["geopotential"].sel(level=lv_use).values.astype(np.float64) / G0
        w  = ds_mean["vertical_velocity"].sel(level=lv_use).values.astype(np.float64)

        # Bracket levels for vertical derivatives
        il, iu = _brackets(avail_lev, lv_use)
        lv_lo, lv_hi = avail_lev[il], avail_lev[iu]
        dp = (lv_hi - lv_lo) * 100.0  # Pa

        u_lo = ds_mean["u_component_of_wind"].sel(level=lv_lo).values.astype(np.float64)
        u_hi = ds_mean["u_component_of_wind"].sel(level=lv_hi).values.astype(np.float64)
        v_lo = ds_mean["v_component_of_wind"].sel(level=lv_lo).values.astype(np.float64)
        v_hi = ds_mean["v_component_of_wind"].sel(level=lv_hi).values.astype(np.float64)
        T_lo = ds_mean["temperature"].sel(level=lv_lo).values.astype(np.float64)
        T_hi = ds_mean["temperature"].sel(level=lv_hi).values.astype(np.float64)

        # ── Vertical wind shear (VWS) ────────────────────────────────────────
        du_dp = (u_hi - u_lo) / dp
        dv_dp = (v_hi - v_lo) / dp
        # convert Pa^-1 → s^-1 via hydrostatic: dp = -ρg dz → dz/dp = -1/(ρg)
        # approximate: dz ≈ -RT/(g*p) dp  →  du/dz = du/dp * (-g*p)/(R_d*T)
        Rd = 287.05
        rho_inv = Rd * T / (lv_use * 100.0)   # 1/ρ  m^3/kg
        du_dz = du_dp * (-G0 / rho_inv) * rho_inv  # simplify: du_dz = du_dp * g
        dv_dz = dv_dp * G0
        VWS = np.sqrt(du_dz ** 2 + dv_dz ** 2)

        # ── Brunt-Väisälä frequency squared (N²) ─────────────────────────────
        dT_dp = (T_hi - T_lo) / dp
        theta_lo = T_lo * (P0 / (lv_lo * 100.0)) ** RD_CP
        theta_hi = T_hi * (P0 / (lv_hi * 100.0)) ** RD_CP
        dtheta_dp = (theta_hi - theta_lo) / dp
        theta_mean = (theta_lo + theta_hi) / 2.0
        dtheta_dz = dtheta_dp * G0   # same hydrostatic approx
        N2 = G0 * dtheta_dz / theta_mean
        N2 = np.clip(N2, -0.1, 0.1)

        # ── Richardson number (Ri) ────────────────────────────────────────────
        VWS2 = VWS ** 2 + 1e-12
        Ri = N2 / VWS2
        Ri = np.clip(Ri, -100, 100)

        # ── Horizontal gradients of u,v ──────────────────────────────────────
        du_dy, du_dx = _hgrad(u, lat, lon)
        dv_dy, dv_dx = _hgrad(v, lat, lon)

        # Deformation = sqrt(stretching² + shearing²)
        stretch = du_dx - dv_dy
        shear   = dv_dx + du_dy
        DEF = np.sqrt(stretch ** 2 + shear ** 2)

        # Divergence
        DIV = du_dx + dv_dy

        # Vorticity
        VORT = dv_dx - du_dy

        # ── Ellrod TI1 / TI2 ─────────────────────────────────────────────────
        TI1 = DEF * VWS
        TI2 = (DEF + np.abs(DIV)) * VWS

        # ── Frontogenesis (simplified scalar) ────────────────────────────────
        # Petterssen: F = 0.5 * d|∇θ|/dt
        # Proxy: magnitude of temperature gradient divergence by deformation
        dT_dy, dT_dx = _hgrad(T, lat, lon)
        grad_T = np.sqrt(dT_dx ** 2 + dT_dy ** 2)
        FRONTO = DEF * grad_T

        # ── WSPD ─────────────────────────────────────────────────────────────
        WSPD = np.sqrt(u ** 2 + v ** 2)

        # ── OMEGA ────────────────────────────────────────────────────────────
        OMEGA = w  # Pa/s

        # ── Vorticity advection proxy ─────────────────────────────────────────
        dVORT_dy, dVORT_dx = _hgrad(VORT, lat, lon)
        VADV = np.abs(u * dVORT_dx + v * dVORT_dy)

        diag_fields = [VWS, N2, Ri, DEF, DIV, TI1, TI2, VORT, FRONTO, WSPD, OMEGA, VADV]

        for field, dname in zip(diag_fields, DIAGNOSTICS):
            field_r = _resize(field, grid_size)
            channels.append(field_r.astype(np.float32))
            names.append(f"{dname}@{lv}")

    X = np.stack(channels, axis=0)  # (C, H, W)
    return X, names


def _resize(field: np.ndarray, grid_size: int) -> np.ndarray:
    H, W = field.shape
    if H == grid_size and W == grid_size:
        return field
    zy = grid_size / H
    zx = grid_size / W
    return zoom(field, (zy, zx), order=1)


def cache_path(event_id: int, base_dir: str = "data/diagnostics") -> Path:
    return Path(base_dir) / f"event_{event_id:04d}.npz"


def load_or_compute(
    event_id: int,
    era5_dir: str = "data/era5",
    cache_dir: str = "data/diagnostics",
    primary_levels: list[int] = DEFAULT_PRIMARY_LEVELS,
    grid_size: int = 24,
) -> tuple[np.ndarray, list[str]] | None:
    """Load cached diagnostics or compute from ERA5."""
    cp = cache_path(event_id, cache_dir)
    names = channel_names(primary_levels)
    if cp.exists():
        data = np.load(cp)
        return data["X"].astype(np.float32), names

    era5_path = Path(era5_dir) / f"event_{event_id:04d}.nc"
    if not era5_path.exists():
        return None
    try:
        ds = xr.open_dataset(era5_path)
        X, names = compute_event_diagnostics(ds, primary_levels, grid_size)
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(cp), X=X)
        return X, names
    except Exception as exc:
        log.warning("Event %04d diagnostics failed: %s", event_id, exc)
        return None


def channel_index(name: str, primary_levels: list[int] = DEFAULT_PRIMARY_LEVELS) -> int:
    return channel_names(primary_levels).index(name)


if __name__ == "__main__":
    import sys
    era5_dir = sys.argv[1] if len(sys.argv) > 1 else "data/era5"
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else "data/diagnostics"
    era5_files = sorted(Path(era5_dir).glob("event_*.nc"))
    print(f"Pre-computing diagnostics for {len(era5_files)} events...")
    for f in era5_files:
        eid = int(f.stem.split("_")[1])
        result = load_or_compute(eid, era5_dir, cache_dir)
        if result is not None:
            print(f"  event_{eid:04d}: {result[0].shape}")
    print("Done.")
