#!/usr/bin/env python3
"""
Global Turbulence Heatmaps — Month × Hour (UTC)
================================================
Uses OpenSky Network ADS-B state vectors (downloaded by scripts/fetch_opensky.py)
to produce a Month × Hour turbulence-frequency heatmap for 15 global aviation
regions.

Turbulence proxy
----------------
ACARS/MADIS directly measures EDR.  OpenSky ADS-B does not.  Instead we use
the **intra-hour standard deviation of vertical rate** per aircraft as a proxy:

  For each icao24 in each hourly file:
    1. Filter to cruise: baroaltitude > 5 500 m (FL180+), onground = False,
       ≥ MIN_OBS_PER_AC valid vertrate readings.
    2. Compute std(vertrate) [m/s] across all readings in that file-hour.
    3. Flag as "turbulent aircraft-hour" if std > VERTRATE_STD_THRESH.
    4. Median lat/lon of that aircraft-hour → region assignment.

  (region, month, hour):
      n_ac_hours   = total aircraft-hours in that cell
      n_turb_hours = turbulent aircraft-hours
      frac_m       = n_turb_hours / n_ac_hours

Calibration note
----------------
VERTRATE_STD_THRESH = 1.5 m/s (≈ 295 ft/min).  Kim et al. (2021) and Sharman
& Lane (2016) show that std(w) > 1–2 m/s corresponds to light–moderate
turbulence; 1.5 m/s is chosen to approximate MEDEDR ≥ 0.20 in terms of
encounter rate.  Adjust if the resulting frac_m values differ substantially
from ACARS baseline.

Data layout expected
--------------------
  data/opensky/{year}/{month:02d}/{day:02d}/hour_{hh:02d}.parquet
  (produced by scripts/fetch_opensky.py)

Outputs
-------
  results/figures/regional_heatmaps.png
  results/cache_regional_mh.parquet
"""
import warnings; warnings.filterwarnings("ignore")
import logging; logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── Config ───────────────────────────────────────────────────────────────────

OPENSKY_DIR        = Path("data/opensky")
OUT                = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)
CACHE              = Path("results/cache_regional_mh.parquet")

MIN_ALT_M          = 5_500    # FL180 in metres
VERTRATE_STD_THRESH = 1.5     # m/s — proxy for moderate turbulence
MIN_OBS_PER_AC     = 5        # minimum vertrate readings to include an aircraft-hour
MIN_AC_HOURS       = 200      # minimum aircraft-hours to draw a panel (else grey)

MO = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ─── Global regions ───────────────────────────────────────────────────────────
# Format: (name, lat_lo, lat_hi, lon_lo, lon_hi, utc_label, crosses_dateline)
# crosses_dateline=True → region spans 180/-180; lon_lo > lon_hi in that case.

REGIONS = [
    # ── Americas ──────────────────────────────────────────────────────────────
    ("W North America",        28,  75, -172, -100, "UTC−8/−7", False),
    ("E North America",        24,  75, -100,  -60, "UTC−6/−5", False),
    ("Caribbean / C. America",  4,  30, -100,  -55, "UTC−7/−5", False),
    ("South America",         -60,   8,  -90,  -30, "UTC−5/−3", False),
    ("N Atlantic (NATS)",      40,  80,  -60,  -10, "UTC−4/0",  False),
    # ── Europe & Africa ───────────────────────────────────────────────────────
    ("NW Europe",              45,  75,  -15,   25, "UTC 0/+2", False),
    ("C & E Europe",           35,  65,   20,   50, "UTC+1/+4", False),
    ("Mediterranean",          25,  47,  -10,   42, "UTC+1/+3", False),
    ("Middle East / N Africa", 12,  42,   30,   65, "UTC+2/+4", False),
    ("Sub-Saharan Africa",    -40,  15,  -20,   55, "UTC 0/+3", False),
    # ── Asia & Oceania ────────────────────────────────────────────────────────
    ("Central Asia / Russia",  35,  75,   50,  120, "UTC+5/+8", False),
    ("South Asia",              5,  35,   60,  100, "UTC+5/+6", False),
    ("East Asia",              20,  58,   95,  150, "UTC+8/+9", False),
    ("SE Asia / Oceania",     -48,  20,   80,  180, "UTC+7/+12",False),
    # ── Pacific ───────────────────────────────────────────────────────────────
    ("North & South Pacific", -60,  65,  140, -100, "UTC±12/−8",True ),
]

NREGIONS = len(REGIONS)


def assign_region(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Vectorised region assignment.  Returns int8 array; -1 = no region.
    Handles the dateline-crossing Pacific region.
    """
    out = np.full(len(lat), -1, dtype=np.int8)
    for idx, (_, la0, la1, lo0, lo1, _, dateline) in enumerate(REGIONS):
        in_lat = (lat >= la0) & (lat < la1)
        if dateline:
            # Pacific: lon >= lo0 (e.g. 140) OR lon < lo1 (e.g. -100)
            in_lon = (lon >= lo0) | (lon < lo1)
        else:
            in_lon = (lon >= lo0) & (lon < lo1)
        mask = (out == -1) & in_lat & in_lon
        out[mask] = idx
    return out


# ─── Ingest ───────────────────────────────────────────────────────────────────

def _process_file(path: Path) -> pd.DataFrame | None:
    """
    Process one hourly Parquet file → per-aircraft-hour turbulence summary.
    Returns a DataFrame with columns [region, month, hour, n_ac, n_turb]
    or None if the file is empty / unusable.
    """
    try:
        df = pd.read_parquet(path, columns=[
            "time", "icao24", "lat", "lon",
            "vertrate", "onground", "baroaltitude",
        ])
    except Exception as exc:
        log.debug("Skip %s: %s", path, exc)
        return None

    if df.empty:
        return None

    # Cruise filter
    df = df[
        (~df["onground"].fillna(True)) &
        (df["baroaltitude"] > MIN_ALT_M) &
        df["vertrate"].notna() &
        df["lat"].notna() &
        df["lon"].notna()
    ]
    if df.empty:
        return None

    # Derive month + hour from the Unix timestamp in the filename path
    # (faster than parsing per-row; all rows in one hourly file share month/hour)
    try:
        # path: .../data/opensky/{year}/{month}/{day}/hour_{hh}.parquet
        parts = path.parts
        month = int(parts[-3])      # month dir
        hour  = int(path.stem.split("_")[1])   # hour_{hh}
    except Exception:
        # Fall back to reading timestamp column
        ts = pd.to_datetime(df["time"].dropna().iloc[0], unit="s", utc=True)
        month, hour = ts.month, ts.hour

    # Per-aircraft std of vertical rate
    ac_grp = df.groupby("icao24", sort=False)
    ac_stats = ac_grp.agg(
        n_obs     =("vertrate", "count"),
        vr_std    =("vertrate", "std"),
        lat_med   =("lat",      "median"),
        lon_med   =("lon",      "median"),
    ).reset_index()

    ac_stats = ac_stats[ac_stats["n_obs"] >= MIN_OBS_PER_AC].copy()
    if ac_stats.empty:
        return None

    ac_stats["vr_std"] = ac_stats["vr_std"].fillna(0.0)
    ac_stats["region"] = assign_region(
        ac_stats["lat_med"].values, ac_stats["lon_med"].values
    )
    ac_stats = ac_stats[ac_stats["region"] >= 0]
    if ac_stats.empty:
        return None

    ac_stats["is_turb"] = (ac_stats["vr_std"] > VERTRATE_STD_THRESH).astype(np.int32)
    ac_stats["month"]   = np.int8(month)
    ac_stats["hour"]    = np.int8(hour)

    agg = (
        ac_stats
        .groupby(["region", "month", "hour"])
        .agg(n_ac=("is_turb", "count"), n_turb=("is_turb", "sum"))
        .reset_index()
    )
    return agg


def ingest() -> pd.DataFrame:
    files = sorted(OPENSKY_DIR.rglob("hour_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No OpenSky Parquet files found under {OPENSKY_DIR}/\n"
            "Run: python scripts/fetch_opensky.py --user YOU --password PASS"
        )

    log.info("Found %d hourly files — processing...", len(files))
    accum = []
    for i, fp in enumerate(files, 1):
        result = _process_file(fp)
        if result is not None:
            accum.append(result)
        if i % 100 == 0:
            log.info("  %d / %d files", i, len(files))

    if not accum:
        raise RuntimeError("No usable data extracted from any OpenSky file.")

    log.info("Aggregating %d partial tables...", len(accum))
    combined = (
        pd.concat(accum, ignore_index=True)
        .groupby(["region", "month", "hour"])[["n_ac", "n_turb"]]
        .sum()
        .reset_index()
    )
    combined["frac_m"] = combined["n_turb"] / combined["n_ac"]
    return combined


def load_or_ingest() -> pd.DataFrame:
    if CACHE.exists():
        log.info("Loading cache %s ...", CACHE)
        return pd.read_parquet(CACHE)
    log.info("Cache not found — ingesting from %s ...", OPENSKY_DIR)
    agg = ingest()
    agg.to_parquet(CACHE, index=False)
    log.info("Cache written → %s", CACHE)
    return agg


# ─── Figure ───────────────────────────────────────────────────────────────────

def make_figure(agg: pd.DataFrame) -> None:
    ncols, nrows = 5, 3
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(28, 14),
        constrained_layout=True,
    )
    axes_flat = axes.flat

    fig.suptitle(
        "Global Turbulence Frequency — Month × Hour (UTC)  |  15 Aviation Regions\n"
        r"OpenSky ADS-B · FL180+  ·  proxy: $\sigma$(vertical rate) > 1.5 m/s per aircraft-hour"
        "  ·  ★ = peak cell  ·  regions ordered W → E",
        fontsize=12, fontweight="bold",
    )

    region_totals = agg.groupby("region")["n_ac"].sum()
    cmap = plt.cm.YlOrRd

    for idx, (ax, (name, *_, utc_label, _dateline)) in enumerate(
            zip(axes_flat, REGIONS)):

        total_ac = int(region_totals.get(idx, 0))
        sub = agg[agg["region"] == idx]

        if total_ac < MIN_AC_HOURS or sub.empty:
            ax.set_facecolor("#d0d0d0")
            ax.text(0.5, 0.5,
                    f"{name}\n(insufficient data)\n{total_ac:,} ac-hrs",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8.5, color="#555")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{name}\n{utc_label}", fontsize=9, fontweight="bold")
            continue

        pivot = sub.pivot_table(
            index="hour", columns="month", values="frac_m", aggfunc="mean"
        )
        pivot = pivot.reindex(index=range(24), columns=range(1, 13))
        pivot *= 100   # → percentage

        vals = pivot.values
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            ax.set_facecolor("#d0d0d0")
            continue

        vmax = max(float(np.nanpercentile(finite, 98)), 1e-4)

        im = ax.pcolormesh(
            pivot.columns, pivot.index, vals,
            cmap=cmap, shading="auto", vmin=0, vmax=vmax,
        )

        # Axis formatting
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MO, fontsize=6.5, rotation=45, ha="right")
        ax.set_yticks(range(0, 24, 3))
        ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], fontsize=6)
        ax.set_xlabel("Month", fontsize=7)
        ax.set_ylabel("Hour UTC", fontsize=7)
        ax.set_title(f"{name}  [{utc_label}]", fontsize=9, fontweight="bold")

        # Peak marker
        flat_idx  = int(np.nanargmax(vals))
        peak_h, peak_m_idx = divmod(flat_idx, 12)
        peak_val  = float(np.nanmax(vals))
        ax.plot(peak_m_idx + 1, peak_h, "w*", ms=10, zorder=5,
                markeredgecolor="black", markeredgewidth=0.4)

        # Seasonal amplitude
        month_means = pivot.mean(axis=0)
        amp         = float(month_means.max() - month_means.min())
        peak_mon    = MO[int(month_means.idxmax()) - 1]

        ax.text(0.02, 0.97,
                f"★ {MO[peak_m_idx]} {peak_h:02d}:00 UTC  {peak_val:.3f}%\n"
                f"Season amp: {amp:.3f}pp  peak mo: {peak_mon}",
                transform=ax.transAxes, fontsize=5.8, color="white",
                va="top", fontweight="bold",
                bbox=dict(fc="black", alpha=0.35, pad=1.5, lw=0))

        ax.text(0.98, 0.02,
                f"{total_ac/1e3:.0f}K ac-hrs",
                transform=ax.transAxes, fontsize=5.8, color="white",
                va="bottom", ha="right", alpha=0.9)

        cb = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.046)
        cb.set_label("% ac-hrs turbulent", fontsize=5.5)
        cb.ax.tick_params(labelsize=5)

    # Hide any spare axes (if NREGIONS < nrows*ncols)
    for ax in list(axes_flat)[NREGIONS:]:
        ax.set_visible(False)

    out_path = OUT / "regional_heatmaps.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("→ %s", out_path)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="Ignore cache and re-ingest from OpenSky files")
    args = ap.parse_args()

    if args.rebuild and CACHE.exists():
        CACHE.unlink()
        log.info("Cache cleared.")

    log.info("=== Global Regional Heatmaps (OpenSky ADS-B) ===")

    agg = load_or_ingest()

    log.info("Aircraft-hours per region:")
    totals = agg.groupby("region")["n_ac"].sum()
    for i, (name, *_) in enumerate(REGIONS):
        log.info("  %-30s %10.0f K ac-hrs", name, totals.get(i, 0) / 1e3)

    log.info("Building figure...")
    make_figure(agg)
    log.info("Done. → results/figures/regional_heatmaps.png")
