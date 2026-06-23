#!/usr/bin/env python3
"""
Regional Turbulence Heatmaps — Month × Hour (UTC)
===================================================
Splits 116 M ACARS EDR reports into 10 geographic regions (arranged west→east)
and produces one Month × Hour heatmap per region, showing the fraction of
reports with MEDEDR ≥ 0.20 (moderate turbulence).

Regions follow jet-stream geography + longitude ordering so that the diurnal
UTC peak shifts left-to-right across panels.

Data source: data/raw_reports.parquet (ACARS/MADIS, 2005–2024)
Output:      results/figures/regional_heatmaps.png
Cache:       results/cache_regional_mh.parquet
"""
import warnings; warnings.filterwarnings("ignore")
import logging; logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pyarrow.parquet as pq
from pathlib import Path

RAW   = Path("data/raw_reports.parquet")
OUT   = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("results/cache_regional_mh.parquet")
BATCH = 5_000_000

MO = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ─── Region definitions (ordered west→east so UTC peak shifts across panels) ──

REGIONS = [
    # name,                  lat_lo, lat_hi, lon_lo, lon_hi,  utc_offset_label
    ("N Central CONUS",        42,    58,   -100,    -84,  "UTC−6/5"),
    ("S Central CONUS",        24,    42,   -100,    -84,  "UTC−7/6"),
    ("NE CONUS",               38,    58,    -84,    -60,  "UTC−5/4"),
    ("SE CONUS",               24,    38,    -84,    -60,  "UTC−5/4"),
    ("W North Atlantic",       40,    72,    -60,    -35,  "UTC−4/−2"),
    ("E North Atlantic",       40,    72,    -35,    -10,  "UTC−2→0"),
    ("NW Europe",              44,    72,    -10,     20,  "UTC 0/+2"),
    ("Mediterranean",          25,    44,    -10,     42,  "UTC+1/+3"),
    ("East Asia",              20,    60,     95,    150,  "UTC+8/+9"),
    ("SE Asia/W Pacific",     -10,    25,     80,    160,  "UTC+7/+10"),
]
REGION_NAMES = [r[0] for r in REGIONS]


def assign_region_vectorised(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Return integer region index (0-9) or -1 if outside all regions."""
    out = np.full(len(lat), -1, dtype=np.int8)
    # Apply in order; earlier entries win where boxes overlap
    for idx, (_, lat_lo, lat_hi, lon_lo, lon_hi, _) in enumerate(REGIONS):
        mask = (
            (out == -1) &
            (lat >= lat_lo) & (lat < lat_hi) &
            (lon >= lon_lo) & (lon < lon_hi)
        )
        out[mask] = idx
    return out


# ─── Batch ingestion ──────────────────────────────────────────────────────────

def ingest() -> pd.DataFrame:
    pf = pq.ParquetFile(RAW)
    accum = []
    total = int(116e6 / BATCH) + 2

    for i, batch in enumerate(pf.iter_batches(batch_size=BATCH), 1):
        df  = batch.to_pandas()
        t   = pd.to_datetime(df["time"], utc=True)
        edr = df["MEDEDR"].values.astype(np.float32)
        lat = df["lat"].values.astype(np.float32)
        lon = df["lon"].values.astype(np.float32)

        region = assign_region_vectorised(lat, lon)
        in_region = region >= 0

        if not in_region.any():
            continue

        mini = pd.DataFrame({
            "region": region[in_region],
            "month":  t.dt.month.values.astype(np.int8)[in_region],
            "hour":   t.dt.hour.values.astype(np.int8)[in_region],
            "n":      np.ones(in_region.sum(), np.int32),
            "is_m":   (edr[in_region] >= 0.20).astype(np.int32),
        })

        accum.append(
            mini.groupby(["region", "month", "hour"])
                .agg(n=("n", "sum"), n_m=("is_m", "sum"))
                .reset_index()
        )

        if i % 5 == 0:
            log.info("  batch %d/~%d", i, total)

    log.info("Finalising...")
    agg = (
        pd.concat(accum)
          .groupby(["region", "month", "hour"])[["n", "n_m"]]
          .sum()
          .reset_index()
    )
    agg["frac_m"] = agg["n_m"] / agg["n"]
    return agg


def load_or_ingest() -> pd.DataFrame:
    if CACHE.exists():
        log.info("Loading cache...")
        return pd.read_parquet(CACHE)
    log.info("Cache not found — ingesting %s (~116 M rows)...", RAW)
    agg = ingest()
    agg.to_parquet(CACHE)
    log.info("Cache written → %s", CACHE)
    return agg


# ─── Figure ───────────────────────────────────────────────────────────────────

MIN_REPORTS = 5_000   # minimum total reports to draw a heatmap; else grey out

def make_figure(agg: pd.DataFrame):
    ncols, nrows = 5, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(26, 11),
                             constrained_layout=True)
    axes = axes.flat

    fig.suptitle(
        "Turbulence Frequency — Month × Hour (UTC)  |  10 Geographic Regions\n"
        "Fraction of ACARS/MADIS reports with MEDEDR ≥ 0.20  |  2005–2024, FL180+\n"
        "Each panel uses its own colour scale  ·  ★ = peak cell  ·  regions ordered west→east",
        fontsize=12, fontweight="bold",
    )

    region_totals = agg.groupby("region")["n"].sum()
    cmap = plt.cm.YlOrRd

    for idx, (ax, (name, *_, utc_label)) in enumerate(zip(axes, REGIONS)):
        sub = agg[agg["region"] == idx]
        total_n = int(region_totals.get(idx, 0))

        if total_n < MIN_REPORTS:
            ax.set_facecolor("#d8d8d8")
            ax.text(0.5, 0.5, f"{name}\n(sparse)\n{total_n:,} reports",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#444444")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{name}\n{utc_label}", fontsize=9, fontweight="bold")
            continue

        pivot = sub.pivot_table(index="hour", columns="month",
                                values="frac_m", aggfunc="mean")
        pivot = pivot.reindex(index=range(24), columns=range(1, 13))
        pivot *= 100   # → %

        # Per-panel scale: 0 → 98th-percentile of this region
        vmax = float(np.nanpercentile(pivot.values[~np.isnan(pivot.values)], 98))
        vmax = max(vmax, 0.005)

        im = ax.pcolormesh(
            pivot.columns, pivot.index, pivot.values,
            cmap=cmap, shading="auto", vmin=0, vmax=vmax,
        )

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MO, fontsize=6.5, rotation=45, ha="right")
        ax.set_yticks(range(0, 24, 3))
        ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], fontsize=6)
        ax.set_xlabel("Month", fontsize=7)
        ax.set_ylabel("Hour UTC", fontsize=7)
        ax.set_title(f"{name}  [{utc_label}]", fontsize=9, fontweight="bold")

        # peak marker
        vals = pivot.values.copy()
        flat_idx = int(np.nanargmax(vals))
        peak_h, peak_m_idx = divmod(flat_idx, 12)
        peak_val = float(np.nanmax(vals))
        ax.plot(peak_m_idx + 1, peak_h, "w*", ms=10, zorder=5,
                markeredgecolor="black", markeredgewidth=0.4)

        # seasonal amplitude annotation (max month mean - min month mean)
        month_means = pivot.mean(axis=0)  # mean over hours
        amp = float(month_means.max() - month_means.min())
        peak_mon = MO[int(month_means.idxmax()) - 1]
        ax.text(0.02, 0.97,
                f"★ {MO[peak_m_idx]} {peak_h:02d}:00 UTC  {peak_val:.3f}%\n"
                f"Season amp: {amp:.3f}pp  peak mo: {peak_mon}",
                transform=ax.transAxes, fontsize=5.8, color="white",
                va="top", fontweight="bold",
                bbox=dict(fc="black", alpha=0.35, pad=1.5, lw=0))

        ax.text(0.98, 0.02, f"{total_n/1e6:.1f}M rpts",
                transform=ax.transAxes, fontsize=5.8, color="white",
                va="bottom", ha="right", alpha=0.9)

        cb = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.046)
        cb.set_label("% of reports", fontsize=6)
        cb.ax.tick_params(labelsize=5.5)

    out = OUT / "regional_heatmaps.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info("→ %s", out)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=== Regional Month×Hour Heatmaps ===")
    agg = load_or_ingest()

    log.info("Reports per region:")
    totals = agg.groupby("region")["n"].sum()
    for i, (name, *_) in enumerate(REGIONS):
        log.info("  %-22s %10.1f M", name, totals.get(i, 0) / 1e6)

    log.info("Building figure...")
    make_figure(agg)
    log.info("Done.")
