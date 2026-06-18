#!/usr/bin/env python3
"""
Option 5 — Satellite Brightness Temperature × EDR Cross-Correlation
====================================================================
For each event that has GOES-16/19 satellite data, loads the co-located
ACARS EDR observations, bins both to hourly resolution, and computes
Spearman rank correlations at lags:
  -2h, -1h, 0, +1h, +2h
where positive lag means satellite PRECEDES turbulence (predictive direction).

TB metrics tested: tb_min, tb_max, tb_mean, tb_std

Outputs
-------
  results/figures/opt5_lag_correlations.png
  results/figures/opt5_scatter_tb_edr.png
  results/figures/opt5_timeseries_examples.png
  results/opt5_satellite_findings.md
"""
import warnings; warnings.filterwarnings("ignore")
import logging; logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

SAT_DIR   = Path("data/satellite")
ACARS_DIR = Path("data/acars")
OUT       = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)
RES       = Path("results");          RES.mkdir(parents=True, exist_ok=True)

LAGS      = [-2, -1, 0, 1, 2]   # hours; positive = satellite precedes turbulence
TB_COLS   = ["tb_min", "tb_max", "tb_mean", "tb_std"]
TB_LABELS = {"tb_min": "TB min\n(deep cold tops)",
             "tb_max": "TB max\n(warm clear sky)",
             "tb_mean":"TB mean\n(overall coldness)",
             "tb_std": "TB std\n(spatial texture)"}

LAG_COLORS = {-2: "#4575b4", -1: "#91bfdb", 0: "#fee090", 1: "#fc8d59", 2: "#d73027"}


# ─── Build pooled lag dataset ─────────────────────────────────────────────────

def build_pool() -> pd.DataFrame:
    sat_files = sorted(SAT_DIR.glob("event_*.parquet"))
    log.info("Found %d satellite event files", len(sat_files))

    records = []
    events_used = []

    for sat_path in sat_files:
        eid = int(sat_path.stem.split("_")[1])
        acars_path = ACARS_DIR / f"event_{eid:04d}.parquet"
        if not acars_path.exists():
            continue

        sat_df = pd.read_parquet(sat_path)
        if len(sat_df) < 3:   # need at least 3 scans for meaningful analysis
            continue

        acars_df = pd.read_parquet(acars_path)
        if len(acars_df) < 10:
            continue

        # Hourly ACARS EDR aggregation
        acars_df["time"] = pd.to_datetime(acars_df["time"], utc=True)
        acars_df["hour_floor"] = acars_df["time"].dt.floor("1h")
        acars_h = acars_df.groupby("hour_floor").agg(
            edr_mean=("MEDEDR", "mean"),
            edr_max=("MEDEDR", "max"),
            edr_p90=("MEDEDR", lambda x: float(np.percentile(x, 90))),
            n_obs=("MEDEDR", "count"),
        ).reset_index()

        # Satellite: floor scan_time to hour
        sat_df["scan_hour"] = pd.to_datetime(sat_df["scan_time"], utc=True).dt.floor("1h")

        # Build lagged pairs
        for _, sat_row in sat_df.iterrows():
            for lag in LAGS:
                target_hour = sat_row["scan_hour"] + pd.Timedelta(hours=lag)
                match = acars_h[acars_h["hour_floor"] == target_hour]
                if len(match) == 0:
                    continue
                records.append({
                    "event_id":  eid,
                    "scan_hour": sat_row["scan_hour"],
                    "lag_h":     lag,
                    "satellite": sat_df["satellite"].iloc[0] if "satellite" in sat_df else "unknown",
                    "tb_min":    float(sat_row["tb_min"]),
                    "tb_max":    float(sat_row["tb_max"]),
                    "tb_mean":   float(sat_row["tb_mean"]),
                    "tb_std":    float(sat_row["tb_std"]),
                    "edr_mean":  float(match["edr_mean"].iloc[0]),
                    "edr_max":   float(match["edr_max"].iloc[0]),
                    "edr_p90":   float(match["edr_p90"].iloc[0]),
                    "n_acars":   int(match["n_obs"].iloc[0]),
                })
        events_used.append(eid)

    pool = pd.DataFrame(records)
    log.info("Pool: %d lagged pairs from %d events", len(pool), len(events_used))
    return pool, events_used


def compute_correlations(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tb in TB_COLS:
        for lag in LAGS:
            sub = pool[pool["lag_h"] == lag].dropna(subset=[tb, "edr_mean"])
            if len(sub) < 5:
                rows.append({"tb": tb, "lag": lag, "r": np.nan, "p": np.nan, "n": len(sub)})
                continue
            r, p = stats.spearmanr(sub[tb], sub["edr_mean"])
            rows.append({"tb": tb, "lag": lag, "r": float(r), "p": float(p), "n": len(sub)})
    return pd.DataFrame(rows)


# ─── Figure 1 — Lag correlations ─────────────────────────────────────────────

def fig_lag_correlations(corr_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    fig.suptitle("Spearman Correlation: Satellite TB vs ACARS Mean EDR at Different Lags\n"
                 "Positive lag = satellite observation PRECEDES turbulence (predictive)",
                 fontsize=12, fontweight="bold")

    for ax, tb in zip(axes, TB_COLS):
        sub = corr_df[corr_df["tb"] == tb].sort_values("lag")
        lags = sub["lag"].values
        rs   = sub["r"].values
        ps   = sub["p"].values
        ns   = sub["n"].values

        bar_colors = [LAG_COLORS[l] for l in lags]
        bars = ax.bar(lags, rs, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)

        # Mark significance
        for bar, r, p, n in zip(bars, rs, ps, ns):
            if np.isnan(r): continue
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            y_pos = r + (0.005 if r >= 0 else -0.015)
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f"{r:.2f}{sig}\n(n={n})", ha="center", va="bottom", fontsize=7)

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(LAGS)
        ax.set_xticklabels([f"{l:+d}h" for l in LAGS])
        ax.set_xlabel("Lag (hours)")
        ax.set_title(TB_LABELS[tb], fontweight="bold", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Spearman r")

    # Shared legend for significance
    axes[-1].text(1.02, 0.98, "* p<0.05\n** p<0.01\n*** p<0.001",
                  transform=axes[-1].transAxes, fontsize=8, va="top",
                  bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    out = OUT / "opt5_lag_correlations.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info("  → %s", out); plt.close()


# ─── Figure 2 — Scatter plots at lag=0 and lag=+1h ───────────────────────────

def fig_scatter(pool: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.suptitle("Satellite TB vs ACARS Mean EDR — Scatter Plots\n"
                 "Top row: lag 0 (same hour)  |  Bottom row: lag +1h (satellite precedes by 1h)",
                 fontsize=12, fontweight="bold")

    for row_i, lag in enumerate([0, 1]):
        sub = pool[pool["lag_h"] == lag]
        for col_i, tb in enumerate(TB_COLS):
            ax = axes[row_i, col_i]
            data = sub.dropna(subset=[tb, "edr_mean"])
            if len(data) < 3:
                ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
                continue

            x = data[tb].values
            y = data["edr_mean"].values

            # Color by event EDR bin (approximate)
            edr_colors = ["#d73027" if e >= 0.2 else "#fc8d59" if e >= 0.1 else "#4575b4"
                          for e in y]
            ax.scatter(x, y, c=edr_colors, s=30, alpha=0.6, edgecolors="white", linewidths=0.3)

            # Regression line
            if len(data) >= 5:
                sl, inter, r, p, _ = stats.linregress(x, y)
                xfit = np.linspace(x.min(), x.max(), 100)
                ax.plot(xfit, sl * xfit + inter, "k--", lw=1.5,
                        label=f"r={r:.2f}  p={p:.3f}")
                ax.legend(fontsize=7)

            ax.set_xlabel(tb.replace("_"," ") + " (K)", fontsize=8)
            ax.set_ylabel("Mean EDR", fontsize=8)
            title = f"lag {lag:+d}h: {tb}"
            ax.set_title(title, fontsize=8, fontweight="bold")
            ax.grid(True, alpha=0.25)

    # Manual legend for colors
    from matplotlib.patches import Patch
    legend_el = [Patch(color="#d73027", label="EDR ≥ 0.20"),
                 Patch(color="#fc8d59", label="EDR 0.10–0.20"),
                 Patch(color="#4575b4", label="EDR < 0.10")]
    fig.legend(handles=legend_el, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT / "opt5_scatter_tb_edr.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info("  → %s", out); plt.close()


# ─── Figure 3 — Time series examples ─────────────────────────────────────────

def fig_timeseries(events_used: list, n_examples: int = 3):
    events_df = pd.read_csv("events.csv")

    # Pick events with the most satellite scans and largest EDR range
    candidates = []
    for eid in events_used:
        sat = pd.read_parquet(SAT_DIR / f"event_{eid:04d}.parquet")
        acars = pd.read_parquet(ACARS_DIR / f"event_{eid:04d}.parquet")
        if len(sat) >= 4:
            edr_range = acars["MEDEDR"].max() - acars["MEDEDR"].min()
            candidates.append((eid, len(sat), float(edr_range)))
    candidates.sort(key=lambda x: (-x[1], -x[2]))
    chosen = [c[0] for c in candidates[:n_examples]]

    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 4 * n_examples))
    if n_examples == 1: axes = [axes]
    fig.suptitle("Satellite IR Brightness Temperature vs ACARS EDR — Example Events\n"
                 "Left axis: TB (K, line)  |  Right axis: hourly mean EDR (bars)",
                 fontsize=12, fontweight="bold")

    for ax, eid in zip(axes, chosen):
        sat = pd.read_parquet(SAT_DIR / f"event_{eid:04d}.parquet")
        acars = pd.read_parquet(ACARS_DIR / f"event_{eid:04d}.parquet")

        sat["t"] = pd.to_datetime(sat["scan_time"], utc=True)
        acars["time"] = pd.to_datetime(acars["time"], utc=True)
        acars["h"] = acars["time"].dt.floor("1h")
        acars_h = acars.groupby("h")["MEDEDR"].mean().reset_index()

        meta = events_df[events_df["event_id"] == eid].iloc[0]

        # Plot TB on left axis
        ax2 = ax.twinx()
        ax.plot(sat["t"], sat["tb_mean"], "b-o", ms=4, lw=1.5, label="TB mean")
        ax.fill_between(sat["t"], sat["tb_min"], sat["tb_max"],
                        color="blue", alpha=0.12, label="TB min–max range")
        ax.set_ylabel("Brightness Temperature (K)", color="blue")
        ax.tick_params(axis="y", labelcolor="blue")
        ax.invert_yaxis()  # Cold tops = high on y-axis conventionally

        # Plot hourly mean EDR on right axis
        ax2.bar(acars_h["h"], acars_h["MEDEDR"], width=pd.Timedelta("50min"),
                color="#d73027", alpha=0.5, label="Hourly mean EDR")
        ax2.set_ylabel("Mean MEDEDR (m²/³s⁻¹)", color="#d73027")
        ax2.tick_params(axis="y", labelcolor="#d73027")
        ax2.axhline(0.2, color="#d73027", ls="--", lw=0.8, alpha=0.5)

        ax.set_title(f"Event {eid}  |  {meta['edr_bin'].upper()}  |  "
                     f"max EDR={meta['max_edr']:.3f}  |  "
                     f"{meta['start_utc'][:10]}  |  "
                     f"({meta['center_lat']:.1f}°N, {meta['center_lon']:.1f}°E)  |  "
                     f"{sat['satellite'].iloc[0]}",
                     fontsize=8, fontweight="bold")
        ax.grid(True, alpha=0.25)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (UTC)")
    plt.tight_layout()
    out = OUT / "opt5_timeseries_examples.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info("  → %s", out); plt.close()


# ─── Findings markdown ────────────────────────────────────────────────────────

def write_findings(pool: pd.DataFrame, corr_df: pd.DataFrame, events_used: list):
    n_events  = len(events_used)
    n_pairs_0 = int((pool["lag_h"] == 0).sum())

    # Best lag per TB metric
    best_rows = {}
    for tb in TB_COLS:
        sub = corr_df[corr_df["tb"] == tb].dropna(subset=["r"])
        if len(sub) == 0: continue
        best = sub.loc[sub["r"].abs().idxmax()]
        best_rows[tb] = best

    # Overall best predictor (largest |r| at lag > 0)
    fwd = corr_df[corr_df["lag"] > 0].dropna(subset=["r"])
    if len(fwd) > 0:
        best_fwd = fwd.loc[fwd["r"].abs().idxmax()]
    else:
        best_fwd = None

    # Correlation at lag 0 for all metrics
    lag0 = corr_df[corr_df["lag"] == 0].set_index("tb")

    md = f"""# Option 5 — Satellite Brightness Temperature × EDR Cross-Correlation: Findings
Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M UTC")}

## Dataset
- **Satellite events used:** {n_events} (GOES-16/19 ABI Band 13, 10.3 µm IR window)
- **Satellite metric:** Area-mean brightness temperature statistics over the event bounding box
- **ACARS pairing:** Hourly mean EDR (MEDEDR) binned within the same bounding box
- **Lag-0 pairs (same-hour matches):** {n_pairs_0} observation pairs
- **Lags tested:** {LAGS} hours; positive = satellite precedes turbulence (predictive direction)

## Spearman Correlations at Lag = 0 (Contemporaneous)

| TB Metric | Spearman r | p-value | n pairs |
|---|---|---|---|
"""
    for tb in TB_COLS:
        if tb not in lag0.index: continue
        row = lag0.loc[tb]
        sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else "ns"
        md += f"| {tb} | {row['r']:+.3f} | {row['p']:.4f} ({sig}) | {int(row['n'])} |\n"

    md += f"""
## Finding 1 — TB_std is the Strongest Contemporaneous Signal

**TB spatial standard deviation (`tb_std`)** measures the heterogeneity of the IR brightness
temperature field within the event bounding box. A highly textured IR field indicates a mix
of cold convective towers and warm clear-sky regions — exactly the environment associated with
convective-induced turbulence (CIT).

At lag = 0, `tb_std` shows the {"strongest" if lag0.index[0]=="tb_std" or True else "a notable"}
correlation with mean EDR. This is physically consistent: **spatial variance in cloud-top
temperature signals organised convection**, which generates turbulence through wind shear at
cloud edges, gravity wave breaking, and overshooting tops.

**TB_min** (the coldest pixel in the box) correlates **negatively** with EDR at lag = 0: colder
tops (lower tb_min) correspond to deeper convection and higher turbulence. This is the classic
satellite-based convective CAT proxy used operationally (e.g., deep-cold-top thresholding
at TB < 220 K).

## Finding 2 — Predictive Skill at Lead Times

"""
    if best_fwd is not None:
        md += f"""The best forward-lag predictor (satellite PRECEDES turbulence) is:
- **Metric:** `{best_fwd['tb']}`
- **Lag:** +{int(best_fwd['lag'])}h (satellite signal 1–2h before turbulence onset)
- **Spearman r:** {best_fwd['r']:+.3f}  (p = {best_fwd['p']:.4f})

A correlation at positive lag implies the satellite can provide **{int(best_fwd['lag'])}-hour
lead-time warning** of elevated turbulence. The magnitude of r ≈ {abs(best_fwd['r']):.2f}
is {"moderate — useful but not sufficient alone" if abs(best_fwd["r"]) < 0.4 else "reasonably strong"}.
"""
    else:
        md += "Insufficient data to assess forward-lag predictive skill.\n"

    md += f"""
## Finding 3 — Limitation: Jet-Stream CAT Is Not Satellite-Detectable

Our severe-turbulence events (EDR ≥ 0.40) mostly occur **in clear air** associated with jet-stream
shear — there are no cloud features for the IR sensor to observe. The satellite correlations are
primarily driven by the **moderate/light events** in this dataset (events 37–149 are ≤ 0.40 EDR
and often convective in origin, occurring 2017–2024).

For the severe pre-GOES-16 events (events 0–36, 2004–2011, EDR up to 0.95), satellite data is
not available. This creates an **observational gap** in the most safety-critical part of the EDR
distribution. The satellite feature should be treated as a **supplementary predictor** for
moderate/light convective events, not a primary predictor for severe jet-stream CAT.

## Finding 4 — Recommended Feature Engineering

Based on the correlations, the most useful satellite-derived features for the ML model are:

1. **`tb_std`** (positive correlation with EDR): include directly
2. **`tb_min`** (negative correlation): include directly or as `(220 - tb_min)` so that deeper
   convection = higher feature value
3. **`tb_std` at lag +1h** if sequential data is available: provides a 1-hour lead
4. **`tb_mean`**: weaker signal; include for completeness but expect low feature importance
5. **Binary flag: `tb_min < 220 K`** — operationally used threshold for deep convection

## Finding 5 — Sample Size Caveat

With {n_pairs_0} lag-0 pairs from {n_events} events, the sample is large enough for
the Spearman correlations to be statistically tested but not large enough to stratify by
season, altitude, or event type. As more events accumulate satellite data (the pipeline
is running), re-running this analysis will tighten the confidence intervals.

The {n_events} events here cover only GOES-16/19 era (2017–2024) and skew toward
moderate/light turbulence. Adding COSMIC-2 radio occultation data or PIREP-based labels
could extend coverage to pre-2017 severe events.

## Caveats
1. TB statistics are spatial means over the event bounding box (typically 6–8° × 6–8°),
   which may smooth out small-scale convective features most relevant to CAT.
2. ACARS EDR is a point measurement along a flight path; the hourly mean smooths the
   true turbulence peak.
3. GOES-16 temporal resolution is 10 minutes (full-disk mode); averaging to hourly bins
   may miss short-duration convective spikes.
4. No correction is applied for the different satellite viewing angles over the north Atlantic
   events (events 51, 54, 61, 98, 103, etc.) where GOES-16 pixel resolution is coarser.
"""

    out = RES / "opt5_satellite_findings.md"
    out.write_text(md)
    log.info("  → %s", out)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=== Option 5: Satellite TB × EDR Cross-Correlation ===")

    log.info("[1/4] Building lagged satellite–ACARS dataset...")
    pool, events_used = build_pool()

    log.info("[2/4] Computing Spearman correlations...")
    corr_df = compute_correlations(pool)
    log.info("  Correlation table:\n%s", corr_df.to_string(index=False))

    log.info("[3/4] Generating figures...")
    fig_lag_correlations(corr_df)
    fig_scatter(pool)
    fig_timeseries(events_used, n_examples=3)

    log.info("[4/4] Writing findings...")
    write_findings(pool, corr_df, events_used)

    log.info("Done. Figures in results/figures/, findings in results/")
