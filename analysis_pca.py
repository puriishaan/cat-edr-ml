"""
PCA analysis of turbulence events.

Figure 1 — ACARS-based PCA  : features derived from per-event aircraft reports
Figure 2 — ERA5-based PCA   : atmospheric state features from reanalysis fields
Figure 3 — Geographic PCA   : PC1 score mapped onto event locations
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xarray as xr

ACARS_DIR  = Path("data/acars")
ERA5_DIR   = Path("data/era5")
EVENTS_CSV = Path("events.csv")
OUT_DIR    = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIN_COLOR = {
    "smooth":   "#4575b4",
    "light":    "#91cf60",
    "moderate": "#fc8d59",
    "severe":   "#d73027",
}
BIN_ORDER = ["smooth", "light", "moderate", "severe"]
BIN_MARKER = {"smooth": "o", "light": "s", "moderate": "^", "severe": "D"}

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def acars_features(event_id: int) -> dict | None:
    path = ACARS_DIR / f"event_{event_id:04d}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if len(df) == 0:
        return None

    edr = df["MEDEDR"].values
    alt = df["alt_m"].values

    return {
        "edr_mean":      float(np.mean(edr)),
        "edr_std":       float(np.std(edr)),
        "edr_p75":       float(np.percentile(edr, 75)),
        "edr_p90":       float(np.percentile(edr, 90)),
        "edr_p95":       float(np.percentile(edr, 95)),
        "frac_light":    float(np.mean(edr >= 0.10)),
        "frac_moderate": float(np.mean(edr >= 0.20)),
        "frac_severe":   float(np.mean(edr >= 0.40)),
        "alt_mean_km":   float(np.mean(alt) / 1000),
        "alt_std_km":    float(np.std(alt)  / 1000),
        "alt_p25_km":    float(np.percentile(alt, 25) / 1000),
        "alt_p75_km":    float(np.percentile(alt, 75) / 1000),
        "lat_spread":    float(df["lat"].std()),
        "lon_spread":    float(df["lon"].std()),
        "n_reports":     float(len(df)),
    }


def load_acars_df(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in events.iterrows():
        feat = acars_features(int(row["event_id"]))
        if feat is None:
            continue
        feat["event_id"]    = int(row["event_id"])
        feat["edr_bin"]     = row["edr_bin"]
        feat["center_lat"]  = float(row["center_lat"])
        feat["center_lon"]  = float(row["center_lon"])
        start = pd.to_datetime(row["start_utc"], utc=True)
        end   = pd.to_datetime(row["end_utc"],   utc=True)
        feat["duration_hr"] = float((end - start).total_seconds() / 3600)
        rows.append(feat)
    return pd.DataFrame(rows).reset_index(drop=True)


FEATURE_LABELS = {
    "edr_mean":      "EDR mean",
    "edr_std":       "EDR std",
    "edr_p75":       "EDR p75",
    "edr_p90":       "EDR p90",
    "edr_p95":       "EDR p95",
    "frac_light":    "frac ≥0.1",
    "frac_moderate": "frac ≥0.2",
    "frac_severe":   "frac ≥0.4",
    "alt_mean_km":   "alt mean",
    "alt_std_km":    "alt std",
    "alt_p25_km":    "alt p25",
    "alt_p75_km":    "alt p75",
    "lat_spread":    "lat spread",
    "lon_spread":    "lon spread",
    "n_reports":     "n reports",
    "duration_hr":   "duration",
}

ACARS_FEATURE_COLS = list(FEATURE_LABELS.keys())


def era5_features(event_id: int) -> dict | None:
    path = ERA5_DIR / f"event_{event_id:04d}.nc"
    if not path.exists():
        return None
    ds = xr.open_dataset(path)

    # Spatial + temporal mean at each pressure level
    m = ds.mean(dim=["latitude", "longitude", "time"])
    u      = m["u_component_of_wind"].values
    v      = m["v_component_of_wind"].values
    T      = m["temperature"].values
    omega  = m["vertical_velocity"].values
    levels = ds.level.values  # hPa, ascending pressure

    feat = {}
    for i, lv in enumerate(levels):
        feat[f"wspd_{lv}"]  = float(np.sqrt(u[i]**2 + v[i]**2))
        feat[f"T_{lv}"]     = float(T[i])
        feat[f"omega_{lv}"] = float(omega[i])

    # Bulk vertical wind shear across full column (500 hPa − 250 hPa analogue)
    # Levels are in the order stored; find indices for key levels
    lvlist = list(levels)
    if 250 in lvlist and 500 in lvlist:
        i250 = lvlist.index(250)
        i500 = lvlist.index(500)
        du = u[i500] - u[i250]
        dv = v[i500] - v[i250]
        feat["bulk_shear"] = float(np.sqrt(du**2 + dv**2))

    # Temperature lapse rate (200 hPa − 500 hPa as stability proxy)
    if 200 in lvlist and 500 in lvlist:
        feat["lapse_200_500"] = float(T[lvlist.index(500)] - T[lvlist.index(200)])

    return feat


def load_era5_df(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in events.iterrows():
        feat = era5_features(int(row["event_id"]))
        if feat is None:
            continue
        feat["event_id"] = int(row["event_id"])
        feat["edr_bin"]  = row["edr_bin"]
        feat["max_edr"]  = float(row["max_edr"])
        rows.append(feat)
    return pd.DataFrame(rows).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — ACARS PCA  (4 panels)
# ─────────────────────────────────────────────────────────────────────────────

def fig_acars_pca(df: pd.DataFrame):
    X = StandardScaler().fit_transform(df[ACARS_FEATURE_COLS].values)
    pca = PCA()
    scores = pca.fit_transform(X)
    ev     = pca.explained_variance_ratio_
    loads  = pca.components_
    labels = [FEATURE_LABELS[c] for c in ACARS_FEATURE_COLS]

    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        f"PCA — ACARS Event Features  ({len(df)} events, 16 features)",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)

    # ── (a) Scree plot ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    n_show = min(12, len(ev))
    ax.bar(range(1, n_show + 1), ev[:n_show] * 100,
           color="#4575b4", alpha=0.75, label="Per-PC")
    ax.plot(range(1, n_show + 1), np.cumsum(ev[:n_show]) * 100,
            "r-o", ms=5, lw=1.5, label="Cumulative")
    ax.axhline(80, color="gray", ls="--", lw=1, label="80% line")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("(a)  Scree Plot", fontweight="bold")
    ax.set_xlim(0.4, n_show + 0.6)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (b) PC1 vs PC2 ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for bn in BIN_ORDER:
        mask = df["edr_bin"].values == bn
        ax.scatter(scores[mask, 0], scores[mask, 1],
                   c=BIN_COLOR[bn], marker=BIN_MARKER[bn],
                   s=55, alpha=0.85, edgecolors="white", linewidths=0.4,
                   label=bn, zorder=3)
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel(f"PC1  ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2  ({ev[1]*100:.1f}% var)")
    ax.set_title("(b)  PC1 vs PC2", fontweight="bold")
    ax.legend(title="EDR bin", fontsize=8, title_fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── (c) PC1 vs PC3 ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    for bn in BIN_ORDER:
        mask = df["edr_bin"].values == bn
        ax.scatter(scores[mask, 0], scores[mask, 2],
                   c=BIN_COLOR[bn], marker=BIN_MARKER[bn],
                   s=55, alpha=0.85, edgecolors="white", linewidths=0.4,
                   label=bn, zorder=3)
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel(f"PC1  ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC3  ({ev[2]*100:.1f}% var)")
    ax.set_title("(c)  PC1 vs PC3", fontweight="bold")
    ax.legend(title="EDR bin", fontsize=8, title_fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── (d) Loading bar chart PC1 & PC2 ───────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    n_feat = len(labels)
    x = np.arange(n_feat)
    w = 0.38
    ax.bar(x - w/2, loads[0], w,
           color=["#4575b4" if v > 0 else "#91bfdb" for v in loads[0]],
           alpha=0.85, label=f"PC1 ({ev[0]*100:.1f}%)")
    ax.bar(x + w/2, loads[1], w,
           color=["#d73027" if v > 0 else "#fc8d59" for v in loads[1]],
           alpha=0.85, label=f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Loading weight")
    ax.set_title("(d)  Feature Loadings — PC1 & PC2", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    out = OUT_DIR / "pca_acars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  → {out}")
    plt.close()
    return pca, scores, ev


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — ERA5 PCA  (2 panels)
# ─────────────────────────────────────────────────────────────────────────────

def fig_era5_pca(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c not in ("event_id", "edr_bin", "max_edr")]
    X = StandardScaler().fit_transform(df[feat_cols].values)
    n_comp = min(len(df) - 1, len(feat_cols))
    pca    = PCA(n_components=n_comp)
    scores = pca.fit_transform(X)
    ev     = pca.explained_variance_ratio_
    loads  = pca.components_

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"PCA — ERA5 Atmospheric Features  ({len(df)} events, {len(feat_cols)} features)",
        fontsize=13, fontweight="bold"
    )

    # ── PC1 vs PC2, colored by max EDR ────────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(scores[:, 0], scores[:, 1],
                    c=df["max_edr"].values, cmap="RdYlGn_r",
                    vmin=0, vmax=0.95,
                    s=80, edgecolors="black", linewidths=0.6, zorder=5)
    cb = plt.colorbar(sc, ax=ax, label="max EDR")
    for i, row in df.iterrows():
        ax.annotate(
            f"E{int(row['event_id'])}",
            (scores[i, 0], scores[i, 1]),
            fontsize=6.5, ha="center", va="bottom",
            xytext=(0, 4), textcoords="offset points", alpha=0.75,
        )
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel(f"PC1  ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2  ({ev[1]*100:.1f}% var)")
    ax.set_title("PC1 vs PC2  (color = max EDR)", fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Legend for EDR bins via edr_bin text in annotations
    for bn in BIN_ORDER:
        mask = df["edr_bin"].values == bn
        ax.scatter([], [], c=BIN_COLOR[bn], s=50, label=bn)
    ax.legend(title="EDR bin (approx)", fontsize=7, title_fontsize=7,
              loc="lower right")

    # ── Top PC1 loadings ──────────────────────────────────────────────────────
    ax = axes[1]
    series = pd.Series(loads[0], index=feat_cols)
    top = series.abs().nlargest(14).index
    vals = series[top]
    bar_colors = ["#4575b4" if v > 0 else "#d73027" for v in vals]
    ax.barh(range(len(top)), vals.values, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top, fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Loading on PC1")
    ax.set_title(f"Top 14 ERA5 features on PC1  ({ev[0]*100:.1f}% var)\nblue = positive  /  red = negative",
                 fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "pca_era5.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Geographic PC1 score map
# ─────────────────────────────────────────────────────────────────────────────

def fig_geo_pca(df: pd.DataFrame, scores: np.ndarray, ev: np.ndarray):
    pc1 = scores[:, 0]
    norm = mcolors.TwoSlopeNorm(vmin=pc1.min(), vcenter=0, vmax=pc1.max())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("Geographic Distribution of PC1 & PC2 Scores",
                 fontsize=13, fontweight="bold")

    for ax_i, (pc_idx, title) in enumerate(
        [(0, f"PC1  ({ev[0]*100:.1f}% var — EDR intensity axis)"),
         (1, f"PC2  ({ev[1]*100:.1f}% var)")]
    ):
        ax = axes[ax_i]
        pc = scores[:, pc_idx]
        norm_i = mcolors.TwoSlopeNorm(vmin=pc.min(), vcenter=0, vmax=pc.max())

        # Rough land outline via scatter background trick — just a scatter plot
        sc = ax.scatter(
            df["center_lon"], df["center_lat"],
            c=pc, cmap="RdBu_r", norm=norm_i,
            s=80, edgecolors="black", linewidths=0.5, zorder=5,
        )
        plt.colorbar(sc, ax=ax, label=f"PC{pc_idx+1} score", shrink=0.85)

        # Annotate edr_bin with small text
        for i, row in df.iterrows():
            ax.annotate(
                row["edr_bin"][0].upper(),  # S / L / M / Se
                (row["center_lon"], row["center_lat"]),
                fontsize=5.5, ha="center", va="center",
                color="white", fontweight="bold", zorder=6,
            )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(df["center_lon"].min() - 10, df["center_lon"].max() + 10)
        ax.set_ylim(df["center_lat"].min() - 5,  df["center_lat"].max() + 5)

    plt.tight_layout()
    out = OUT_DIR / "pca_geo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    events = pd.read_csv(EVENTS_CSV)
    print(f"Events loaded: {len(events)}")

    print("\n[1/3]  Building ACARS features...")
    df_acars = load_acars_df(events)
    print(f"       {len(df_acars)} events  |  bin counts: {df_acars['edr_bin'].value_counts().to_dict()}")
    pca_obj, scores, ev = fig_acars_pca(df_acars)

    print("\n[2/3]  Building ERA5 features...")
    df_era5 = load_era5_df(events)
    print(f"       {len(df_era5)} events  |  bin counts: {df_era5['edr_bin'].value_counts().to_dict()}")
    if len(df_era5) >= 4:
        fig_era5_pca(df_era5)
    else:
        print("       Skipping — need ≥4 events.")

    print("\n[3/3]  Geographic PC1/PC2 map...")
    fig_geo_pca(df_acars, scores, ev)

    print("\nAll figures saved to results/figures/")
