"""
Step 1b — Cluster events from raw_reports.parquet
Runs after step1 download completes. Filters to turbulent rows only
before clustering so it fits in 8GB RAM.

Usage:
    python scripts/step1b_cluster_events.py --raw data/raw_reports.parquet --out events.csv
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EDR_CLUSTER_MIN = 0.1   # only cluster turbulent reports to find events
CLUSTER_KM      = 200
CLUSTER_HR      = 6
BOX_PAD         = 2.0

EDR_BINS = [
    ("smooth",   0.00, 0.10),
    ("light",    0.10, 0.20),
    ("moderate", 0.20, 0.40),
    ("severe",   0.40, 9.99),
]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat1) / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def cluster_reports(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").reset_index(drop=True)
    ids = np.full(len(df), -1, dtype=int)
    nxt = 0
    for i in range(len(df)):
        if ids[i] != -1:
            continue
        ids[i] = nxt
        for j in range(i + 1, len(df)):
            if ids[j] != -1:
                continue
            if (df.loc[j, "time"] - df.loc[i, "time"]).total_seconds() / 3600 > CLUSTER_HR:
                break
            if haversine_km(df.loc[i, "lat"], df.loc[i, "lon"],
                            df.loc[j, "lat"], df.loc[j, "lon"]) <= CLUSTER_KM:
                ids[j] = nxt
        nxt += 1
    df["cluster_id"] = ids
    return df


def edr_bin(v: float) -> str:
    for name, lo, hi in EDR_BINS:
        if lo <= v < hi:
            return name
    return "severe"


def build_events(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    records = []
    for cid, grp in df.groupby("cluster_id"):
        mx = float(grp["MEDEDR"].max())
        records.append({
            "event_id":   int(cid),
            "center_lat": round(float(grp["lat"].mean()), 4),
            "center_lon": round(float(grp["lon"].mean()), 4),
            "start_utc":  grp["time"].min().isoformat(),
            "end_utc":    grp["time"].max().isoformat(),
            "max_edr":    round(mx, 4),
            "mean_edr":   round(float(grp["MEDEDR"].mean()), 4),
            "n_reports":  len(grp),
            "edr_bin":    edr_bin(mx),
            "lat_min":    round(float(grp["lat"].min()) - BOX_PAD, 4),
            "lat_max":    round(float(grp["lat"].max()) + BOX_PAD, 4),
            "lon_min":    round(float(grp["lon"].min()) - BOX_PAD, 4),
            "lon_max":    round(float(grp["lon"].max()) + BOX_PAD, 4),
        })

    events = pd.DataFrame(records)
    per_bin = max(1, top_n // len(EDR_BINS))
    selected = pd.concat([
        events[events["edr_bin"] == b].nlargest(per_bin, "n_reports")
        for b, *_ in EDR_BINS
    ]).drop_duplicates("event_id")

    remaining = top_n - len(selected)
    if remaining > 0:
        extra = (events[~events["event_id"].isin(selected["event_id"])]
                 .nlargest(remaining, "n_reports"))
        selected = pd.concat([selected, extra])

    selected = selected.sort_values("max_edr", ascending=False).reset_index(drop=True)
    selected["event_id"] = selected.index
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw_reports.parquet")
    parser.add_argument("--out", default="events.csv")
    parser.add_argument("--top", type=int, default=150)
    args = parser.parse_args()

    log.info("Loading raw reports...")
    df = pd.read_parquet(args.raw)
    log.info("Total rows: %d", len(df))
    log.info("EDR distribution:\n%s",
             df["MEDEDR"].describe(percentiles=[.1,.25,.5,.75,.9,.99]).to_string())

    # Filter to turbulent only for clustering
    df_turb = df[df["MEDEDR"] >= EDR_CLUSTER_MIN].copy()
    log.info("Turbulent rows (MEDEDR >= %.2f): %d", EDR_CLUSTER_MIN, len(df_turb))
    del df  # free RAM immediately

    log.info("Clustering %d turbulent reports...", len(df_turb))
    df_turb = cluster_reports(df_turb)
    log.info("Clusters found: %d", df_turb["cluster_id"].nunique())

    events = build_events(df_turb, top_n=args.top)
    log.info("Events per EDR bin:\n%s", events["edr_bin"].value_counts().to_string())

    events.to_csv(args.out, index=False)
    log.info("Saved %d events → %s", args.top, args.out)
    print(events.to_string(index=False))


if __name__ == "__main__":
    main()
