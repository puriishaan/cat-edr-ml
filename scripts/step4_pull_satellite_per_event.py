"""
Step 4 — Pull GOES Band 13 brightness temperature per event (anonymous S3, no API key).
Uses GOES-16 (pre-2025) and GOES-19 (2025+) from the NOAA public S3 bucket.
Crops to event bounding box immediately — never stores full-disk images.

No API key required — NOAA GOES S3 buckets are publicly accessible anonymously.

Usage:
    python scripts/step4_pull_satellite_per_event.py --events events.csv --out data/satellite/
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# GOES-19 became operational GOES-East in early 2025
def _goes_bucket(year: int) -> str:
    return "noaa-goes19" if year >= 2025 else "noaa-goes16"

TIME_PAD_MIN = 30   # fetch satellite images within ±30 min of event window
BOX_PAD_DEG  = 0.5  # extra padding on satellite crop


def _goes_files(fs: s3fs.S3FileSystem, dt: pd.Timestamp, bucket: str) -> list[str]:
    doy = dt.timetuple().tm_yday
    pattern = f"s3://{bucket}/ABI-L2-CMIPF/{dt.year}/{doy:03d}/{dt.hour:02d}/*C13*.nc"
    try:
        return sorted(fs.glob(pattern))
    except Exception:
        return []


def _latlon_to_xy(lat: float, lon: float, ds: xr.Dataset):
    """Convert lat/lon to GOES fixed-grid (x, y) in radians."""
    import pyproj
    proj_info = ds["goes_imager_projection"]
    p = pyproj.Proj(
        proj="geos",
        h=float(proj_info.attrs["perspective_point_height"]),
        lon_0=float(proj_info.attrs["longitude_of_projection_origin"]),
        sweep=proj_info.attrs.get("sweep_angle_axis", "x"),
    )
    x_m, y_m = p(lon, lat)
    h = float(proj_info.attrs["perspective_point_height"])
    return x_m / h, y_m / h


def pull_event(row: pd.Series, fs: s3fs.S3FileSystem, out_dir: Path):
    out_path = out_dir / f"event_{int(row['event_id']):04d}.parquet"
    if out_path.exists():
        log.info("Event %d satellite already exists, skipping", row["event_id"])
        return

    start = pd.to_datetime(row["start_utc"]) - pd.Timedelta(minutes=TIME_PAD_MIN)
    end   = pd.to_datetime(row["end_utc"])   + pd.Timedelta(minutes=TIME_PAD_MIN)

    # Collect all timestamps in the event window (GOES scans every 10 min)
    times = pd.date_range(start.floor("10min"), end.ceil("10min"), freq="10min")
    bucket = _goes_bucket(start.year)

    records = []
    seen_hours = set()

    for t in times:
        hour_key = (t.year, t.timetuple().tm_yday, t.hour)
        if hour_key in seen_hours:
            continue
        seen_hours.add(hour_key)

        files = _goes_files(fs, t, bucket)
        if not files:
            continue

        # Pick the file closest to t
        for fpath in files:
            try:
                with fs.open(fpath) as f:
                    ds = xr.open_dataset(f, engine="scipy")

                # Crop to event bounding box
                # GOES uses fixed-grid (x, y) in radians — convert bbox corners
                x_min, y_min = _latlon_to_xy(
                    float(row["lat_min"]) - BOX_PAD_DEG,
                    float(row["lon_min"]) - BOX_PAD_DEG, ds
                )
                x_max, y_max = _latlon_to_xy(
                    float(row["lat_max"]) + BOX_PAD_DEG,
                    float(row["lon_max"]) + BOX_PAD_DEG, ds
                )

                cropped = ds["CMI"].sel(
                    x=slice(min(x_min, x_max), max(x_min, x_max)),
                    y=slice(max(y_min, y_max), min(y_min, y_max)),
                )

                if cropped.size == 0:
                    continue

                # Parse scan start time from dataset
                scan_time = pd.to_datetime(
                    ds["t"].values, unit="s",
                    origin=pd.Timestamp("2000-01-01 12:00:00"), utc=True
                )

                records.append({
                    "scan_time":    scan_time,
                    "tb_min":       float(cropped.min()),
                    "tb_max":       float(cropped.max()),
                    "tb_mean":      float(cropped.mean()),
                    "tb_std":       float(cropped.std()),
                    "source_file":  fpath,
                })
                break  # one file per 10-min slot is enough

            except Exception as e:
                log.debug("Could not open %s: %s", fpath, e)
                continue

    if records:
        df = pd.DataFrame(records)
        df["event_id"] = int(row["event_id"])
        df.to_parquet(out_path, index=False)
        log.info("Event %d — %d satellite snapshots saved", row["event_id"], len(df))
    else:
        log.warning("Event %d — no satellite data found", row["event_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="events.csv")
    parser.add_argument("--out",    default="data/satellite/")
    args = parser.parse_args()

    events  = pd.read_csv(args.events)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Connecting to NOAA GOES S3 (anonymous)...")
    fs = s3fs.S3FileSystem(anon=True)

    log.info("Pulling satellite Band 13 for %d events...", len(events))
    for _, row in events.iterrows():
        pull_event(row, fs, out_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
