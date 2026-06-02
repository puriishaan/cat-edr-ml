"""
Step 3 — Pull ERA5 reanalysis per event via ARCO-ERA5 on Google Cloud (anonymous, no API key).
For each event, slices u, v, T, z, omega, q over the event bounding box and pressure levels
used at cruise altitude, and saves as NetCDF.

No API key required — ARCO-ERA5 is publicly accessible on GCS anonymously.

Usage:
    python scripts/step3_pull_era5_per_event.py --events events.csv --out data/era5/
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ARCO_URI = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Cruise altitude pressure levels (hPa)
LEVELS = [150, 175, 200, 225, 250, 300, 350, 400, 500]

# Variables needed for CAT diagnostics (TI1, TI2, Ri, N², frontogenesis)
VARIABLES = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "geopotential",
    "vertical_velocity",
    "specific_humidity",
]

TIME_PAD_HR = 2   # fetch ±2 hours around event window


def pull_event(row: pd.Series, ds_full: xr.Dataset, out_dir: Path):
    out_path = out_dir / f"event_{int(row['event_id']):04d}.nc"
    if out_path.exists():
        log.info("Event %d ERA5 already exists, skipping", row["event_id"])
        return

    start = pd.to_datetime(row["start_utc"], utc=True).tz_localize(None) - pd.Timedelta(hours=TIME_PAD_HR)
    end   = pd.to_datetime(row["end_utc"],   utc=True).tz_localize(None) + pd.Timedelta(hours=TIME_PAD_HR)

    # ERA5 longitude is 0-360; convert event box if needed
    lon_min = float(row["lon_min"]) % 360
    lon_max = float(row["lon_max"]) % 360

    try:
        ds_event = ds_full[VARIABLES].sel(
            time=slice(start, end),
            level=LEVELS,
            latitude=slice(float(row["lat_max"]), float(row["lat_min"])),  # ERA5 lat descends
            longitude=slice(lon_min, lon_max),
        ).compute()

        ds_event.to_netcdf(out_path)
        log.info("Event %d ERA5 saved (%s)", row["event_id"], out_path.name)
    except Exception as e:
        log.warning("Event %d ERA5 failed: %s", row["event_id"], e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="events.csv")
    parser.add_argument("--out",    default="data/era5/")
    args = parser.parse_args()

    events  = pd.read_csv(args.events)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Opening ARCO-ERA5 Zarr store (anonymous, no API key needed)...")
    ds_full = xr.open_zarr(
        ARCO_URI,
        chunks=None,
        storage_options={"token": "anon"},
    )
    available_levels = ds_full.level.values.tolist()
    log.info("Available pressure levels: %s", available_levels)

    # Use only levels that exist in the dataset
    global LEVELS
    LEVELS = [l for l in LEVELS if l in available_levels]
    log.info("Using levels: %s", LEVELS)
    log.info("ERA5 store opened. Pulling %d events...", len(events))

    for _, row in events.iterrows():
        pull_event(row, ds_full, out_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
