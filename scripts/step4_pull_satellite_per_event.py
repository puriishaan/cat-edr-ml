"""
Step 4 — Pull IR brightness temperature per event (anonymous S3, no API key).

Satellite coverage by era:
  pre-2010:   No suitable public S3 data → skip, mark NaN
  2010-2017:  GOES-13 (GOES-East), Channel 4 (10.7 µm IR window)
  2017-2025:  GOES-16 ABI L2-CMIPF, Band 13 (10.3 µm)
  2025+:      GOES-19 ABI L2-CMIPF, Band 13 (10.3 µm)

Crops to event bounding box immediately — never stores full-disk images.

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

TIME_PAD_MIN = 30
BOX_PAD_DEG  = 0.5


# ---------------------------------------------------------------------------
# Satellite era routing
# ---------------------------------------------------------------------------

def _era(year: int) -> str:
    if year < 2010:
        return "none"
    elif year < 2018:
        return "goes13"
    elif year < 2025:
        return "goes16"
    else:
        return "goes19"


# ---------------------------------------------------------------------------
# GOES-16/19 ABI L2-CMIPF (2017+)
# ---------------------------------------------------------------------------

def _abi_files(fs, dt: pd.Timestamp, bucket: str) -> list[str]:
    doy = dt.timetuple().tm_yday
    pattern = f"s3://{bucket}/ABI-L2-CMIPF/{dt.year}/{doy:03d}/{dt.hour:02d}/*C13*.nc"
    try:
        return sorted(fs.glob(pattern))
    except Exception:
        return []


def _latlon_to_xy(lat: float, lon: float, ds: xr.Dataset):
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


def _pull_abi(fs, t: pd.Timestamp, bucket: str, row: pd.Series) -> dict | None:
    files = _abi_files(fs, t, bucket)
    if not files:
        return None
    for fpath in files:
        try:
            with fs.open(fpath) as f:
                ds = xr.open_dataset(f, engine="netcdf4")
            x_min, y_min = _latlon_to_xy(float(row["lat_min"]) - BOX_PAD_DEG,
                                          float(row["lon_min"]) - BOX_PAD_DEG, ds)
            x_max, y_max = _latlon_to_xy(float(row["lat_max"]) + BOX_PAD_DEG,
                                          float(row["lon_max"]) + BOX_PAD_DEG, ds)
            cropped = ds["CMI"].sel(
                x=slice(min(x_min, x_max), max(x_min, x_max)),
                y=slice(max(y_min, y_max), min(y_min, y_max)),
            )
            if cropped.size == 0:
                continue
            scan_time = pd.to_datetime(
                ds["t"].values, unit="s",
                origin=pd.Timestamp("2000-01-01 12:00:00"), utc=True
            )
            return {
                "scan_time": scan_time,
                "tb_min":    float(cropped.min()),
                "tb_max":    float(cropped.max()),
                "tb_mean":   float(cropped.mean()),
                "tb_std":    float(cropped.std()),
                "source":    fpath,
            }
        except Exception as e:
            log.debug("ABI file error %s: %s", fpath, e)
    return None


# ---------------------------------------------------------------------------
# GOES-13 (2010-2017)
# GOES-13 S3 bucket: noaa-goes13
# Path: GOES13/YYYY/DDD/HH/goes13.YYYY.DDD.HHMM.G.nc
# IR window is Channel 4 (~10.7 µm), variable name varies by file
# ---------------------------------------------------------------------------

def _goes13_files(fs, dt: pd.Timestamp) -> list[str]:
    doy = dt.timetuple().tm_yday
    pattern = f"s3://noaa-goes13/GOES13/{dt.year}/{doy:03d}/{dt.hour:02d}/*.nc"
    try:
        return sorted(fs.glob(pattern))
    except Exception:
        return []


def _pull_goes13(fs, t: pd.Timestamp, row: pd.Series) -> dict | None:
    files = _goes13_files(fs, t)
    if not files:
        return None

    lat_min = float(row["lat_min"]) - BOX_PAD_DEG
    lat_max = float(row["lat_max"]) + BOX_PAD_DEG
    lon_min = float(row["lon_min"]) - BOX_PAD_DEG
    lon_max = float(row["lon_max"]) + BOX_PAD_DEG

    for fpath in files:
        try:
            with fs.open(fpath) as f:
                ds = xr.open_dataset(f, engine="netcdf4")

            # GOES-13 files have lat/lon coordinates directly
            # Find IR channel variable — typically 'IR' or 'data' or channel 4
            ir_var = None
            for candidate in ["IR", "IR_WV", "data", "channel_4", "ch4"]:
                if candidate in ds.data_vars:
                    ir_var = candidate
                    break
            if ir_var is None:
                # Try first non-coordinate variable
                candidates = [v for v in ds.data_vars
                              if ds[v].ndim >= 2]
                if candidates:
                    ir_var = candidates[0]
                else:
                    continue

            da = ds[ir_var]

            # Crop by lat/lon if coordinates exist
            if "lat" in ds.coords and "lon" in ds.coords:
                da = da.where(
                    (ds.lat >= lat_min) & (ds.lat <= lat_max) &
                    (ds.lon >= lon_min) & (ds.lon <= lon_max),
                    drop=True
                )
            elif "latitude" in ds.coords and "longitude" in ds.coords:
                da = da.where(
                    (ds.latitude >= lat_min) & (ds.latitude <= lat_max) &
                    (ds.longitude >= lon_min) & (ds.longitude <= lon_max),
                    drop=True
                )

            if da.size == 0 or float(da.count()) == 0:
                continue

            # Convert to Kelvin if needed (GOES-13 may store as counts or Celsius)
            vals = da.values.astype(float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue

            # Heuristic: if values look like counts (0-1023) convert to K
            if vals.max() < 400 and vals.min() >= 0:
                pass  # already looks like K or close
            elif vals.max() > 1000:
                # Raw counts — skip, can't convert without calibration table
                continue

            scan_time = pd.Timestamp(t, tz="UTC")
            return {
                "scan_time": scan_time,
                "tb_min":    float(vals.min()),
                "tb_max":    float(vals.max()),
                "tb_mean":   float(vals.mean()),
                "tb_std":    float(vals.std()),
                "source":    fpath,
            }
        except Exception as e:
            log.debug("GOES-13 file error %s: %s", fpath, e)
    return None


# ---------------------------------------------------------------------------
# Main event puller
# ---------------------------------------------------------------------------

def pull_event(row: pd.Series, fs: s3fs.S3FileSystem, out_dir: Path):
    out_path = out_dir / f"event_{int(row['event_id']):04d}.parquet"
    if out_path.exists():
        log.info("Event %d satellite already exists, skipping", row["event_id"])
        return

    start = pd.to_datetime(row["start_utc"], utc=True) - pd.Timedelta(minutes=TIME_PAD_MIN)
    end   = pd.to_datetime(row["end_utc"],   utc=True) + pd.Timedelta(minutes=TIME_PAD_MIN)
    era   = _era(start.year)

    if era == "none":
        log.warning("Event %d — pre-2010, no public satellite data available", row["event_id"])
        return

    times = pd.date_range(start.floor("10min"), end.ceil("10min"), freq="10min")
    bucket = {"goes13": "noaa-goes13", "goes16": "noaa-goes16", "goes19": "noaa-goes19"}[era]

    records = []
    seen_hours = set()

    for t in times:
        hour_key = (t.year, t.timetuple().tm_yday, t.hour)
        if hour_key in seen_hours:
            continue
        seen_hours.add(hour_key)

        if era in ("goes16", "goes19"):
            rec = _pull_abi(fs, t, bucket, row)
        else:
            rec = _pull_goes13(fs, t, row)

        if rec:
            records.append(rec)

    if records:
        df = pd.DataFrame(records)
        df["event_id"] = int(row["event_id"])
        df["satellite"] = era
        df.to_parquet(out_path, index=False)
        log.info("Event %d (%s) — %d snapshots saved", row["event_id"], era, len(df))
    else:
        log.warning("Event %d (%s, %d) — no satellite data found", row["event_id"], era, start.year)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="events.csv")
    parser.add_argument("--out",    default="data/satellite/")
    args = parser.parse_args()

    events  = pd.read_csv(args.events)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Connecting to NOAA S3 (anonymous)...")
    fs = s3fs.S3FileSystem(anon=True)

    log.info("Pulling satellite IR for %d events...", len(events))
    for _, row in events.iterrows():
        pull_event(row, fs, out_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
