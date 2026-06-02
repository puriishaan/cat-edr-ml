"""
Step 2 — Pull full ACARS data per event
For each event in events.csv, pull ALL ACARS reports (turbulent + smooth)
within the event bounding box and time window from public MADIS archive.
These form both positive and negative training samples.

Usage:
    python scripts/step2_pull_acars_per_event.py --events events.csv --out data/acars/
"""

import argparse
import asyncio
import gzip
import logging
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MADIS_BASE = (
    "https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/"
    "{year}/{month:02d}/{day:02d}/point/acars/netcdf/{stem}.gz"
)

MAX_CONCURRENT = 20
MISSING        = -100
ALT_MIN_M      = 5500
TIME_PAD_HR    = 2      # fetch ±2 hours around event window


def _hours_for_event(row: pd.Series) -> list[datetime]:
    start = pd.to_datetime(row["start_utc"]).to_pydatetime().replace(tzinfo=None) - timedelta(hours=TIME_PAD_HR)
    end   = pd.to_datetime(row["end_utc"]).to_pydatetime().replace(tzinfo=None) + timedelta(hours=TIME_PAD_HR)
    hours = []
    t = start.replace(minute=0, second=0, microsecond=0)
    while t <= end:
        hours.append(t)
        t += timedelta(hours=1)
    return hours


def _parse(content: bytes, row: pd.Series) -> pd.DataFrame | None:
    try:
        with gzip.open(BytesIO(content)) as f:
            ds = xr.open_dataset(f, engine="scipy")
    except Exception:
        return None

    needed = {"medEDR", "maxEDR", "latitude", "longitude", "altitude", "timeObs"}
    if needed - set(ds.data_vars) - set(ds.coords):
        return None

    df = pd.DataFrame({
        "time":   pd.to_datetime(ds["timeObs"].values, unit="s", utc=True),
        "lat":    ds["latitude"].values.astype("float32"),
        "lon":    ds["longitude"].values.astype("float32"),
        "alt_m":  ds["altitude"].values.astype("float32"),
        "MEDEDR": ds["medEDR"].values.astype("float32"),
        "MAXEDR": ds["maxEDR"].values.astype("float32"),
    })

    for col in ["lat", "lon", "alt_m", "MEDEDR", "MAXEDR"]:
        df[col] = df[col].where(df[col] > MISSING)

    df = df.dropna(subset=["lat", "lon", "MEDEDR", "time"])
    df = df[df["alt_m"] >= ALT_MIN_M]

    # Spatial filter to event bounding box
    df = df[
        (df["lat"] >= row["lat_min"]) & (df["lat"] <= row["lat_max"]) &
        (df["lon"] >= row["lon_min"]) & (df["lon"] <= row["lon_max"])
    ]

    return df.reset_index(drop=True) if len(df) > 0 else None


async def _fetch_one(session, ts: datetime, row: pd.Series, sem) -> pd.DataFrame | None:
    url = MADIS_BASE.format(
        year=ts.year, month=ts.month, day=ts.day,
        stem=ts.strftime("%Y%m%d_%H00"),
    )
    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as r:
                if r.status != 200:
                    return None
                content = await r.read()
        except Exception:
            return None
    return _parse(content, row)


async def fetch_event(row: pd.Series, out_dir: Path):
    out_path = out_dir / f"event_{int(row['event_id']):04d}.parquet"
    if out_path.exists():
        log.info("Event %d already exists, skipping", row["event_id"])
        return

    hours = _hours_for_event(row)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    frames = []

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_fetch_one(session, ts, row, sem) for ts in hours]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is not None:
                frames.append(result)

    if frames:
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time", "lat", "lon"])
        df["event_id"] = int(row["event_id"])
        df.to_parquet(out_path, index=False)
        log.info("Event %d — %d reports saved", row["event_id"], len(df))
    else:
        log.warning("Event %d — no reports found", row["event_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="events.csv")
    parser.add_argument("--out",    default="data/acars/")
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Pulling ACARS for %d events", len(events))
    for _, row in events.iterrows():
        asyncio.run(fetch_event(row, out_dir))

    log.info("Done. Files in %s:", out_dir)
    files = list(out_dir.glob("*.parquet"))
    log.info("%d event files written", len(files))


if __name__ == "__main__":
    main()
