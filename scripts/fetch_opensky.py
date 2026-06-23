#!/usr/bin/env python3
"""
Download OpenSky Network historical state-vector files and convert to Parquet.

OpenSky publishes 4-second global ADS-B snapshots as hourly CSV.gz files:
  https://opensky-network.org/datasets/states/{year}/{month:02d}/{day:02d}/

A free OpenSky account is required:
  https://opensky-network.org/index.php?option=com_users&view=registration

Strategy
--------
Downloading every hour of every day would be ~TB scale. Instead this script
samples one configurable day per month (default: the 15th). With 4 years ×
12 months × 24 hours = 1,152 files at ~80 MB compressed each, the full default
run is ~90 GB raw but each hourly file is converted to Parquet (~20–30 MB)
immediately after download, so disk at rest is ~25–35 GB.

Usage examples
--------------
  python scripts/fetch_opensky.py --user YOU --password PASS
  python scripts/fetch_opensky.py --user YOU --password PASS --years 2022 2023 --day 10
  python scripts/fetch_opensky.py --user YOU --password PASS --years 2019 2021 2022 2023 --days 8 22

Output layout
-------------
  data/opensky/{year}/{month:02d}/{day:02d}/hour_{hh:02d}.parquet
"""
import argparse
import gzip
import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL   = "https://opensky-network.org/datasets/states"
OUT_DIR    = Path("data/opensky")
CHUNK      = 1 << 20   # 1 MB read chunks

# Column names in OpenSky CSV (positional, no header in file)
_ALL_COLS = [
    "time", "icao24", "lat", "lon", "velocity", "heading",
    "vertrate", "callsign", "onground", "alert", "spi",
    "squawk", "baroaltitude", "geoaltitude", "lastposupdate", "lastcontact",
]

# Only keep what the turbulence analysis needs
KEEP = ["time", "icao24", "lat", "lon", "vertrate", "onground",
        "baroaltitude", "callsign"]

_NUMERIC = {"time": "int32", "lat": "float32", "lon": "float32",
            "vertrate": "float32", "baroaltitude": "float32"}


def _make_session(username: str | None, password: str | None) -> requests.Session:
    s = requests.Session()
    if username and password:
        s.auth = (username, password)
    retry = Retry(total=5, backoff_factor=2,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _url(year: int, month: int, day: int, hour: int) -> str:
    tag = f"{year}-{month:02d}-{day:02d}-{hour:02d}"
    return f"{BASE_URL}/{year}/{month:02d}/{day:02d}/{tag}.csv.gz"


def _download_hour(session: requests.Session,
                   year: int, month: int, day: int, hour: int,
                   out: Path) -> bool:
    """Download one hour, convert CSV.gz → Parquet. Returns True on success."""
    if out.exists():
        log.debug("Skip (exists): %s", out)
        return True

    url = _url(year, month, day, hour)
    try:
        resp = session.get(url, stream=True, timeout=180)
        if resp.status_code == 404:
            log.debug("404: %s", url)
            return False
        resp.raise_for_status()

        buf = io.BytesIO()
        for chunk in resp.iter_content(CHUNK):
            buf.write(chunk)
        buf.seek(0)

        with gzip.open(buf, "rt", encoding="utf-8", errors="replace") as fh:
            df = pd.read_csv(
                fh,
                names=_ALL_COLS,
                usecols=KEEP,
                dtype=str,
                na_values=["", "None", "null", "NaN"],
                low_memory=False,
            )

        # Type coercion
        for col, dtype in _NUMERIC.items():
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
        df["onground"] = df["onground"].map(
            {"True": True, "False": False, "1": True, "0": False, "true": True, "false": False}
        ).fillna(False).astype(bool)

        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, compression="snappy")
        log.debug("  saved %s  (%d rows)", out.name, len(df))
        return True

    except Exception as exc:
        log.warning("Failed %s: %s", url, exc)
        return False


def _download_day(session: requests.Session,
                  year: int, month: int, day: int) -> int:
    log.info("  %04d-%02d-%02d", year, month, day)
    ok = 0
    for hour in range(24):
        out = OUT_DIR / str(year) / f"{month:02d}" / f"{day:02d}" / f"hour_{hour:02d}.parquet"
        if _download_hour(session, year, month, day, hour, out):
            ok += 1
        time.sleep(0.4)   # polite rate limit
    log.info("    %d/24 hours", ok)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch OpenSky state vectors → Parquet (one day/month sample)"
    )
    ap.add_argument("--user",     default=None, help="OpenSky username (required for historical data)")
    ap.add_argument("--password", default=None, help="OpenSky password")
    ap.add_argument("--years",    nargs="+", type=int, default=[2019, 2021, 2022, 2023],
                    help="Years to download (default: 2019 2021 2022 2023)")
    ap.add_argument("--months",   nargs="+", type=int, default=list(range(1, 13)),
                    help="Months (1–12, default: all)")
    ap.add_argument("--day",      type=int,   default=15,
                    help="Day of month to sample (default: 15)")
    ap.add_argument("--days",     nargs="+",  type=int, default=None,
                    help="Multiple days to sample per month (overrides --day)")
    ap.add_argument("--out",      default=str(OUT_DIR),
                    help=f"Output root (default: {OUT_DIR})")
    args = ap.parse_args()

    global OUT_DIR
    OUT_DIR = Path(args.out)

    sample_days = args.days if args.days else [args.day]
    session = _make_session(args.user, args.password)

    log.info("=== OpenSky downloader ===")
    log.info("Years: %s  |  Months: %s  |  Days: %s", args.years, args.months, sample_days)
    log.info("Output: %s", OUT_DIR)

    total = 0
    for year in sorted(args.years):
        for month in sorted(args.months):
            for day in sorted(sample_days):
                total += _download_day(session, year, month, day)

    log.info("Done. %d hourly Parquet files in %s/", total, OUT_DIR)


if __name__ == "__main__":
    main()
