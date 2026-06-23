#!/usr/bin/env python3
"""
Download OpenSky Network state-vector samples and convert to Parquet.

OpenSky's old hourly archive (datasets/states/{y}/{m}/{d}/{tag}.csv.gz) is no
longer served.  The free public data is now a weekly-snapshot S3 bucket:

  https://s3.opensky-network.org/data-samples/states/{YYYY-MM-DD}/{hh}/states_{YYYY-MM-DD}-{hh}.csv.tar

Each snapshot is one Monday, with 24 hourly ``*.csv.tar`` files (~130 MB each;
each tar wraps a single ``*.csv.gz`` plus LICENSE/README).  The bucket is public
— no credentials needed (``--user``/``--password`` are accepted for backwards
compatibility but ignored).  The freely downloadable range is roughly
2020-05-25 → 2022-06-27.

This script:
  1. Lists the snapshot dates actually present for the requested year(s).
  2. Builds the full (date, hour) work list and shuffles it (``--seed``), so a
     size-capped run spreads coverage across the whole year rather than
     front-loading January.
  3. Downloads each tar, extracts the inner CSV, keeps only the columns the
     turbulence analysis needs, and (by default) filters to cruise rows
     (FL180+, airborne, valid vertical rate) so a full year fits in a few GB.
  4. Writes Parquet in the layout the analysis scripts expect and stops once
     the on-disk total reaches ``--max-gb``.

Output layout
-------------
  data/opensky/{year}/{month:02d}/{day:02d}/hour_{hh:02d}.parquet
"""
from __future__ import annotations

import argparse
import gzip
import io
import logging
import random
import tarfile
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BUCKET_BASE = "https://s3.opensky-network.org/data-samples"
LIST_URL    = "https://s3.opensky-network.org/data-samples/"
OUT_DIR     = Path("data/opensky")
CHUNK       = 1 << 20   # 1 MB read chunks
MIN_ALT_M   = 5_500     # FL180 — cruise filter floor (matches analysis)

_ALL_COLS = [
    "time", "icao24", "lat", "lon", "velocity", "heading",
    "vertrate", "callsign", "onground", "alert", "spi",
    "squawk", "baroaltitude", "geoaltitude", "lastposupdate", "lastcontact",
]
KEEP = ["time", "icao24", "lat", "lon", "vertrate", "onground",
        "baroaltitude", "callsign"]
_NUMERIC = {"time": "int32", "lat": "float32", "lon": "float32",
            "vertrate": "float32", "baroaltitude": "float32"}


def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=2,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def list_snapshot_dates(session: requests.Session, year: int) -> list[str]:
    """Return sorted YYYY-MM-DD snapshot folders present for ``year``."""
    dates, token = [], None
    prefix = f"states/{year}-"
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    while True:
        params = {"list-type": "2", "prefix": prefix, "delimiter": "/"}
        if token:
            params["continuation-token"] = token
        r = session.get(LIST_URL, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for cp in root.findall(f"{ns}CommonPrefixes"):
            p = cp.findtext(f"{ns}Prefix", "")          # e.g. states/2022-01-03/
            name = p.rstrip("/").split("/")[-1]
            if name.startswith("."):                    # hidden / deprecated
                continue
            if len(name) == 10 and name[:4] == str(year):
                dates.append(name)
        if root.findtext(f"{ns}IsTruncated") == "true":
            token = root.findtext(f"{ns}NextContinuationToken")
        else:
            break
    return sorted(set(dates))


def _hour_url(date: str, hour: int) -> str:
    return f"{BUCKET_BASE}/states/{date}/{hour:02d}/states_{date}-{hour:02d}.csv.tar"


def _out_path(date: str, hour: int) -> Path:
    y, m, d = date.split("-")
    return OUT_DIR / y / m / d / f"hour_{hour:02d}.parquet"


def _read_inner_csv(tar_bytes: bytes) -> pd.DataFrame:
    """Extract the single inner *.csv.gz from a downloaded *.csv.tar."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith(".csv.gz"))
        gz = tf.extractfile(member)
        with gzip.open(gz, "rt", encoding="utf-8", errors="replace") as fh:
            return pd.read_csv(
                fh,
                header=0,                 # inner CSV now ships a header row
                names=_ALL_COLS,          # enforce known column order anyway
                usecols=KEEP,
                dtype=str,
                na_values=["", "None", "null", "NaN"],
                low_memory=False,
            )


def _download_hour(session: requests.Session, date: str, hour: int,
                   cruise_only: bool) -> bool:
    out = _out_path(date, hour)
    if out.exists():
        log.debug("skip (exists) %s", out)
        return True

    url = _hour_url(date, hour)
    try:
        resp = session.get(url, stream=True, timeout=300)
        if resp.status_code == 404:
            log.debug("404 %s", url)
            return False
        resp.raise_for_status()
        buf = io.BytesIO()
        for chunk in resp.iter_content(CHUNK):
            buf.write(chunk)

        df = _read_inner_csv(buf.getvalue())

        for col, dtype in _NUMERIC.items():
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
        df["onground"] = df["onground"].map(
            {"True": True, "False": False, "1": True, "0": False,
             "true": True, "false": False}
        ).fillna(False).astype(bool)

        if cruise_only:
            df = df[
                (~df["onground"]) &
                (df["baroaltitude"] > MIN_ALT_M) &
                df["vertrate"].notna() &
                df["lat"].notna() & df["lon"].notna()
            ]

        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, compression="snappy")
        log.info("  saved %s  (%d rows, %.1f MB)",
                 out, len(df), out.stat().st_size / 1e6)
        return True
    except Exception as exc:
        log.warning("failed %s: %s", url, exc)
        return False


def _disk_gb() -> float:
    return sum(p.stat().st_size for p in OUT_DIR.rglob("*.parquet")) / 1e9


def main() -> None:
    global OUT_DIR
    ap = argparse.ArgumentParser(
        description="Fetch OpenSky state-vector samples (S3 bucket) → Parquet")
    ap.add_argument("--user",     default=None, help="(ignored; bucket is public)")
    ap.add_argument("--password", default=None, help="(ignored; bucket is public)")
    ap.add_argument("--years",  nargs="+", type=int, default=[2022],
                    help="Years to sample (default: 2022)")
    ap.add_argument("--max-gb", type=float, default=18.0,
                    help="Stop once on-disk Parquet reaches this many GB (default 18)")
    ap.add_argument("--seed",   type=int, default=42, help="Shuffle seed")
    ap.add_argument("--workers", type=int, default=6,
                    help="Concurrent download/convert workers (default 6)")
    ap.add_argument("--no-filter", action="store_true",
                    help="Keep all rows (default: cruise-only, FL180+ airborne)")
    ap.add_argument("--out",    default=str(OUT_DIR), help=f"Output root ({OUT_DIR})")
    args = ap.parse_args()
    OUT_DIR = Path(args.out)

    session = _make_session()
    cruise_only = not args.no_filter

    work: list[tuple[str, int]] = []
    for year in sorted(args.years):
        dates = list_snapshot_dates(session, year)
        log.info("%d: %d snapshot dates available (%s … %s)",
                 year, len(dates), dates[0] if dates else "-",
                 dates[-1] if dates else "-")
        for d in dates:
            for h in range(24):
                work.append((d, h))

    random.Random(args.seed).shuffle(work)
    log.info("=== OpenSky S3 downloader ===")
    log.info("Work items: %d hourly files | cap: %.1f GB | cruise_only=%s | out=%s",
             len(work), args.max_gb, cruise_only, OUT_DIR)

    counters = {"saved": 0, "done": 0}
    lock = threading.Lock()
    stop = threading.Event()

    def _job(item: tuple[str, int]) -> None:
        if stop.is_set():
            return
        date, hour = item
        ok = _download_hour(session, date, hour, cruise_only)
        with lock:
            counters["done"] += 1
            if ok:
                counters["saved"] += 1
            n = counters["done"]
        if n % 20 == 0:
            gb = _disk_gb()
            log.info("progress %d/%d  saved=%d  disk=%.2f GB",
                     n, len(work), counters["saved"], gb)
            if gb >= args.max_gb:
                log.info("Reached %.1f GB cap — stopping new downloads.", args.max_gb)
                stop.set()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_job, it) for it in work]
        for f in as_completed(futures):
            f.result()

    log.info("Done. %d files, %.2f GB on disk under %s/",
             counters["saved"], _disk_gb(), OUT_DIR)


if __name__ == "__main__":
    main()
