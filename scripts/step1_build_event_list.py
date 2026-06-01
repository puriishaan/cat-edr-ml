"""
Step 1 — Build the Event List (efficient 20-year scan)

Designed for: 24 GB RAM, 200 GB+ storage, ~50-100 Mbps connection.

Strategy:
  1. Sample every 3 hours (CAT events last 2-6 hrs; nothing missed, 3x fewer files).
  2. HEAD request first — skip files under MIN_FILE_BYTES (sparse overnight hours).
  3. 20 concurrent async downloads — saturates bandwidth instead of waiting serially.
  4. Stream in memory, extract 6 columns only, never write raw NetCDF to disk.
  5. Flush to Parquet every FLUSH_EVERY files — RAM stays flat at ~1-2 GB throughout.
  6. Checkpoint: if script crashes, resume from the last flushed Parquet.
  7. Stratified event selection across EDR bins so output covers full EDR range.

Estimated output: ~5-15 GB Parquet for 20 years (6 columns only).
Estimated runtime: 4-10 hours depending on connection speed.

Usage:
    # Quick test — one month
    python scripts/step1_build_event_list.py --start 2023-01-01 --end 2023-01-31 --top 20

    # Full 20-year scan
    python scripts/step1_build_event_list.py --start 2004-01-01 --end 2024-12-31 --top 150
"""

import argparse
import asyncio
import gzip
import logging
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MADIS_BASE = (
    "https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/"
    "{year}/{month:02d}/{day:02d}/point/acars/netcdf/{stem}.gz"
)

SAMPLE_HOURS   = [0, 3, 6, 9, 12, 15, 18, 21]
MIN_FILE_BYTES = 200_000   # skip files smaller than this (~sparse traffic)
MAX_CONCURRENT = 20        # parallel downloads
FLUSH_EVERY    = 500       # write to Parquet after this many completed downloads
ALT_MIN_M      = 5500      # FL100+
MISSING        = -100

EDR_BINS = [
    ("smooth",   0.00, 0.10),
    ("light",    0.10, 0.20),
    ("moderate", 0.20, 0.40),
    ("severe",   0.40, 9.99),
]

CLUSTER_KM = 200
CLUSTER_HR = 6
BOX_PAD    = 2.0

PARQUET_SCHEMA = pa.schema([
    ("time",   pa.timestamp("ms", tz="UTC")),
    ("lat",    pa.float32()),
    ("lon",    pa.float32()),
    ("alt_m",  pa.float32()),
    ("MEDEDR", pa.float32()),
    ("MAXEDR", pa.float32()),
])


# ---------------------------------------------------------------------------
# Download + parse
# ---------------------------------------------------------------------------

def _url(ts: datetime) -> str:
    return MADIS_BASE.format(
        year=ts.year, month=ts.month, day=ts.day,
        stem=ts.strftime("%Y%m%d_%H00"),
    )


def _parse(content: bytes) -> pd.DataFrame | None:
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
    df = df[(df["alt_m"] >= ALT_MIN_M) & (df["MEDEDR"] >= 0.0) & (df["MEDEDR"] < 2.0)]
    return df.reset_index(drop=True) if len(df) > 0 else None


async def _fetch_one(
    session: aiohttp.ClientSession,
    ts: datetime,
    semaphore: asyncio.Semaphore,
) -> pd.DataFrame | None:
    url = _url(ts)
    async with semaphore:
        try:
            async with session.head(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status != 200:
                    return None
                if int(r.headers.get("Content-Length", 0)) < MIN_FILE_BYTES:
                    return None
        except Exception:
            return None

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as r:
                if r.status != 200:
                    return None
                content = await r.read()
        except Exception as e:
            log.warning("Download failed %s: %s", ts, e)
            return None

    return _parse(content)


# ---------------------------------------------------------------------------
# Streaming Parquet writer
# ---------------------------------------------------------------------------

class ParquetSink:
    """Accumulates DataFrames and flushes to Parquet every flush_every rows."""

    def __init__(self, path: Path, flush_every: int = FLUSH_EVERY):
        self.path = path
        self.flush_every = flush_every
        self._buf: list[pd.DataFrame] = []
        self._buf_rows = 0
        self._writer: pq.ParquetWriter | None = None
        self._total_rows = 0
        self._files_seen = 0

    def add(self, df: pd.DataFrame | None):
        self._files_seen += 1
        if df is not None and len(df) > 0:
            self._buf.append(df)
            self._buf_rows += len(df)

        if self._files_seen % self.flush_every == 0:
            self._flush()
            log.info("Files processed: %d | Rows on disk: %d", self._files_seen, self._total_rows)

    def _flush(self):
        if not self._buf:
            return
        combined = pd.concat(self._buf, ignore_index=True)
        table = pa.Table.from_pandas(combined, schema=PARQUET_SCHEMA, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, PARQUET_SCHEMA)
        self._writer.write_table(table)
        self._total_rows += len(combined)
        self._buf = []
        self._buf_rows = 0

    def close(self) -> int:
        self._flush()
        if self._writer:
            self._writer.close()
        return self._total_rows


# ---------------------------------------------------------------------------
# Async download loop
# ---------------------------------------------------------------------------

async def run_downloads(timestamps: list[datetime], sink: ParquetSink):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = {
            asyncio.ensure_future(_fetch_one(session, ts, semaphore)): ts
            for ts in timestamps
        }
        for future in asyncio.as_completed(tasks):
            result = await future
            sink.add(result)


# ---------------------------------------------------------------------------
# Resume: figure out which timestamps are already done
# ---------------------------------------------------------------------------

def already_downloaded_range(parquet_path: Path) -> tuple[datetime | None, datetime | None]:
    if not parquet_path.exists():
        return None, None
    df = pd.read_parquet(parquet_path, columns=["time"])
    t = pd.to_datetime(df["time"])
    return t.min().to_pydatetime(), t.max().to_pydatetime()


# ---------------------------------------------------------------------------
# Clustering + event building
# ---------------------------------------------------------------------------

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


def build_events(df_all: pd.DataFrame, top_n: int) -> pd.DataFrame:
    records = []
    for cid, grp in df_all.groupby("cluster_id"):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",      default="2004-01-01")
    parser.add_argument("--end",        default="2024-12-31")
    parser.add_argument("--top",        type=int, default=150)
    parser.add_argument("--out",        default="events.csv")
    parser.add_argument("--raw",        default="data/raw_reports.parquet",
                        help="Streaming Parquet of all extracted reports (kept for reuse)")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip timestamps already covered in existing Parquet")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Ignore existing Parquet and download everything fresh")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end   = datetime.strptime(args.end,   "%Y-%m-%d")
    raw_path = Path(args.raw)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Step A: download (or resume) ---
    done_min, done_max = already_downloaded_range(raw_path)
    if done_min and done_max and not args.overwrite:
        log.info("Existing raw data covers %s → %s", done_min.date(), done_max.date())
        if args.resume:
            all_ts = [
                ts for ts in _build_timestamps(start, end)
                if not (done_min.replace(tzinfo=None) <= ts <= done_max.replace(tzinfo=None))
            ]
            log.info("Resuming: %d timestamps remaining", len(all_ts))
        else:
            log.info("Use --resume to skip existing data or --overwrite to start fresh.")
            all_ts = _build_timestamps(start, end)
    else:
        all_ts = _build_timestamps(start, end)

    if all_ts:
        log.info(
            "Downloading %d timestamps (%d concurrent). RAM stays flat; flushing every %d files.",
            len(all_ts), MAX_CONCURRENT, FLUSH_EVERY,
        )
        sink = ParquetSink(raw_path)
        t0 = time.time()
        asyncio.run(run_downloads(all_ts, sink))
        total_rows = sink.close()
        log.info("Download complete: %d rows written in %.0f s", total_rows, time.time() - t0)
    else:
        log.info("No new timestamps to download.")

    # --- Step B: cluster + build event list ---
    log.info("Loading raw reports from %s …", raw_path)
    df_all = pd.read_parquet(raw_path)
    log.info("Total reports: %d", len(df_all))
    log.info("EDR distribution:\n%s",
             df_all["MEDEDR"].describe(percentiles=[.1, .25, .5, .75, .9, .99]).to_string())

    log.info("Clustering …")
    df_all = cluster_reports(df_all)
    log.info("Clusters: %d", df_all["cluster_id"].nunique())

    events = build_events(df_all, top_n=args.top)
    log.info("Events per EDR bin:\n%s", events["edr_bin"].value_counts().to_string())
    events.to_csv(args.out, index=False)
    log.info("Saved %d events → %s", len(events), args.out)
    print(events.to_string(index=False))


def _build_timestamps(start: datetime, end: datetime) -> list[datetime]:
    stamps, dt = [], start
    while dt <= end:
        for h in SAMPLE_HOURS:
            stamps.append(dt.replace(hour=h))
        dt += timedelta(days=1)
    return stamps


if __name__ == "__main__":
    main()
