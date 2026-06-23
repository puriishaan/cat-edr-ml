#!/usr/bin/env python3
"""
Climate-index correlation of EEMD IMF modes — ACARS turbulence, Eastern N. America
==================================================================================
Goal (Ishaan): correlate the multi-year variability captured by the EEMD IMF
modes of the ACARS turbulence record against large-scale climate indices
(ENSO, QBO, PDO, NAO, AO, MJO) and decide which modes are physically forced.

What this script does
---------------------
  1. Rebuilds a monthly turbulence signal from results/cache_ym_agg.parquet
     using the SAME preprocessing as analysis_emd.py (count thresholds, the
     2009-2011 event-window exclusion, COVID Apr-May 2020 exclusion, linear
     interpolation).  The signal is the monthly fraction of ACARS reports with
     MEDEDR >= 0.20 (turbulence occurrence rate).

       NOTE: analysis_emd.py decomposes the monthly *mean above-threshold EDR*
       (intensity), which lives only in data/raw_reports.parquet (116 M rows,
       not present locally).  frac_m (occurrence rate) is the closest signal
       fully reproducible from the committed cache and is highly correlated
       with intensity.  If raw_reports.parquet is restored, set
       SIGNAL = "edr_mean_above" and it will decompose the exact EEMD signal.

  2. Runs EEMD (identical params to analysis_emd.py) -> IMFs + residue.
  3. Fetches monthly climate indices from NOAA PSL (+ MJO RMM from BoM).
  4. For each index, finds the IMF whose dominant period matches the index's
     native timescale, then computes a LAGGED cross-correlation (index leading
     turbulence by -12..+12 months).  Reports r at best lag, the lag, and a
     phase-randomised surrogate p-value (because narrowband IMFs correlate with
     anything narrowband — the analytic p-value is meaningless here).
  5. Also correlates each index against the IMF's Hilbert amplitude envelope
     (does turbulence variability strengthen during, e.g., El Nino?).
  6. Writes results/climate_index_findings.md and
     results/figures/climate_indices.png plus a full IMF x index matrix.

Outputs
-------
  results/figures/climate_indices.png
  results/climate_index_findings.md
"""
import warnings; warnings.filterwarnings("ignore")
import logging; logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

import io
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import linregress
from PyEMD import EEMD

# ─── Config (mirrors analysis_emd.py) ─────────────────────────────────────────
YM_CACHE     = Path("results/cache_ym_agg.parquet")
OUT          = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)
RES          = Path("results")

SIGNAL       = "frac_m"                     # frac_m | frac_s | edr_mean_above
MIN_N_TOTAL  = 10_000                       # min total reports per month
BAD_YEARS    = {2009, 2010, 2011}           # event-window sampling bias
COVID_MONTHS = ["2020-04", "2020-05"]       # COVID reduced-traffic anomaly

EEMD_TRIALS      = 100
EEMD_NOISE_WIDTH = 0.05
SEED             = 42

MAX_LAG_MONTHS = 12                          # cross-correlation search window
N_SURROGATES   = 1000                        # phase-randomised significance

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# Index -> (URL, native-period-in-months band, kind).  band is used to pick the
# matching IMF; "envelope" indices are amplitude-modulators we also test on the
# IMF Hilbert envelope.
PSL = "https://psl.noaa.gov/data/correlation/{}.data"
INDICES = {
    "ONI (ENSO)":  dict(url=PSL.format("oni"),    band=(30, 60),  kind="psl"),
    "Nino3.4":     dict(url=PSL.format("nina34"),  band=(30, 60),  kind="psl"),
    "QBO":         dict(url=PSL.format("qbo"),     band=(24, 32),  kind="psl"),
    "PDO":         dict(url=PSL.format("pdo"),     band=(60, 200), kind="psl"),
    "NAO":         dict(url=PSL.format("nao"),     band=(2, 18),   kind="psl"),
    "AO":          dict(url=PSL.format("ao"),      band=(2, 18),   kind="psl"),
    "MJO (RMM amp)": dict(url="http://www.bom.gov.au/clim_data/IDCKGEM000/rmm.74toRealtime.txt",
                          band=(2, 4), kind="rmm"),
}


# ─── Signal construction ──────────────────────────────────────────────────────
def prepare_series() -> tuple[pd.Series, pd.Series]:
    """Monthly turbulence signal from the committed cache, EMD preprocessing.

    Returns (ts_clean, valid_mask) on a continuous monthly index.  valid_mask is
    True for months that were *actually observed* (not interpolated/excluded) —
    correlations are computed only on those months.
    """
    ym = pd.read_parquet(YM_CACHE)
    ym["date"] = pd.to_datetime(ym["ym_key"].astype(str), format="%Y%m")
    ym = ym.sort_values("date").reset_index(drop=True)
    ym = ym[ym["n"] >= MIN_N_TOTAL]

    full = pd.date_range(ym["date"].min(), ym["date"].max(), freq="MS")
    ts_raw = ym.set_index("date")[SIGNAL].reindex(full)

    valid = ts_raw.notna().copy()
    valid[valid.index.year.isin(BAD_YEARS)] = False
    for cm in COVID_MONTHS:
        if pd.Timestamp(cm) in valid.index:
            valid[pd.Timestamp(cm)] = False

    ts_clean = ts_raw.copy()
    ts_clean[~valid] = np.nan
    ts_clean = ts_clean.interpolate("linear").bfill().ffill()

    log.info("Signal '%s': %d months (%s–%s), %d valid / %d interpolated",
             SIGNAL, len(ts_clean), full[0].strftime("%Y-%m"),
             full[-1].strftime("%Y-%m"), int(valid.sum()), int((~valid).sum()))
    return ts_clean, valid


# ─── EEMD ─────────────────────────────────────────────────────────────────────
def run_eemd(ts: pd.Series) -> np.ndarray:
    mean, std = float(ts.mean()), float(ts.std())
    y = (ts.values - mean) / std
    log.info("EEMD: %d trials, noise_width=%.3f ...", EEMD_TRIALS, EEMD_NOISE_WIDTH)
    eemd = EEMD(trials=EEMD_TRIALS, noise_width=EEMD_NOISE_WIDTH)
    eemd.noise_seed(SEED)
    imfs = eemd.eemd(y, max_imf=12) * std
    log.info("EEMD produced %d IMFs", len(imfs))
    return imfs


def dominant_period_months(imf: np.ndarray) -> float:
    if np.std(imf) < 1e-14:
        return np.nan
    signs = np.sign(imf); signs[signs == 0] = 1
    zc = np.where(np.diff(signs) != 0)[0]
    return float(2.0 * np.mean(np.diff(zc))) if len(zc) >= 2 else np.nan


def period_label(m: float) -> str:
    if np.isnan(m):           return "—"
    if m < 14:                return f"~{m:.0f} mo"
    return f"~{m/12:.1f} yr"


# ─── Index fetching / parsing ─────────────────────────────────────────────────
def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def parse_psl(text: str) -> pd.Series:
    """PSL '.data' format: header 'startyr endyr', then 'year v1..v12' rows."""
    recs = {}
    for line in text.splitlines():
        t = line.split()
        if len(t) != 13:
            continue
        try:
            yr = int(t[0]); vals = [float(x) for x in t[1:]]
        except ValueError:
            continue
        if not (1900 <= yr <= 2100):
            continue
        for mo, v in enumerate(vals, start=1):
            recs[pd.Timestamp(yr, mo, 1)] = np.nan if v <= -90 else v
    s = pd.Series(recs).sort_index()
    return s


def parse_rmm(text: str) -> pd.Series:
    """BoM RMM: header lines then 'YYYY MM DD RMM1 RMM2 phase amplitude ...'.
    Returns monthly-mean MJO amplitude."""
    rows = []
    for line in text.splitlines():
        t = line.split()
        if len(t) < 7:
            continue
        try:
            yr, mo, dy = int(t[0]), int(t[1]), int(t[2]); amp = float(t[6])
        except ValueError:
            continue
        if not (1974 <= yr <= 2100) or amp > 90:
            continue
        rows.append((pd.Timestamp(yr, mo, dy), amp))
    if not rows:
        return pd.Series(dtype=float)
    daily = pd.Series(dict(rows)).sort_index()
    return daily.resample("MS").mean()


def fetch_indices() -> dict[str, pd.Series]:
    out = {}
    for name, spec in INDICES.items():
        try:
            txt = _get(spec["url"])
            s = parse_rmm(txt) if spec["kind"] == "rmm" else parse_psl(txt)
            if s.dropna().empty:
                raise ValueError("no data parsed")
            out[name] = s
            log.info("  fetched %-14s %d months (%s–%s)", name, s.notna().sum(),
                     s.dropna().index.min().strftime("%Y-%m"),
                     s.dropna().index.max().strftime("%Y-%m"))
        except Exception as e:
            log.warning("  SKIP %-14s (%s)", name, e)
    return out


def deseasonalise(s: pd.Series) -> pd.Series:
    """Remove monthly climatology (harmless for anomaly indices, fixes absolute
    ones like raw Nino3.4 SST)."""
    clim = s.groupby(s.index.month).transform("mean")
    return s - clim


# ─── Lagged cross-correlation + surrogate significance ────────────────────────
def lagged_xcorr(sig: pd.Series, idx: pd.Series, mask: pd.Series,
                 max_lag: int) -> tuple[float, int, np.ndarray, np.ndarray]:
    """Return (best_r, best_lag, lags, r_at_each_lag).
    lag>0 => index LEADS turbulence by `lag` months (predictive direction)."""
    lags = np.arange(-max_lag, max_lag + 1)
    rs = []
    for L in lags:
        shifted = idx.shift(L)                       # value at t-L -> t  (lead)
        df = pd.concat([sig.rename("s"), shifted.rename("i")], axis=1)
        df = df[mask.reindex(df.index, fill_value=False)].dropna()
        rs.append(df["s"].corr(df["i"]) if len(df) > 6 else np.nan)
    rs = np.array(rs)
    if np.all(np.isnan(rs)):
        return np.nan, 0, lags, rs
    j = int(np.nanargmax(np.abs(rs)))
    return float(rs[j]), int(lags[j]), lags, rs


def phase_randomise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fourier phase-randomised surrogate: same power spectrum, random phases."""
    X = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0
    if len(x) % 2 == 0:
        ph[-1] = 0
    Xs = np.abs(X) * np.exp(1j * ph)
    return np.fft.irfft(Xs, n=len(x))


def surrogate_p(sig: pd.Series, idx: pd.Series, mask: pd.Series,
                max_lag: int, r_obs: float, rng: np.random.Generator) -> float:
    """Fraction of phase-randomised index surrogates whose best |lagged r| >=
    |observed|.  Preserves index autocorrelation -> honest significance."""
    base = idx.dropna()
    common = sig.index.intersection(base.index)
    if len(common) < 24:
        return np.nan
    base = base.reindex(common).interpolate().bfill().ffill()
    vals = base.values
    hits = 0
    for _ in range(N_SURROGATES):
        surr = pd.Series(phase_randomise(vals, rng), index=common)
        r, _, _, _ = lagged_xcorr(sig, surr, mask, max_lag)
        if not np.isnan(r) and abs(r) >= abs(r_obs):
            hits += 1
    return (hits + 1) / (N_SURROGATES + 1)


# ─── Amplitude-envelope (amplitude-modulation) analysis ───────────────────────
def bandpass(x: np.ndarray, period_mo: float, width: float = 1.7) -> np.ndarray:
    """Zero-phase band-pass around a target period (months), fs = 1 cycle/month."""
    hi_p, lo_p = period_mo * width, period_mo / width          # long/short period
    f_lo, f_hi = 1.0 / hi_p, 1.0 / lo_p                         # cycles / month
    nyq = 0.5
    wn = [max(f_lo / nyq, 1e-4), min(f_hi / nyq, 0.99)]
    b, a = butter(2, wn, btype="band")
    return filtfilt(b, a, x - np.mean(x))


def envelope_corr(ts, valid, imfs, periods, indices, rng):
    """Correlate the Hilbert amplitude ENVELOPE of each IMF against the envelope
    of the band-matched index. Answers: do the two modes wax and wane together
    (even if phase-shifted)?  Lag search + phase-randomised surrogate p."""
    rows = []
    for name in indices:
        spec = INDICES[name]
        centre = np.sqrt(spec["band"][0] * spec["band"][1])
        cand = [(abs((periods[k] or 1e9) - centre), k)
                for k in range(len(imfs)) if not np.isnan(periods[k])]
        if not cand:
            continue
        k = min(cand)[1]
        P = periods[k]

        idx_m = (deseasonalise(indices[name]).reindex(ts.index)
                 .interpolate().bfill().ffill())
        idx_band = bandpass(idx_m.values, P)
        env_idx = pd.Series(np.abs(hilbert(idx_band)), index=ts.index)
        env_imf = pd.Series(np.abs(hilbert(imfs[k])), index=ts.index)

        # trim filter/Hilbert edge transients (~half a period each end)
        trim = int(min(round(P), len(ts) // 4))
        emask = valid.copy()
        if trim > 0:
            emask.iloc[:trim] = False
            emask.iloc[-trim:] = False

        r, lag, lags, rs = lagged_xcorr(env_imf, env_idx, emask, MAX_LAG_MONTHS)

        # proportionality at best lag: slope of IMF-env on index-env (z-scored)
        sh = env_idx.shift(lag)
        df = pd.concat([env_imf.rename("e"), sh.rename("i")], axis=1)
        df = df[emask.reindex(df.index, fill_value=False)].dropna()
        if len(df) > 6:
            ez = (df["e"] - df["e"].mean()) / df["e"].std()
            iz = (df["i"] - df["i"].mean()) / df["i"].std()
            slope = float(linregress(iz, ez).slope)
        else:
            slope = np.nan

        # surrogate p: phase-randomise the band-passed index, re-envelope
        hits = 0
        for _ in range(N_SURROGATES):
            surr = phase_randomise(idx_band, rng)
            es = pd.Series(np.abs(hilbert(surr)), index=ts.index)
            rr, _, _, _ = lagged_xcorr(env_imf, es, emask, MAX_LAG_MONTHS)
            if not np.isnan(rr) and abs(rr) >= abs(r):
                hits += 1
        p = (hits + 1) / (N_SURROGATES + 1)

        rows.append(dict(index=name, imf=k + 1, period=period_label(P),
                         r=r, lag=lag, p=p, slope=slope, lags=lags, rs=rs,
                         env_idx=env_idx, env_imf=env_imf, emask=emask))
        log.info("  AMP %-14s -> IMF %d (%s): env-env r=%+.2f @ lag %+d mo "
                 "(p_surr=%.3f, slope=%+.2f)", name, k + 1, period_label(P),
                 r, lag, p, slope)
    return rows


def write_amplitude_findings(rows):
    def stars(p):
        if np.isnan(p):  return ""
        if p < 0.01:     return " ***"
        if p < 0.05:     return " **"
        if p < 0.10:     return " *"
        return " (ns)"
    tbl = "\n".join(
        f"| {r['index']} | IMF {r['imf']} ({r['period']}) | {r['r']:+.2f} | "
        f"{r['lag']:+d} mo | {r['slope']:+.2f} | {r['p']:.3f}{stars(r['p'])} |"
        for r in rows)
    md = f"""# Amplitude-Modulation Test — IMF envelope vs climate-index envelope
Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M UTC")}

## Question
Do the turbulence modes and the climate modes **wax and wane together** — i.e. do
their *amplitudes* (not phases) rise and fall proportionally, even if shifted in
time? This is the amplitude-modulation question, separate from phase correlation.

## Method
For each index↔IMF pairing, the index is band-passed to the IMF's period band,
then the **Hilbert amplitude envelope** of both the index and the IMF is taken.
The two envelopes are cross-correlated (lag>0 ⇒ index envelope leads). `slope` is
the z-scored regression of IMF-envelope on index-envelope at the best lag — ~+1
means amplitudes scale ~proportionally; ~0 means no proportional link. Edge
months (~one period each end) are trimmed; significance from {N_SURROGATES}
phase-randomised surrogates of the band-passed index. ***p<0.01,**p<0.05,*p<0.10.

## Results — amplitude (envelope) coupling

| Climate index | Matched mode | env–env r | Lead/lag | prop. slope | p (surrogate) |
|---|---|---|---|---|---|
{tbl}

## Reading it
- **High env–env r with p<0.05 and slope≈+1** ⇒ the turbulence mode genuinely
  strengthens when the climate mode strengthens — a real amplitude link worth
  using even if the phases don't align.
- **High r but p>0.1** ⇒ envelopes are smooth/trending, so they correlate by
  chance; not a real link (the surrogate test is doing its job).
- A non-zero **lag** with the index envelope *leading* is the predictively useful
  case (climate amplitude foreshadows a turbulence-variability burst).
"""
    (RES / "climate_amplitude_findings.md").write_text(md)


def make_amplitude_figure(ts, rows):
    rows = [r for r in rows if r["index"] in ("ONI (ENSO)", "QBO", "PDO", "NAO")]
    if not rows:
        return
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(15, 2.7 * n),
                             gridspec_kw={"width_ratios": [3, 1], "hspace": 0.45,
                                          "wspace": 0.18})
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Amplitude modulation — IMF envelope vs climate-index envelope  "
                 "|  ACARS, Eastern N. America", fontsize=12, fontweight="bold",
                 y=1.005)

    def z(x):
        x = np.asarray(x, float)
        return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

    for i, r in enumerate(rows):
        ax = axes[i, 0]
        ax.plot(ts.index, z(r["env_imf"].values), lw=1.8, color="#c0392b",
                label=f"IMF {r['imf']} envelope ({r['period']})")
        ax.plot(ts.index, z(r["env_idx"].values), lw=1.6, color="#1e8449",
                alpha=0.85, label=f"{r['index']} envelope")
        ax.set_title(f"{r['index']} amplitude ↔ IMF {r['imf']} amplitude   "
                     f"r={r['r']:+.2f} @ lag {r['lag']:+d} mo  (p={r['p']:.3f}, "
                     f"slope={r['slope']:+.2f})", fontsize=9, fontweight="bold",
                     loc="left")
        ax.legend(fontsize=7.5, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.15); ax.set_ylabel("env z-score", fontsize=8)

        axl = axes[i, 1]
        axl.axhline(0, color="k", lw=0.5); axl.axvline(0, color="gray", lw=0.5, ls=":")
        axl.plot(r["lags"], r["rs"], color="#117a65", lw=1.5)
        axl.scatter([r["lag"]], [r["r"]], color="#e74c3c", zorder=5, s=30)
        axl.set_title("env lagged xcorr", fontsize=8)
        axl.set_xlabel("index-env lead (mo) →", fontsize=7)
        axl.tick_params(labelsize=7); axl.grid(True, alpha=0.15)

    axes[-1, 0].set_xlabel("Date", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 1])
    out = OUT / "climate_amplitude.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    log.info("  → %s", out)


# ─── Main analysis ────────────────────────────────────────────────────────────
def main():
    log.info("=== Climate-index correlation of EEMD IMF modes ===")
    ts, valid = prepare_series()
    imfs = run_eemd(ts)
    periods = [dominant_period_months(m) for m in imfs]
    for k, p in enumerate(periods):
        log.info("  IMF %d  period %-8s  var %.1f%%", k + 1, period_label(p),
                 100 * np.var(imfs[k]) / np.var(ts.values))

    indices = fetch_indices()
    rng = np.random.default_rng(SEED)

    # Pre-wrap IMFs and envelopes as monthly Series on ts.index
    imf_series = [pd.Series(m, index=ts.index) for m in imfs]
    env_series = [pd.Series(np.abs(hilbert(m)), index=ts.index) for m in imfs]

    def pick_imf(band) -> int:
        """IMF whose dominant period is closest to the centre of the index band."""
        centre = np.sqrt(band[0] * band[1])
        cand = [(abs((periods[k] or 1e9) - centre), k)
                for k in range(len(imfs)) if not np.isnan(periods[k])]
        return min(cand)[1] if cand else 0

    rows = []
    for name in indices:
        spec = INDICES[name]
        idx = deseasonalise(indices[name])
        k = pick_imf(spec["band"])
        sig = imf_series[k]

        r, lag, lags, rs = lagged_xcorr(sig, idx, valid, MAX_LAG_MONTHS)
        p = surrogate_p(sig, idx, valid, MAX_LAG_MONTHS, r, rng)

        # amplitude-modulation: index vs IMF Hilbert envelope (lag 0)
        env = env_series[k]
        df = pd.concat([env.rename("e"), idx.rename("i")], axis=1)
        df = df[valid.reindex(df.index, fill_value=False)].dropna()
        r_env = df["e"].corr(df["i"]) if len(df) > 6 else np.nan

        rows.append(dict(index=name, imf=k + 1, period=period_label(periods[k]),
                         r=r, lag=lag, p=p, r_env=r_env, lags=lags, rs=rs,
                         band=spec["band"]))
        log.info("  %-14s -> IMF %d (%s): r=%+.2f @ lag %+d mo (p_surr=%.3f), "
                 "env r=%+.2f", name, k + 1, period_label(periods[k]), r, lag,
                 p if not np.isnan(p) else float("nan"), r_env)

    # Full IMF x index matrix (best-lag |r|), for the appendix
    matrix = {}
    for name in indices:
        idx = deseasonalise(indices[name])
        matrix[name] = [lagged_xcorr(imf_series[k], idx, valid, MAX_LAG_MONTHS)[0]
                        for k in range(len(imfs))]

    write_findings(ts, valid, imfs, periods, rows, matrix)
    make_figure(ts, imfs, periods, indices, rows)

    log.info("--- Amplitude-modulation (envelope vs envelope) ---")
    amp_rows = envelope_corr(ts, valid, imfs, periods, indices, rng)
    write_amplitude_findings(amp_rows)
    make_amplitude_figure(ts, amp_rows)

    log.info("Done. -> results/climate_index_findings.md, "
             "results/climate_amplitude_findings.md, results/figures/")


# ─── Output: markdown ─────────────────────────────────────────────────────────
def write_findings(ts, valid, imfs, periods, rows, matrix):
    def sig_stars(p):
        if np.isnan(p):  return ""
        if p < 0.01:     return " ***"
        if p < 0.05:     return " **"
        if p < 0.10:     return " *"
        return " (ns)"

    main_tbl = "\n".join(
        f"| {r['index']} | IMF {r['imf']} ({r['period']}) | {r['r']:+.2f} | "
        f"{r['lag']:+d} mo | {r['p']:.3f}{sig_stars(r['p'])} | {r['r_env']:+.2f} |"
        for r in rows)

    imf_hdr = " | ".join(f"IMF{k+1}" for k in range(len(imfs)))
    imf_sub = " | ".join(period_label(p) for p in periods)
    mat_rows = "\n".join(
        f"| {name} | " + " | ".join(
            ("—" if np.isnan(v) else f"{v:+.2f}") for v in vals) + " |"
        for name, vals in matrix.items())

    n_valid = int(valid.sum())
    md = f"""# Climate-Index Correlation of EEMD IMF Modes — Findings
Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M UTC")}

## Question
Do the multi-year EEMD IMF modes of the ACARS turbulence record (Eastern North
America) correlate with large-scale climate indices — ENSO, QBO, PDO, NAO, AO,
MJO — and at what lead/lag?

## Signal & method
- **Signal:** monthly `{SIGNAL}` (fraction of ACARS reports with MEDEDR ≥ 0.20),
  rebuilt from `results/cache_ym_agg.parquet` with the analysis_emd.py
  preprocessing (≥{MIN_N_TOTAL:,} reports/month; {sorted(BAD_YEARS)} and COVID
  {COVID_MONTHS} excluded + interpolated). **{n_valid} observed months** used for
  correlation (interpolated months masked out).
  *Proxy note:* analysis_emd.py decomposes mean above-threshold EDR (intensity),
  which needs `data/raw_reports.parquet` (absent locally). Occurrence rate is the
  closest reproducible signal; set `SIGNAL="edr_mean_above"` to use the exact one.
- **Decomposition:** EEMD ({EEMD_TRIALS} trials, noise={EEMD_NOISE_WIDTH}, seed={SEED}).
- **Each index** is matched to the IMF whose dominant period is closest to the
  index's native band, then a **lagged cross-correlation** (±{MAX_LAG_MONTHS} mo,
  lag>0 ⇒ index *leads* turbulence) gives r at the best lag.
- **Significance** is from **{N_SURROGATES} Fourier phase-randomised surrogates**
  of the index (same power spectrum, scrambled phase). This is the honest test:
  two narrowband series correlate at r≈0.5+ by chance, so the analytic p-value is
  not used. ***p<0.01, **p<0.05, *p<0.10.
- **env r** = correlation of the index with the IMF's Hilbert amplitude envelope
  (does turbulence variability *strengthen* in a given climate phase?).

## Results — primary index↔mode pairings

| Climate index | Matched mode | r (best lag) | Lead/lag | p (surrogate) | env r |
|---|---|---|---|---|---|
{main_tbl}

*Lead/lag sign: **+** = the climate index leads turbulence (potential predictor);
**−** = turbulence leads the index (likely spurious / shared trend).*

## Full IMF × index matrix (|best-lag r|, all modes)

| Index \\ Mode | {imf_hdr} |
|---|{'---|' * len(imfs)}
| *period* | {imf_sub} |
{mat_rows}

## How to read this
- **ENSO (ONI / Niño3.4)** is the mode your 20-yr record can actually resolve
  (~4–5 cycles). A positive lag with surrogate p<0.05 is the result worth
  trusting and worth feeding to the ML model as an exogenous feature.
- **QBO (~2.4 yr)** maps to the inter-annual IMF; check the lag — stratospheric
  influence on upper-tropospheric shear is not instantaneous.
- **PDO** is decadal (20–30 yr): in a 20-yr record it is **under-resolved** and
  any correlation here is effectively a shared trend — treat as not testable, not
  as a real teleconnection.
- **NAO / AO** are winter-dominant and broadband; a DJF-only test (future work)
  will be more sensitive than the all-season r above.
- **MJO** is sub-seasonal (30–90 d); at monthly resolution only its envelope
  survives, so it can at most touch the fastest IMF.

## Caveats
1. Surrogate p-values reflect autocorrelation but **not** the small number of
   independent multi-year cycles — even a "significant" multi-year r rests on
   ~4–5 cycles. Treat as suggestive, not confirmatory.
2. Interpolated months (2009–2011, COVID) are excluded from correlation but still
   shape the IMFs; modes localised to those gaps are unreliable.
3. Occurrence-rate proxy ≠ intensity signal; re-run with raw_reports.parquet to
   confirm on the exact EEMD series.
"""
    (RES / "climate_index_findings.md").write_text(md)


# ─── Output: figure ───────────────────────────────────────────────────────────
def make_figure(ts, imfs, periods, indices, rows):
    rows_to_plot = [r for r in rows if r["index"] in
                    ("ONI (ENSO)", "QBO", "PDO", "NAO")]
    n = len(rows_to_plot)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(15, 2.7 * n),
                             gridspec_kw={"width_ratios": [3, 1], "hspace": 0.45,
                                          "wspace": 0.18})
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("EEMD turbulence modes vs climate indices  |  ACARS, "
                 "Eastern N. America  |  occurrence rate (MEDEDR ≥ 0.20)",
                 fontsize=12, fontweight="bold", y=1.005)

    def z(x):
        x = np.asarray(x, float)
        return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

    for i, r in enumerate(rows_to_plot):
        k = r["imf"] - 1
        ax = axes[i, 0]
        imf_s = pd.Series(z(imfs[k]), index=ts.index)
        idx = deseasonalise(indices[r["index"]]).reindex(ts.index)
        ax.plot(ts.index, imf_s.values, lw=1.6, color="#c0392b",
                label=f"IMF {r['imf']} ({r['period']})")
        ax.plot(ts.index, z(idx.values), lw=1.4, color="#2471a3", alpha=0.85,
                label=r["index"])
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
        ax.set_title(f"{r['index']}  ↔  IMF {r['imf']}   "
                     f"r={r['r']:+.2f} @ lag {r['lag']:+d} mo  "
                     f"(p={r['p']:.3f})", fontsize=9, fontweight="bold", loc="left")
        ax.legend(fontsize=7.5, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.15)
        ax.set_ylabel("z-score", fontsize=8)

        axl = axes[i, 1]
        axl.axhline(0, color="k", lw=0.5)
        axl.axvline(0, color="gray", lw=0.5, ls=":")
        axl.plot(r["lags"], r["rs"], color="#7d3c98", lw=1.5)
        axl.scatter([r["lag"]], [r["r"]], color="#e74c3c", zorder=5, s=30)
        axl.set_title("lagged xcorr", fontsize=8)
        axl.set_xlabel("index lead (mo) →", fontsize=7)
        axl.tick_params(labelsize=7)
        axl.grid(True, alpha=0.15)

    axes[-1, 0].set_xlabel("Date", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 1])
    out = OUT / "climate_indices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  → %s", out)


if __name__ == "__main__":
    main()
