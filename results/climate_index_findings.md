# Climate-Index Correlation of EEMD IMF Modes — Findings
Generated: 2026-06-23 17:50 UTC

## Question
Do the multi-year EEMD IMF modes of the ACARS turbulence record (Eastern North
America) correlate with large-scale climate indices — ENSO, QBO, PDO, NAO, AO,
MJO — and at what lead/lag?

## Signal & method
- **Signal:** monthly `frac_m` (fraction of ACARS reports with MEDEDR ≥ 0.20),
  rebuilt from `results/cache_ym_agg.parquet` with the analysis_emd.py
  preprocessing (≥10,000 reports/month; [2009, 2010, 2011] and COVID
  ['2020-04', '2020-05'] excluded + interpolated). **170 observed months** used for
  correlation (interpolated months masked out).
  *Proxy note:* analysis_emd.py decomposes mean above-threshold EDR (intensity),
  which needs `data/raw_reports.parquet` (absent locally). Occurrence rate is the
  closest reproducible signal; set `SIGNAL="edr_mean_above"` to use the exact one.
- **Decomposition:** EEMD (100 trials, noise=0.05, seed=42).
- **Each index** is matched to the IMF whose dominant period is closest to the
  index's native band, then a **lagged cross-correlation** (±12 mo,
  lag>0 ⇒ index *leads* turbulence) gives r at the best lag.
- **Significance** is from **1000 Fourier phase-randomised surrogates**
  of the index (same power spectrum, scrambled phase). This is the honest test:
  two narrowband series correlate at r≈0.5+ by chance, so the analytic p-value is
  not used. ***p<0.01, **p<0.05, *p<0.10.
- **env r** = correlation of the index with the IMF's Hilbert amplitude envelope
  (does turbulence variability *strengthen* in a given climate phase?).

## Results — primary index↔mode pairings

| Climate index | Matched mode | r (best lag) | Lead/lag | p (surrogate) | env r |
|---|---|---|---|---|---|
| ONI (ENSO) | IMF 4 (~3.7 yr) | +0.31 | -7 mo | 0.727 (ns) | -0.30 |
| Nino3.4 | IMF 4 (~3.7 yr) | +0.31 | -7 mo | 0.728 (ns) | -0.25 |
| QBO | IMF 3 (~1.5 yr) | +0.37 | -10 mo | 0.281 (ns) | -0.01 |
| PDO | IMF 5 (~10.2 yr) | +0.42 | +12 mo | 0.581 (ns) | +0.30 |
| NAO | IMF 2 (~8 mo) | -0.17 | +0 mo | 0.710 (ns) | -0.15 |
| AO | IMF 2 (~8 mo) | +0.16 | -9 mo | 0.793 (ns) | -0.09 |
| MJO (RMM amp) | IMF 1 (~3 mo) | -0.14 | +10 mo | 0.848 (ns) | -0.02 |

*Lead/lag sign: **+** = the climate index leads turbulence (potential predictor);
**−** = turbulence leads the index (likely spurious / shared trend).*

## Full IMF × index matrix (|best-lag r|, all modes)

| Index \ Mode | IMF1 | IMF2 | IMF3 | IMF4 | IMF5 | IMF6 | IMF7 |
|---|---|---|---|---|---|---|---|
| *period* | ~3 mo | ~8 mo | ~1.5 yr | ~3.7 yr | ~10.2 yr | — | — |
| ONI (ENSO) | -0.04 | +0.12 | +0.23 | +0.31 | -0.33 | -0.27 | -0.24 |
| Nino3.4 | -0.06 | +0.13 | +0.22 | +0.31 | -0.30 | -0.22 | -0.20 |
| QBO | -0.05 | -0.11 | +0.37 | +0.46 | +0.18 | -0.21 | -0.16 |
| PDO | +0.09 | -0.14 | +0.24 | +0.14 | +0.42 | +0.11 | +0.31 |
| NAO | -0.13 | -0.17 | -0.18 | -0.15 | -0.14 | -0.15 | -0.10 |
| AO | +0.14 | +0.16 | +0.18 | -0.20 | +0.11 | -0.06 | -0.05 |
| MJO (RMM amp) | -0.14 | +0.14 | +0.21 | +0.16 | +0.17 | -0.08 | -0.13 |

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
