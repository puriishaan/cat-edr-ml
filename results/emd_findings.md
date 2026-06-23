# EMD — Empirical Mode Decomposition of Monthly ACARS Turbulence Rate: Findings
Generated: 2026-06-06 09:50 UTC

## Method

**Empirical Mode Decomposition (EMD)** is a fully data-adaptive, nonlinear signal
decomposition that extracts oscillatory Intrinsic Mode Functions (IMFs) directly
from the data without assuming any fixed basis (unlike Fourier or wavelet analysis).
IMFs are ordered from fastest to slowest timescale; their sum exactly reconstructs
the original signal.

**Input signal:** monthly fraction of ACARS reports with MEDEDR ≥ 0.20 (`frac_m`)
from `cache_ym_agg.parquet`.  Monthly aggregation was chosen over daily because
daily `frac_m` is identically zero on >50% of clean-year days, while `edr_mean`
has a 30–50× long-term instrumental decline (fleet composition change) that would
dominate EMD.  The fractional metric is instrumentally stable.

**Pre-processing:**
  - 23 missing months (no ACARS data)
  - 36 months in anomalous years [2009, 2010, 2011] (event-window bias)
  - 2 COVID months (2020-04, 2020-05) (reduced-traffic bias)

All NaN'd months were linearly interpolated before EMD.  Signal normalised to
zero-mean / unit-std before sifting (de-normalised to frac_m units afterwards);
normalisation is required because PyEMD's stopping criteria trigger prematurely
on raw amplitudes of ~10⁻⁴.

**Overall mean turbulence rate:** 1.016 × 10⁻⁴ (i.e., ~0.0102% of
ACARS reports exceed MEDEDR 0.20 on a typical month; the signal is strongly right-skewed
toward rare turbulent episodes).

## Results: IMF Variance Partition

Total signal variance: 8.97e-09  (units: frac_m²)

| Component | Dom. Period | Physical Interpretation | % of Variance |
|---|---|---|---|
| IMF 1 | ~4 mo | Sub-seasonal synoptic variability | 37.5% |
| IMF 2 | ~9 mo | Semi-annual / annual cycle | 33.9% |
| IMF 3 | ~1.5 yr | Inter-annual (ENSO / QBO related) | 7.1% |
| IMF 4 | ~3 yr | Multi-year variability | 11.3% |
| IMF 5 | ~8 yr | Decadal / long-term trend | 37.4% |
| IMF 6 | — | Trend / very slow drift | 7.4% |
| Residue | — | Mean turbulence rate (DC level) | 0.0% |

*(Variance percentages may sum to >100% because IMFs can be mutually correlated;
this is expected for EMD applied to non-stationary signals.)*

## IMF-by-IMF Physical Interpretation

### IMF 1  (~4 mo)
The fastest component captures **sub-seasonal synoptic variability** — the
month-to-month randomness in whether the ACARS flight network happened to
encounter turbulent air. Individual baroclinic storms, jet-stream excursions,
and mesoscale convective systems each last 2–7 days, so their imprint on a
monthly mean produces oscillations at the 3–5 month scale.

This is the dominant *noise* floor for any monthly-scale turbulence model.
Its 37% variance share confirms that much of the
month-to-month fluctuation is driven by transient synoptic weather that cannot
be predicted from climatological inputs alone.

### IMF 2  (~9 mo)
The second component, with a period near ~9 mo, captures
**sub-annual to annual-scale variability** — likely a blended representation of
the seasonal jet-stream cycle and sub-seasonal oscillations (e.g. the Madden-Julian
Oscillation, which modulates North American weather on 30–90 day cycles).

Note that no clean 12-month IMF appears: the seasonal cycle in monthly `frac_m`
is weaker than sub-seasonal noise (consistent with the seasonal amplitude in
`opt2_climatology.png` being only 2–4× the inter-monthly standard deviation).
Mode mixing between seasonal and sub-seasonal IMFs is expected when EMD is applied
to monthly rather than daily data.

### IMF 2  (~9 mo)
The IMF whose period is closest to 12 months (~9 mo) carries the
**annual/inter-annual climate signal** — related to the North Atlantic Oscillation
(NAO), Pacific/North American (PNA) pattern, and ENSO teleconnections that modulate
North American jet-stream strength and thus clear-air turbulence.
This mode accounts for 34% of total signal variance.

### Multi-year IMFs  (~3 yr, ~8 yr)
Components with periods of 3–10 years (49% combined variance)
represent **inter-annual and multi-year variability**:

- **ENSO cycle (~3–5 years):** El Niño strengthens the subtropical jet while
  La Niña intensifies the polar jet; both modulate turbulence frequency via
  different mechanisms and tend to shift activity regionally.
- **Quasi-biennial oscillation (QBO, ~2.4 years):** Stratospheric zonal wind
  reversal propagates into upper-tropospheric shear affecting FL350–FL390
  turbulence.
- **Instrumental drift:** The multi-year modes also capture the gradual shift
  in fleet composition (increasing ACARS reports from calmer commercial widebodies
  post-2012 lowers frac_m); this non-atmospheric signal cannot be separated from
  climate variability without independent calibration.

## Key Findings

### 1. Sub-seasonal synoptic variability dominates monthly frac_m
The fastest IMFs (sub-seasonal timescales) collectively explain the largest share
of variance.  This means that **individual storms and jet-stream excursions are the
primary drivers of month-to-month turbulence variability** — not the background
seasonal cycle.  A skill ML model must capture synoptic-scale ERA5 features (shear,
ageostrophic wind, Richardson number) rather than relying on calendar date alone.

### 2. The annual seasonal cycle is a secondary mode
In contrast to the classical Savitzky-Golay decomposition (`opt2_decomposition.png`),
which explicitly removes the day-of-year mean, EMD does not isolate a clean 12-month
IMF.  This is physically meaningful: **the seasonal signal is real but weaker than
synoptic noise at monthly resolution**.  Including seasonal features (sin/cos of DOY)
is important for model calibration, but these features will have lower importance
than synoptic ERA5 fields for event-level prediction.

### 3. Multi-year variability is physically rich but contaminated
The 3–8 year IMFs carry mixed signals from ENSO, QBO, NAO, and instrumental fleet
drift.  Separating these requires exogenous indices (Niño 3.4 SST, QBO 30 hPa wind,
NAO index).  **Recommendation:** include ENSO and QBO phase as feature variables in
the ML model to capture inter-annual turbulence modulation.

### 4. Comparison with classical decomposition
| | Savitzky-Golay (`opt2_decomposition.png`) | **EMD (this analysis)** |
|---|---|---|
| Method | Fixed 365-day kernel | Data-adaptive sifting |
| Seasonal extraction | Explicit DOY mean subtraction | Autonomous IMF |
| Handles non-stationarity | No | Yes |
| Dominant finding | Seasonal cycle clearly isolated | Sub-seasonal noise dominates |

Both analyses are complementary: the classical decomposition explicitly separates
the seasonal cycle (revealing its shape and amplitude); EMD reveals that the seasonal
mode is only one of several competing timescale processes in this dataset.

## Caveats

1. **Mode mixing:** Standard EMD (not EEMD) may blend timescales within a single IMF.
   Ensemble EMD (adding white-noise perturbations before sifting) would produce cleaner
   IMF separation but requires ~100× computation.

2. **Interpolated gaps:** Three anomalous-year windows (2009–2011, ~36 months) and 2
   COVID months were linearly interpolated.  Any IMF structure localised to these periods
   should not be physically interpreted.

3. **Period non-stationarity:** Zero-crossing period estimates assume each IMF oscillates
   at a roughly constant frequency.  Climate-driven frequency changes (e.g. a lengthening
   seasonal cycle under climate change) would appear as period instability within an IMF;
   Hilbert-Huang instantaneous frequency provides a more general characterisation.

4. **frac_m vs. physical turbulence probability:** The metric is a ratio of ACARS counts,
   not a probability over all airspace.  Spatial coverage changes with airline network
   evolution and may introduce low-frequency artefacts not related to atmospheric dynamics.
