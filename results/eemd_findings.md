# EEMD — Ensemble EMD of Monthly Mean ACARS Turbulence Intensity: Findings
Generated: 2026-06-23 07:19 UTC

## Method

**Ensemble Empirical Mode Decomposition (EEMD)** extends standard EMD by adding
white-noise perturbations (100 trials, noise_width=0.05) before
sifting and averaging the resulting IMFs. This suppresses the mode-mixing artifact
that afflicts single-trial EMD, producing cleaner timescale separation.

**Signal:** monthly mean MEDEDR among ACARS reports where MEDEDR ≥ 0.2
(`edr_mean_above`), computed from 116 M raw records in `data/raw_reports.parquet`.

This metric captures turbulence *intensity* (m²/³ s⁻¹) rather than event frequency.
Conditioning on MEDEDR ≥ 0.2 focuses the mean on the turbulent regime and
reduces (though does not eliminate) contamination from fleet composition drift,
since calm widebody aircraft rarely exceed the threshold and do not enter the mean.

**Pre-processing:**
  - 34 missing months (no ACARS data or below count thresholds)
  - 36 months in anomalous years [2009, 2010, 2011] (event-window bias)
  - 2 COVID months (2020-04, 2020-05) (reduced-traffic bias)

All NaN'd months linearly interpolated. Signal normalised to zero-mean/unit-std
before EEMD; de-normalised to m²/³ s⁻¹ units afterwards.

**Overall mean turbulence intensity:** 0.2690 m²/³ s⁻¹
(mean MEDEDR of above-threshold reports across the full record).

## Results: IMF Variance Partition

Total signal variance: 2.83e-03  (units: (m²/³ s⁻¹)²)

| Component | Dom. Period | Physical Interpretation | % of Variance |
|---|---|---|---|
| IMF 1 | ~3.1 mo | Sub-seasonal synoptic variability | 5.1% |
| IMF 2 | ~7 mo | Semi-annual / annual cycle | 3.7% |
| IMF 3 | ~1.6 yr | Inter-annual (ENSO / QBO related) | 2.1% |
| IMF 4 | ~4 yr | Multi-year variability | 11.7% |
| IMF 5 | ~8 yr | Decadal / long-term trend | 18.8% |
| IMF 6 | — | Trend / very slow drift | 20.7% |
| IMF 7 | — | Trend / very slow drift | 20.2% |
| Residue | — | Mean turbulence intensity (DC level) | 14.4% |

*(Variance percentages may sum to >100% because IMFs can be mutually correlated.)*

## Key Findings

### 1. Sub-seasonal synoptic variability
The fastest IMFs (period < 8 months, 9% combined variance) capture
month-to-month fluctuations in turbulence *intensity* driven by individual baroclinic
storms and jet-stream excursions. EEMD produces cleaner separation of this band from
the seasonal signal compared to single-trial EMD.

### 2. Annual / semi-annual cycle
IMF 2 (period ≈ ~7 mo, 4% variance) captures the
seasonal modulation of turbulence intensity — strongest in winter when the polar jet is
at peak strength and clearest-air turbulence risk is highest.

### 3. Multi-year variability
Components with periods ≥ 2 years (31% combined variance) represent
inter-annual and multi-year climate forcing:
- **ENSO (~3–5 yr):** jet-stream reorganisation modulates both CAT frequency and intensity.
- **QBO (~2.4 yr):** stratospheric wind reversal affects upper-tropospheric shear.
- **PDO / AMO:** decadal modes may modulate the long-term baseline intensity.
- **Residual fleet drift:** a gradual decline in mean above-threshold EDR post-2012
  (more low-EDR widebodies) may contaminate the slow IMFs.

### 4. EEMD vs EMD
EEMD with 100 trials suppresses mode mixing between the sub-seasonal and annual
bands, producing IMFs whose periods are more physically interpretable. The trade-off
is ~100× compute cost versus single-trial EMD.

## Caveats

1. **Residual fleet drift:** conditioning on MEDEDR ≥ 0.2 reduces but does not
   eliminate instrumental trends. The slow IMFs may carry non-atmospheric signal.
2. **Interpolated gaps:** 2009–2011 and COVID months are linearly interpolated;
   IMF structure localised to these periods should not be interpreted.
3. **Threshold sensitivity:** the mean above 0.2 m²/³ s⁻¹ is sensitive to
   the choice of threshold; results should be cross-checked at 0.15 and 0.25.
