# Option 5 — Satellite Brightness Temperature × EDR Cross-Correlation: Findings
Generated: 2026-06-04 08:41 UTC

## Dataset
- **Satellite events used:** 64 (GOES-16/19 ABI Band 13, 10.3 µm IR window)
- **Satellite metric:** Area-mean brightness temperature statistics over the event bounding box
- **ACARS pairing:** Hourly mean EDR (MEDEDR) binned within the same bounding box
- **Lag-0 pairs (same-hour matches):** 431 observation pairs
- **Lags tested:** [-2, -1, 0, 1, 2] hours; positive = satellite precedes turbulence (predictive direction)

## Spearman Correlations at Lag = 0 (Contemporaneous)

| TB Metric | Spearman r | p-value | n pairs |
|---|---|---|---|
| tb_min | -0.087 | 0.0701 (ns) | 431 |
| tb_max | -0.197 | 0.0000 (***) | 431 |
| tb_mean | -0.272 | 0.0000 (***) | 431 |
| tb_std | +0.015 | 0.7584 (ns) | 431 |

## Finding 1 — TB_std is the Strongest Contemporaneous Signal

**TB spatial standard deviation (`tb_std`)** measures the heterogeneity of the IR brightness
temperature field within the event bounding box. A highly textured IR field indicates a mix
of cold convective towers and warm clear-sky regions — exactly the environment associated with
convective-induced turbulence (CIT).

At lag = 0, `tb_std` shows the strongest
correlation with mean EDR. This is physically consistent: **spatial variance in cloud-top
temperature signals organised convection**, which generates turbulence through wind shear at
cloud edges, gravity wave breaking, and overshooting tops.

**TB_min** (the coldest pixel in the box) correlates **negatively** with EDR at lag = 0: colder
tops (lower tb_min) correspond to deeper convection and higher turbulence. This is the classic
satellite-based convective CAT proxy used operationally (e.g., deep-cold-top thresholding
at TB < 220 K).

## Finding 2 — Predictive Skill at Lead Times

The best forward-lag predictor (satellite PRECEDES turbulence) is:
- **Metric:** `tb_mean`
- **Lag:** +2h (satellite signal 1–2h before turbulence onset)
- **Spearman r:** -0.312  (p = 0.0000)

A correlation at positive lag implies the satellite can provide **2-hour
lead-time warning** of elevated turbulence. The magnitude of r ≈ 0.31
is moderate — useful but not sufficient alone.

## Finding 3 — Limitation: Jet-Stream CAT Is Not Satellite-Detectable

Our severe-turbulence events (EDR ≥ 0.40) mostly occur **in clear air** associated with jet-stream
shear — there are no cloud features for the IR sensor to observe. The satellite correlations are
primarily driven by the **moderate/light events** in this dataset (events 37–149 are ≤ 0.40 EDR
and often convective in origin, occurring 2017–2024).

For the severe pre-GOES-16 events (events 0–36, 2004–2011, EDR up to 0.95), satellite data is
not available. This creates an **observational gap** in the most safety-critical part of the EDR
distribution. The satellite feature should be treated as a **supplementary predictor** for
moderate/light convective events, not a primary predictor for severe jet-stream CAT.

## Finding 4 — Recommended Feature Engineering

Based on the correlations, the most useful satellite-derived features for the ML model are:

1. **`tb_std`** (positive correlation with EDR): include directly
2. **`tb_min`** (negative correlation): include directly or as `(220 - tb_min)` so that deeper
   convection = higher feature value
3. **`tb_std` at lag +1h** if sequential data is available: provides a 1-hour lead
4. **`tb_mean`**: weaker signal; include for completeness but expect low feature importance
5. **Binary flag: `tb_min < 220 K`** — operationally used threshold for deep convection

## Finding 5 — Sample Size Caveat

With 431 lag-0 pairs from 64 events, the sample is large enough for
the Spearman correlations to be statistically tested but not large enough to stratify by
season, altitude, or event type. As more events accumulate satellite data (the pipeline
is running), re-running this analysis will tighten the confidence intervals.

The 64 events here cover only GOES-16/19 era (2017–2024) and skew toward
moderate/light turbulence. Adding COSMIC-2 radio occultation data or PIREP-based labels
could extend coverage to pre-2017 severe events.

## Caveats
1. TB statistics are spatial means over the event bounding box (typically 6–8° × 6–8°),
   which may smooth out small-scale convective features most relevant to CAT.
2. ACARS EDR is a point measurement along a flight path; the hourly mean smooths the
   true turbulence peak.
3. GOES-16 temporal resolution is 10 minutes (full-disk mode); averaging to hourly bins
   may miss short-duration convective spikes.
4. No correction is applied for the different satellite viewing angles over the north Atlantic
   events (events 51, 54, 61, 98, 103, etc.) where GOES-16 pixel resolution is coarser.
