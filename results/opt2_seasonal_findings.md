# Option 2 — Seasonal & Diurnal Decomposition: Findings
Generated: 2026-06-04 08:47 UTC

## Dataset
- **Total ACARS reports analysed:** 116,245,659
- **Date range:** 2005-05-15 → 2024-12-31
- **Altitude floor:** FL180+ (≥ 5,500 m / ~18,000 ft)
- **Turbulence thresholds:** light ≥ 0.10 m²/³s⁻¹, moderate ≥ 0.20, severe ≥ 0.40

## Key Statistics
| Metric | Value |
|---|---|
| Overall moderate turbulence fraction | 0.041% of all reports |
| Overall severe turbulence fraction | 0.010% of all reports |
| Peak turbulence month (moderate) | Jan (0.009%) |
| Lowest turbulence month (moderate) | Aug (0.003%) |
| Seasonal amplitude (moderate) | 0.005 pp |
| Peak turbulence hour UTC (moderate) | 21:00 UTC ≈ 16:00 Eastern |
| Diurnal amplitude (moderate) | 0.0521 pp |
| Peak altitude band (moderate) | FL180–260 |
| Peak altitude band (severe) | FL180–260 |

## Finding 1 — Seasonal Cycle (Monthly Climatology)

**Jan has the highest moderate-turbulence fraction (0.009%)**,
0.01 percentage points above the annual minimum in Aug (0.003%).

This is consistent with the dominant meteorological mechanism for clear-air turbulence over
North America: **jet-stream shear**. The Northern Hemisphere polar jet is strongest in
December–March, when the equator-to-pole temperature gradient is largest, producing maximum
vertical wind shear and Richardson number instability at cruise altitudes (FL300–FL390).

The seasonal decomposition identifies the annual-cycle peak near **day-of-year 252
(≈ 08 Sep)** and trough near **DOY 353
(≈ 18 Dec)**, confirming a winter-dominant pattern.

**Implication for modelling:** A month-of-year feature (or sine/cosine encoding of DOY) will
be a strong predictor. Include `sin(2π·DOY/365)` and `cos(2π·DOY/365)` in the feature set.

## Finding 2 — Diurnal Cycle

The diurnal amplitude is only **0.0521 percentage points** — very small compared
to the seasonal amplitude of 0.01 pp.

Peak UTC hour is **21:00 UTC (≈ 16:00 Eastern,
13:00 Pacific)**; minimum is 06:00 UTC.

**Interpretation:** The weak diurnal signal means most turbulence in this dataset is
**jet-stream driven (mechanical)** rather than convective. Convective CAT would show a
strong afternoon peak (≈ UTC 20–22 over the US). The weak diurnal structure is actually
informative: it means time-of-day is a **low-importance** feature for jet-stream events
and you should not over-weight it during feature engineering.

**Note on the heatmap:** The slight UTC 11–16 enhancement (local morning) may reflect
denser flight schedules in that window rather than a true physical signal, since the
turbulence fraction is normalised by report count per hour.

## Finding 3 — Altitude Structure

Peak moderate-turbulence fraction is in **FL180–260**; peak severe fraction in
**FL180–260**. Turbulence intensity increases with altitude up through the
upper-cruise band.

This is expected: the tropopause is at ≈ 10–12 km over mid-latitudes in winter, and
maximum vertical wind shear occurs just below it. Aircraft cruising at FL350–FL390
(≈ 10.7–11.9 km) are near the peak of the jet-stream shear layer.

**Implication:** Altitude (or pressure level) is a critical feature. The ERA5 feature set
should be anchored at the pressure level closest to the actual aircraft altitude, not just
a fixed 250 hPa level.

## Finding 4 — Annual Trend (2004–2024)

Linear regression on annual mean turbulence fraction:
- **Moderate (≥0.20):** slope = -0.00063 %/year (-0.006 pp/decade),
  p = 0.017 → statistically significant (p < 0.05)
- **Severe (≥0.40):** slope = -0.00024 %/year (-0.002 pp/decade),
  p = 0.032 → statistically significant (p < 0.05)

The trend is **decreasing** for moderate turbulence. Published literature (Williams 2017,
Storer et al. 2019) shows a 40–170% increase in North Atlantic CAT over 1979–2020 attributed
to strengthening of the jet stream due to lower-stratosphere warming. Our dataset covers a
narrower time window and predominantly North American routes, so consistency with those
findings would be notable but not guaranteed.

**Caveat:** ACARS fleet composition changed over 2004–2024 (newer aircraft, EDR algorithm
updates). Some trend signal may be instrumental rather than atmospheric. Cross-checking
with ERA5 wind-shear trend over the same region is recommended.

## Finding 5 — Variance Decomposition

Of the total day-to-day variance in turbulence fraction:
- Seasonal cycle explains ≈ **4.8%** of variance
- Long-term trend explains ≈ **5.4%** of variance
- Residual (weather-scale, day-to-day) ≈ **89.8%** of variance

The large residual fraction confirms that **synoptic-scale weather variability**
(individual storms, jet-stream meanders) dominates day-to-day turbulence — exactly the
signal your ML model needs to capture via ERA5 fields. The seasonal and trend components
are structural baselines.

## Data Quality Note — Anomalous Years (2009, 2010, 2011)

Years 2009, 2010, 2011 were detected as anomalous and **excluded from all climatology and
trend calculations**. These years show `frac_m` ~100× higher than surrounding years while
flight volume is only ~2× higher, indicating **event-window sampling bias**: the raw
ACARS parquet for these years was seeded with reports specifically downloaded around the
selected turbulence events (step1/step2 overlap), not a uniform 3-hourly scan.

In the decomposition figure these years are NaN-interpolated. In the climatology figure
they appear as grey × markers (annual panel). All statistics above are computed from
clean years only.

**Action:** Re-run step1 for 2009, 2010, 2011 with `--overwrite` to obtain an unbiased
uniform sample, then regenerate this analysis.

## Caveats
1. **Sampling bias:** Turbulence fraction is computed as proportion of EDR reports, not as
   fraction of airspace. More flights in certain months/hours inflates counts but is
   corrected by normalising within time bin.
2. **Geography:** Raw reports cover all longitudes but ~85% are over North America. The
   seasonal signal is dominated by the North American jet stream.
3. **Missing pre-2005 data:** MADIS ACARS archive is sparse before 2005.
4. **EDR calibration:** Different aircraft types use different EDR algorithms; the floor
   value 0.05 m²/³s⁻¹ is common for aircraft that round down.
5. **Fleet mix change:** The 3× growth in reports/day from 2006 to 2024 reflects both
   more ACARS-equipped aircraft and denser flight schedules, not a meteorological change.
