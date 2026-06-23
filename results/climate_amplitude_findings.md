# Amplitude-Modulation Test — IMF envelope vs climate-index envelope
Generated: 2026-06-23 17:51 UTC

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
months (~one period each end) are trimmed; significance from 1000
phase-randomised surrogates of the band-passed index. ***p<0.01,**p<0.05,*p<0.10.

## Results — amplitude (envelope) coupling

| Climate index | Matched mode | env–env r | Lead/lag | prop. slope | p (surrogate) |
|---|---|---|---|---|---|
| ONI (ENSO) | IMF 4 (~3.7 yr) | -0.75 | -12 mo | -0.75 | 0.326 (ns) |
| Nino3.4 | IMF 4 (~3.7 yr) | -0.74 | -12 mo | -0.74 | 0.335 (ns) |
| QBO | IMF 3 (~1.5 yr) | -0.32 | -12 mo | -0.32 | 0.655 (ns) |
| PDO | IMF 5 (~10.2 yr) | +0.98 | -12 mo | +0.98 | 0.186 (ns) |
| NAO | IMF 2 (~8 mo) | +0.21 | +2 mo | +0.21 | 0.761 (ns) |
| AO | IMF 2 (~8 mo) | +0.18 | +7 mo | +0.18 | 0.867 (ns) |
| MJO (RMM amp) | IMF 1 (~3 mo) | -0.22 | -12 mo | -0.22 | 0.504 (ns) |

## Reading it
- **High env–env r with p<0.05 and slope≈+1** ⇒ the turbulence mode genuinely
  strengthens when the climate mode strengthens — a real amplitude link worth
  using even if the phases don't align.
- **High r but p>0.1** ⇒ envelopes are smooth/trending, so they correlate by
  chance; not a real link (the surrogate test is doing its job).
- A non-zero **lag** with the index envelope *leading* is the predictively useful
  case (climate amplitude foreshadows a turbulence-variability burst).
