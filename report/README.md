# Paper: Physics-Informed CNN for CAT EDR Prediction

LaTeX source for the research paper compiling the `cat-edr-ml` project.
Two-column article; all exploratory-analysis figures embedded.

## Files
- `main.tex` — the paper (Abstract, Intro, Background, Data, EDA,
  Methodology, Results [scaffolding], Discussion, Conclusion, Appendices).
- `references.bib` — BibTeX references.
- `figures/` — the 14 analysis PNGs (copies of `../results/figures/`).

## Compile
```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
Output: `main.pdf`.

## Status notes
- **Sections 1–5** (Intro → Methodology) are written in full prose.
- **§6.1 has REAL preliminary results**: the pure-NumPy baseline CNN
  (`models/cat_cnn_eval.csv`) — overall Pearson 0.885 / RMSE 0.120,
  with a generated predicted-vs-observed figure (`cnn_pred_vs_obs.png`,
  built from the eval CSV). This is a simple 80/20 split, NOT the
  rigorous grouped-CV protocol, and is labelled as such.
- **§6.2–6.5 remain scaffolding**: the physics-informed CNN, GTG
  surrogate, and XGBoost have not been run under the rigorous protocol
  yet — red `[PENDING]` markers. Fill once validation metrics exist.
- §5.6 now has a full **XGBoost-comparison** subsection (why CNN-vs-XGB
  isolates the value of spatial structure).

## 15 figures embedded
14 analysis PNGs from `results/figures/` + `cnn_pred_vs_obs.png`
(generated here from the baseline eval CSV).

## Reconciliations handled in-text
1. Title centers the built CNN (older draft said "physics-residual
   hybrid"; residual is now an explicit future-work A/B).
2. CNN uses **36** input channels (12 diagnostics × 3 levels), not the
   ~21 raw-field list from the early brainstorm. The NumPy *reference*
   uses 18 raw channels (documented separately).
3. **EMD/EEMD reconciled**: not a conflict — two different signals.
   EMD on occurrence rate `frac_m` (IMF1≈37.5%) vs. EEMD on intensity
   `edr_mean_above` (IMF1≈5.1%). Both shown; paper leads with the
   robust ~90%-synoptic figure from the seasonal decomposition.
