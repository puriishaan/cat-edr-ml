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
- **Section 6 (Results)** is intentionally scaffolding: the model is
  built but **not yet validated**, so it carries red `[PLACEHOLDER]`
  markers. Fill these once validation metrics exist.
- The physics **surrogate** and **residual-hybrid** are marked as
  specified/planned; the **CNN** and **XGBoost** baselines are built.

## Known items to reconcile (flagged in-text)
1. Title/framing centers the built CNN (the older draft said
   "physics-residual hybrid"; residual is now future work).
2. CNN uses **36** input channels (12 diagnostics × 3 levels), not the
   ~21 raw-field list from the early brainstorm.
3. The EEMD per-IMF variance percentages are method-sensitive; the
   paper leads with the **~90%-synoptic** figure from the seasonal
   decomposition instead. The embedded EEMD figure shows its own
   annotations — keep captions qualitative.
