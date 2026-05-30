# CAT-EDR-ML: Satellite-Augmented ML for Clear-Air Turbulence Prediction

## Overview

This project develops a physics-informed machine learning pipeline for predicting clear-air turbulence (CAT) intensity, quantified via eddy dissipation rate (EDR). ERA5 reanalysis fields (wind shear, Richardson number, divergence, deformation) are combined with satellite-derived proxies — GOES-16/18 water vapor imagery, cloud-top height gradients, and COSMIC-2 radio occultation bending angles — to build a feature set grounded in the dynamics of jet-stream-driven turbulence. A hybrid modelling approach fuses an XGBoost surrogate trained on physics-based diagnostics with a neural-network correction layer, targeting ACARS/MADIS in-situ EDR observations as ground truth. The goal is skillful short-range CAT forecasting that generalises across seasons and flight levels without requiring NWP output at inference time.

## Setup

### Recommended: conda (handles eccodes/cartopy system deps automatically)

```bash
conda env create -f environment.yml
conda activate cat-edr-ml
pip install -e .          # installs src/ as editable package (optional)
```

### Fallback: pip

Install system libraries first (macOS example):

```bash
brew install eccodes geos proj
pip install -r requirements.txt
```

### ERA5 / CDS API credentials

Create `~/.cdsapirc` (never commit this file):

```
url: https://cds.climate.copernicus.eu/api/v2
key: <YOUR_UID>:<YOUR_API_KEY>
```

---

## Folder Structure

```
cat-edr-ml/
├── data/                    # all data files — gitignored
│   ├── era5/               # ERA5 GRIB/NetCDF downloads
│   ├── satellite/          # GOES-16/18, Himawari, COSMIC-2
│   ├── acars/              # MADIS ACARS EDR observations
│   ├── pireps/             # IEM PIREPs
│   └── processed/          # collocation tables, derived features
├── src/
│   ├── data/               # acquisition: ERA5, GOES, ACARS downloaders
│   ├── features/           # turbulence diagnostics & feature engineering
│   ├── models/             # XGBoost, surrogate, hybrid model code
│   └── viz/                # plotting utilities (cartopy maps, PDPs, etc.)
├── notebooks/              # exploratory analysis
├── results/
│   ├── figures/            # publication-ready plots
│   ├── tables/             # skill scores, feature importance CSVs
│   └── checkpoints/        # trained model artifacts — gitignored
└── tests/                  # unit tests
```

---

## Results

_Placeholder — to be populated as experiments complete._

| Model | AUROC | CSI (moderate+) | Brier score |
|-------|-------|-----------------|-------------|
| Baseline (Ri only) | — | — | — |
| XGBoost (ERA5 features) | — | — | — |
| Hybrid (+ satellite) | — | — | — |

---

## License

MIT — see [LICENSE](LICENSE).
