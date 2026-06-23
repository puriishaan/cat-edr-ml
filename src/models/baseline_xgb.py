"""
XGBoost accuracy baseline — the number the CNN must beat.

This is a *standalone* gradient-boosted-trees regressor on spatially-pooled physics
diagnostics (+ climate + cyclic time + satellite stats), predicting log1p(max_edr).
It is NOT a residual corrector and NOT an ensemble partner: its only job is to tell us
whether the CNN's spatial structure adds skill over per-pixel/pooled physics features.
If the CNN can't beat this, the convolutions aren't earning their keep.

Out-of-fold predictions (grouped/stratified by edr_bin, so no event leaks across folds)
give an honest skill estimate; a final model is fit on all events for reuse.

    python -m src.models.baseline_xgb            # CV + fit + report
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import (
    CLIMATE_NAMES, SAT_FEATURES, build_raw_samples, make_folds,
)

log = logging.getLogger(__name__)
MODELS_DIR = Path("models")


def pool_features(rs) -> tuple[np.ndarray, list[str]]:
    """Spatial mean/max/std per diagnostic channel + climate + time + satellite."""
    d = rs.diag                                  # (N,C,H,W)
    mean = d.mean(axis=(2, 3))
    mx = d.max(axis=(2, 3))
    sd = d.std(axis=(2, 3))
    X = np.concatenate([mean, mx, sd, rs.climate, rs.time, rs.sat], axis=1)

    names = (
        [f"{n}|mean" for n in rs.names]
        + [f"{n}|max" for n in rs.names]
        + [f"{n}|std" for n in rs.names]
        + CLIMATE_NAMES
        + ["sinDOY", "cosDOY", "sinHour", "cosHour"]
        + SAT_FEATURES
    )
    return X.astype(np.float32), names


def _metrics(y_true_edr, y_pred_edr) -> dict:
    err = y_pred_edr - y_true_edr
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    r = float(np.corrcoef(y_true_edr, y_pred_edr)[0, 1]) if len(y_true_edr) > 1 else float("nan")
    # log-space RMSE (the CNN's training metric) for apples-to-apples comparison
    rmse_log = float(np.sqrt(np.mean((np.log1p(y_pred_edr) - np.log1p(y_true_edr)) ** 2)))
    return {"rmse_edr": rmse, "mae_edr": mae, "pearson": r, "rmse_log": rmse_log}


def run(n_folds: int = 5, seed: int = 42) -> dict:
    import xgboost as xgb

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rs = build_raw_samples()
    X, feat_names = pool_features(rs)
    y = np.log1p(rs.y_max)                        # train in log space, like the CNN

    params = dict(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        reg_lambda=1.0, objective="reg:squarederror", random_state=seed,
    )

    all_idx = np.arange(len(rs))
    oof = np.full(len(rs), np.nan, dtype=np.float64)
    for tr, va in make_folds(rs, all_idx, n_folds=n_folds, seed=seed):
        model = xgb.XGBRegressor(**params)
        model.fit(X[tr], y[tr])
        oof[va] = model.predict(X[va])

    m = _metrics(rs.y_max, np.expm1(oof))
    log.info("XGBoost OOF | RMSE=%.4f EDR  MAE=%.4f  Pearson=%.3f  RMSE(log)=%.4f",
             m["rmse_edr"], m["mae_edr"], m["pearson"], m["rmse_log"])

    # final model on all data + feature importances
    final = xgb.XGBRegressor(**params).fit(X, y)
    MODELS_DIR.mkdir(exist_ok=True)
    final.save_model(str(MODELS_DIR / "xgb_baseline.json"))

    imp = pd.DataFrame({"feature": feat_names, "gain": final.feature_importances_}) \
        .sort_values("gain", ascending=False)
    imp.to_csv(MODELS_DIR / "xgb_baseline_importance.csv", index=False)
    pd.DataFrame({
        "event_id": rs.eids, "edr_bin": rs.bins,
        "true_max_edr": rs.y_max, "pred_max_edr": np.expm1(oof),
    }).to_csv(MODELS_DIR / "xgb_baseline_oof.csv", index=False)

    log.info("Top features: %s", ", ".join(imp["feature"].head(8)))
    log.info("Saved → models/xgb_baseline.json (+ importance, OOF predictions)")
    return m


def main():
    ap = argparse.ArgumentParser(description="XGBoost CAT accuracy baseline (no residual)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.folds, args.seed)


if __name__ == "__main__":
    main()
