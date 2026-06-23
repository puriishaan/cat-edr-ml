"""
XGBoost baseline for CAT turbulence intensity prediction.

Features: spatial mean/max/std per diagnostic channel + climate + time + sat.
Output: models/xgb_baseline.json + OOF predictions + feature importances.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
import xgboost as xgb

log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MAX_EDR    = 0.95


def pool_features(samples) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Flatten per-event diagnostics to a feature vector."""
    rows, targets = [], []
    feat_names = None

    for s in samples:
        diag = s.diag  # (C, H, W)
        C = diag.shape[0]
        row = []
        names = []
        for c in range(C):
            ch = diag[c].ravel()
            row.extend([ch.mean(), ch.max(), ch.std()])
            names.extend([f"diag{c}_mean", f"diag{c}_max", f"diag{c}_std"])
        row.extend(s.climate.tolist())
        names.extend(["ONI", "Nino34", "PDO", "QBO"])
        row.extend(s.time.tolist())
        names.extend(["sin_doy", "cos_doy", "sin_hr", "cos_hr"])
        row.extend(s.sat.tolist())
        row.append(float(s.sat_mask[0]))
        names.extend(["tb_cold", "tb_mean", "tb_std", "tb_max", "tb_cooling", "sat_mask"])
        rows.append(row)
        targets.append(float(s.y_max) * MAX_EDR)
        if feat_names is None:
            feat_names = names

    return np.array(rows, dtype=np.float32), np.array(targets, dtype=np.float32), feat_names


def _metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    corr = float(pd.Series(y_true).corr(pd.Series(y_pred)))
    mae  = float(np.abs(y_true - y_pred).mean())
    return dict(rmse=rmse, pearson=corr, mae=mae)


def run(samples, n_folds: int = 5, seed: int = 42) -> dict:
    X, y, feat_names = pool_features(samples)
    groups = np.arange(len(samples))  # each event is its own group

    oof_pred = np.zeros_like(y)
    gkf = GroupKFold(n_splits=n_folds)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            tree_method="hist",
            verbosity=0,
        )
        model.fit(X[tr_idx], y[tr_idx],
                  eval_set=[(X[va_idx], y[va_idx])],
                  verbose=False)
        oof_pred[va_idx] = model.predict(X[va_idx])
        fold_metrics.append(_metrics(y[va_idx], oof_pred[va_idx]))
        log.info("Fold %d  RMSE=%.4f  r=%.3f", fold + 1,
                 fold_metrics[-1]["rmse"], fold_metrics[-1]["pearson"])

    oof_m = _metrics(y, oof_pred)
    log.info("OOF  RMSE=%.4f  r=%.3f  MAE=%.4f", oof_m["rmse"], oof_m["pearson"], oof_m["mae"])

    # Final model on all data
    final = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, tree_method="hist", verbosity=0,
    )
    final.fit(X, y)

    MODELS_DIR.mkdir(exist_ok=True)
    final.save_model(str(MODELS_DIR / "xgb_baseline.json"))

    imp_df = pd.DataFrame({"feature": feat_names, "importance": final.feature_importances_})
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(str(MODELS_DIR / "xgb_importance.csv"), index=False)

    oof_df = pd.DataFrame({
        "event_id": [s.event_id for s in samples],
        "y_true":   y,
        "y_pred":   oof_pred,
        "edr_bin":  [s.edr_bin for s in samples],
    })
    oof_df.to_csv(str(MODELS_DIR / "xgb_oof.csv"), index=False)

    return dict(oof=oof_m, fold_metrics=fold_metrics,
                model=final, importance=imp_df, oof_df=oof_df)


def main():
    import sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from src.data.dataset import build_raw_samples
    samples = build_raw_samples()
    run(samples)


if __name__ == "__main__":
    main()
