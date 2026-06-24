#!/usr/bin/env python3
"""
Generate all validation plots and metrics summary for CatCNN + XGBoost baseline.
Outputs to results/figures/.
"""
import sys
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path

from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, mean_squared_error, confusion_matrix,
    ConfusionMatrixDisplay
)

FIGURES = Path("results/figures")
FIGURES.mkdir(parents=True, exist_ok=True)
MODELS  = Path("models")
MAX_EDR = 0.95

plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
})


def severity(edr):
    if edr < 0.15: return "smooth"
    if edr < 0.35: return "light"
    if edr < 0.55: return "moderate"
    return "severe"

SEV_COLOR = {"smooth": "#2196F3", "light": "#4CAF50",
             "moderate": "#FF9800", "severe": "#F44336"}


# ── Load data ──────────────────────────────────────────────────────────────────

def load_cnn_eval():
    df = pd.read_csv(MODELS / "cat_cnn_torch_eval.csv")
    return df

def load_xgb_oof():
    return pd.read_csv(MODELS / "xgb_oof.csv")

def load_fold_metrics():
    return pd.read_csv(MODELS / "cat_cnn_torch_folds.csv", index_col=0)

def load_xgb_importance():
    return pd.read_csv(MODELS / "xgb_importance.csv")


# ── 1. OOF scatter: CNN + XGB side-by-side ───────────────────────────────────

def plot_oof_scatter(cnn_df, xgb_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Out-of-Fold Predictions vs True EDR", fontweight="bold")

    for ax, (df, name, ycol, xcol) in zip(
        axes,
        [
            (cnn_df, "CatCNN (PyTorch)", "y_max_true", "y_max_pred"),
            (xgb_df, "XGBoost Baseline", "y_true", "y_pred"),
        ]
    ):
        y_true = df[xcol].values if name.startswith("XGB") else df[ycol].values
        y_pred = df[ycol].values if name.startswith("XGB") else df[xcol].values

        if name.startswith("XGB"):
            y_true = xgb_df["y_true"].values
            y_pred = xgb_df["y_pred"].values
        else:
            oof_mask = df["split"].str.contains("oof", na=False)
            sub = df[oof_mask] if oof_mask.any() else df
            y_true = sub["y_max_true"].values
            y_pred = sub["y_max_pred"].values

        colors = [SEV_COLOR[severity(v)] for v in y_true]
        ax.scatter(y_true, y_pred, c=colors, alpha=0.75, s=45, edgecolors="white", linewidths=0.5)

        lo = min(y_true.min(), y_pred.min()) - 0.02
        hi = max(y_true.max(), y_pred.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.5, label="Perfect")

        r  = float(pd.Series(y_true).corr(pd.Series(y_pred)))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(np.abs(y_true - y_pred).mean())
        ax.set_title(f"{name}\nRMSE={rmse:.3f}  r={r:.3f}  MAE={mae:.3f}")
        ax.set_xlabel("True max EDR")
        ax.set_ylabel("Predicted max EDR")

        from matplotlib.lines import Line2D
        handles = [Line2D([0],[0], marker="o", color="w", markerfacecolor=c, markersize=9, label=s)
                   for s, c in SEV_COLOR.items()]
        ax.legend(handles=handles, title="Severity", fontsize=8, title_fontsize=8)

    plt.tight_layout()
    out = FIGURES / "01_oof_scatter.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── 2. Residual analysis ──────────────────────────────────────────────────────

def plot_residuals(cnn_df, xgb_df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Residual Analysis", fontweight="bold")

    for row, (df, name, ycol, xcol) in enumerate([
        (cnn_df, "CatCNN", "y_max_true", "y_max_pred"),
        (xgb_df, "XGBoost", "y_true", "y_pred"),
    ]):
        if name == "CatCNN":
            mask = cnn_df["split"].str.contains("oof", na=False)
            sub  = df[mask] if mask.any() else df
            y_t  = sub["y_max_true"].values
            y_p  = sub["y_max_pred"].values
        else:
            y_t  = df["y_true"].values
            y_p  = df["y_pred"].values

        res = y_p - y_t

        # Residuals vs true
        ax1 = axes[row, 0]
        ax1.scatter(y_t, res, alpha=0.6, s=30, c="steelblue", edgecolors="white", lw=0.4)
        ax1.axhline(0, color="red", linestyle="--", lw=1.2)
        ax1.set_xlabel("True EDR")
        ax1.set_ylabel("Residual (pred − true)")
        ax1.set_title(f"{name} — Residuals vs True")

        # Histogram of residuals
        ax2 = axes[row, 1]
        ax2.hist(res, bins=25, color="steelblue", edgecolor="white", alpha=0.85)
        ax2.axvline(res.mean(), color="red", linestyle="--", label=f"mean={res.mean():.3f}")
        ax2.axvline(0, color="black", linestyle=":", lw=0.8)
        ax2.set_xlabel("Residual")
        ax2.set_ylabel("Count")
        ax2.set_title(f"{name} — Residual Distribution  σ={res.std():.3f}")
        ax2.legend(fontsize=8)

    plt.tight_layout()
    out = FIGURES / "02_residuals.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── 3. ROC + PR curves ────────────────────────────────────────────────────────

def plot_roc_pr(cnn_df, xgb_df, threshold=0.35):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"Binary Classification (threshold={threshold} EDR = moderate+)",
                 fontweight="bold")

    colors = {"CatCNN": "#E91E63", "XGBoost": "#2196F3"}

    for name, df, yt_col, yp_col in [
        ("CatCNN",  cnn_df, "y_max_true", "y_max_pred"),
        ("XGBoost", xgb_df, "y_true",     "y_pred"),
    ]:
        if name == "CatCNN":
            mask = cnn_df["split"].str.contains("oof", na=False)
            sub  = df[mask] if mask.any() else df
            y_t  = sub[yt_col].values
            y_p  = sub[yp_col].values
        else:
            y_t  = df["y_true"].values
            y_p  = df["y_pred"].values

        labels = (y_t >= threshold).astype(int)
        if labels.sum() == 0 or labels.sum() == len(labels):
            print(f"  Skipping {name} ROC/PR: degenerate class distribution")
            continue

        fpr, tpr, _ = roc_curve(labels, y_p)
        auroc = roc_auc_score(labels, y_p)
        axes[0].plot(fpr, tpr, color=colors[name], lw=2,
                     label=f"{name}  AUC={auroc:.3f}")

        prec, rec, _ = precision_recall_curve(labels, y_p)
        auprc = average_precision_score(labels, y_p)
        axes[1].plot(rec, prec, color=colors[name], lw=2,
                     label=f"{name}  AP={auprc:.3f}")

    axes[0].plot([0,1], [0,1], "k--", lw=0.8)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend()

    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall Curve"); axes[1].legend()

    plt.tight_layout()
    out = FIGURES / "03_roc_pr.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── 4. Per-bin performance ────────────────────────────────────────────────────

def plot_per_bin(cnn_df, xgb_df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Per-Severity-Bin Performance", fontweight="bold")

    bins_order = ["light", "moderate", "severe"]

    for ax, metric_name in zip(axes, ["RMSE", "MAE", "Pearson r"]):
        cnn_vals, xgb_vals = [], []

        for b in bins_order:
            # CNN OOF
            mask = cnn_df["split"].str.contains("oof", na=False)
            sub  = cnn_df[mask] if mask.any() else cnn_df
            # use event bins from events.csv
            events = pd.read_csv("events.csv").set_index("event_id")
            sub2 = sub.copy()
            sub2["edr_bin"] = sub2["event_id"].map(lambda x: events.at[x, "edr_bin"] if x in events.index else "?")
            sub_b = sub2[sub2["edr_bin"] == b]

            if len(sub_b) < 2:
                cnn_vals.append(float("nan"))
            else:
                yt = sub_b["y_max_true"].values
                yp = sub_b["y_max_pred"].values
                if metric_name == "RMSE":
                    cnn_vals.append(float(np.sqrt(mean_squared_error(yt, yp))))
                elif metric_name == "MAE":
                    cnn_vals.append(float(np.abs(yt - yp).mean()))
                else:
                    cnn_vals.append(float(pd.Series(yt).corr(pd.Series(yp))))

            # XGBoost
            xgb_b = xgb_df[xgb_df["edr_bin"] == b]
            if len(xgb_b) < 2:
                xgb_vals.append(float("nan"))
            else:
                yt = xgb_b["y_true"].values
                yp = xgb_b["y_pred"].values
                if metric_name == "RMSE":
                    xgb_vals.append(float(np.sqrt(mean_squared_error(yt, yp))))
                elif metric_name == "MAE":
                    xgb_vals.append(float(np.abs(yt - yp).mean()))
                else:
                    xgb_vals.append(float(pd.Series(yt).corr(pd.Series(yp))))

        x = np.arange(len(bins_order))
        w = 0.35
        b1 = ax.bar(x - w/2, cnn_vals, width=w, label="CatCNN", color="#E91E63", alpha=0.85)
        b2 = ax.bar(x + w/2, xgb_vals, width=w, label="XGBoost", color="#2196F3", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(bins_order)
        ax.set_title(metric_name)
        ax.set_ylabel(metric_name)
        ax.legend()

    plt.tight_layout()
    out = FIGURES / "04_per_bin.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── 5. Fold-level CV summary ──────────────────────────────────────────────────

def plot_cv_folds(fold_df):
    metrics = [c for c in fold_df.columns if c not in ("fold",) and not c.startswith("Unnamed")]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle("5-Fold Cross-Validation Metrics (CatCNN)", fontweight="bold")

    for ax, m in zip(axes, metrics):
        vals = fold_df[m].values
        ax.bar(range(1, len(vals)+1), vals, color="steelblue", alpha=0.8)
        ax.axhline(vals.mean(), color="red", linestyle="--", lw=1.2, label=f"mean={vals.mean():.3f}")
        ax.set_title(m)
        ax.set_xlabel("Fold")
        ax.set_xticks(range(1, len(vals)+1))
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = FIGURES / "05_cv_folds.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── 6. XGBoost feature importance (top 30) ───────────────────────────────────

def plot_feature_importance(imp_df):
    top = imp_df.head(30)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#E91E63" if "diag" in n else "#2196F3" if n in ("ONI","Nino34","PDO","QBO")
              else "#4CAF50" for n in top["feature"]]
    ax.barh(range(len(top)), top["importance"].values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values[::-1], fontsize=8)
    ax.set_xlabel("Feature Importance")
    ax.set_title("XGBoost Feature Importance (Top 30)")

    from matplotlib.patches import Patch
    handles = [Patch(color="#E91E63", label="ERA5 diagnostic"),
               Patch(color="#2196F3", label="Climate index"),
               Patch(color="#4CAF50", label="Other (time/sat)")]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    plt.tight_layout()
    out = FIGURES / "06_feature_importance.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── 7. EDR distribution: true vs predicted ───────────────────────────────────

def plot_edr_distribution(cnn_df):
    mask = cnn_df["split"].str.contains("oof", na=False)
    sub  = cnn_df[mask] if mask.any() else cnn_df
    y_t  = sub["y_max_true"].values
    y_p  = sub["y_max_pred"].values

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("EDR Distribution: True vs Predicted", fontweight="bold")

    bins = np.linspace(0, 0.96, 25)
    axes[0].hist(y_t, bins=bins, alpha=0.7, color="#E91E63", label="True")
    axes[0].hist(y_p, bins=bins, alpha=0.7, color="#2196F3", label="Predicted")
    axes[0].set_xlabel("max EDR"); axes[0].set_ylabel("Count")
    axes[0].set_title("Histogram overlay")
    axes[0].legend()

    axes[1].scatter(y_t, y_p, alpha=0.6, s=35, c=[SEV_COLOR[severity(v)] for v in y_t])
    axes[1].axline((0, 0), slope=1, color="k", linestyle="--", lw=1)
    for thr, lbl in [(0.15, "light"), (0.35, "moderate"), (0.55, "severe")]:
        axes[1].axvline(thr, color="gray", linestyle=":", lw=0.8, alpha=0.7)
        axes[1].axhline(thr, color="gray", linestyle=":", lw=0.8, alpha=0.7)
    axes[1].set_xlabel("True max EDR"); axes[1].set_ylabel("Predicted max EDR")
    axes[1].set_title("Scatter with severity thresholds")

    plt.tight_layout()
    out = FIGURES / "07_edr_distribution.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── 8. Severity confusion matrix ─────────────────────────────────────────────

def plot_confusion_matrix(cnn_df):
    mask = cnn_df["split"].str.contains("oof", na=False)
    sub  = cnn_df[mask] if mask.any() else cnn_df
    y_t  = sub["y_max_true"].values
    y_p  = sub["y_max_pred"].values

    labels = ["light", "moderate", "severe"]
    y_t_cat = [severity(v) for v in y_t]
    y_p_cat = [severity(v) for v in y_p]

    # Normalize to only labels that appear
    present = sorted(set(y_t_cat + y_p_cat),
                     key=lambda x: labels.index(x) if x in labels else 99)

    cm = confusion_matrix(y_t_cat, y_p_cat, labels=present)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Severity Classification — CatCNN OOF", fontweight="bold")

    # Counts
    disp = ConfusionMatrixDisplay(cm, display_labels=present)
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Counts")

    # Normalized
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=present)
    disp2.plot(ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title("Row-normalised (recall)")

    plt.tight_layout()
    out = FIGURES / "08_confusion_matrix.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── 9. CNN vs XGB comparison summary ─────────────────────────────────────────

def plot_model_comparison(cnn_df, xgb_df):
    # CNN OOF
    mask = cnn_df["split"].str.contains("oof", na=False)
    sub  = cnn_df[mask] if mask.any() else cnn_df
    cnn_yt = sub["y_max_true"].values
    cnn_yp = sub["y_max_pred"].values

    xgb_yt = xgb_df["y_true"].values
    xgb_yp = xgb_df["y_pred"].values

    def get_metrics(yt, yp, thr=0.35):
        labels = (yt >= thr).astype(int)
        m = dict(
            RMSE    = float(np.sqrt(mean_squared_error(yt, yp))),
            MAE     = float(np.abs(yt - yp).mean()),
            Pearson = float(pd.Series(yt).corr(pd.Series(yp))),
        )
        if labels.sum() > 0 and labels.sum() < len(labels):
            m["AUROC"] = float(roc_auc_score(labels, yp))
            m["AUPRC"] = float(average_precision_score(labels, yp))
        return m

    cnn_m = get_metrics(cnn_yt, cnn_yp)
    xgb_m = get_metrics(xgb_yt, xgb_yp)

    shared_keys = [k for k in cnn_m if k in xgb_m]
    x = np.arange(len(shared_keys))
    cnn_vals = [cnn_m[k] for k in shared_keys]
    xgb_vals = [xgb_m[k] for k in shared_keys]

    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.35
    ax.bar(x - w/2, cnn_vals, w, label="CatCNN",  color="#E91E63", alpha=0.85)
    ax.bar(x + w/2, xgb_vals, w, label="XGBoost", color="#2196F3", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(shared_keys)
    ax.set_title("OOF Metrics — CatCNN vs XGBoost", fontweight="bold")
    ax.set_ylabel("Metric value")
    ax.legend()

    # Annotate
    for xi, (cv, xv) in enumerate(zip(cnn_vals, xgb_vals)):
        ax.text(xi - w/2, cv + 0.005, f"{cv:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(xi + w/2, xv + 0.005, f"{xv:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out = FIGURES / "09_model_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # Print summary table
    print("\n─── Metrics Summary ───────────────────────────────────────────────")
    print(f"{'Metric':<12} {'CatCNN':>10} {'XGBoost':>10}")
    print("─" * 36)
    for k in shared_keys:
        print(f"{k:<12} {cnn_m[k]:>10.4f} {xgb_m[k]:>10.4f}")
    print("─" * 36)

    # Save metrics JSON
    metrics_out = MODELS / "metrics_summary.json"
    with open(metrics_out, "w") as f:
        json.dump({"CatCNN_OOF": cnn_m, "XGBoost_OOF": xgb_m}, f, indent=2)
    print(f"Metrics saved → {metrics_out}")


# ── 10. Sample prediction heatmaps ───────────────────────────────────────────

def generate_sample_heatmaps(n_samples=12):
    """Run inference on a subset of events and save heatmap grid."""
    import yaml
    from src.data.dataset import build_raw_samples, load_norm, channel_names as _cnames
    from src.models.cnn_torch import build_model, DEFAULT_CFG
    from predict_cat_torch import predict_event

    cfg_path = "configs/cnn.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    plevels = cfg.get("primary_levels", [225, 250, 300])
    cnames  = _cnames(plevels)
    device  = torch.device("cpu")

    norm    = load_norm(str(MODELS / "cat_cnn_torch_norm.npz"))
    model_cfg = {**DEFAULT_CFG, **cfg.get("model", {})}
    model   = build_model(model_cfg, len(cnames)).to(device)
    model.load_state_dict(torch.load(str(MODELS / "cat_cnn_torch.pt"), map_location=device))
    model.eval()

    samples = build_raw_samples(primary_levels=plevels, grid_size=cfg.get("grid_size", 24))
    # Pick diverse events: severe, moderate, light
    events = pd.read_csv("events.csv").set_index("event_id")
    severe  = [s for s in samples if events.at[s.event_id, "edr_bin"] == "severe"][:4]
    moderate= [s for s in samples if events.at[s.event_id, "edr_bin"] == "moderate"][:4]
    light   = [s for s in samples if events.at[s.event_id, "edr_bin"] == "light"][:4]
    picks   = severe + moderate + light
    picks   = picks[:n_samples]

    cols = 4
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes_flat = axes.flatten() if rows > 1 else axes

    fig.suptitle("Sample Turbulence Intensity Fields — CatCNN Predictions",
                 fontweight="bold", fontsize=13)

    cmap = plt.cm.YlOrRd
    from predict_cat_torch import predict_event as pe

    for i, s in enumerate(picks):
        ax = axes_flat[i]
        with torch.no_grad():
            r = pe(s, model, norm, cnames, device)
        field = r["field"]
        vmax  = max(field.max(), 0.3)
        im = ax.imshow(field, cmap=cmap, vmin=0, vmax=vmax, origin="upper", aspect="auto")
        sev_true = severity(r["y_max"])
        sev_pred = severity(r["max_hat"])
        color = "red" if sev_true != sev_pred else "black"
        ax.set_title(
            f"Event {s.event_id:04d}\n"
            f"True {r['y_max']:.2f} ({sev_true})\n"
            f"Pred {r['max_hat']:.2f} ({sev_pred})",
            fontsize=7.5, color=color
        )
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    for j in range(len(picks), len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    out = FIGURES / "10_sample_heatmaps.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading evaluation data...")
    cnn_df  = load_cnn_eval()
    xgb_df  = load_xgb_oof()
    fold_df = load_fold_metrics()
    imp_df  = load_xgb_importance()

    print(f"CNN eval rows: {len(cnn_df)}  XGB OOF rows: {len(xgb_df)}")

    print("\nGenerating plots...")
    plot_oof_scatter(cnn_df, xgb_df)
    plot_residuals(cnn_df, xgb_df)
    plot_roc_pr(cnn_df, xgb_df)
    plot_per_bin(cnn_df, xgb_df)
    plot_cv_folds(fold_df)
    plot_feature_importance(imp_df)
    plot_edr_distribution(cnn_df)
    plot_confusion_matrix(cnn_df)
    plot_model_comparison(cnn_df, xgb_df)
    generate_sample_heatmaps()

    print("\nAll plots saved to results/figures/")
    figs = sorted(FIGURES.glob("*.png"))
    for f in figs:
        print(f"  {f.name}")
