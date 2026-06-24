"""
Generate the real results figures from committed model outputs:
  report/figures/cnn_pred_vs_obs.png  — CatCNN OOF predicted vs observed + per-bin RMSE
  report/figures/model_comparison.png — CatCNN vs XGBoost OOF, grouped bars
  report/figures/xgb_importance.png   — XGBoost top-15 feature importance (named)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

# ── diagnostic channel index → name  (level-major: 12 diagnostics × [225,250,300]) ─
DIAGS = ["VWS","N2","Ri","DEF","DIV","TI1","TI2","VORT","FRONTO","WSPD","OMEGA","VADV"]
LEVELS = [225, 250, 300]
def diag_name(idx):
    lv, d = divmod(idx, 12)
    return f"{DIAGS[d]}@{LEVELS[lv]}"
def feat_label(f):
    # e.g. diag23_std → VADV@250 (std);  ONI → ONI;  cos_doy → cos(DOY)
    if f.startswith("diag"):
        body, _, stat = f.partition("_")
        n = int(body[4:])
        return f"{diag_name(n)} ({stat})"
    return {"cos_doy":"cos(DOY)","sin_doy":"sin(DOY)","cos_hour":"cos(hr)",
            "sin_hour":"sin(hr)","ONI":"ONI (ENSO)","Nino34":"Niño3.4",
            "PDO":"PDO","QBO":"QBO"}.get(f, f)

C_LIGHT, C_MOD, C_SEV = '#2980b9', '#e67e22', '#c0392b'
BINCOL = {'light':C_LIGHT,'moderate':C_MOD,'severe':C_SEV}

# ════════════════════════════════════════════════════════════════════════════════
# 1. CatCNN OOF predicted vs observed  (+ per-bin RMSE)
# ════════════════════════════════════════════════════════════════════════════════
c = pd.read_csv('models/cat_cnn_torch_eval.csv')
oof = c[c.split == 'train_oof'].copy()
oof['bin'] = pd.cut(oof.y_max_true, [-1,0.199,0.399,9], labels=['light','moderate','severe'])

from scipy.stats import pearsonr
r_all = pearsonr(oof.y_max_true, oof.y_max_pred)[0]
rmse_all = np.sqrt(np.mean((oof.y_max_true - oof.y_max_pred)**2))

fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.6), dpi=160,
                               gridspec_kw={'width_ratios':[1.25,1]})

# (a) scatter
for b in ['light','moderate','severe']:
    s = oof[oof.bin == b]
    axa.scatter(s.y_max_true, s.y_max_pred, s=46, alpha=0.78,
                color=BINCOL[b], edgecolor='white', lw=0.5, label=f'{b} (n={len(s)})')
axa.plot([0,1],[0,1],'k--',lw=1.2,alpha=0.7)
axa.set_xlim(0,1); axa.set_ylim(0,1)
axa.set_xlabel('Observed max EDR  (m$^{2/3}$ s$^{-1}$)', fontsize=11)
axa.set_ylabel('Predicted max EDR', fontsize=11)
axa.set_title(f'(a) CatCNN out-of-fold predictions  (n=127)\n'
              f'Pearson r = {r_all:.3f}   ·   RMSE = {rmse_all:.3f}',
              fontsize=11, fontweight='bold')
axa.legend(fontsize=9, loc='upper left', framealpha=0.9)
axa.grid(alpha=0.25)

# (b) per-bin RMSE — CatCNN vs XGBoost
x = pd.read_csv('models/xgb_oof.csv')
bins = ['light','moderate','severe']
def binrmse(df, yt, yp, binner):
    out=[]
    for b in bins:
        s = df[binner(df)==b]
        out.append(np.sqrt(np.mean((s[yt]-s[yp])**2)))
    return out
cnn_b = binrmse(oof,'y_max_true','y_max_pred', lambda d: d['bin'])
xgb_b = binrmse(x,'y_true','y_pred', lambda d: d['edr_bin'])
xi = np.arange(3); w=0.38
axb.bar(xi-w/2, cnn_b, w, label='CatCNN', color='#16a085', edgecolor='white')
axb.bar(xi+w/2, xgb_b, w, label='XGBoost', color='#7f8c8d', edgecolor='white')
axb.set_xticks(xi); axb.set_xticklabels([f'{b}' for b in bins], fontsize=10)
axb.set_ylabel('RMSE  (m$^{2/3}$ s$^{-1}$)', fontsize=11)
axb.set_title('(b) Per-severity RMSE', fontsize=11, fontweight='bold')
axb.legend(fontsize=10); axb.grid(axis='y', alpha=0.25)
for i,(cv,xv) in enumerate(zip(cnn_b,xgb_b)):
    axb.text(i-w/2, cv+0.004, f'{cv:.3f}', ha='center', fontsize=7.5)
    axb.text(i+w/2, xv+0.004, f'{xv:.3f}', ha='center', fontsize=7.5)

plt.tight_layout()
plt.savefig('report/figures/cnn_pred_vs_obs.png', dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved cnn_pred_vs_obs.png')

# ════════════════════════════════════════════════════════════════════════════════
# 2. Model comparison grouped bars (OOF)
# ════════════════════════════════════════════════════════════════════════════════
M = json.load(open('models/metrics_summary.json'))
cat, xgb = M['CatCNN_OOF'], M['XGBoost_OOF']
metrics = [('Pearson','Pearson r',1),('AUPRC','AUPRC@0.20',1),
           ('AUROC','AUROC@0.20',1),('RMSE','RMSE (↓)',1),('MAE','MAE (↓)',1)]
fig, ax = plt.subplots(figsize=(11, 5.6), dpi=160)
xi = np.arange(len(metrics)); w=0.36
cvals=[cat[k] for k,_,_ in metrics]; xvals=[xgb[k] for k,_,_ in metrics]
b1=ax.bar(xi-w/2, cvals, w, label='Physics CatCNN', color='#16a085', edgecolor='white')
b2=ax.bar(xi+w/2, xvals, w, label='XGBoost (pooled)', color='#7f8c8d', edgecolor='white')
ax.set_xticks(xi); ax.set_xticklabels([lbl for _,lbl,_ in metrics], fontsize=10.5)
ax.set_ylabel('score', fontsize=11)
ax.set_title('Out-of-fold model comparison — Physics CatCNN vs XGBoost\n'
             '(higher is better for Pearson/AUPRC/AUROC; lower for RMSE/MAE)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='upper right'); ax.grid(axis='y', alpha=0.25)
ax.set_ylim(0, 1.0)
for rects in (b1,b2):
    for r in rects:
        ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.012,
                f'{r.get_height():.3f}', ha='center', fontsize=8.5, fontweight='bold')
plt.tight_layout()
plt.savefig('report/figures/model_comparison.png', dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved model_comparison.png')

# ════════════════════════════════════════════════════════════════════════════════
# 3. XGBoost feature importance (top 15, named)
# ════════════════════════════════════════════════════════════════════════════════
imp = pd.read_csv('models/xgb_importance.csv').head(15).iloc[::-1]
labels = [feat_label(f) for f in imp.feature]
def fcolor(f):
    if f.startswith('diag'): return '#2c6fbb'
    if f in ('ONI','Nino34','PDO','QBO'): return '#8e44ad'
    return '#27ae60'   # time
colors = [fcolor(f) for f in imp.feature]
fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
ax.barh(range(len(imp)), imp.importance, color=colors, edgecolor='white')
ax.set_yticks(range(len(imp))); ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel('XGBoost gain importance', fontsize=11)
ax.set_title('XGBoost feature importance (top 15)\n'
             'physics diagnostics dominate; climate & time enter only weakly',
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.25)
import matplotlib.patches as mpatches
ax.legend(handles=[mpatches.Patch(color='#2c6fbb',label='physics diagnostic'),
                   mpatches.Patch(color='#8e44ad',label='climate index'),
                   mpatches.Patch(color='#27ae60',label='cyclic time')],
          fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('report/figures/xgb_importance.png', dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved xgb_importance.png')
print(f'\nCatCNN OOF: r={cat["Pearson"]:.3f} RMSE={cat["RMSE"]:.3f} AUPRC={cat["AUPRC"]:.3f}')
print(f'XGBoost OOF: r={xgb["Pearson"]:.3f} RMSE={xgb["RMSE"]:.3f} AUPRC={xgb["AUPRC"]:.3f}')
print(f'Top feature: {feat_label(imp.feature.iloc[-1])}')
