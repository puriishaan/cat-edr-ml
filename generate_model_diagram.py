"""
Generate model.png — detailed CatCNNTorch architecture diagram.
Run from repo root: python generate_model_diagram.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Canvas
# ─────────────────────────────────────────────────────────────────────────────
FW, FH = 26, 16
fig, ax = plt.subplots(figsize=(FW, FH), dpi=160)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')
BG = '#f0f2f5'
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ─────────────────────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────────────────────
CD  = '#1565c0'   # diag input
CC  = '#6a1b9a'   # conditioning
CS  = '#e65100'   # satellite
CE  = '#1976d2'   # encoder conv
CF  = '#7b1fa2'   # FiLM
CB  = '#00695c'   # bottleneck
CSK = '#2e7d32'   # skip
CDK = '#0277bd'   # decoder conv
CH  = '#b71c1c'   # head / softplus
CA  = '#e65100'   # aggregation
CL  = '#263238'   # loss box
CHY = '#4e342e'   # hyperparams box

# ─────────────────────────────────────────────────────────────────────────────
# Helper: draw a rounded box with multi-line text
# ─────────────────────────────────────────────────────────────────────────────
def box(cx, cy, w, h, color, text, fs=7.5, tc='white', alpha=0.93, lw=1.4, zorder=4):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                        boxstyle='round,pad=0.07,rounding_size=0.12',
                        facecolor=color, edgecolor='white',
                        linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold', zorder=zorder+1,
            multialignment='center', linespacing=1.3)

def arr(x1, y1, x2, y2, color='#37474f', lw=1.6, asz=10, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=asz))

def arr_curve(x1, y1, x2, y2, color, lw=1.5, rad=0.0, asz=10):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=asz,
                                connectionstyle=f'arc3,rad={rad}'))

def label(cx, cy, text, fs=6.5, color='#455a64', bold=False, italic=False):
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, color=color,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal')

# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────
ax.text(13, 15.55, 'CatCNNTorch — Physics-Informed U-Net Architecture',
        ha='center', va='center', fontsize=15, fontweight='bold', color='#0d1b2a')
ax.text(13, 15.1, '36 input channels (12 diagnostics × 3 levels)  |  '
                  'depth=3  |  base_filters=32  |  ≈80k parameters  |  '
                  'grid: 24×24 (≈0.25° × 2° event box)',
        ha='center', va='center', fontsize=8.5, color='#546e7a', style='italic')

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN LAYOUT
#   col 0: labels/headers
#   col 1 (x=2.0):  input streams
#   col 2 (x=4.8):  encoder stage 1
#   col 3 (x=7.5):  encoder stage 2
#   col 4 (x=10.2): encoder stage 3
#   col 5 (x=12.9): bottleneck + satellite fusion
#   col 6 (x=15.6): decoder stage 3 (→ 6×6)
#   col 7 (x=18.3): decoder stage 2 (→ 12×12)
#   col 8 (x=21.0): decoder stage 1 (→ 24×24)
#   col 9 (x=23.5): head + aggregation
#   loss panel: right side (floated)
# ─────────────────────────────────────────────────────────────────────────────

# Row heights (top to bottom)
# R_TOP = 12.8  encoder / decoder bodies
# R_MID = 10.7  FiLM modules
# R_BOT =  8.5  pool labels / skip labels
# R_BTL =  6.2  bottleneck row
# R_SAT =  4.0  satellite MLP

R_TOP = 12.5
R_FILM = 11.0
R_SKIP = 9.8
R_BTL  = 7.2
R_SAT  = 4.8

# ─── SECTION HEADER BANDS ────────────────────────────────────────────────────
def hband(y1, y2, color, text, xa=0.45):
    r = FancyBboxPatch((xa, y1), 1.3, y2-y1,
                        boxstyle='round,pad=0.05,rounding_size=0.1',
                        facecolor=color, edgecolor='none', alpha=0.18, zorder=2)
    ax.add_patch(r)
    ax.text(xa + 0.65, (y1+y2)/2, text, ha='center', va='center', fontsize=7,
            color=color, fontweight='bold', rotation=90, zorder=3)

hband(R_BTL+0.6, R_TOP+0.6, CD, 'ENCODER', 0.3)
hband(R_BTL+0.6, R_TOP+0.6, CDK, 'DECODER', 23.3)

# ─── SECTION LABEL TAPE ──────────────────────────────────────────────────────
for xc, lbl, col in [
    (2.0, 'Inputs', '#37474f'),
    (4.8, 'Enc-1\n24×24', CE),
    (7.5, 'Enc-2\n12×12', CE),
    (10.2, 'Enc-3\n6×6', CE),
    (12.9, 'Bottleneck\n3×3', CB),
    (15.6, 'Dec-3\n6×6', CDK),
    (18.3, 'Dec-2\n12×12', CDK),
    (21.0, 'Dec-1\n24×24', CDK),
    (23.5, 'Head\n& Agg.', CH),
]:
    ax.text(xc, 14.6, lbl, ha='center', va='center', fontsize=7.5,
            color=col, fontweight='bold', multialignment='center')

# ─────────────────────────────────────────────────────────────────────────────
# INPUT COLUMN  (x = 2.0)
# ─────────────────────────────────────────────────────────────────────────────
IX = 2.0
box(IX, R_TOP, 2.8, 1.4, CD,
    'Physics Diagnostics\n'
    '(B, 36, 24, 24)\n'
    '12 diag × 3 levels\n'
    '(225 / 250 / 300 hPa)',
    fs=7.5)

box(IX, 11.3, 2.8, 0.9, CC,
    'Climate indices  (B, 4)\n'
    'ONI · Niño3.4 · PDO · QBO', fs=7)
box(IX, 10.3, 2.8, 0.9, CC,
    'Cyclic time  (B, 4)\n'
    'sin/cos(DOY) · sin/cos(hour)', fs=7)

# concat climate+time
box(IX, 9.3, 1.6, 0.65, CC, 'cat → cond  (B, 8)', fs=7)
arr(IX, 11.3-0.45, IX, 9.3+0.33, color=CC, lw=1.3)
arr(IX, 10.3-0.45, IX, 9.3+0.33, color=CC, lw=1.3)

box(IX, R_SAT, 2.8, 1.5, CS,
    'Satellite MLP  (optional)\n'
    'Input: (B, 6)  [5 TB stats + mask]\n'
    '  tb_cold · tb_mean · tb_std\n'
    '  tb_max · Δtb · avail. mask\n'
    '→ Lin(32, GELU) → Lin(16)',
    fs=7)

label(IX, R_SAT-1.0,
      'Covers 64/150 events (2017+)\nzero-padded when absent',
      fs=6.5, color=CS, italic=True)

# ─────────────────────────────────────────────────────────────────────────────
# ENCODER  (three stages)
# ─────────────────────────────────────────────────────────────────────────────
ENC_XS  = [4.8, 7.5, 10.2]
ENC_INS = [36,  32,  64]
ENC_OTS = [32,  64,  128]
ENC_SPS = ['24×24', '12×12', '6×6']
ENC_SP2 = ['12×12', '6×6', '3×3']

for i, (ex, ei, eo, sp, sp2) in enumerate(
        zip(ENC_XS, ENC_INS, ENC_OTS, ENC_SPS, ENC_SP2)):

    # ConvBlock
    box(ex, R_TOP, 2.2, 1.6, CE,
        f'ConvBlock  {sp}\n'
        f'Conv(3×3) {ei}→{eo}\n'
        f'BatchNorm → GELU\n'
        f'Dropout2d(p=0.10)\n'
        f'Conv(3×3) {eo}→{eo}\n'
        f'+ residual 1×1',
        fs=7)

    # FiLM after ConvBlock
    box(ex, R_FILM, 2.0, 0.85, CF,
        f'FiLM  (cond_dim=8 → {2*eo})\n'
        f'x ← x·(γ+1) + β  [identity init]\n'
        f'Lin(8,{2*eo}) → chunk(γ,β)',
        fs=6.5)

    # cond → FiLM
    arr(IX, 9.3-0.32, ex, R_FILM+0.42, color=CC, lw=1.2, asz=8)

    # ConvBlock → FiLM
    arr(ex, R_TOP-0.8, ex, R_FILM+0.42, color=CE, lw=1.3)

    # FiLM → pool
    pool_y = R_SKIP
    box(ex, pool_y, 2.0, 0.7, CE,
        f'MaxPool2d(2)  {sp} → {sp2}',
        fs=7, alpha=0.8)
    arr(ex, R_FILM-0.42, ex, pool_y+0.35, color=CE, lw=1.3)

    # Pool → next encoder (or bottleneck)
    if i < 2:
        arr(ex + 1.1, pool_y, ENC_XS[i+1]-1.1, R_TOP-0.8, color=CE, lw=1.5)
    else:
        # Pool → bottleneck
        arr(ex + 1.1, pool_y, 12.9-1.1, R_BTL+0.7, color=CE, lw=1.5)

# Input → Enc-1
arr(IX+1.4, R_TOP, ENC_XS[0]-1.1, R_TOP, color=CD, lw=2)

# ─────────────────────────────────────────────────────────────────────────────
# SKIP CONNECTIONS  (at FiLM-output level → decoder concat level)
# ─────────────────────────────────────────────────────────────────────────────
DEC_XS = [15.6, 18.3, 21.0]
SKIP_YS = [R_FILM, R_FILM, R_FILM]

# enc1 skip → dec1   enc2 → dec2   enc3 → dec3
for i, (ex, dx, ch, sp) in enumerate(
        zip(ENC_XS, reversed(DEC_XS), ENC_OTS, ENC_SPS)):
    # draw dashed horizontal skip line at an offset y so they don't overlap
    y_skip = R_FILM - 0.0 + i * 0.0
    ax.annotate('', xy=(dx, R_SKIP+0.15 + i*0.18), xytext=(ex, R_SKIP+0.15 + i*0.18),
                arrowprops=dict(arrowstyle='->', color=CSK, lw=1.6,
                                linestyle='dashed', mutation_scale=10))
    ax.text((ex + dx)/2, R_SKIP+0.15 + i*0.18 + 0.18,
            f'skip  ({ch} ch, {sp})',
            ha='center', va='bottom', fontsize=6.5, color=CSK, style='italic')

# ─────────────────────────────────────────────────────────────────────────────
# BOTTLENECK  (x = 12.9)
# ─────────────────────────────────────────────────────────────────────────────
BX = 12.9
box(BX, R_BTL, 2.5, 1.55, CB,
    'Bottleneck  3×3\n'
    'ConvBlock  128→256\n'
    '3×3, BatchNorm, GELU\n'
    'Dropout2d(0.10)\n'
    '+ residual 1×1',
    fs=7.5)

box(BX, R_BTL-1.15, 2.4, 0.85, CF,
    'FiLM  (cond=8 → 512)\n'
    'x ← x·(γ+1) + β  [identity init]',
    fs=6.8)
arr(BX, R_BTL-0.77, BX, R_BTL-1.15+0.42, color=CF, lw=1.2)
# cond → bottleneck FiLM
arr(IX, 9.3-0.32, BX, R_BTL-0.73, color=CC, lw=1.2, asz=8)

# Satellite fusion
box(BX, R_SAT+0.3, 2.5, 1.4, CS,
    'Satellite fusion\n'
    'sat_vec (B,16) broadcast\n'
    '→ (B,16,3,3)\n'
    'cat( bottleneck, sat_map )\n'
    '→ (B, 272, 3, 3)',
    fs=7)
arr(IX+1.4, R_SAT, BX-1.25, R_SAT+0.3, color=CS, lw=1.5)   # satellite MLP → fusion
arr(BX, R_BTL-1.55, BX, R_SAT+1.0, color=CB, lw=1.5)        # bottleneck → fusion

# fusion → first decoder
arr(BX+1.25, R_SAT+0.55, DEC_XS[0]-1.1, R_BTL, color=CB, lw=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# DECODER  (three stages)
# ─────────────────────────────────────────────────────────────────────────────
DEC_INS  = [272, 128, 64]    # in channels (after prev upsample)
DEC_SKIP = [128, 64, 32]     # skip channels
DEC_CATS = [256, 128, 64]    # after concat
DEC_OTS  = [128, 64, 32]     # output channels
DEC_SP_IN= ['3×3', '6×6', '12×12']
DEC_SP_OT= ['6×6', '12×12', '24×24']

for i, (dx, di, ds, dc, do_, si, so) in enumerate(
        zip(DEC_XS, DEC_INS, DEC_SKIP, DEC_CATS, DEC_OTS, DEC_SP_IN, DEC_SP_OT)):

    # Upsample + 1×1
    box(dx, R_BTL+0.2, 2.2, 0.9, CDK,
        f'↑2× bilinear  {si}→{so}\n'
        f'Conv1×1  {di}→{ds}  |  cat skip({ds})\n'
        f'→ ({dc}, {so})',
        fs=6.8)

    # ConvBlock
    box(dx, R_TOP, 2.2, 1.6, CDK,
        f'ConvBlock  {so}\n'
        f'Conv(3×3) {dc}→{do_}\n'
        f'BatchNorm → GELU\n'
        f'Dropout2d(0.10)\n'
        f'Conv(3×3) {do_}→{do_}\n'
        f'+ residual 1×1',
        fs=7)

    # Upsample → ConvBlock
    arr(dx, R_BTL+0.65, dx, R_TOP-0.8, color=CDK, lw=1.3)

    if i < 2:
        arr(dx+1.1, R_BTL+0.2, DEC_XS[i+1]-1.1, R_BTL+0.2, color=CDK, lw=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# HEAD + AGGREGATION  (x ≈ 23.5)
# ─────────────────────────────────────────────────────────────────────────────
HX = 23.5
# Head
box(HX, R_TOP-0.15, 2.4, 1.35, CH,
    'Head\n'
    'Conv1×1(32 → 1)\n'
    'Softplus  →  ≥ 0\n'
    'EDR field  (B,1,24,24)',
    fs=7.5)
# dec1 → head
arr(DEC_XS[2]+1.1, R_TOP, HX-1.2, R_TOP-0.15, color=CDK, lw=1.8)

# Aggregation
box(HX, R_FILM-0.1, 2.4, 1.5, CA,
    'Soft Aggregation\n'
    '─────────────────\n'
    'max_hat: logsumexp(field)\n'
    '         − log(576)  (τ=8.0)\n'
    'mean_hat: field.mean()\n'
    '─────────────────\n'
    'alt: top-k mean  (k=4)',
    fs=7)
arr(HX, R_TOP-0.82, HX, R_FILM+0.65, color=CA, lw=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# LOSS PANEL  (box below, centred under the full diagram)
# ─────────────────────────────────────────────────────────────────────────────
LX, LY, LW, LH = 13.0, 2.65, 23.5, 3.5
p = FancyBboxPatch((LX-LW/2, LY-LH/2), LW, LH,
                    boxstyle='round,pad=0.1,rounding_size=0.2',
                    facecolor=CL, edgecolor='white', linewidth=2,
                    alpha=0.93, zorder=3)
ax.add_patch(p)

ax.text(LX, LY+LH/2-0.28, 'Physics-Informed Loss Function',
        ha='center', va='center', fontsize=11, color='white',
        fontweight='bold', zorder=5)

LCOLS = 4
loss_items = [
    ('①  Base: log1p-Huber(max_hat, y_max)',
     'Huber(δ=0.10) on log1p-space predictions vs labels\n'
     'Magnitude weight: w = (1 + y_max)²  [amplifies rare severe events]'),
    ('②  Auxiliary mean  [λ = 0.20]',
     'Huber(log1p(mean_hat), log1p(y_mean))\n'
     'Soft field-average constraint'),
    ('③  Ri / Shear Gating  [λ = 0.10]',
     'If Ri > 0.25 everywhere: flow is stable → penalise non-zero field\n'
     'Encodes Kelvin–Helmholtz theory directly in the loss'),
    ('④  TI1 Consistency  [λ = 0.10]',
     'Penalise negative rank-correlation between TI1 and field max\n'
     'High Ellrod index should co-locate with high predicted EDR'),
    ('⑤  Total Variation  [λ = 0.05]',
     'TV(field) = mean(|∂field/∂x| + |∂field/∂y|)\n'
     'Spatial coherence: penalises salt-and-pepper noise'),
    ('⑥  Climatological Cap  [λ = 0.05]',
     'ReLU(field − 1.05).mean()\n'
     'Softplus can exceed 1.0; penalise predictions above observed max EDR'),
    ('Optimiser:  AdamW',
     'lr=1e-3 · wd=1e-4 · batch=16\ngrad-clip norm=5 · patience=30 · ≤200 epochs'),
    ('Hyperparameter search:  Optuna (TPE)',
     '40 trials × 4-fold · 80-epoch budget\n'
     'Objective: log-RMSE + (1 − AUPRC@EDR≥0.20)'),
]

ncols = 4
nrows = 2
cell_w = LW / ncols
cell_h = (LH - 0.6) / nrows

for idx, (title, body) in enumerate(loss_items):
    col = idx % ncols
    row = idx // ncols
    cx = LX - LW/2 + cell_w*(col+0.5)
    cy = LY + LH/2 - 0.6 - cell_h*(row+0.5)
    ax.text(cx, cy+0.15, title, ha='center', va='top', fontsize=7.5,
            color='#ffecb3', fontweight='bold', zorder=5)
    ax.text(cx, cy-0.05, body, ha='center', va='top', fontsize=6.8,
            color='#b0bec5', zorder=5, multialignment='center', linespacing=1.3)

# arrows from aggregation down to loss box
arr(HX-0.6, R_FILM-0.85, HX-0.6, LY+LH/2+0.05, color=CA, lw=1.5)
arr(HX+0.6, R_FILM-0.85, HX+0.6, LY+LH/2+0.05, color=CA, lw=1.5)
ax.text(HX-1.6, (R_FILM-0.85+LY+LH/2)/2, 'max_hat', fontsize=7,
        color=CA, ha='right', va='center', fontweight='bold')
ax.text(HX+1.6, (R_FILM-0.85+LY+LH/2)/2, 'mean_hat', fontsize=7,
        color=CA, ha='left', va='center', fontweight='bold')

# ACARS ground truth arrow into loss
ax.text(LX+14.0, LY+0.3, 'ACARS in-situ EDR\n(y_max, y_mean)',
        ha='center', va='center', fontsize=8, color='#ef9a9a',
        fontweight='bold', zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color=CD, label='Physics Diagnostic Input  (36 ch)'),
    mpatches.Patch(color=CC, label='FiLM Conditioning  (cond_dim=8)'),
    mpatches.Patch(color=CS, label='Satellite Stream  (optional, masked)'),
    mpatches.Patch(color=CE, label='Encoder ConvBlock  + FiLM'),
    mpatches.Patch(color=CB, label='Bottleneck  (256→272 ch)'),
    mpatches.Patch(color=CSK, label='Skip Connection'),
    mpatches.Patch(color=CDK, label='Decoder ConvBlock  + Upsample'),
    mpatches.Patch(color=CH, label='Head  (Conv1×1 + Softplus)'),
    mpatches.Patch(color=CA, label='Soft Aggregation  (logsumexp / top-k)'),
    mpatches.Patch(color=CL, label='Physics-Informed Loss'),
]
leg = ax.legend(handles=patches, loc='upper left', fontsize=7.5,
                bbox_to_anchor=(0.01, 0.99), framealpha=0.9,
                ncol=2, title='Component', title_fontsize=8,
                edgecolor='#90a4ae')

plt.tight_layout(pad=0.3)
out = 'results/figures/model.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'Saved → {out}')
