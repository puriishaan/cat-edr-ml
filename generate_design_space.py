"""
design_space.png — how the Optuna design-space search works.
Left: the search space (knobs). Middle: the TPE→k-fold→prune loop.
Right: the objective and the selected configuration.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

FW, FH = 18, 10
fig, ax = plt.subplots(figsize=(FW, FH), dpi=160)
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis('off')
fig.patch.set_facecolor('white'); ax.set_facecolor('white')

C_SPACE = '#2e5d8a'
C_ARCH  = '#3a7bd5'
C_FILM  = '#7d3c98'
C_SAT   = '#ca6f1e'
C_LOSS  = '#1a252f'
C_LOOP  = '#1e8449'
C_PRUNE = '#c0392b'
C_OBJ   = '#b7950b'
C_BEST  = '#117a65'

def box(cx, cy, w, h, color, title, body='', fs_t=9, fs_b=7.5,
        tc='white', alpha=0.95, zr=4):
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle='round,pad=0.06,rounding_size=0.12',
                       facecolor=color, edgecolor='white', lw=1.4,
                       alpha=alpha, zorder=zr)
    ax.add_patch(p)
    if body:
        ax.text(cx, cy+h/2-0.26, title, ha='center', va='top', fontsize=fs_t,
                color=tc, fontweight='bold', zorder=zr+1)
        ax.text(cx, cy+h/2-0.62, body, ha='center', va='top', fontsize=fs_b,
                color=tc, zorder=zr+1, linespacing=1.4, multialignment='center')
    else:
        ax.text(cx, cy, title, ha='center', va='center', fontsize=fs_t,
                color=tc, fontweight='bold', zorder=zr+1,
                linespacing=1.4, multialignment='center')

def arrow(x1, y1, x2, y2, color='#34495e', lw=2.0, rad=0.0, asz=16):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=asz, color=color, lw=lw,
                 connectionstyle=f'arc3,rad={rad}', zorder=3))

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(FW/2, FH-0.4, 'Design-Space Search (Optuna · TPE sampler · median pruner)',
        ha='center', va='center', fontsize=14, fontweight='bold', color='#0d1b2a')
ax.text(FW/2, FH-0.85, '40 trials  ×  4-fold event-grouped CV  ·  80-epoch budget per fold',
        ha='center', va='center', fontsize=9.5, color='#5d6d7e', style='italic')

# ── LEFT: search space (the knobs) ────────────────────────────────────────────
ax.text(3.0, 8.55, 'SEARCH SPACE', ha='center', fontsize=10, fontweight='bold',
        color=C_SPACE)

box(3.0, 7.5, 4.6, 1.05, C_ARCH, 'Architecture',
    'depth {2,3,4} · base width {16,32,48}\nkernel {3,5} · norm {batch,group}\n'
    'act {relu,gelu,silu} · pool {max,avg,stride} · dropout [0,0.3]',
    fs_t=8.5, fs_b=6.8)
box(3.0, 6.25, 4.6, 0.95, C_FILM, 'Conditioning & streams',
    'FiLM {on, off}\nsatellite stream {on, off}\nhead depth {1,2,3} · width {32,64,128}',
    fs_t=8.5, fs_b=6.8)
box(3.0, 5.05, 4.6, 0.85, '#0e7b67', 'Aggregation',
    'type {logsumexp, top-k}\n$\\tau \\in [2,16]$  ·  $k \\in \\{2,4,8\\}$',
    fs_t=8.5, fs_b=6.8)
box(3.0, 3.85, 4.6, 1.05, C_LOSS, 'Loss weights',
    '$\\lambda_{Ri},\\lambda_{TI},\\lambda_{TV},\\lambda_{cap}$ (each log-scale)\n'
    '$w_{mag}$ (severe up-weight) · $w_{aux}$\n'
    'optimizer {AdamW} · lr · weight-decay',
    fs_t=8.5, fs_b=6.8)

# bracket → sampler
arrow(5.4, 5.6, 7.0, 6.0, color=C_SPACE, lw=2.2, rad=-0.05)

# ── MIDDLE: the loop ──────────────────────────────────────────────────────────
box(8.6, 6.6, 3.0, 0.95, C_LOOP, 'TPE sampler',
    'proposes a config from the\nposterior over good regions',
    fs_t=9, fs_b=7)
box(8.6, 4.9, 3.0, 1.05, C_ARCH, 'Train candidate',
    'build CatCNN(config)\n4-fold event-grouped CV\n80 epochs, early-stop',
    fs_t=9, fs_b=7)
box(8.6, 3.1, 3.0, 0.85, C_PRUNE, 'Median pruner',
    'kill trials worse than the\nrunning median at each step',
    fs_t=9, fs_b=7)

arrow(8.6, 6.1, 8.6, 5.45, color='#34495e')
arrow(8.6, 4.35, 8.6, 3.55, color='#34495e')
# feedback loop pruner → sampler
arrow(10.15, 3.1, 10.15, 6.6, color=C_LOOP, lw=1.8, rad=-0.55)
ax.text(11.05, 4.9, 'update\nposterior\n(repeat ×40)', ha='center', va='center',
        fontsize=7.5, color=C_LOOP, style='italic', fontweight='bold')

# ── RIGHT: objective + best config ────────────────────────────────────────────
box(14.7, 6.6, 4.3, 1.15, C_OBJ,
    'Objective (minimise)',
    'log-RMSE  +  (1 $-$ AUPRC@0.20)\n'
    '─────────────────\n'
    'balances tail-accurate regression\nwith rare-event (severe) discrimination',
    fs_t=9, fs_b=7)
arrow(10.1, 4.9, 12.55, 6.4, color=C_PRUNE, lw=1.8, rad=0.12)

box(14.7, 4.0, 4.3, 2.3, C_BEST,
    'Selected configuration',
    'depth 3 · width 32 · kernel 3\nGroupNorm · GELU · max-pool · drop 0.1\n'
    'FiLM on · satellite searched\nhead depth 2 / width 64\n'
    'logsumexp  $\\tau=8$\n'
    '$\\lambda_{Ri}{=}0.1,\\ \\lambda_{TI}{=}0.05,\\ \\lambda_{TV}{=}0.01$\n'
    'AdamW  lr $10^{-3}$  wd $10^{-4}$\n$\\approx$80k parameters',
    fs_t=9.5, fs_b=7.2)
arrow(14.7, 6.0, 14.7, 5.2, color=C_OBJ, lw=2.0)

# footer note
ax.text(FW/2, 1.0,
        'Event-grouped folds prevent leakage (no event split across train/val).  '
        'The same search jointly tunes architecture, conditioning, aggregation, and every physics-loss weight — '
        'so each design choice is selected by held-out skill, not by hand.',
        ha='center', va='center', fontsize=8.5, color='#34495e', style='italic',
        wrap=True)

plt.tight_layout(pad=0.4)
out = 'report/figures/design_space.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved → {out}')
