"""
Generate model.png — 3D perspective CatCNNTorch architecture diagram.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
import numpy as np
import colorsys

# ── Canvas ────────────────────────────────────────────────────────────────────
FW, FH = 30, 14
fig, ax = plt.subplots(figsize=(FW, FH), dpi=160)
ax.set_xlim(-0.5, FW + 0.5)
ax.set_ylim(-4.5, FH)
ax.axis('off')
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')

# ── Colours ──────────────────────────────────────────────────────────────────
BLUE   = '#3a7bd5'   # encoder conv
BLUE2  = '#2563b0'   # second conv in same block
RED    = '#e04e3a'   # max pool
GREEN  = '#1e8449'   # decoder conv
GREEN2 = '#166a3a'
PURPLE = '#7d3c98'   # FiLM
ORANGE = '#ca6f1e'   # satellite
TEAL   = '#0e7b67'   # bottleneck
GOLD   = '#b7950b'   # head
GRAY_I = '#566573'   # input
SK_C   = '#27ae60'   # skip connection

def _rgb(h):
    h = h.lstrip('#')
    return (int(h[:2], 16)/255, int(h[2:4], 16)/255, int(h[4:], 16)/255)

def lc(c, a=0.28):
    r,g,b = _rgb(c) if isinstance(c,str) else c
    h,s,v = colorsys.rgb_to_hsv(r,g,b)
    return colorsys.hsv_to_rgb(h, max(0,s-0.15), min(1,v+a))

def dc(c, a=0.22):
    r,g,b = _rgb(c) if isinstance(c,str) else c
    h,s,v = colorsys.rgb_to_hsv(r,g,b)
    return colorsys.hsv_to_rgb(h, min(1,s+0.05), max(0,v-a))

PX, PY = 0.42, 0.26   # perspective offset per unit depth

# ── Core drawing primitive ────────────────────────────────────────────────────
def blk(x, y0, w, h, d, color, alpha=1.0, lw=0.8, zr=3, ec='white'):
    dx, dy = PX*d, PY*d
    for pts, fc in [
        ([[x,y0],[x+w,y0],[x+w,y0+h],[x,y0+h]], color),
        ([[x,y0+h],[x+w,y0+h],[x+w+dx,y0+h+dy],[x+dx,y0+h+dy]], lc(color)),
        ([[x+w,y0],[x+w+dx,y0+dy],[x+w+dx,y0+h+dy],[x+w,y0+h]], dc(color)),
    ]:
        ax.add_patch(Polygon(np.array(pts, dtype=float), closed=True,
                             fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=zr))
    return dict(x=x, y0=y0, w=w, h=h, d=d, dx=dx, dy=dy,
                xR=x+w+dx, xFR=x+w, xFL=x,
                yMid=y0+h/2, yTop=y0+h+dy, yTopF=y0+h,
                yMidR=y0+h/2+dy/2)

def arr(x1,y1,x2,y2, color='#2c3e50', lw=1.5, asz=10):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=asz), zorder=12)

def arcc(x1,y1,x2,y2, color, lw=1.5, rad=0.3, asz=11):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=asz,
                                connectionstyle=f'arc3,rad={rad}'), zorder=12)

def txt(x,y,s,fs=7,c='#1c2833',ha='center',va='center',
        bold=False,italic=False,zr=14):
    ax.text(x,y,s,ha=ha,va=va,fontsize=fs,color=c,zorder=zr,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal',
            multialignment='center',linespacing=1.35)

# ── Spatial geometry ──────────────────────────────────────────────────────────
YC = 4.8   # vertical centre
# block heights keyed by spatial dim
H  = {24: 6.0, 12: 4.5, 6: 3.0, 3: 1.8}
# visual depth keyed by channel count
D  = {36:1.5, 32:1.35, 64:1.85, 128:2.6, 256:3.5, 272:3.6, 1:0.5}

def y0(sp): return YC - H[sp]/2

# ── X-layout  ─────────────────────────────────────────────────────────────────
# Each ConvBlock = 2 thin slabs (w=0.25 each) + FiLM (w=0.16)
# MaxPool = thin red slab (w=0.20)
# Gaps tuned for ~28 total units

SLAB = 0.25   # single conv layer slab width
FILM = 0.17
POOL = 0.20
BTL  = 0.28
GAP  = 0.14   # gap between slabs in same block
STG  = 0.40   # gap between stages

x = 0.5
positions = {}

def _adv(name, w, d):
    positions[name] = x

def advance(name, w, d, gap_after=0.0):
    global x
    positions[name] = x
    x += w + PX*d + gap_after

# INPUT
advance('inp', 0.40, D[36], 0.30)

# ENC-1  (24×24, 36→32)
advance('e1a', SLAB, D[32], GAP)   # conv slab 1
advance('e1b', SLAB, D[32], GAP)   # conv slab 2
advance('e1f', FILM, D[32], GAP)   # FiLM
advance('e1p', POOL, D[32], STG)   # MaxPool

# ENC-2  (12×12, 32→64)
advance('e2a', SLAB, D[64], GAP)
advance('e2b', SLAB, D[64], GAP)
advance('e2f', FILM, D[64], GAP)
advance('e2p', POOL, D[64], STG)

# ENC-3  (6×6, 64→128)
advance('e3a', SLAB, D[128], GAP)
advance('e3b', SLAB, D[128], GAP)
advance('e3f', FILM, D[128], GAP)
advance('e3p', POOL, D[128], STG)

# BOTTLENECK (3×3, 128→256) + satellite fusion slab
advance('btla', BTL, D[256], GAP)
advance('btlb', BTL, D[256], GAP)
advance('btlf', FILM, D[256], GAP)
advance('bsat', FILM, D[272], STG)   # satellite fusion marker

# DEC-3  (6×6, 272→128)
advance('d3a', SLAB, D[128], GAP)
advance('d3b', SLAB, D[128], STG)

# DEC-2  (12×12, 128→64)
advance('d2a', SLAB, D[64], GAP)
advance('d2b', SLAB, D[64], STG)

# DEC-1  (24×24, 64→32)
advance('d1a', SLAB, D[32], GAP)
advance('d1b', SLAB, D[32], 0.35)

# HEAD (24×24, 32→1)
advance('hd', 0.30, D[1], 0.55)

AGG_X = x + 0.6

print(f"Total width: {x:.1f}")

# ── Draw all blocks ───────────────────────────────────────────────────────────
B = {}   # name → block info dict

def enc_blk(n, sp, ch, color):
    B[n] = blk(positions[n], y0(sp), SLAB, H[sp], D[ch], color)

def pool_blk(n, sp, ch):
    B[n] = blk(positions[n], y0(sp), POOL, H[sp], D[ch], RED)

def film_blk(n, sp, ch):
    B[n] = blk(positions[n], y0(sp), FILM, H[sp], D[ch], PURPLE, alpha=0.9)

# Input
B['inp'] = blk(positions['inp'], y0(24), 0.40, H[24], D[36], GRAY_I)

# Encoder 1
enc_blk('e1a', 24, 32, BLUE)
enc_blk('e1b', 24, 32, BLUE2)
film_blk('e1f', 24, 32)
pool_blk('e1p', 24, 32)

# Encoder 2
enc_blk('e2a', 12, 64, BLUE)
enc_blk('e2b', 12, 64, BLUE2)
film_blk('e2f', 12, 64)
pool_blk('e2p', 12, 64)

# Encoder 3
enc_blk('e3a', 6, 128, BLUE)
enc_blk('e3b', 6, 128, BLUE2)
film_blk('e3f', 6, 128)
pool_blk('e3p', 6, 128)

# Bottleneck
B['btla'] = blk(positions['btla'], y0(3), BTL, H[3], D[256], TEAL)
B['btlb'] = blk(positions['btlb'], y0(3), BTL, H[3], D[256], dc(TEAL, 0.08))
B['btlf'] = blk(positions['btlf'], y0(3), FILM, H[3], D[256], PURPLE, alpha=0.9)
B['bsat'] = blk(positions['bsat'], y0(3), FILM, H[3], D[272], ORANGE)

# Decoder 3
enc_blk('d3a', 6, 128, GREEN)
enc_blk('d3b', 6, 128, GREEN2)

# Decoder 2
enc_blk('d2a', 12, 64, GREEN)
enc_blk('d2b', 12, 64, GREEN2)

# Decoder 1
enc_blk('d1a', 24, 32, GREEN)
enc_blk('d1b', 24, 32, GREEN2)

# Head
B['hd'] = blk(positions['hd'], y0(24), 0.30, H[24], D[1], GOLD)

# ── Stage-group labels (below blocks) ─────────────────────────────────────────
def grp_label(keys, text, fs=7, c='#1c2833'):
    xs = [positions[k] for k in keys]
    xe = max(positions[k] + (SLAB if k not in ('inp','btla','btlb','bsat','hd') else 0.4)
             + PX*D.get(36 if k=='inp' else 32, 1.5) for k in keys)
    mid = (xs[0] + xe) / 2
    txt(mid, y0(24)-0.32, text, fs=fs, c=c)

def dim_label(key, sp, ch, below=True):
    b = B[key]
    y = b['y0'] - 0.22 if below else b['yTop'] + 0.12
    txt(b['x'] + b['w']/2, y, f'{ch}×{sp}×{sp}', fs=6, c='#1c2833', italic=True)

# Labels under first slab of each group
txt(positions['inp'] + 0.2, y0(24)-0.5,
    '36 ch\n24×24\nPhysics\nDiag.', fs=6.5, c=GRAY_I)

for (ka, kb, kf, kp, sp, chin, chout) in [
    ('e1a','e1b','e1f','e1p', 24, 36, 32),
    ('e2a','e2b','e2f','e2p', 12, 32, 64),
    ('e3a','e3b','e3f','e3p',  6, 64, 128),
]:
    cx = (positions[ka] + positions[kp] + POOL + PX*D[chout]) / 2
    txt(cx, y0(sp)-0.42, f'Conv+BN+GELU×2\n{chin}→{chout}ch  ·  {sp}×{sp}', fs=6.5)
    txt(positions[kf]+FILM/2, y0(sp)-0.42, 'FiLM\n(cond8)', fs=5.8, c=PURPLE)
    txt(positions[kp]+POOL/2, y0(sp)-0.42, 'MaxPool\n÷2', fs=5.8, c='#922b21')

txt((positions['btla']+positions['btlf']+FILM+PX*D[256])/2, y0(3)-0.42,
    'Bottleneck  128→256ch  ·  3×3\n+ FiLM', fs=6.5, c=TEAL)
txt(positions['bsat']+FILM/2+PX*D[272]/2, y0(3)-0.42,
    'Sat\nFusion\n→272ch', fs=5.8, c=ORANGE)

for (ka, kb, sp, ch) in [('d3a','d3b',6,128), ('d2a','d2b',12,64), ('d1a','d1b',24,32)]:
    cx = (positions[ka] + positions[kb] + SLAB + PX*D[ch]) / 2
    txt(cx, y0(sp)-0.42, f'↑2× biliear · skip cat\n{ch*2}→{ch}ch  ·  {sp}×{sp}', fs=6.5, c=GREEN)

txt(positions['hd']+0.15, y0(24)-0.52, 'Conv1×1\n1ch\nSoftplus\nEDR field', fs=6.5, c=GOLD)

# ── Sequential flow arrows ─────────────────────────────────────────────────────
def flow(n1, n2, yf=None):
    b1, b2 = B[n1], B[n2]
    y1 = yf if yf else b1['yMid']
    y2 = yf if yf else b2['yMid']
    arr(b1['xFR'], y1, b2['xFL'], y2)

def pool_transition(pname, nxt, sp_from, sp_to, ch):
    """Arrow from pool block to next encoder stage (height changes)."""
    b1, b2 = B[pname], B[nxt]
    arr(b1['xFR'], b1['yMid'], b2['xFL'], b2['yMid'], color='#922b21')

# Within-stage flows
for pairs in [
    [('inp','e1a'), ('e1a','e1b'), ('e1b','e1f'), ('e1f','e1p')],
    [('e2a','e2b'), ('e2b','e2f'), ('e2f','e2p')],
    [('e3a','e3b'), ('e3b','e3f'), ('e3f','e3p')],
    [('btla','btlb'), ('btlb','btlf'), ('btlf','bsat')],
    [('d3a','d3b')],
    [('d2a','d2b')],
    [('d1a','d1b'), ('d1b','hd')],
]:
    for n1, n2 in pairs:
        flow(n1, n2)

# Pool transitions (changing height)
pool_transition('e1p', 'e2a', 24, 12, 32)
pool_transition('e2p', 'e3a', 12,  6, 64)
pool_transition('e3p', 'btla', 6,  3, 128)

# Bottleneck sat → decoder
arr(B['bsat']['xFR'], B['bsat']['yMid'], B['d3a']['xFL'], B['d3a']['yMid'],
    color=ORANGE, lw=1.5)

# Decoder stage transitions (height increases)
arr(B['d3b']['xFR'], B['d3b']['yMid'], B['d2a']['xFL'], B['d2a']['yMid'], color=GREEN)
arr(B['d2b']['xFR'], B['d2b']['yMid'], B['d1a']['xFL'], B['d1a']['yMid'], color=GREEN)

# Head → aggregation
arr(B['hd']['xFR'], B['hd']['yMid']+0.6, AGG_X-0.05, YC+1.1)
arr(B['hd']['xFR'], B['hd']['yMid']-0.6, AGG_X-0.05, YC-0.8)

# ── Skip connections (arcs over the top) ──────────────────────────────────────
# Enc-1 → Dec-1  (widest, highest arc)
arcc(B['e1b']['xFR']+B['e1b']['dx']*0.5, B['e1b']['yTop'],
     B['d1a']['xFL']+B['d1a']['dx']*0.3, B['d1a']['yTop'],
     color=SK_C, lw=1.8, rad=-0.22)
txt((B['e1b']['xFR']+B['d1a']['xFL'])/2 + 2.0, 12.4, 'skip  32 ch', fs=6.5, c=SK_C, italic=True)

# Enc-2 → Dec-2
arcc(B['e2b']['xFR']+B['e2b']['dx']*0.5, B['e2b']['yTop'],
     B['d2a']['xFL']+B['d2a']['dx']*0.3, B['d2a']['yTop'],
     color=SK_C, lw=1.8, rad=-0.20)
txt((B['e2b']['xFR']+B['d2a']['xFL'])/2 + 1.0, 11.3, 'skip  64 ch', fs=6.5, c=SK_C, italic=True)

# Enc-3 → Dec-3
arcc(B['e3b']['xFR']+B['e3b']['dx']*0.5, B['e3b']['yTop'],
     B['d3a']['xFL']+B['d3a']['dx']*0.3, B['d3a']['yTop'],
     color=SK_C, lw=1.8, rad=-0.18)
txt((B['e3b']['xFR']+B['d3a']['xFL'])/2, 10.2, 'skip  128 ch', fs=6.5, c=SK_C, italic=True)

# ── FiLM conditioning strip ───────────────────────────────────────────────────
COND_Y = 12.95
ax.text(0.3, COND_Y,
        'FiLM conditioning  (cond_dim = 8):   '
        'Climate (ONI, Niño3.4, PDO, QBO)  +  '
        'Cyclic time (sin/cos DOY, sin/cos hour-UTC)',
        ha='left', va='center', fontsize=7.5, color='white', zorder=14,
        bbox=dict(boxstyle='round,pad=0.35', facecolor=PURPLE, ec='#5b2c6f', lw=1.2))

# arrows from cond strip down to each FiLM slab
for fname, sp in [('e1f',24), ('e2f',12), ('e3f',6), ('btlf',3)]:
    bx = positions[fname] + FILM/2
    arr(bx, COND_Y-0.22, bx, B[fname]['yTopF']+0.06, color=PURPLE, lw=1.1, asz=8)

# ── Satellite input (below, into fusion slab) ─────────────────────────────────
sat_cx = positions['bsat'] + FILM/2 + PX*D[272]*0.3
sat_iy = -3.4
ax.text(sat_cx, sat_iy,
        'Satellite MLP\n'
        '[tb_cold, tb_mean, tb_std, tb_max, Δtb]\n'
        '+ availability mask  (0 = absent, 1 = present)\n'
        'Lin(6→32, GELU) → Lin(32→16)\n'
        'broadcast → (16, H, W) → cat\n'
        'Covers 64/150 events (2017+)',
        ha='center', va='center', fontsize=7, color='white', zorder=14,
        bbox=dict(boxstyle='round,pad=0.4', facecolor=ORANGE, ec='#873600', lw=1.3))
arr(sat_cx, sat_iy + 0.98, sat_cx, B['bsat']['y0'] - 0.1, color=ORANGE, lw=1.6)
txt(sat_cx+1.5, sat_iy+0.5, 'ablatable\n(zero when absent)', fs=6.5, c=ORANGE, italic=True)

# ── Aggregation boxes ─────────────────────────────────────────────────────────
ax.text(AGG_X+0.25, YC+1.1,
        'LogSumExp  (τ=8.0)\n'
        'or Top-k mean  (k=4)\n'
        '→ max_hat  (B,)',
        ha='center', va='center', fontsize=7.5, color='#1c2833',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#fdebd0', ec='#ca6f1e', lw=1.5),
        zorder=14)
ax.text(AGG_X+0.25, YC-0.8,
        'field.mean()\n→ mean_hat  (B,)',
        ha='center', va='center', fontsize=7.5, color='#1c2833',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#d6eaf8', ec='#1a6fa0', lw=1.5),
        zorder=14)

# ── Physics Loss panel ────────────────────────────────────────────────────────
loss_items = [
    '① Base:  log1p-Huber(max_hat, y_max)\n   weight = (1 + y_max)²   [amplify severe tail]',
    '② λ=0.20 · Huber(mean_hat, y_mean)\n   soft field-average constraint',
    '③ λ=0.10 · Ri gating\n   Ri > 0.25 → penalise non-zero field\n   [KH instability theory]',
    '④ λ=0.10 · TI1 consistency\n   penalise −corr(TI1, field max)\n   [Ellrod index]',
    '⑤ λ=0.05 · Total Variation\n   TV(field) = |∂x| + |∂y|\n   spatial coherence',
    '⑥ λ=0.05 · Cap penalty\n   ReLU(field − 1.05)\n   bounded by obs max EDR',
]
LX, LY0 = 0.3, -4.2
LW_cell = (FW - 1.0) / len(loss_items)
for i, txt_s in enumerate(loss_items):
    cx = LX + (i + 0.5) * LW_cell
    ax.text(cx, LY0, txt_s,
            ha='center', va='center', fontsize=6.8, color='white',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#1a252f',
                      ec='#566573', lw=0.9),
            zorder=14, multialignment='center', linespacing=1.3)
ax.text(FW/2, LY0 + 1.25,
        'Physics-Informed Loss Function',
        ha='center', va='center', fontsize=9.5, color='#1a252f',
        fontweight='bold')

# Arrows from aggregation down to loss
arr(AGG_X+0.25, YC-1.4, AGG_X+0.25, LY0+1.0, color='#566573', lw=1.2)
txt(AGG_X+1.2, (YC-1.4+LY0+1.0)/2, '← ACARS\ny_max, y_mean', fs=6.5, c='#e74c3c')

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(FW/2, FH - 0.35,
        'CatCNNTorch — Physics-Informed U-Net Architecture',
        ha='center', va='center', fontsize=15, fontweight='bold', color='#0d1b2a', zorder=15)
ax.text(FW/2, FH - 0.78,
        '36 input channels  (12 diagnostics × 3 pressure levels: 225 / 250 / 300 hPa)   ·   '
        'depth = 3   ·   base_filters = 32   ·   ≈80k parameters   ·   '
        'FiLM conditioning   ·   ablatable satellite stream',
        ha='center', va='center', fontsize=8, color='#5d6d7e',
        fontstyle='italic', zorder=15)

# ── Legend ────────────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(fc=GRAY_I, ec='w', label='Physics Diag. Input (36 ch)'),
    mpatches.Patch(fc=BLUE,   ec='w', label='Encoder Conv Layer (3×3, BN, GELU, Dropout2d)'),
    mpatches.Patch(fc=RED,    ec='w', label='MaxPool2d (÷2)'),
    mpatches.Patch(fc=PURPLE, ec='w', label='FiLM  (γ,β ← cond_dim=8; identity-init)'),
    mpatches.Patch(fc=TEAL,   ec='w', label='Bottleneck (128→256 ch, 3×3)'),
    mpatches.Patch(fc=ORANGE, ec='w', label='Satellite Fusion (ablatable)'),
    mpatches.Patch(fc=GREEN,  ec='w', label='Decoder Conv (↑2× bilinear + skip concat)'),
    mpatches.Patch(fc=GOLD,   ec='w', label='Head: Conv1×1 + Softplus → EDR field'),
    mpatches.Patch(fc=SK_C,   ec='w', label='U-Net Skip Connection (cat)'),
]
leg = ax.legend(handles=patches, loc='upper left',
                bbox_to_anchor=(0.0, 0.93), fontsize=7.5,
                framealpha=0.97, ncol=3, edgecolor='#bdc3c7',
                title='Component legend', title_fontsize=8.5)

# ── Component bounding boxes (numbered, for elaboration in the paper) ─────────
from matplotlib.patches import FancyBboxPatch, Circle

def zone(x0, x1, y0_, y1_, num, color, lw=2.2):
    r = FancyBboxPatch((x0, y0_), x1 - x0, y1_ - y0_,
                       boxstyle='round,pad=0.04,rounding_size=0.18',
                       fill=False, edgecolor=color, lw=lw,
                       linestyle=(0, (6, 3)), zorder=11, alpha=0.95)
    ax.add_patch(r)
    # numbered badge at top-left
    bx, by = x0 + 0.32, y1_ - 0.05
    ax.add_patch(Circle((bx, by), 0.30, facecolor=color, edgecolor='white',
                        lw=1.4, zorder=16))
    ax.text(bx, by, str(num), ha='center', va='center', fontsize=11,
            color='white', fontweight='bold', zorder=17)

Zc = '#1a252f'
# block-body vertical extents
yb_lo = y0(24) - 0.75
yb_24 = B['e1a']['yTop'] + 0.15
yb_btl = B['btla']['yTop'] + 0.15

# ① Physics-diagnostic inputs
zone(B['inp']['x'] - 0.18, B['inp']['xR'] + 0.18, yb_lo, yb_24, 1, GRAY_I)
# ③ Encoder
zone(B['e1a']['x'] - 0.18, B['e3p']['xR'] + 0.22, yb_lo, yb_24, 3, BLUE)
# ④ Bottleneck + satellite fusion
zone(B['btla']['x'] - 0.18, B['bsat']['xR'] + 0.22, yb_lo, yb_btl + 0.3, 4, TEAL)
# ⑤ Decoder
zone(B['d3a']['x'] - 0.18, B['d1b']['xR'] + 0.22, yb_lo, yb_24, 5, GREEN)
# ⑥ Head + soft aggregation
zone(B['hd']['x'] - 0.20, AGG_X + 2.0, yb_lo, yb_24, 6, GOLD)

# ② FiLM badge on the conditioning strip
ax.add_patch(Circle((0.05, COND_Y), 0.30, facecolor=PURPLE, edgecolor='white',
                    lw=1.4, zorder=16, clip_on=False))
ax.text(0.05, COND_Y, '2', ha='center', va='center', fontsize=11,
        color='white', fontweight='bold', zorder=17)
# ⑦ Loss badge
ax.add_patch(Circle((0.05, LY0), 0.30, facecolor='#1a252f', edgecolor='white',
                    lw=1.4, zorder=16, clip_on=False))
ax.text(0.05, LY0, '7', ha='center', va='center', fontsize=11,
        color='white', fontweight='bold', zorder=17)

plt.tight_layout(pad=0.3)
out = 'results/figures/model.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved → {out}')
