"""
single_region.png — Month×Hour turbulence-proxy heatmap for ONE region
(E North America), used in the CNN Architecture section to motivate
time/location FiLM conditioning.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REGION_ID = 1            # E North America
REGION_NAME = "E North America"

df = pd.read_parquet('results/cache_regional_mh.parquet')
d = df[df.region == REGION_ID].copy()

# pivot to month (rows) × hour (cols)
grid = d.pivot(index='month', columns='hour', values='frac_m').reindex(
    index=range(1, 13), columns=range(0, 24))
nac = d.pivot(index='month', columns='hour', values='n_ac').reindex(
    index=range(1, 13), columns=range(0, 24))

MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
im = ax.imshow(grid.values, aspect='auto', cmap='inferno', origin='upper',
               extent=[-0.5, 23.5, 11.5, -0.5])

ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels(range(0, 24, 2), fontsize=9)
ax.set_yticks(range(12))
ax.set_yticklabels(MON, fontsize=9)
ax.set_xlabel('Hour (UTC)', fontsize=11)
ax.set_ylabel('Month', fontsize=11)
ax.set_title(f'Turbulence-encounter proxy — {REGION_NAME}\n'
             '(intra-hour std of vertical rate $>1.5$ m s$^{-1}$, FL180+)',
             fontsize=12, fontweight='bold')

# mark the peak cell
flat = grid.values
pi, pj = np.unravel_index(np.nanargmax(flat), flat.shape)
ax.scatter([pj], [pi], s=260, marker='*', facecolor='cyan',
           edgecolor='black', lw=1.2, zorder=5)
ax.annotate(f'peak  {MON[pi]} {pj:02d}UTC\nfrac={flat[pi,pj]:.2f}',
            (pj, pi), xytext=(pj + 1.5, pi + 1.6), fontsize=9, color='cyan',
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='cyan', lw=1.4))

# seasonal amplitude annotation
monthly = np.nanmean(flat, axis=1)
amp = np.nanmax(monthly) - np.nanmin(monthly)
hourly = np.nanmean(flat, axis=0)
damp = np.nanmax(hourly) - np.nanmin(hourly)
ax.text(0.015, 1.02,
        f'seasonal amplitude {amp:.2f}   ·   diurnal amplitude {damp:.2f}'
        f'   ·   {int(np.nansum(nac.values)):,} aircraft-hours',
        transform=ax.transAxes, fontsize=8.5, color='#333', va='bottom')

cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cb.set_label('turbulence-proxy fraction', fontsize=10)

plt.tight_layout()
out = 'report/figures/single_region.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved → {out}  | seasonal amp {amp:.3f}, diurnal amp {damp:.3f}')
