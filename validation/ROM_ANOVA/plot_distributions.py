"""Distribution figure: hip (top row) and knee (bottom row) flexion ROM, Left
and Right columns.

Box-and-whisker per cohort with one jittered point per participant, coloured
by cohort category; faint dotted lines separate the categories. Panels in a
row share a y-axis so L and R compare directly.
Reads participant_rom.csv; writes figures/distribution.png.
"""
import os
import os.path as osp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _figstyle  # noqa: F401  (sets the Agg backend)
from _figstyle import CATEGORY_COLOUR, GRID, INK, MUTED, SIDE_NAME, tidy

HERE = osp.dirname(osp.abspath(__file__))
OUT = osp.join(HERE, 'figures')
JOINTS = ('hip', 'knee')
CATEGORIES = {
    'Healthy Overground': ['138H', 'G120', '10KH'],
    'Healthy Treadmill': ['9WRT', '24MP'],
    'Stroke Overground': ['50ST', '10KS'],
}
ORDER = [g for gs in CATEGORIES.values() for g in gs]
CAT_OF = {g: c for c, gs in CATEGORIES.items() for g in gs}
# x positions between categories: after 10KH, after 24MP
DIVIDERS = np.cumsum([len(gs) for gs in CATEGORIES.values()])[:-1] - 0.5


def draw_panel(ax, t, col, rng):
    for i, g in enumerate(ORDER):
        v = t.loc[t.group == g, col].dropna().values
        ax.boxplot(v, positions=[i], widths=0.52, showfliers=False,
                   medianprops=dict(color=INK, lw=1.4),
                   boxprops=dict(color=MUTED, lw=0.9),
                   whiskerprops=dict(color=MUTED, lw=0.9),
                   capprops=dict(color=MUTED, lw=0.9))
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=11, alpha=0.65,
                   linewidths=0, color=CATEGORY_COLOUR[CAT_OF[g]], zorder=3)
    for x in DIVIDERS:
        ax.axvline(x, color=MUTED, lw=0.8, ls=':', alpha=0.6, zorder=0)
    tidy(ax)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, rotation=30, ha='right', fontsize=8)


def main():
    t = pd.read_csv(osp.join(HERE, 'participant_rom.csv'))
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.6), sharey='row')
    for row, joint in zip(axes, JOINTS):
        cols = [f'{s}_{joint}_flexion_ROM' for s in ('L', 'R')]
        for ax, c in zip(row, cols):
            draw_panel(ax, t, c, rng)
            ax.set_title(f'{SIDE_NAME[c[0]]} {joint}', fontsize=9.5, color=INK)
        # n labels once both panels are drawn, so they sit on the final ylim.
        y0 = row[0].get_ylim()[0]
        for ax, c in zip(row, cols):
            for i, g in enumerate(ORDER):
                ax.text(i, y0, f'n={t.loc[t.group == g, c].notna().sum()}',
                        ha='center', va='bottom', fontsize=6.5, color=MUTED)
        row[0].set_ylabel(f'{joint} flexion ROM (°)\none point = one participant',
                          fontsize=9, color=INK)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=k, ms=6)
               for k, c in CATEGORY_COLOUR.items()]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles),
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.07, top=0.93,
                        wspace=0.06, hspace=0.32)
    out = osp.join(OUT, 'distribution.png')
    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    print('wrote', osp.relpath(out, HERE))


if __name__ == '__main__':
    main()
