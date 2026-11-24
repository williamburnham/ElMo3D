"""Shared styling for plot_distributions.py."""
import matplotlib
matplotlib.use('Agg')

# Keyed by cohort category (plot_distributions.CATEGORIES).
CATEGORY_COLOUR = {'Healthy Overground': '#4C6E8F',
                   'Healthy Treadmill': '#6FA3B8',
                   'Stroke Overground': '#C1553B'}

GRID = '#D9DFE4'
INK = '#2B3539'
MUTED = '#71808A'

SIDE_NAME = {'L': 'Left', 'R': 'Right'}


def tidy(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
