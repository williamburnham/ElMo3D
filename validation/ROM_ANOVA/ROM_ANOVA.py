"""Cohort agreement on hip and knee flexion ROM (participant means, no exclusion).

Healthy overground gait (138H, G120, 10KH): Welch's ANOVA, L and R separately.
Stroke overground gait (10KS vs 50ST): Welch's t-test on bilateral ROM (mean of
L and R), since the paretic side is not documented for 10KS.

Reads participant_rom.csv (from extract_ROM.py); writes ROM_ANOVA.csv.
Usage:  python ROM_ANOVA.py
"""
import os.path as osp

import pandas as pd
from scipy import stats
from statsmodels.stats.oneway import anova_oneway

HERE = osp.dirname(osp.abspath(__file__))
HEALTHY = ['138H', 'G120', '10KH']
STROKE = ['10KS', '50ST']
JOINTS = ['hip', 'knee']


def describe(groups):
    return ' vs '.join(f'{g} {v.mean():.1f} ± {v.std(ddof=1):.1f} (n={len(v)})'
                       for g, v in groups.items())


def main():
    t = pd.read_csv(osp.join(HERE, 'participant_rom.csv'))
    rows = []

    for j in JOINTS:
        for side in 'LR':
            f = f'{side}_{j}_flexion_ROM'
            d = t.loc[t.group.isin(HEALTHY), ['group', f]].dropna()
            r = anova_oneway(d[f], d.group, use_var='unequal')
            rows.append(dict(test="Welch's ANOVA", cohorts='/'.join(HEALTHY),
                             feature=f, descriptives=describe(
                                 {g: d.loc[d.group == g, f] for g in HEALTHY}),
                             statistic=r.statistic, df1=r.df_num,
                             df2=r.df_denom, p=r.pvalue))

    for j in JOINTS:
        s = t[t.group.isin(STROKE)]
        bi = s[[f'L_{j}_flexion_ROM', f'R_{j}_flexion_ROM']].mean(axis=1, skipna=False)
        g = {c: bi[s.group == c].dropna() for c in STROKE}
        r = stats.ttest_ind(*g.values(), equal_var=False)
        ci = r.confidence_interval()
        rows.append(dict(test="Welch's t-test", cohorts=' - '.join(STROKE),
                         feature=f'bilateral_{j}_flexion_ROM',
                         descriptives=describe(g),
                         mean_diff=g[STROKE[0]].mean() - g[STROKE[1]].mean(),
                         ci_lo=ci.low, ci_hi=ci.high,
                         statistic=r.statistic, df2=r.df, p=r.pvalue))

    res = pd.DataFrame(rows)
    res.to_csv(osp.join(HERE, 'ROM_ANOVA.csv'), index=False)
    with pd.option_context('display.width', 250, 'display.max_colwidth', 90,
                           'display.float_format', '{:.3g}'.format):
        print(res.to_string(index=False))


if __name__ == '__main__':
    main()
