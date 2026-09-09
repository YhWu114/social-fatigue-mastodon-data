"""
Generate the two deliverables suggested by the AN-10 annotation:
  1) an10_core_contrast_table.csv  -- the pre-specified core contrasts with
     direction-retention counts and the range of differences (pp) across seeds.
  2) an10_fluctuation_plot.png     -- fluctuation (mean proportion across seeds
     with min-max range) for each of the 4 theme topics across the 9 language
     groups; plus a second panel showing per-topic mean Jaccard match quality.

Reads results/an10_topic_stability_stats.json (produced by an10_topic_stability.py).
No manuscript is modified.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STATS = '/home/user/social-fatigue/results/an10_topic_stability_stats.json'
OUT_TABLE = '/home/user/social-fatigue/results/an10_core_contrast_table.csv'
OUT_FIG = '/home/user/social-fatigue/results/an10_fluctuation_plot.png'

stats = json.load(open(STATS))
flu = stats['fluctuation_pp']        # key "lang|topic" -> {mean,std,min,max,...}
contrasts = stats['contrasts']
langs = ['en', 'zh-S', 'ja', 'de', 'zh-T', 'fr', 'es', 'pt', 'nl']

# ---- 1) contrast table ----
rows = []
for theme, rec in contrasts.items():
    rt = rec['ref_topic']
    for pr in rec['pairs']:
        rows.append({
            'theme': theme,
            'ref_topic': rt,
            'comparison': f"{pr['a']} vs {pr['b']}",
            'expected': pr['expected'],
            'seeds_retaining_direction': pr['seeds_retaining'],
            'seeds_total': pr['seeds_total'],
            'mean_diff_pp': round(pr['mean_diff_pp'], 2),
            'min_diff_pp': round(pr['min_diff_pp'], 2),
            'max_diff_pp': round(pr['max_diff_pp'], 2),
        })
ctab = pd.DataFrame(rows)
ctab.to_csv(OUT_TABLE, index=False)
print(f'[save] {OUT_TABLE}')
print(ctab.to_string(index=False))

# ---- 2) fluctuation plot ----
core = [(1, 'family'), (2, 'platform'), (0, 'daily-life'), (3, 'fascist')]
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

# Left: mean proportion with min-max whisker per language, for each theme topic
x = np.arange(len(langs))
width = 0.2
theme_color = {'family': '#c0392b', 'platform': '#2980b9',
               'daily-life': '#27ae60', 'fascist': '#8e44ad'}
for i, (rt, theme) in enumerate(core):
    means, mins, maxs = [], [], []
    for l in langs:
        v = flu.get(f'{l}|{rt}')
        if v:
            means.append(v['mean']); mins.append(v['min']); maxs.append(v['max'])
        else:
            means.append(np.nan); mins.append(np.nan); maxs.append(np.nan)
    means = np.array(means); mins = np.array(mins); maxs = np.array(maxs)
    axL.bar(x + (i - 1.5) * width, means, width, yerr=[means - mins, maxs - means],
            capsize=2, label=theme, color=theme_color[theme], alpha=0.85)
axL.set_xticks(x); axL.set_xticklabels(langs, rotation=30)
axL.set_ylabel('Topic proportion within language (%)')
axL.set_title('Core-theme topic proportions across 10 seeds\n(bars=mean, whiskers=min-max range)')
axL.legend()
axL.grid(axis='y', alpha=0.3)

# Right: per-seed mean Jaccard match quality (topic alignment stability)
mq = stats['match_quality_mean']
seeds = [str(s) for s in stats['seeds']]
vals = [mq[s] for s in seeds]
axR.bar(range(len(seeds)), vals, color='#34495e', alpha=0.85)
axR.axhline(0.5, color='red', ls='--', lw=1, label='0.50 reliability threshold')
axR.set_xticks(range(len(seeds))); axR.set_xticklabels(seeds, rotation=30)
axR.set_ylim(0, 1.05)
axR.set_ylabel('Mean Jaccard match quality vs reference')
axR.set_title('Topic alignment across seeds\n(Jaccard on post overlap; reference = seed 42)')
axR.legend()
axR.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=140)
print(f'[save] {OUT_FIG}')
