#!/usr/bin/env python3
"""
Figure 5 - Per-encounter BERTScore consistency distribution.

Single metric (the validated semantic headline): box plots of the ~100
per-encounter cell means per model across Q6->Q3 (1x3, compact, very thin
lines). Shows the degradation is more than a mean shift - Qwen's box drops and
widens toward Q3 (tail to ~0.48), Gemma stays high/tight, Phi-4 mid/flat. Rare
high outliers (a few near-identical encounters) are omitted for clarity; the
other metrics' central tendency lives in Fig 2.

Read-only on figures/data/per_encounter.csv. Model names only.

Run: venv_metrics/bin/python scripts/figures/fig_distribution.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import _style as S

S.apply()
pe = pd.read_csv(S.DATA / "per_encounter.csv")
SPINE = "#b9b9b9"
# Phi-4 teal in this plot (grey reads washed-out); line figures keep grey.
COLOR = {"Qwen-MediCare-BD": S.MODEL_COLOR["Qwen-MediCare-BD"],
         "Gemma-3-4B-it": S.MODEL_COLOR["Gemma-3-4B-it"],
         "Phi-4-mini-instruct": "#1B9E77"}


def cell_data(model):
    return [pe[(pe.model == model) & (pe.quant_level == q)]["bertscore_f1_mean"].dropna().values
            for q in S.QUANTS]


fig, axes = plt.subplots(1, 3, figsize=(S.COL, 2.55), sharey=True)
for ax, model in zip(axes, S.MODELS):
    c = COLOR[model]
    ax.boxplot(cell_data(model), positions=list(range(4)), widths=0.62,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=to_rgba(c, 0.32), edgecolor=c, linewidth=0.7),
               medianprops=dict(color="#222222", linewidth=0.9),
               whiskerprops=dict(color="#8a8a8a", linewidth=0.7),
               capprops=dict(color="#8a8a8a", linewidth=0.7))
    ax.set_title(S.MODEL_SHORT[model], fontsize=8, color="#333333", pad=4)
    ax.set_xticks(range(4)); ax.set_xticklabels(S.QUANT_TICK, fontsize=6.8)
    ax.set_ylim(0.45, 0.90); ax.set_yticks([0.5, 0.7, 0.9])
    ax.grid(axis="y", color="#d2d2d2", lw=0.6); ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(SPINE); ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=6.8, colors="#555555")
axes[0].set_ylabel("BERTScore F1  (per encounter)", fontsize=7.5)
fig.tight_layout(w_pad=1.0)
S.save(fig, "fig5_bertscore_distribution")
print("Figure 5 (BERTScore per-encounter box, 1x3) done.")
