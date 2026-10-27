#!/usr/bin/env python3
"""
Figure 2 v2 (attractive variant) - dose-response with data-point markers + slightly
thicker lines, for a more finished "connected-scatter" look.

Identical data, palette, scales, panels, and legend to the v1 raw figure
(fig_dose_response.py); v1 is left untouched. The only changes are: uniform circle
markers at each point (white-edged, same shape for all three models so there are no
mixed o/s/^ shapes), line width 1.4 -> 1.7, and a touch more x-padding for the end
markers. Read-only on figures/data/dose_response.csv. Output: fig2_dose_response_v2.

Run: venv_metrics/bin/python scripts/figures/fig_dose_response_v2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import _style as S

S.apply()
d = pd.read_csv(S.DATA / "dose_response.csv")

PANELS = [("bertscore_f1_mean",            "BERTScore",         "↑", (0.50, 0.81),  [0.50, 0.60, 0.70, 0.80]),
          ("rougeL_f1_mean",               "ROUGE-L",           "↑", (0.10, 0.56),  [0.10, 0.25, 0.40, 0.55]),
          ("medcon_f1_mean",               "Concept-set F1",    "↑", (0.27, 0.77),  [0.30, 0.45, 0.60, 0.75]),
          ("nli_contradiction_mean_mean",  "NLI contradiction", "↓", (0.07, 0.255), [0.10, 0.15, 0.20, 0.25])]
X = [0, 1, 2, 3]
SPINE = "#b9b9b9"
LW, MS = 1.7, 3.4

fig, axes = plt.subplots(2, 2, figsize=(S.COL, 2.85))
for ax, (col, name, arrow, ylim, yticks) in zip(axes.flat, PANELS):
    for model in S.MODELS:
        sub = d[(d.metric == col) & (d.model == model)].sort_values("quant_idx")
        ax.plot(X, sub["cond_mean"], color=S.MODEL_COLOR[model], ls=S.MODEL_LS[model], lw=LW,
                marker="o", markersize=MS, markerfacecolor=S.MODEL_COLOR[model],
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ax.set_title(f"{name} {arrow}", fontsize=8, color="#333333", pad=4)
    ax.set_xticks(X)
    ax.set_xlim(-0.25, 3.25)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, color="#d2d2d2", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=6.8, colors="#555555")

for ax in axes[0]:
    ax.set_xticklabels([])
for ax in axes[1]:
    ax.set_xticklabels(S.QUANT_TICK)

handles = [Line2D([0], [0], color=S.MODEL_COLOR[m], ls=S.MODEL_LS[m], lw=1.9,
                  marker="o", markersize=MS, markerfacecolor=S.MODEL_COLOR[m],
                  markeredgecolor="white", markeredgewidth=0.6, label=S.MODEL_SHORT[m])
           for m in S.MODELS]
fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995),
           columnspacing=2.0, handlelength=1.8, handletextpad=0.5, fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.945), h_pad=1.3, w_pad=1.8)
S.save(fig, "fig2_dose_response_v2")
print("Figure 2 v2 (markers + thicker lines) done.")
