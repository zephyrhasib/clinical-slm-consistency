#!/usr/bin/env python3
"""
Figure 4 v2 (attractive variant) - numeric suppression with data-point markers +
slightly thicker lines, matching the Figure 2 v2 look.

Identical data, palette, scales, panels, and legend to v1 (fig_numeric_suppression.py),
which is left untouched. Only changes: uniform white-edged circle markers at each
point (same shape for all models), line width 1.4 -> 1.7. Read-only on
figures/data/numeric_suppression.csv. Output: fig4_numeric_suppression_v2.

Run: venv_metrics/bin/python scripts/figures/fig_numeric_suppression_v2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import _style as S

S.apply()
ns = pd.read_csv(S.DATA / "numeric_suppression.csv")
X = [0, 1, 2, 3]
SPINE = "#b9b9b9"
LW, MS = 1.7, 3.4


def qsub(m):
    return ns[ns.model == m].sort_values("quant_idx")


def style_ax(ax):
    ax.grid(True, color="#d2d2d2", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=6.6, colors="#555555")


def plot_metric(ax, colname):
    for m in S.MODELS:
        s = qsub(m)
        ax.plot(X, s[colname], color=S.MODEL_COLOR[m], ls=S.MODEL_LS[m], lw=LW,
                marker="o", markersize=MS, markerfacecolor=S.MODEL_COLOR[m],
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)


fig, axes = plt.subplots(1, 2, figsize=(S.COL, 1.85))

ax = axes[0]
plot_metric(ax, "numeric_jaccard_mean")
ax.set_title("Numeric Jaccard", fontsize=7.6, color="#333333", pad=3)
ax.set_ylim(0.13, 0.82); ax.set_yticks([0.20, 0.40, 0.60, 0.80])
ax.set_xticks(X); ax.set_xlim(-0.25, 3.25); ax.set_xticklabels(S.QUANT_TICK)
style_ax(ax)

ax = axes[1]
plot_metric(ax, "mean_fact_count")
ax.set_title("Mean facts / note", fontsize=7.6, color="#333333", pad=3)
ax.set_ylim(0.80, 2.20); ax.set_yticks([1.00, 1.40, 1.80, 2.20])
ax.set_xticks(X); ax.set_xlim(-0.25, 3.25); ax.set_xticklabels(S.QUANT_TICK)
style_ax(ax)

handles = [Line2D([0], [0], color=S.MODEL_COLOR[m], ls=S.MODEL_LS[m], lw=1.9,
                  marker="o", markersize=MS, markerfacecolor=S.MODEL_COLOR[m],
                  markeredgecolor="white", markeredgewidth=0.6, label=S.MODEL_SHORT[m])
           for m in S.MODELS]
fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
           columnspacing=2.0, handletextpad=0.5, fontsize=7.2)
fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.6)
S.save(fig, "fig4_numeric_suppression_v2")
print("Figure 4 v2 (markers + thicker lines) done.")
