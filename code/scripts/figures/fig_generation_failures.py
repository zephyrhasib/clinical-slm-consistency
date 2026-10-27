#!/usr/bin/env python3
"""
Figure 5 - Generation-failure composition (candidate).

Per cell, the mutually-exclusive output composition across Q6 -> Q3:
clean / repetition loop / anomaly / empty (fractions summing to 1). Explains two
things the other figures only hint at: why Phi-4 looks flat (it loops at every
bit-width) and what the Gemma-Q3 missing (NaN) metric cells are (empty outputs).

Single combined figure: three model panels (model names only), stacked bars.
Read-only on figures/data/generation_failures.csv.

Run: venv_metrics/bin/python scripts/figures/fig_generation_failures.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _style as S

S.apply()
gf = pd.read_csv(S.DATA / "generation_failures.csv")
X = [0, 1, 2, 3]
SPINE = "#b9b9b9"
# mutually-exclusive output categories (no yellow)
CATS = [("frac_clean", "clean", "#4C9F70"),
        ("frac_looped", "loop", "#D1605E"),
        ("frac_anom_only", "anomaly", "#856FAE"),
        ("frac_empty", "empty", "#B8B8B8")]

fig, axes = plt.subplots(1, 3, figsize=(S.COL, 2.2), sharey=True)
for ax, model in zip(axes, S.MODELS):
    s = gf[gf.model == model].sort_values("quant_idx")
    bottom = np.zeros(4)
    for col, lab, color in CATS:
        vals = s[col].to_numpy()
        ax.bar(X, vals, bottom=bottom, width=0.74, color=color,
               label=lab if ax is axes[0] else None, edgecolor="white", linewidth=0.4)
        bottom += vals
    ax.set_title(S.MODEL_SHORT[model], fontsize=7.8, color="#333333", pad=4)
    ax.set_xticks(X); ax.set_xticklabels(S.QUANT_TICK, fontsize=6.6)
    ax.set_ylim(0, 1.0); ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(axis="y", color="#e2e2e2", lw=0.6); ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(SPINE); ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=6.6, colors="#555555")
axes[0].set_ylabel("Fraction of generations", fontsize=7)
fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03),
           columnspacing=1.3, handletextpad=0.4, handlelength=1.1, fontsize=6.6)
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.2)
S.save(fig, "fig5_generation_failures")
print("Figure 5 (generation failures) done.")
