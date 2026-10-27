#!/usr/bin/env python3
"""
Figure 3 (side-by-side) - cross-metric agreement (contribution C3), two panels.

(a) Metric-metric agreement as a STRICT lower triangle (upper triangle AND the
    trivial 1.0 diagonal removed - the matrix is exactly symmetric, so only the 10
    informative off-diagonal cells remain): the four content metrics agree 0.87-0.99
    (mutually redundant); the NLI row is decorrelated (-0.08 to -0.22).
(b) NLI vs each content metric WITHIN each model (sign-flipped, + = agree), on its
    own colour scale so the per-model contrast is visible: the specialist (Qwen)
    agrees; the generalist/edge (Gemma/Phi-4) do not.

Two-column-width single figure (replaces two separate figures). Read-only on
figures/data/. Clinical axis = customized UMLS concept-set F1 ("Concept"), NOT a
full token-level F1. Model names only. The backup heatmaps/network are untouched.

Run: venv_metrics/bin/python scripts/figures/fig_agreement_sidebyside.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _style as S

S.apply()
am = pd.read_csv(S.DATA / "agreement_matrix_oriented.csv", index_col=0)
pm = pd.read_csv(S.DATA / "nli_content_agreement_permodel.csv")

FULL = ["BERT", "ROUGE", "MEDCON", "Numeric", "NLImean"]
TICK = {"BERT": "BERT", "ROUGE": "ROUGE", "MEDCON": "Concept", "Numeric": "Num", "NLImean": "NLI"}
CONTENT = ["BERT", "ROUGE", "MEDCON", "Numeric"]

fig, (axa, axb) = plt.subplots(1, 2, figsize=(S.DBL, 2.55))

# ---------------------------------------------------------------- (a) strict lower triangle
rows_a = ["ROUGE", "MEDCON", "Numeric", "NLImean"]   # drop empty BERT row
cols_a = ["BERT", "ROUGE", "MEDCON", "Numeric"]      # drop empty NLI column
A = np.full((len(rows_a), len(cols_a)), np.nan)
for i, r in enumerate(rows_a):
    for j, c in enumerate(cols_a):
        if FULL.index(c) < FULL.index(r):            # strictly below diagonal only
            A[i, j] = am.loc[r, c]
cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("white")
im = axa.imshow(np.ma.masked_invalid(A), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
axa.set_xticks(range(len(cols_a))); axa.set_yticks(range(len(rows_a)))
axa.set_xticklabels([TICK[c] for c in cols_a], fontsize=7.2)
axa.set_yticklabels([TICK[r] for r in rows_a], fontsize=7.2)
axa.tick_params(length=0); axa.grid(False)
for sp in axa.spines.values():
    sp.set_visible(False)
for i in range(len(rows_a)):
    for j in range(len(cols_a)):
        if not np.isnan(A[i, j]):
            v = A[i, j]
            axa.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.7,
                     color="white" if abs(v) > 0.55 else "#222222")
axa.set_title("(a)  Metric–metric agreement", fontsize=7.8, color="#333333", pad=6)
cba = fig.colorbar(im, ax=axa, fraction=0.046, pad=0.04)
cba.set_label("Spearman ρ  (+ = agree)", fontsize=7.6)
cba.ax.tick_params(labelsize=6.8, length=0); cba.outline.set_visible(False)

# ---------------------------------------------------------------- (b) per-model NLI vs content
mat = pm.pivot(index="model", columns="content_metric", values="rho_oriented").loc[S.MODELS, CONTENT]
im2 = axb.imshow(mat.values, cmap="RdBu_r", vmin=-0.35, vmax=0.35, aspect="auto")
axb.set_xticks(range(len(CONTENT))); axb.set_yticks(range(len(S.MODELS)))
axb.set_xticklabels([TICK[c] for c in CONTENT], fontsize=7.2)
axb.set_yticklabels([S.MODEL_SHORT[m] for m in S.MODELS], fontsize=7.4)
axb.tick_params(length=0); axb.grid(False)
for sp in axb.spines.values():
    sp.set_visible(False)
for i in range(len(S.MODELS)):
    for j in range(len(CONTENT)):
        v = mat.values[i, j]
        axb.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.9,
                 color="white" if abs(v) > 0.22 else "#222222")
axb.set_title("(b)  NLI vs content, by model", fontsize=7.8, color="#333333", pad=6)
cbb = fig.colorbar(im2, ax=axb, fraction=0.046, pad=0.04)
cbb.set_label("NLI–content ρ  (+ = agree)", fontsize=7.6)
cbb.ax.tick_params(labelsize=6.8, length=0); cbb.outline.set_visible(False)

fig.tight_layout(w_pad=2.0)
S.save(fig, "fig3_agreement_sidebyside")
print("Figure 3 side-by-side (strict lower triangle + per-model) done.")
