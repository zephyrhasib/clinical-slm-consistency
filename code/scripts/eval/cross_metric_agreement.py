"""
Cross-metric agreement (C3 artifact). READ-ONLY on existing metric CSVs.
Computes Spearman correlation among the five reference-free metric axes
(BERTScore, ROUGE-L, MEDCON/UMLS-F1, numeric Jaccard, NLI mean_mean):
  (1) across the 12 (model x quant) condition means  -> cross_metric_agreement.csv
  (2) within each model, across per-encounter cells   -> cross_metric_agreement_permodel.csv
No generation, no NLI recompute. Pure arithmetic on existing per-cell values.
Orientation note: BERTScore/ROUGE/MEDCON/Jaccard are higher=more consistent;
NLI contradiction is LOWER=more consistent. So a NEGATIVE NLI-vs-content
correlation means the two axes AGREE on which cells are consistent.
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr

B = "outputs/main_eval/"
M = {"bertscore_f1_mean": "BERT", "rougeL_f1_mean": "ROUGE", "medcon_f1_mean": "MEDCON",
     "numeric_jaccard_mean": "Numeric", "nli_contradiction_mean_mean": "NLImean"}
cols = list(M)

mr = pd.read_csv(B + "metric_results.csv")
nli = pd.read_csv(B + "metric_results_nli.csv")
df = mr.merge(nli, on=["input_id", "model", "quant_level"], suffixes=("", "_nli"))
df = df.rename(columns=M)
short = list(M.values())

print("merged rows:", len(df), "| NaN dropped for corr where present")

# (1) condition-level: 12 (model x quant) means
cond = df.groupby(["model", "quant_level"])[short].mean().reset_index()
print("\n=== (1) CELL-LEVEL 5x5 Spearman over", len(cond), "(model x quant) condition means ===")
cm = cond[short].corr(method="spearman").round(3)
print(cm.to_string())
cm.to_csv(B + "cross_metric_agreement.csv")
print("saved:", B + "cross_metric_agreement.csv")
print("  -- p-values (n=%d) --" % len(cond))
for i in range(len(short)):
    for j in range(i + 1, len(short)):
        r, p = spearmanr(cond[short[i]], cond[short[j]])
        print(f"    {short[i]:8s} vs {short[j]:8s}  rho={r:+.3f}  p={p:.3f}")

# (2) per-model, across per-encounter cells (pool quants)
print("\n=== (2) PER-MODEL 5x5 Spearman over per-encounter cells (quants pooled) ===")
rows = []
for model in sorted(df.model.unique()):
    sub = df[df.model == model][short].dropna()
    print(f"\n  --- {model}  (n={len(sub)} encounters x quants) ---")
    pm = sub.corr(method="spearman").round(3)
    print(pm.to_string())
    for i in range(len(short)):
        for j in range(i + 1, len(short)):
            rows.append({"model": model, "metric_a": short[i], "metric_b": short[j],
                         "spearman_rho": round(float(pm.iloc[i, j]), 3)})
pd.DataFrame(rows).to_csv(B + "cross_metric_agreement_permodel.csv", index=False)
print("\nsaved:", B + "cross_metric_agreement_permodel.csv")

# pooled per-encounter (all 1200) for reference
print("\n=== (ref) POOLED 5x5 Spearman over all", len(df[short].dropna()), "per-encounter cells ===")
print(df[short].dropna().corr(method="spearman").round(3).to_string())

# (item 2) NLI three-aggregation availability + rank consistency
print("\n=== NLI THREE AGGREGATIONS: availability + rank consistency ===")
naggs = ["nli_contradiction_max_mean", "nli_contradiction_mean_mean", "nli_contradiction_q90_mean"]
print("  present:", [c for c in naggs if c in nli.columns], "| NaN cells:",
      {c: int(nli[c].isna().sum()) for c in naggs})
ncond = nli.groupby(["model", "quant_level"])[naggs].mean()
print("  inter-aggregation Spearman over 12 conditions (do they rank cells consistently?):")
print(ncond.corr(method="spearman").round(3).to_string())
