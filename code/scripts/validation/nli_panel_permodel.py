"""
Per-model NLI panel breakdown (main study, 60-pair panel). READ-ONLY on the
main-eval outputs; writes one reproducible CSV.

For each model (Gemma / Phi-4 / Qwen) and each of the 3 NLI aggregations
(mean_mean / q90_mean / max_mean), computes Spearman rho (n, p) between the
3-judge human consensus and:
  - the production CELL-level NLI value (metric_results_nli.csv, joined on
    input_id+model+quant_level — exactly what aggregate_panel_scores.py uses), and
  - the per-PAIR NLI value (recomputed on the exact sampled pairs, read from
    nli_panel_perpair_diag.csv).
Also recomputes Krippendorff's alpha (ordinal) per model and overall.

Inputs (all in outputs/main_eval/):
  nli_validation_pairs.csv      pair_id -> input_id, model, quant_level
  metric_results_nli.csv        per-cell NLI aggregations
  nli_panel_scores.csv          pair_id, judge_id, score (3 judges x 60)
  nli_panel_perpair_diag.csv    per-pair NLI (pair_max_mean/mean_mean/q90_mean) + consensus
Output:
  nli_panel_permodel.csv

Run in venv_metrics:
  python scripts/validation/nli_panel_permodel.py
"""
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr
try:
    import krippendorff
    HAVE_K = True
except ImportError:
    HAVE_K = False

OUT = Path(__file__).resolve().parent.parent.parent / "outputs/main_eval"
AGGS = ["mean_mean", "q90_mean", "max_mean"]   # report order

pairs   = pd.read_csv(OUT / "nli_validation_pairs.csv")
cellnli = pd.read_csv(OUT / "metric_results_nli.csv")
scores  = pd.read_csv(OUT / "nli_panel_scores.csv")
perpair = pd.read_csv(OUT / "nli_panel_perpair_diag.csv")   # has pair_* + consensus

# consensus = median across judges per pair (same as the production aggregator)
cons = scores.groupby("pair_id")["score"].median().rename("consensus").reset_index()

# CELL-level join (production validator path)
cellcols = ["input_id", "model", "quant_level"] + [f"nli_contradiction_{a}" for a in AGGS]
cj = pairs.merge(cellnli[cellcols], on=["input_id", "model", "quant_level"], how="left")
cj = cj.merge(cons, on="pair_id", how="left")

# PER-PAIR values (already merged with consensus in the diag file)
pp = perpair.copy()

def kripp_alpha(score_df, pair_ids):
    if not HAVE_K:
        return float("nan")
    sub = score_df[score_df["pair_id"].isin(pair_ids)]
    wide = sub.pivot(index="judge_id", columns="pair_id", values="score")
    return float(krippendorff.alpha(reliability_data=wide.values,
                                    level_of_measurement="ordinal"))

rows = []
def add(scope, pair_ids, cj_sub, pp_sub):
    alpha = kripp_alpha(scores, pair_ids)
    for a in AGGS:
        # cell-level
        s = cj_sub.dropna(subset=["consensus", f"nli_contradiction_{a}"])
        r_c, p_c = spearmanr(s["consensus"], s[f"nli_contradiction_{a}"]) if len(s) >= 3 else (np.nan, np.nan)
        # pair-level
        s2 = pp_sub.dropna(subset=["consensus", f"pair_{a}"])
        r_p, p_p = spearmanr(s2["consensus"], s2[f"pair_{a}"]) if len(s2) >= 3 else (np.nan, np.nan)
        rows.append({
            "scope": scope, "aggregation": f"nli_contradiction_{a}",
            "n": int(len(s)),
            "spearman_rho_cell": round(float(r_c), 4), "spearman_p_cell": round(float(p_c), 4),
            "spearman_rho_pair": round(float(r_p), 4), "spearman_p_pair": round(float(p_p), 4),
            "krippendorff_alpha_ordinal": round(alpha, 4),
        })

# overall
add("ALL", set(pairs["pair_id"]), cj, pp)
# per model
for model in ["Gemma-3-4B-it", "Phi-4-mini-instruct", "Qwen-MediCare-BD"]:
    pids = set(pairs.loc[pairs["model"] == model, "pair_id"])
    add(model, pids, cj[cj["model"] == model], pp[pp["model"] == model])

df = pd.DataFrame(rows)
df.to_csv(OUT / "nli_panel_permodel.csv", index=False)

print("Wrote", OUT / "nli_panel_permodel.csv")
print("\n" + "=" * 100)
print("PER-MODEL NLI PANEL BREAKDOWN (main, 60-pair). consensus = median of 3 judges (1=consistent..5=contradictory)")
print("cell = production join (per-cell NLI);  pair = per-pair recomputed NLI")
print("=" * 100)
print(df.to_string(index=False))
print("\nPer-model mean values (context):")
for model in ["Gemma-3-4B-it", "Phi-4-mini-instruct", "Qwen-MediCare-BD"]:
    sc = cj[cj["model"] == model]
    print(f"  {model:<22} n={len(sc)}  mean cell-NLI(mean_mean)={sc['nli_contradiction_mean_mean'].mean():.3f}"
          f"  mean human consensus={sc['consensus'].mean():.2f}")
