#!/usr/bin/env python3
"""
build_figure_data.py - Stage 1 of the Results-figure pipeline (READ-ONLY).

Reads the frozen evaluation outputs in outputs/main_eval/ and emits tidy,
figure-ready CSV tables to outputs/figures/data/. NOTHING is recomputed:
no metric is re-scored, no generation is run, no NLI matrix is rebuilt. This
is pure reshaping / condition-level aggregation of the existing per-cell
values, so every figure is reproducible from the committed eval outputs alone.

Run (evaluation venv):
    venv_metrics/bin/python scripts/figures/build_figure_data.py

Inputs (outputs/main_eval/):
    metric_results.csv                per-cell BERTScore/ROUGE/MEDCON/numeric + mean_fact_count
    metric_results_nli.csv            per-cell NLI contradiction (max/mean/q90 aggregations)
    stat_pairs.csv                    pairwise Wilcoxon-Pratt + rank-biserial + BCa + Holm
    stat_omnibus.csv                  Friedman + Kendall W per (model, metric)
    cross_metric_agreement.csv        pooled 5x5 Spearman (condition-level)
    cross_metric_agreement_permodel.csv  per-model 5x5 Spearman
    main_combined.csv                 24,000 generations + is_anomalous / is_looped flags

Outputs (outputs/figures/data/):
    dose_response.csv                 (model, quant, metric): condition mean, SE, n, Q6-normalized
    effect_sizes.csv                  Q6_K-vs-lower rank-biserial + Holm + BCa (degradation-oriented)
    numeric_suppression.csv           (model, quant): numeric Jaccard + mean fact count
    generation_failures.csv           (model, quant): anomaly / loop / empty / clean rates + NaN-cell count
    agreement_matrix_oriented.csv     5x5 Spearman, NLI sign-flipped so +1 = agreement everywhere
    nli_content_agreement_permodel.csv  per model: NLI-vs-content Spearman (sign-flipped)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs" / "main_eval"
OUT = ROOT / "figures" / "data"
OUT.mkdir(parents=True, exist_ok=True)

# ---- canonical orders / labels -------------------------------------------------
MODELS = ["Qwen-MediCare-BD", "Gemma-3-4B-it", "Phi-4-mini-instruct"]
ARCHETYPE = {"Qwen-MediCare-BD": "specialist",
             "Gemma-3-4B-it": "generalist",
             "Phi-4-mini-instruct": "edge"}
QUANTS = ["Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M"]  # decreasing precision, left->right

# the five reported axes (column name -> display label, higher_is_better)
METRICS = [
    ("bertscore_f1_mean",          "BERTScore (semantic)",   True),
    ("rougeL_f1_mean",             "ROUGE-L (lexical)",      True),
    ("medcon_f1_mean",             "UMLS concept-set F1 (clinical)", True),
    ("numeric_jaccard_mean",       "Numeric Jaccard",        True),
    ("nli_contradiction_mean_mean","NLI contradiction",      False),  # lower = more consistent
]
LABEL = {k: v for k, v, _ in METRICS}
HIB = {k: h for k, _, h in METRICS}


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ================================================================================
banner("LOAD")
mr = pd.read_csv(SRC / "metric_results.csv")
nli = pd.read_csv(SRC / "metric_results_nli.csv")
df = mr.merge(nli, on=["input_id", "model", "quant_level"], suffixes=("", "_nlifile"))
print(f"metric_results      {mr.shape}")
print(f"metric_results_nli  {nli.shape}")
print(f"merged              {df.shape}  (expect 1200 = 100 x 3 x 4)")
assert df["model"].isin(MODELS).all(), "unexpected model name"
assert df["quant_level"].isin(QUANTS).all(), "unexpected quant level"

# NaN cell audit (cells with too few valid seed pairs -> NaN metric)
banner("NaN CELL AUDIT (generation-failure cells)")
nan_cells = df[df["nli_contradiction_mean_mean"].isna()][["model", "quant_level", "input_id"]]
print(f"NLI mean_mean NaN cells: {len(nan_cells)}")
print(nan_cells.groupby(["model", "quant_level"]).size().to_string() or "  (none)")


# ================================================================================
# 1. DOSE-RESPONSE: condition means + Q6-normalized
# ================================================================================
banner("[1] dose_response.csv")
rows = []
for model in MODELS:
    for mcol, mlabel, hib in METRICS:
        sub = df[df["model"] == model]
        g = sub.groupby("quant_level")[mcol]
        cond = {q: (g.get_group(q).mean() if q in g.groups else np.nan) for q in QUANTS}
        se = {q: (g.get_group(q).sem() if q in g.groups else np.nan) for q in QUANTS}
        n = {q: (g.get_group(q).notna().sum() if q in g.groups else 0) for q in QUANTS}
        q6 = cond["Q6_K"]
        for q in QUANTS:
            v = cond[q]
            if hib:
                norm = v / q6 if q6 not in (0, np.nan) else np.nan
            else:  # NLI: flip so down = worse consistency, starts at 1.0 at Q6
                norm = q6 / v if (v not in (0, np.nan)) else np.nan
            rows.append({
                "model": model, "archetype": ARCHETYPE[model],
                "metric": mcol, "metric_label": mlabel, "higher_is_better": hib,
                "quant_level": q, "quant_idx": QUANTS.index(q),
                "cond_mean": v, "cond_sem": se[q], "n_cells": int(n[q]),
                "norm_to_q6": norm,
            })
dose = pd.DataFrame(rows)
dose.to_csv(OUT / "dose_response.csv", index=False)
print(f"wrote {len(dose)} rows  (3 models x 5 metrics x 4 quants = 60)")
# headline check: BERTScore Q6 vs Q3 per model
piv = dose[dose.metric == "bertscore_f1_mean"].pivot(index="model", columns="quant_level", values="cond_mean")
print("\nBERTScore condition means (Q6 -> Q3):")
print(piv[QUANTS].round(4).to_string())


# ================================================================================
# 2. EFFECT SIZES: Q6_K-vs-lower pairwise (degradation-oriented)
# ================================================================================
banner("[2] effect_sizes.csv")
sp = pd.read_csv(SRC / "stat_pairs.csv")
keep = list(LABEL.keys())
ev = sp[(sp["quant_a"] == "Q6_K") & (sp["metric"].isin(keep))].copy()
# orient so POSITIVE = Q6 more consistent than the lower-precision level (degradation)
ev["effect_oriented"] = np.where(ev["higher_is_better"], ev["effect_r_rb"], -ev["effect_r_rb"])
ev["metric_label"] = ev["metric"].map(LABEL)
ev["archetype"] = ev["model"].map(ARCHETYPE)
ev["comparison"] = "Q6_K vs " + ev["quant_b"]
ev = ev[["model", "archetype", "metric", "metric_label", "quant_b", "comparison",
         "effect_r_rb", "effect_oriented", "median_diff_a_minus_b",
         "bca_ci_lo", "bca_ci_hi", "p_holm", "significant_05", "higher_is_better", "n_patients"]]
ev.to_csv(OUT / "effect_sizes.csv", index=False)
print(f"wrote {len(ev)} rows  (Q6_K vs Q5/Q4/Q3 x 5 metrics x 3 models = 45)")
q63 = ev[ev.quant_b == "Q3_K_M"].pivot(index="model", columns="metric_label", values="effect_oriented")
print("\nQ6->Q3 oriented rank-biserial (|>0| = Q6 more consistent):")
print(q63.round(3).to_string())


# ================================================================================
# 3. NUMERIC SUPPRESSION: Jaccard up / fact count down
# ================================================================================
banner("[3] numeric_suppression.csv")
ns_rows = []
for model in MODELS:
    sub = df[df["model"] == model]
    for q in QUANTS:
        s = sub[sub.quant_level == q]
        ns_rows.append({
            "model": model, "archetype": ARCHETYPE[model],
            "quant_level": q, "quant_idx": QUANTS.index(q),
            "numeric_jaccard_mean": s["numeric_jaccard_mean"].mean(),
            "mean_fact_count": s["mean_fact_count"].mean(),
        })
ns = pd.DataFrame(ns_rows)
ns.to_csv(OUT / "numeric_suppression.csv", index=False)
print(f"wrote {len(ns)} rows")
print("\nfact count Q6 -> Q3 (suppression check):")
fp = ns.pivot(index="model", columns="quant_level", values="mean_fact_count")[QUANTS]
jp = ns.pivot(index="model", columns="quant_level", values="numeric_jaccard_mean")[QUANTS]
for m in MODELS:
    print(f"  {m:22s} fact {fp.loc[m,'Q6_K']:.3f}->{fp.loc[m,'Q3_K_M']:.3f}   "
          f"jaccard {jp.loc[m,'Q6_K']:.3f}->{jp.loc[m,'Q3_K_M']:.3f}")


# ================================================================================
# 4. GENERATION FAILURES: anomaly / loop / empty / clean rates + NaN cells
# ================================================================================
banner("[4] generation_failures.csv")
mc = pd.read_csv(SRC / "main_combined.csv")
print(f"main_combined {mc.shape}")


def as_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


mc["_anom"] = as_bool(mc["is_anomalous"])
mc["_loop"] = as_bool(mc["is_looped"])
mc["_empty"] = mc["generated_text"].isna() | (mc["generated_text"].astype(str).str.strip() == "")
mc["_clean"] = ~(mc["_anom"] | mc["_loop"] | mc["_empty"])
# NaN metric cells per (model, quant)
nanc = (df.assign(_nan=df["nli_contradiction_mean_mean"].isna())
          .groupby(["model", "quant_level"])["_nan"].sum().rename("nan_cells"))
gf_rows = []
for model in MODELS:
    for q in QUANTS:
        s = mc[(mc.model == model) & (mc.quant_level == q)]
        nrow = len(s)
        # mutually-exclusive buckets (priority empty > loop > anomaly-only > clean), sum to 1
        empty = s["_empty"]
        loop = s["_loop"] & ~empty
        anom = s["_anom"] & ~s["_loop"] & ~empty
        clean = ~(empty | s["_loop"] | s["_anom"])
        gf_rows.append({
            "model": model, "archetype": ARCHETYPE[model],
            "quant_level": q, "quant_idx": QUANTS.index(q), "n_generations": nrow,
            "anomalous_rate": s["_anom"].mean(), "looped_rate": s["_loop"].mean(),
            "empty_rate": s["_empty"].mean(), "clean_rate": s["_clean"].mean(),
            "frac_clean": clean.mean(), "frac_looped": loop.mean(),
            "frac_anom_only": anom.mean(), "frac_empty": empty.mean(),
            "nan_cells": int(nanc.get((model, q), 0)),
        })
gf = pd.DataFrame(gf_rows)
gf.to_csv(OUT / "generation_failures.csv", index=False)
print(f"wrote {len(gf)} rows")
print("\nclean / loop / empty rate by cell:")
print(gf[["model", "quant_level", "clean_rate", "looped_rate", "empty_rate", "nan_cells"]]
      .round(3).to_string(index=False))


# ================================================================================
# 5. CROSS-METRIC AGREEMENT (sign-flip NLI so +1 = agreement everywhere)
# ================================================================================
banner("[5] agreement_matrix_oriented.csv + nli_content_agreement_permodel.csv")
am = pd.read_csv(SRC / "cross_metric_agreement.csv", index_col=0)
amo = am.copy()
# flip every correlation that involves NLImean, except the diagonal
for i in amo.index:
    for j in amo.columns:
        if (("NLImean" in (i, j)) and (i != j)):
            amo.loc[i, j] = -amo.loc[i, j]
amo.to_csv(OUT / "agreement_matrix_oriented.csv")
print("oriented pooled matrix (NLI negated; +1 = agree):")
print(amo.round(3).to_string())

pm = pd.read_csv(SRC / "cross_metric_agreement_permodel.csv")
nli_pm = pm[(pm.metric_a == "NLImean") | (pm.metric_b == "NLImean")].copy()
nli_pm["content_metric"] = np.where(nli_pm.metric_a == "NLImean", nli_pm.metric_b, nli_pm.metric_a)
nli_pm["rho_oriented"] = -nli_pm["spearman_rho"]   # +ve = NLI agrees with content
nli_pm = nli_pm[["model", "content_metric", "spearman_rho", "rho_oriented"]]
nli_pm.to_csv(OUT / "nli_content_agreement_permodel.csv", index=False)
print("\nNLI-vs-content oriented Spearman per model (+ve = agree):")
print(nli_pm.pivot(index="model", columns="content_metric", values="rho_oriented").round(3).to_string())

# ================================================================================
# 6. PER-ENCOUNTER values (for the distribution figure)
# ================================================================================
banner("[6] per_encounter.csv")
pe = df[["model", "quant_level", "input_id",
         "bertscore_f1_mean", "rougeL_f1_mean", "medcon_f1_mean",
         "numeric_jaccard_mean", "nli_contradiction_mean_mean"]].copy()
pe.to_csv(OUT / "per_encounter.csv", index=False)
print(f"wrote {len(pe)} rows (100 patients x 12 conditions; per-encounter cell means)")
b = pe.dropna(subset=["bertscore_f1_mean"])
print("BERTScore per-encounter range:", round(b["bertscore_f1_mean"].min(), 3), "-", round(b["bertscore_f1_mean"].max(), 3))
print(b.groupby(["model", "quant_level"])["bertscore_f1_mean"].agg(["min", "median", "max"]).round(3).to_string())

banner("DONE - all tidy tables written to outputs/figures/data/")
for f in sorted(OUT.glob("*.csv")):
    print(f"  {f.name}")
