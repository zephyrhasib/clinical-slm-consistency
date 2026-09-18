# Data dictionary

Every file here is an actual computed output of the study. Model names are
`Qwen-MediCare-BD`, `Gemma-3-4B-it`, `Phi-4-mini-instruct`; quantization levels are
`Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q3_K_M`; encounter IDs are `MAIN-001` … `MAIN-100`.

A **cell** is one (encounter, model, quantization level) triple, scored over its
C(20,2) = 190 seed pairs. A **condition** is one (model, quantization level) pair
(12 conditions).

Metric column naming: `bertscore_f1`, `rougeL_f1`, `medcon_f1` (concept-set F1,
adapted from MEDCON), `numeric_jaccard`, and `nli_contradiction_{mean,q90,max}` (the
three NLI row-aggregations). For all metrics except NLI, higher = more consistent;
for NLI contradiction, lower = more consistent.

---

## Headline tables

**`table3_per_cell.csv`** — Table III of the paper. One row per condition (12 rows).
`BERTScore`, `ROUGE-L`, `Concept-set F1`, `Numeric Jaccard` are condition means over
100 encounters; `NLI` is the contradiction mean-aggregation (lower = more consistent).
Q6_K is each model's own baseline.

**`table4_effects.csv`** — Table IV of the paper. Q6_K → Q3_K_M oriented rank-biserial
effect size per metric × model; positive = more consistent at Q6_K. `✓` = significant
after Holm correction; `n.s.` = not significant. The Friedman omnibus was significant
for all 21 families.

---

## Full metric results (1,200 cells)

**`metric_results.csv`** — the four content metrics per cell.

| Column | Meaning |
|---|---|
| `input_id`, `model`, `quant_level` | cell identity |
| `n_pairs` | seed pairs scored (190 when all 20 seeds produced output) |
| `n_skipped_pairs_x`, `n_skipped_pairs_y`, `n_skipped_numeric` | pairs skipped because a note was empty (per metric family) |
| `bertscore_f1_mean`, `bertscore_f1_std` | BERTScore F1 over the pairs |
| `rougeL_f1_mean`, `rougeL_f1_std` | ROUGE-L F1 over the pairs |
| `medcon_f1_mean`, `medcon_f1_std` | UMLS concept-set F1 over the pairs |
| `mean_cui_count` | mean number of UMLS concepts per note in the cell |
| `numeric_jaccard_mean`, `numeric_jaccard_std` | numerical-attribute Jaccard over the pairs |
| `mean_fact_count` | mean number of extracted numerical facts per note — read alongside `numeric_jaccard_mean` to detect suppression (rising Jaccard with falling fact count) |

**`metric_results_nli.csv`** — the NLI contradiction metric per cell, all three
aggregations: `nli_contradiction_{max,mean,q90}_mean` and matching `_std`.

---

## Statistical results

**`stat_omnibus.csv`** — Friedman omnibus per (model, metric) family: `n_patients`
(complete blocks after listwise drop), `k_conditions` (4), `friedman_chi2`,
`friedman_p`, `kendall_w`, `friedman_significant_05`.

**`stat_pairs.csv`** — the **primary, omnibus-gated** pairwise stream. One row per
(model, metric, level pair).

| Column | Meaning |
|---|---|
| `quant_a`, `quant_b` | the pair, always higher-precision first |
| `n_patients` | encounters with both levels present |
| `median_a`, `median_b`, `median_diff_a_minus_b` | paired medians and their difference |
| `bca_ci_lo`, `bca_ci_hi`, `ci_method` | 95% bootstrap CI on the median difference (BCa, 10,000 resamples; `percentile` fallback if n < 30) |
| `ci_excludes_zero` | whether the CI excludes 0 |
| `wilcoxon_W`, `wilcoxon_p`, `alternative` | Wilcoxon signed-rank (Pratt) statistic, one-sided p, fixed direction |
| `effect_r_rb` | rank-biserial effect size |
| `n_nonzero_diffs` | encounters with a non-zero paired difference |
| `higher_is_better` | metric orientation |
| `omnibus_gated` | `True` in this file |
| `p_holm`, `significant_05` | Holm-corrected p within the 6-pair family, and its α = 0.05 verdict |
| `sig_ci_agreement` | whether the Holm verdict and the CI agree |

**`stat_pairs_ungated.csv`** — the **sensitivity** stream: identical columns, but
pairwise tests run for every family regardless of the omnibus result.

**`stat_summary.csv`** — descriptive statistics (`count`, `mean`, `median`, `std`) per
(model, quant_level, metric).

**`power_analysis.csv`** — simulated power curve: `effect_size` (standardized δ),
`power` with a 95% CI, `n_rejections` / `n_sims` (2,000 simulations each). The design
reaches 80% power at δ ≈ 0.246; Type-I rate at δ = 0 is 0.046.

---

## Cross-metric agreement

**`cross_metric_agreement.csv`** — 5×5 Spearman ρ matrix across the 12 condition means.
Labels: `BERT`, `ROUGE`, `MEDCON` (concept-set F1), `Numeric`, `NLImean`.

**`cross_metric_agreement_permodel.csv`** — pairwise Spearman ρ between metrics
computed **within** each model across its per-encounter cells (`model`, `metric_a`,
`metric_b`, `spearman_rho`).

---

## Judge panel (`judge_panel/`)

Three LLM judges (`claude`, `gemini`, `deepseek`) scored in blinded sessions.

**`nli_panel_scores.csv`** — every contradiction score: `pair_id` (P01–P60),
`judge_id`, `score` (1–5 ordinal). 60 pairs × 3 judges = 180 rows.

**`nli_panel_validation_report.csv`** — Spearman ρ between the panel consensus and
each NLI aggregation, at `cell` level (n = 12) and `pair` level (n = 60), with
`krippendorff_alpha` (ordinal, = 0.548 pooled). `is_primary` marks the paper's lead
aggregation (`mean`).

**`nli_panel_permodel.csv`** — the same validation broken down per model (`scope` =
`ALL` or a model name), exposing the cross-model verbosity confound.

**`nli_panel_perpair_diag.csv`** — per-pair diagnostic: the machine scores
(`pair_max_mean`, `pair_mean_mean`, `pair_q90_mean`) and the panel `consensus` for
each sampled pair.

**`bertscore_panel_categories.csv`** — every category vote for the 25 most
BERTScore-degraded pairs: `pair_id` (BVAL-001 … BVAL-025), `judge_id`, `category`
(A = content disagreement, B = format variation, C = degeneracy, D = mixed). 75 rows.

**`bertscore_validation_report.csv`** — category distribution (`n_pairs`, `fraction`)
and `krippendorff_alpha` (nominal, = 0.292).

---

## FP16-vs-FP32 verification (`fp16_vs_fp32/`)

This folder concerns **only the NLI scoring model** (DeBERTa-v3-large), not the SLMs
under study. `precision` is the precision the *scoring model* ran at.

**`results.csv`** — matched contradiction scores per cell under three conditions:
`GOLD` (FP32, all 190 pairs — the reference), `DIAG_fp32_60` (FP32, 60-pair
subsample), and the FP16 production run; columns `mean_mean`, `q90_mean`, `max_mean`,
`elapsed_seconds`.

**`differential_results.csv`** — FP32 gold `mean_mean` across all four quantization
levels for the differential check.

**`subsample_vs_effect.csv`** — why the full 190 pairs were retained: for each
(metric, model, level pair), the between-quantization effect vs the subsample noise,
`noise_pct_of_effect`, and the `verdict`.

**`batch_speed_results.csv`** — FP16 scores are identical at batch sizes 32/64/128;
`fp16_speedup_vs_fp32_gold` ≈ 2.9×.

---

## Provenance (`provenance/`)

**`master_hashes.json`** — the pinned `llama.cpp` tag/commit and CUDA build flags,
SHA-256 of the `llama-completion` and `llama-quantize` binaries, and SHA-256 + size +
source HuggingFace repo of every FP16 master and every quantized GGUF file. Every
generation run verified these hashes at start-up.

**`aci_bench_tui_codes.json`** — the 66 UMLS semantic-type identifiers (TUIs) across
7 semantic groups that the concept-set metric restricts to, derived from ACI-Bench's
own `semantic_types.txt` and evaluation filter (with their SHA-256).
