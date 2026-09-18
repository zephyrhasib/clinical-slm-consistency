# Supplement — Statistical Analysis

This document records the statistical methodology exactly as executed. All values are
taken verbatim from the project's statistical-analysis scripts (to be released with the
full code on publication). The computed outputs are in [`../data/`](../data/).

Consistency is measured pairwise between the 20 seeds of each cell (C(20,2) = 190 seed
pairs), then aggregated to a per-encounter cell value; the tests below operate on those
per-encounter values with the 100 encounters as matched blocks.

---

## 1. Omnibus test — Friedman + Kendall's W

For each (model, metric) family the four quantization levels are compared with the
Friedman test, encounters as matched blocks:

- **Test:** Friedman rank test across the four levels Q6_K, Q5_K_M, Q4_K_M, Q3_K_M.
- **Effect size:** Kendall's W = χ² / (n · (k − 1)), where n = encounters and k = 4
  levels.
- **Blocking / missing data:** the data are pivoted to encounter × level; any encounter
  missing one or more levels is dropped listwise (`dropna(how="any")`) so every retained
  encounter contributes a complete block.
- **Significance:** α = 0.05.

The Friedman result gates the pairwise tests in the primary stream (below).

---

## 2. Pairwise test — Wilcoxon signed-rank

Within each Friedman-significant family, all C(4, 2) = 6 level pairs are tested:

- **Test:** Wilcoxon signed-rank with **Pratt zero-handling** (`zero_method="pratt"` —
  tied encounters are kept in the ranking pool and their zero-difference ranks dropped,
  rather than discarding tied encounters outright).
- **Effect size:** rank-biserial correlation.
- **Direction:** one-sided alternatives, fixed in advance per metric (higher-precision
  level expected to be more consistent: "greater" for BERTScore, ROUGE-L, concept-set
  F1, and numeric Jaccard; "less" for NLI contradiction, where lower = more
  consistent). Pairs are always ordered higher-precision vs lower-precision, so the
  direction is unambiguous.

---

## 3. Two analysis streams

- **Primary (omnibus-protected):** pairwise tests run only for families whose Friedman
  omnibus is significant at α = 0.05.
- **Sensitivity (unprotected):** pairwise tests run for all families regardless of the
  omnibus result (`--no-gate`). Reported alongside the primary stream so no finding
  depends on the gating choice.

---

## 4. Confidence intervals — BCa bootstrap

Per-pair confidence intervals on the paired median difference:

| Parameter | Value |
|---|---|
| Resamples | 10,000 |
| Method | bias-corrected and accelerated (BCa) |
| Bias term (z0) | ties-corrected: (strict-less count + 0.5 · equal count) / B, clamped to avoid ±∞ |
| Acceleration | full leave-one-out jackknife of the median |
| Fallback | percentile bootstrap when n < 30 encounters |

**Reproducible per-pair seed scheme.** Each pair draws an independent, reproducible
32-bit child seed derived from a master seed (42) and the pair's identity via SHA-256:

```
child_seed = int.from_bytes(
    sha256( f"{master_seed}|{model}|{metric}|{quant_a}|{quant_b}" ).digest()[:4],
    "big"
)
```

The `|`-joined string is hashed and its first four bytes are read as a big-endian
integer. This gives every (model, metric, level-pair) an independent but fully
reproducible bootstrap RNG, with no reliance on the `SeedSequence.spawn` API.

---

## 5. Multiple-comparison correction

- **Method:** Holm-Bonferroni.
- **Family:** the 6 level-pairs within each (model, metric) family.
- One-sided alternatives (Section 2) are fixed in advance, not chosen from the data.

---

## 6. Power analysis

Simulation-based, applying the exact analysis pipeline above:

| Parameter | Value |
|---|---|
| Encounters (n) | 100 |
| Pairs per family | 6 (C(4,2)) |
| Simulations per effect size | 2,000 |
| α | 0.05 |
| Master seed | 42 |

**The design reaches 80% power for a standardized effect size δ ≈ 0.246.** The Type-I
rate at δ = 0 was 0.046, confirming calibration. Full power curve
([`../data/power_analysis.csv`](../data/power_analysis.csv)):

| δ | power |
|---|---|
| 0.00 | 0.046 |
| 0.05 | 0.097 |
| 0.10 | 0.214 |
| 0.15 | 0.430 |
| 0.20 | 0.641 |
| 0.25 | 0.813 |
| 0.30 | 0.934 |
| 0.40 | 0.997 |
| 0.50 | 1.000 |
| 0.70 | 1.000 |

Effects smaller than δ ≈ 0.246 may produce non-significant results because of limited
power rather than absence of an effect; this is noted in the paper's discussion of null
findings.

---

## 7. Cross-metric agreement

To characterize how far the five reference-free axes agree, their per-cell values are
correlated with Spearman's ρ:

- **Cell level:** a 5×5 Spearman matrix computed across the twelve (model ×
  quantization) condition means.
- **Within model:** the same correlation repeated separately for each model, across its
  per-encounter cells (quantization levels pooled), exposing where the axes agree for
  one model but not another.

A positive correlation means two axes agree on which cells are consistent. The content
axes (BERTScore, ROUGE-L, concept-set F1, numeric Jaccard) correlate strongly with one
another; the NLI contradiction axis is largely decorrelated from them at the condition
level, and its within-model agreement is model-dependent.
