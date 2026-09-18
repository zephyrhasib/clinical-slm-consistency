# Supplement — Consistency Metrics

Five complementary reference-free metrics measure run-to-run agreement between the 20
seeds of each cell (C(20,2) = 190 seed pairs per cell, symmetric). A three-judge LLM
panel provides a coarse external check. All values are taken verbatim from the project's
metric pipeline (to be released with the full code on publication).

## Precision — two separate topics, never to be conflated

This paper involves two entirely separate precision settings. They must not be confused
in this document.

1. **The models under study (the subject of the paper).** The three SLMs — Qwen, Gemma,
   Phi-4 — are GGUF-quantized to Q6_K / Q5_K_M / Q4_K_M / Q3_K_M, all built from FP16
   masters. This quantization of the models is what the paper measures.
2. **The evaluation tooling (a scoring-pipeline detail).** Separately, the NLI scoring
   model (DeBERTa-v3-large) was run in FP16 as a speed choice, verified against FP32.
   This has nothing to do with the SLM quantization the paper studies.

The FP16-vs-FP32 verification described under the NLI metric below concerns **only the
second topic** — the NLI scoring model — and never the models under test.

---

## 1. ROUGE-L F1 (lexical)

- Library: `rouge-score`, ROUGE-L F-measure.
- Porter stemming enabled.
- Higher = more lexically consistent across seeds.

---

## 2. BERTScore F1 (semantic)

- Model: `microsoft/deberta-xlarge-mnli`, layer 40.
- Batch size 4, `rescale_with_baseline=False`, **FP32**.
- Tokenizer patch: `deberta-xlarge-mnli` ships `model_max_length = sys.maxsize`, which
  overflows the fast tokenizer; it is clamped to 512 around each encode call and
  restored afterward.
- Higher = more semantically consistent across seeds.

---

## 3. Concept-set F1 (clinical)

- Extractor: QuickUMLS.
- Similarity threshold 0.7, window 5, `ignore_syntax=True`.
- UMLS release 2026AA (English), restricted to the benchmark's clinically relevant
  semantic types: 66 TUIs across 7 semantic groups (the ACI-Bench filter; full list in
  [`../data/provenance/aci_bench_tui_codes.json`](../data/provenance/aci_bench_tui_codes.json)).
- Adapted from **MEDCON**: a symmetric F1 between the UMLS concept sets extracted from
  two generations. Edge cases match the numeric metric: both concept sets empty → 1.0
  (trivially agreeing); exactly one empty → 0.0.
- This is a customized concept-set overlap F1, **not** a full token-level F1.
- Higher = more consistent clinical concepts across seeds.

---

## 4. Numerical-attribute Jaccard

Pairwise Jaccard over `(entity, value, unit)` triples extracted by six fixed pattern
families (text lowercased before matching). The families:

| # | Family | Captures |
|---|---|---|
| 1 | Drug + dose + unit | e.g. "metformin 500 mg" (mg/mcg/g/ml/units/IU) |
| 2 | Administration frequency | "3 times daily", "twice a day", "BID/TID/QID/QD", "q12h" |
| 3 | Treatment duration | "for 7 days", "x 14 days", "for 2 weeks" |
| 4 | Vital signs | BP, HR, RR, SpO2, temp, pulse, blood pressure, heart rate |
| 5 | Laboratory values | Hgb/HbA1c/glucose/creatinine/BUN/WBC/Na/K/Cl/bicarbonate |
| 6 | Patient age | "32-year-old", "65 year old" |

- Comparison: `|A ∩ B| / |A ∪ B|` over the triple sets of two generations.
- Edge cases: both sets empty → 1.0; one empty → 0.0.
- Higher = more consistent numerical attributes across seeds. A companion `mean fact
  count` is reported per cell, because a rising Jaccard alongside a falling fact count
  indicates numerical **suppression** (fewer facts to disagree on), not improvement.
- Same six patterns applied to every generation; not tuned per model or per quant.

---

## 5. NLI contradiction

- Model: `cross-encoder/nli-deberta-v3-large`, contradiction label index 0.
- **FP16** cross-encoder, batch size 32.
- SummaC-style sentence-level contradiction matrix: for a pair of notes, every
  premise-sentence × hypothesis-sentence entry is the contradiction probability.
- Symmetrized bidirectionally: `score(A,B) = ½ · (score_{A→B} + score_{B→A})`, so the
  metric does not privilege either note.
- Three row-aggregations are computed from the same matrix and averaged over rows:
  - **mean** — mean over columns per row, then mean over rows (least length-inflating;
    the value the paper leads with);
  - **q0.9** — 90th percentile over columns per row, then mean over rows;
  - **max** — max over columns per row, then mean over rows (the original SummaC row-max
    form; most length-inflating).
- Document-level fallback: when either note has fewer than two sentences, a single
  whole-note bidirectional contradiction is used instead of the sentence matrix.
- Lower = more consistent (less contradiction) across seeds.

**Precision (evaluation tooling only — not the models under study).** The NLI
contradiction cross-encoder (DeBERTa-v3-large) was run in FP16 to save time. The
supplement verification confirms that this FP16 choice shifted contradiction scores by
under 1×10⁻⁴ compared to FP32, so the speed optimization did not affect the reported
results. At equal 190-pair sampling the primary mean-aggregation contradiction score was
0.227764 in FP16 versus 0.227806 in FP32 — a difference of about 4×10⁻⁵ — and FP16
results were additionally bit-identical across reruns and checkpoint resumes, at roughly
2.9× the speed. **BERTScore and all other neural scoring used FP32.** Data:
[`../data/fp16_vs_fp32/`](../data/fp16_vs_fp32/).

---

## 6. Judge panel (coarse external check)

- Three LLM judges from different families: **Claude, Gemini, DeepSeek**.
- Blinded sessions (no model/quant identity revealed).
- Two tracks: a contradiction-scale check on sampled seed pairs, and a categorization
  of the largest-BERTScore-degradation pairs (content disagreement / format variation /
  degeneracy / mixture).
- Inter-judge agreement: Krippendorff's α = **0.55** (NLI, ordinal) and **0.29**
  (BERTScore categorization, nominal).
- This is a **coarse ordinal instrument** — a small panel scoring on a few-level scale.
  It is used to probe where the metric signals diverge, not to certify any metric as
  correct. The panel's contradiction scores did not show positive convergent validity
  with the NLI aggregations at main scale, which the paper reports transparently.
