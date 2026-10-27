# Quantization Effects on Stochastic Consistency in Clinical Note Generation Using Small Language Models

Source code, computed analysis outputs and methodology documents for a study of how
GGUF quantization affects the **run-to-run (stochastic) consistency** of small language
models that generate clinical notes.

> Quantization compresses small language models for offline clinical use; however, its
> impact on stochastic consistency, that is, whether the same input generates the same
> output on each run, is not well understood. This study evaluates three sub-4B models
> across four GGUF compression levels from Q6 to Q3 using five reference-free metrics.
> A quantization effect was observed in all 21 model-metric families, with varying
> results for each model. Gemma led all content metrics and saw only a slight drop in
> BERTScore (0.789 to 0.765). Qwen ranked the worst on content metrics and showed the
> biggest fall on the same measure, from 0.652 to 0.565. The flat curve for Phi-4 was
> not due to robustness but because it already loops on almost half of its outputs at
> full precision. No single bit width or metric was proven safe across all models.
> Contradiction was the least reliable despite being the most trusted, diverging from
> the other four axes in two of the three models. As factual content was suppressed,
> numerical agreement increased, which single-run accuracy checks did not detect. These
> findings indicate that consistency should not be assumed from accuracy results; it
> must be verified per model and metric.

---

## Study design in one line

Three sub-4B instruction-tuned models, each at four `llama.cpp` K-quantization levels
(Q6_K, Q5_K_M, Q4_K_M, Q3_K_M), generate a clinical note for each of 100 ACI-Bench
encounters under 20 fixed seeds = **24,000 generations**. Consistency is measured
**reference-free** between the 20 seeds of each cell (C(20,2) = 190 seed pairs per cell)
along five complementary metric axes, tested with a Friedman-gated Wilcoxon analysis,
and characterized by cross-metric agreement.

| Model | Role | Weights |
|---|---|---|
| Qwen-MediCare-BD (3.1B) | domain specialist, fine-tuned from Qwen2.5-3B-Instruct | [huggingface.co/CBrootA/Qwen-MediCare-BD](https://huggingface.co/CBrootA/Qwen-MediCare-BD) |
| Gemma-3-4B-it (3.9B text decoder) | multilingual generalist | [huggingface.co/google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) |
| Phi-4-mini-instruct (3.8B) | efficiency-oriented edge model | [huggingface.co/microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) |

Dataset: **ACI-Bench** (Yim et al., 2023, *Scientific Data*) —
[github.com/wyim/aci-bench](https://github.com/wyim/aci-bench).

---

## Repository contents

### `code/` — the full pipeline

All scripts used in the study, from dataset preparation and the contamination audit to
generation, metrics, statistics, the judge-panel validation, the summary tables and the
figures. Setup, the expected folder layout and the run order are in
[`code/README.md`](code/README.md).

| Stage | Where |
|---|---|
| Dataset preparation and quality audit | `code/scripts/prep_dataset.py`, `audit_dataset.py` |
| Contamination audit | `code/scripts/audit_01`–`audit_04`, `verify_audit.py` |
| UMLS / QuickUMLS setup | `code/scripts/umls_01`–`umls_04`, `apply_quickumls_py312_patches.py` |
| Model download and quantization | `code/scripts/download_models.py`, `quantize_all.py` |
| Generation (pilot and main) | `code/scripts/pilot_run/`, `code/scripts/main_run/` |
| Metrics and statistics | `code/scripts/eval/`, `code/scripts/main_eval/`, `code/scripts/pilot_eval/` |
| Judge-panel validation | `code/scripts/validation/` |
| Summary tables and figures | `code/scripts/make_summary_tables.py`, `code/scripts/figures/` |

### `supplement/` — methodology documents

| File | What it covers |
|---|---|
| [`contamination.md`](supplement/contamination.md) | Pre-generation contamination audit: MinHash corpus-overlap test (Qwen) and verbatim-memorization probe (Gemma, Phi-4), all constants, and the 0/100 result. |
| [`generation_config.md`](supplement/generation_config.md) | Full generation protocol: the zero-shot instruction, the three native chat templates (verbatim), all decoding parameters, the complete stop-sequence lists (including Qwen's nine role-injection stops), Phi-4 looping/truncation handling, the quality-flag regexes, and sampling. |
| [`statistics.md`](supplement/statistics.md) | Friedman + Kendall's W omnibus, Wilcoxon signed-rank (Pratt) pairwise, BCa bootstrap with the reproducible per-pair seed formula, Holm-Bonferroni, the two analysis streams, power analysis, and cross-metric agreement. |
| [`metrics.md`](supplement/metrics.md) | The five consistency metrics (ROUGE-L, BERTScore, UMLS concept-set F1, numerical-attribute Jaccard, NLI contradiction), the three-judge panel, and the FP16-vs-FP32 verification of the NLI scoring model. |
| [`judge_panel.md`](supplement/judge_panel.md) | The three-judge blinded panel: judges and modes, blinding, sampling of the 60 contradiction pairs and 25 categorisation pairs, batching, consensus rules, and both prompts and rubrics verbatim. |
| [`numerical_suppression_example.md`](supplement/numerical_suppression_example.md) | Worked example of numerical suppression (encounter MAIN-013, Qwen): fact counts and numerical Jaccard at all four levels, with two generations reproduced verbatim. |

### `data/` — computed outputs behind every table and figure

All values are the actual computed outputs of the study. Nothing is regenerated or
reconstructed. A column-by-column dictionary is in [`data/README.md`](data/README.md).

| Group | Files |
|---|---|
| **Summary tables** | `table3_per_cell.csv` (per-condition means), `table4_effects.csv` (Q6_K → Q3_K_M effect sizes) |
| **Full metric results** (1,200 cells) | `metric_results.csv`, `metric_results_nli.csv` |
| **Statistical results** | `stat_omnibus.csv`, `stat_pairs.csv` (primary, gated), `stat_pairs_ungated.csv` (sensitivity), `stat_summary.csv`, `power_analysis.csv` |
| **Cross-metric agreement** | `cross_metric_agreement.csv`, `cross_metric_agreement_permodel.csv` |
| **Judge panel** (`judge_panel/`) | all 180 contradiction scores and 75 category votes from the blinded three-judge panel, with validation reports and Krippendorff's α |
| **FP16-vs-FP32 proof** (`fp16_vs_fp32/`) | the NLI scoring-model precision verification (≈ 4×10⁻⁵ shift; see note below) |
| **Provenance** (`provenance/`) | `master_hashes.json` — SHA-256 of the pinned `llama.cpp` build, its binaries, and every FP16 master and quantized GGUF file; `aci_bench_tui_codes.json` — the 66-TUI UMLS semantic-type filter used by the concept-set metric |

---

## Two precision topics — never conflate

This study involves two entirely separate uses of "FP16":

1. **The models under study.** The three SLMs are GGUF-quantized to
   Q6_K / Q5_K_M / Q4_K_M / Q3_K_M, all built from FP16 masters. This quantization is
   what the study measures.
2. **The evaluation tooling (a scoring detail).** Separately, the NLI scoring model
   (DeBERTa-v3-large) was run in FP16 for speed and verified against FP32. The
   `data/fp16_vs_fp32/` folder concerns **only this second topic** and has nothing to
   do with the SLM quantization. On matched 190-pair sampling the primary contradiction
   score differed by ≈ 4×10⁻⁵ (under 1×10⁻⁴); BERTScore and all other neural scoring
   ran at FP32.

---

## What is not included

- **The 24,000 generated notes** are not distributed here; every metric and statistic
  computed from them is in `data/`.
- **Model weights** (FP16 and quantized GGUF files, ~74 GB) are re-derivable from the
  Hugging Face repositories above with `code/scripts/download_models.py` and
  `code/scripts/quantize_all.py`; every file's SHA-256 is in
  `data/provenance/master_hashes.json`.
- **UMLS 2026AA** cannot be redistributed under the NLM licence. It needs a free UTS
  account; `code/scripts/umls_01`–`umls_04` build the QuickUMLS index from it.
- **ACI-Bench** is available from its upstream repository.
- **Raw judge text outputs** are not included; the parsed scores are in
  `data/judge_panel/`.

---

## Citation

```bibtex
@misc{clinical_slm_consistency_2026,
  title  = {Quantization Effects on Stochastic Consistency in Clinical Note Generation Using Small Language Models},
  year   = {2026},
  note   = {Code, analysis outputs and methodology documents},
  url    = {https://github.com/zephyrhasib/clinical-slm-consistency}
}
```

A `CITATION.cff` file is included so GitHub's **"Cite this repository"** button works.

## License

- **Data and documentation** (`data/`, `supplement/`, this README) are released under
  [CC BY 4.0](LICENSE): free to use with attribution.
- **Code** (`code/`) is released under the Apache License 2.0 — see
  [`code/LICENSE`](code/LICENSE).
- ACI-Bench, UMLS and the model weights remain under their own upstream licences.
