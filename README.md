# Quantization Effects on Stochastic Consistency in Clinical Note Generation Using Small Language Models

**Supplementary material and analysis outputs** for the paper of the same title
(under review, 2026).

> Quantization compresses small language models for offline clinical use; however, its
> impact on stochastic consistency, that is, whether the same input generates the same
> output on each run, is not well understood. This study evaluates three sub-4B models
> across four GGUF compression levels from Q6 to Q3 using five reference free metrics.
> A quantization effect was observed in all 21 model-metric families, with varying
> results for each model. Gemma led all content metrics and saw only a slight drop in
> BERTScore (0.789 to 0.765). Qwen ranked the worst on content metrics and showed the
> biggest fall on the same measure, from 0.652 to 0.565. The flat curve for Phi-4 was
> not due to robustness but because it already loops on almost half of its outputs at
> full precision. No single bit width or metric was proven safe across all models.
> Contradiction was the least reliable despite being the most trusted, diverging from
> the other four axes in two of the three models. As factual content was suppressed,
> numerical agreement increased, which single run accuracy checks did not detect. These
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

### `supplement/` — methodology documents (cited from the paper)

| File | What the paper points here for |
|---|---|
| [`contamination.md`](supplement/contamination.md) | Pre-generation contamination audit: MinHash corpus-overlap test (Qwen) and verbatim-memorization probe (Gemma, Phi-4), all constants, and the 0/100 result. |
| [`generation_config.md`](supplement/generation_config.md) | Full generation protocol: the zero-shot instruction, the three native chat templates (verbatim), all decoding parameters, the complete stop-sequence lists (including Qwen's nine role-injection stops), Phi-4 looping/truncation handling, the quality-flag regexes, and sampling. |
| [`statistics.md`](supplement/statistics.md) | Friedman + Kendall's W omnibus, Wilcoxon signed-rank (Pratt) pairwise, BCa bootstrap with the reproducible per-pair seed formula, Holm-Bonferroni, the two analysis streams, power analysis, and cross-metric agreement. |
| [`metrics.md`](supplement/metrics.md) | The five consistency metrics (ROUGE-L, BERTScore, UMLS concept-set F1, numerical-attribute Jaccard, NLI contradiction), the three-judge panel, and the FP16-vs-FP32 verification of the NLI scoring model. |

### `data/` — computed outputs behind every table and figure

All values are the actual computed outputs of the study. Nothing is regenerated or
reconstructed. A column-by-column dictionary is in [`data/README.md`](data/README.md).

| Group | Files |
|---|---|
| **Headline tables** | `table3_per_cell.csv` (Table III), `table4_effects.csv` (Table IV) |
| **Full metric results** (1,200 cells) | `metric_results.csv`, `metric_results_nli.csv` |
| **Statistical results** | `stat_omnibus.csv`, `stat_pairs.csv` (primary, gated), `stat_pairs_ungated.csv` (sensitivity), `stat_summary.csv`, `power_analysis.csv` |
| **Cross-metric agreement** | `cross_metric_agreement.csv`, `cross_metric_agreement_permodel.csv` |
| **Judge panel** (`judge_panel/`) | all 180 contradiction scores and 75 category votes from the blinded three-judge panel, with validation reports and Krippendorff's α |
| **FP16-vs-FP32 proof** (`fp16_vs_fp32/`) | the NLI scoring-model precision verification (≈ 4×10⁻⁵ shift; see note below) |
| **Provenance** (`provenance/`) | `master_hashes.json` — SHA-256 of the pinned `llama.cpp` build, its binaries, and every FP16 master and quantized GGUF file; `aci_bench_tui_codes.json` — the 66-TUI UMLS semantic-type filter used by the concept-set metric |

---

## Two precision topics — never conflate

This study involves two entirely separate uses of "FP16":

1. **The models under study (the subject of the paper).** The three SLMs are
   GGUF-quantized to Q6_K / Q5_K_M / Q4_K_M / Q3_K_M, all built from FP16 masters.
   This is what the paper measures.
2. **The evaluation tooling (a scoring detail).** Separately, the NLI scoring model
   (DeBERTa-v3-large) was run in FP16 for speed and verified against FP32. The
   `data/fp16_vs_fp32/` folder concerns **only this second topic** and has nothing to
   do with the SLM quantization the paper studies. On matched 190-pair sampling the
   primary contradiction score differed by ≈ 4×10⁻⁵ (under 1×10⁻⁴); BERTScore and all
   other neural scoring ran at FP32.

---

## What is intentionally not included (yet)

The runnable source code — the generation harness, the metric pipeline, and the
statistical scripts — is **withheld until the paper is published**. The four
methodology documents specify every parameter, template, stop sequence, threshold,
regex, algorithm, and formula in enough detail to interpret and independently
reproduce the study, and the released data files are the complete computed outputs
behind every reported number. The raw 24,000 generated notes are also not
distributed here.

The full code will be released in this repository on publication.

---

## Citation

The paper is under review, so author details are withheld here for now. Until it is
published, please cite this repository:

```bibtex
@misc{clinical_slm_consistency_2026,
  title  = {Quantization Effects on Stochastic Consistency in Clinical Note Generation Using Small Language Models},
  year   = {2026},
  note   = {Supplementary material. Under review; author details withheld.},
  url    = {https://github.com/zephyrhasib/clinical-slm-consistency}
}
```

A `CITATION.cff` file is included so GitHub's **"Cite this repository"** button works.
Both will be updated with the full author list and the published reference on
acceptance.

## License

Data and documentation in this repository are released under
[CC BY 4.0](LICENSE): free to use with attribution.
