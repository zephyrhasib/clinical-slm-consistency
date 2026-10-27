# Code

Every script used in the study, from dataset preparation to the final figures. The
computed outputs these scripts produced are in [`../data/`](../data/), and the
methodology they implement is documented in [`../supplement/`](../supplement/).

```
code/
├── LICENSE                          Apache-2.0
├── requirements-venv.txt            generation environment (pinned)
├── requirements-venv_metrics.txt    evaluation environment (pinned)
├── hashes/llama_cpp_commit.txt      pinned llama.cpp commit
└── scripts/
    ├── prep_dataset.py, audit_dataset.py          dataset preparation and quality audit
    ├── audit_01 … audit_04, verify_audit.py       contamination audit
    ├── umls_01 … umls_04, apply_quickumls_py312_patches.py   UMLS / QuickUMLS setup
    ├── download_models.py, quantize_all.py        model download and quantization
    ├── make_demo_eval_data.py, verify_metric_stack.py        metric-stack self-test
    ├── make_summary_tables.py                     per-condition means and effect-size tables
    ├── pilot_run/      12 generation scripts, pilot (20 pilot encounters)
    ├── main_run/       12 generation scripts, main (100 encounters × 20 seeds) + runners
    ├── pilot_eval/     combine + verify + run the evaluation on the pilot
    ├── main_eval/      combine + verify + run the evaluation on the main run
    ├── eval/           metric pipeline, statistics, cross-metric agreement, power, NLI precision proofs
    ├── validation/     judge-panel sampling, blinding, parsing and aggregation
    └── figures/        figure data and plotting scripts
```

---

## What each stage does

| Stage | Script(s) | Produces |
|---|---|---|
| **1. Dataset** | `prep_dataset.py` | Stratified, disjoint 100 main + 20 pilot ACI-Bench encounters (seed 42), `main_inputs.csv`, `pilot_inputs.csv`, `all_references.csv`, `stratification_metadata.json` |
| | `audit_dataset.py` | Quality checks on the stratified sets |
| **2. Contamination audit** | `audit_01_download_training_data.py` | Qwen-MediCare-BD's two public fine-tuning corpora |
| | `audit_02_model_a_minhash.py` | MinHash LSH overlap test for Qwen |
| | `audit_03_models_bc_probe.py` | 50-word-prefix memorisation probe for Gemma and Phi-4 |
| | `audit_04_summary.py`, `verify_audit.py` | Final decision (0/100 flagged) and a check of the audit outputs |
| **3. UMLS** | `umls_01_extract_meta.py` → `umls_04_smoke_test.py` | MRCONSO/MRSTY extraction, the ACI-Bench 66-TUI filter, the QuickUMLS index, and a smoke test |
| **4. Models** | `download_models.py` | Hugging Face weights for the three models |
| | `quantize_all.py` | Q6_K / Q5_K_M / Q4_K_M / Q3_K_M GGUF files from the FP16 masters, with SHA-256 checks before and after |
| **5. Self-test** | `make_demo_eval_data.py`, `verify_metric_stack.py` | Synthetic data and a PASS/FAIL run of every metric and statistical component |
| **6. Generation** | `pilot_run/gen_*_test.py` | 960 pilot generations |
| | `main_run/run_all_main.sh` → `run_{qwen,gemma,phi4}.sh` → `gen_*_main.py` | 24,000 main generations (12 cells × 100 encounters × 20 seeds), resume-safe |
| **7. Evaluation** | `main_eval/run_main_eval.sh` | Combine → verify → `eval/metric_pipeline.py` (pairwise between seeds, reference-free) → `eval/stat_omnibus.py` (Friedman + Kendall's W) → `eval/stat_pairwise.py` (Wilcoxon-Pratt + BCa + Holm; gated primary and ungated sensitivity) → panel sampling and blinding |
| | `main_eval/run_main_eval_from_nli.sh` | Same, resuming from the NLI step when the other metrics already exist |
| **8. Judge panel** | `main_eval/run_main_panels.sh` → `validation/parse_panel_outputs.py`, `aggregate_panel_scores.py`, `aggregate_bertscore_categories.py` | 180 contradiction scores and 75 category votes, Spearman ρ against the three NLI aggregations, Krippendorff's α |
| | `validation/nli_panel_permodel.py` | Per-model panel breakdown |
| **9. Agreement and power** | `eval/cross_metric_agreement.py`, `eval/power_analysis.py` | Spearman agreement between the five metric axes; simulated minimum detectable effect |
| **10. NLI precision check** | `eval/nli_subsample_proof.py`, `nli_batch_speed_proof.py`, `nli_differential_proof.py` | FP16-vs-FP32 and batch-size checks of the NLI scorer (`data/fp16_vs_fp32/`) |
| **11. Tables and figures** | `make_summary_tables.py`, `figures/build_figure_data.py`, `figures/fig_*.py` | `table3_per_cell.csv`, `table4_effects.csv`, and the dose-response, cross-metric agreement, numerical-suppression, distribution and generation-failure figures |

---

## Environment

Tested on Ubuntu 24.04 under WSL2, Python 3.12, CUDA 12.8, NVIDIA RTX 5060 8 GB
(SM_120).

Two virtual environments are used, each pinned in its own requirements file:

| Environment | Used for | Why separate |
|---|---|---|
| `venv` (`requirements-venv.txt`) | model download, dataset preparation, contamination audit, quantization and generation | Generation itself runs in the `llama.cpp` binary; Python only drives it. |
| `venv_metrics` (`requirements-venv_metrics.txt`) | every metric, statistic, panel aggregation, table and figure | QuickUMLS 1.4.2 and spaCy 3.7 need `numpy` 1.26 and `transformers` 4.35, which conflict with the generation stack. |

All 24,000 outputs are scored in the same `venv_metrics` environment, so the split
cannot differ between models or quantization levels.

---

## Setup

The scripts expect a project root at `~/Research` (the shell runners `cd ~/Research`,
and `verify_audit.py`, `eval/nli_batch_speed_proof.py` and `eval/nli_differential_proof.py`
set `ROOT` to that path; edit those lines if your project lives elsewhere). Copy
`code/scripts/`, `code/hashes/` and the two requirements files there, then:

```bash
cd ~/Research

# 1. Two virtual environments
python3.12 -m venv venv
venv/bin/pip install -r requirements-venv.txt --extra-index-url https://download.pytorch.org/whl/cu128
python3.12 -m venv venv_metrics
venv_metrics/bin/pip install -r requirements-venv_metrics.txt --extra-index-url https://download.pytorch.org/whl/cu128
source venv_metrics/bin/activate
python scripts/apply_quickumls_py312_patches.py     # one-time QuickUMLS fix for Python 3.12
deactivate

# 2. llama.cpp at the pinned build (tag b9283, commit bbce619)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout bbce619adb409880fb6db850a1c5a5f36a4dc7b1
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
cd ..

# 3. Provenance files from this repository
cp <repo>/data/provenance/master_hashes.json hashes/

# 4. ACI-Bench
git clone https://github.com/wyim/aci-bench data/aci-bench-main
mkdir -p data/raw
for f in train valid clinicalnlp_taskB_test1 clinicalnlp_taskC_test2 clef_taskC_test3; do
  cp data/aci-bench-main/data/challenge_data/$f.csv data/raw/
done
```

**UMLS.** Create a free UTS account at <https://uts.nlm.nih.gov>, download the UMLS
2026AA Metathesaurus (`2026aa-1-meta.nlm`) into `umls/`, then run `umls_01` to
`umls_04` in `venv_metrics`. `umls_02` writes `umls/aci_bench_tui_codes.json`; the copy
in `data/provenance/` lets you check it matches.

**Models.** Accept the Gemma licence on Hugging Face and run `huggingface-cli login`,
then `python scripts/download_models.py`. Convert each model to an FP16 GGUF in
`models/f16/` with `llama.cpp/convert_hf_to_gguf.py --outtype f16`, using the file
names in `master_hashes.json`. Then run `python scripts/quantize_all.py` (in `venv`):
it checks the llama.cpp commit, the `llama-quantize` binary and every FP16 file against
`master_hashes.json`, stops on any mismatch, and writes the 12 quantized files to
`models/quantized/`. The quantized models are needed before the contamination probe.

---

## Run order

```bash
cd ~/Research

# Dataset and contamination audit (venv)
source venv/bin/activate
python scripts/prep_dataset.py
python scripts/audit_dataset.py
python scripts/audit_01_download_training_data.py
python scripts/audit_02_model_a_minhash.py
python scripts/audit_03_models_bc_probe.py
python scripts/audit_04_summary.py
python scripts/verify_audit.py
deactivate

# Metric-stack self-test (venv_metrics)
venv_metrics/bin/python scripts/make_demo_eval_data.py
venv_metrics/bin/python scripts/verify_metric_stack.py

# Main generation, ~38 h on one RTX 5060
bash scripts/main_run/run_all_main.sh

# Main evaluation (stages 1–8), then the judge panel (stages 9–11)
bash scripts/main_eval/run_main_eval.sh
#   → score the two blinded CSVs with the three judges, put the six text files
#     in panel_raw_main/ (prompts and rubrics: supplement/judge_panel.md)
bash scripts/main_eval/run_main_panels.sh

# Agreement, power, tables and figures (venv_metrics)
source venv_metrics/bin/activate
python scripts/eval/cross_metric_agreement.py
python scripts/eval/power_analysis.py
python scripts/make_summary_tables.py
python scripts/figures/build_figure_data.py
for f in scripts/figures/fig_*.py; do python "$f"; done
```

Outputs go to `outputs/` (`main_run/`, `main_eval/`, `figures/`). The pilot follows the
same pattern with `pilot_run/` and `pilot_eval/run_pilot_eval.sh`.

Exact SHA-256 matches for the quantized files are only expected with the same
`llama.cpp` commit. Each generation script checks the commit, the `llama-completion`
binary and its GGUF file against `master_hashes.json` before it starts, and writes a
metadata JSON next to its output CSV.
