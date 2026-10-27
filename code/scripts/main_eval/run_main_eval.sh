#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation Pipeline (Stages 1–8 of 11)
#
# Runs ALL automated evaluation stages on the 24,000-row main dataset, then
# stops before the manual LLM-judge step. After judges score, run the
# companion script `run_main_panels.sh` (Stages 9–11) to finish.
#
# Stage map:
#   1. Combine 12 main CSVs → main_combined.csv (24,000 rows)
#   2. Pre-flight verify (UMLS + NLI + schema checks)
#   3. Metric pipeline — 7 metrics, ~30-40 h on RTX 5060 8 GB
#   4. Friedman omnibus + Kendall W
#   5. Pairwise gated (Wilcoxon-pratt + BCa + Holm)  ← PRIMARY
#   5b. Pairwise ungated (sensitivity counterpart, Issue 4)
#   6. Sample 60 NLI panel pairs (--scenario main)
#   7. Sample 25 BERTScore inspection pairs (--scenario main)
#   8. Export blinded judge-ready NLI CSV
#
# After this finishes:
#   • Share outputs/main_eval/nli_validation_pairs_blinded.csv with 3 judges
#   • Share outputs/main_eval/bertscore_inspection_pairs_blinded.csv with 3 judges
#   • Each judge produces a .txt → put 6 files in panel_raw_main/
#   • Then run: bash scripts/main_eval/run_main_panels.sh
#
# Run from project root:
#   bash scripts/main_eval/run_main_eval.sh
#
# Resume-safe: each python script's outputs are atomic; re-running the shell
# will re-do stages that already finished. Metric pipeline has NO mid-run
# resume — interrupting stage 3 loses progress on the current cell.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd ~/Research
source venv_metrics/bin/activate

OUT=outputs/main_eval
mkdir -p "$OUT" logs

ALT="${ALT:-auto}"
NO_GATE_FLAG=""
if [[ -n "${NO_GATE:-}" ]]; then
    NO_GATE_FLAG="--no-gate"
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/main_eval_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
PIPELINE_START=$(date +%s)

echo "════════════════════════════════════════════════════════════════════════"
echo "MAIN EVALUATION PIPELINE — Stages 1-8 (automated)"
echo "════════════════════════════════════════════════════════════════════════"
echo "  Started        : $START_TS"
echo "  Log            : $LOG"
echo "  ALT            : $ALT"
echo "  NO_GATE        : ${NO_GATE_FLAG:-<gating enabled>}"
echo "  Expected total : ~30-40 hours (dominated by NLI sentence-matrix step)"
echo ""

# Stage 1
echo ""
echo "[1/8] Combining 12 main CSVs ..."
T0=$(date +%s)
python scripts/main_eval/combine_main_csvs.py
T_COMBINE=$(($(date +%s) - T0))

# Stage 2
echo ""
echo "[2/8] Pre-flight verification ..."
T0=$(date +%s)
python scripts/main_eval/verify_combined.py
T_VERIFY=$(($(date +%s) - T0))

# Stage 3 — long step
echo ""
echo "[3/8] Metric pipeline (LONG step, 30-40 hours) ..."
T0=$(date +%s)
python scripts/eval/metric_pipeline.py \
    --generations "$OUT/main_combined.csv" \
    --out         "$OUT/metric_results.csv" \
    --out-nli     "$OUT/metric_results_nli.csv"
T_METRIC=$(($(date +%s) - T0))

# Stage 4
echo ""
echo "[4/8] Statistical omnibus (Friedman + Kendall's W) ..."
T0=$(date +%s)
python scripts/eval/stat_omnibus.py \
    --pairwise-metrics "$OUT/metric_results.csv" \
    --nli-metrics      "$OUT/metric_results_nli.csv" \
    --out              "$OUT/stat_omnibus.csv"
T_OMNIBUS=$(($(date +%s) - T0))

# Stage 5 — GATED PRIMARY
echo ""
echo "[5/8] Statistical pairwise (gated on Friedman, PRIMARY) ..."
T0=$(date +%s)
python scripts/eval/stat_pairwise.py \
    --pairwise-metrics "$OUT/metric_results.csv" \
    --nli-metrics      "$OUT/metric_results_nli.csv" \
    --omnibus          "$OUT/stat_omnibus.csv" \
    --summary          "$OUT/stat_summary.csv" \
    --pairs            "$OUT/stat_pairs.csv" \
    --alternative      "$ALT" \
    $NO_GATE_FLAG
T_PAIRWISE=$(($(date +%s) - T0))

# Stage 5b — UNGATED SENSITIVITY
echo ""
echo "[5b/8] Statistical pairwise (UN-GATED sensitivity, Issue 4) ..."
T0=$(date +%s)
python scripts/eval/stat_pairwise.py \
    --pairwise-metrics "$OUT/metric_results.csv" \
    --nli-metrics      "$OUT/metric_results_nli.csv" \
    --omnibus          "$OUT/stat_omnibus.csv" \
    --summary          "$OUT/stat_summary_ungated.csv" \
    --pairs            "$OUT/stat_pairs_ungated.csv" \
    --alternative      "$ALT" \
    --no-gate
T_PAIRWISE_UNGATED=$(($(date +%s) - T0))

# Stage 6 — NLI panel sampling (main scenario)
echo ""
echo "[6/8] Sampling 60 NLI panel pairs (--scenario main) ..."
T0=$(date +%s)
python scripts/validation/sample_nli_validation_pairs.py \
    --scenario    main \
    --nli-metrics "$OUT/metric_results_nli.csv" \
    --generations "$OUT/main_combined.csv" \
    --out         "$OUT/nli_validation_pairs.csv"
T_SAMPLE_NLI=$(($(date +%s) - T0))

# Stage 7 — BERTScore inspection sampling (main scenario)
echo ""
echo "[7/8] Sampling 25 BERTScore inspection pairs (--scenario main) ..."
T0=$(date +%s)
python scripts/validation/sample_bertscore_inspection_pairs.py \
    --scenario       main \
    --metric-results "$OUT/metric_results.csv" \
    --generations    "$OUT/main_combined.csv" \
    --un-blinded     "$OUT/bertscore_inspection_pairs.csv" \
    --blinded        "$OUT/bertscore_inspection_pairs_blinded.csv"
T_SAMPLE_BERT=$(($(date +%s) - T0))

# Stage 8 — blind NLI CSV for judges
echo ""
echo "[8/8] Exporting blinded NLI panel CSV ..."
T0=$(date +%s)
python scripts/validation/export_blinded_pairs.py \
    --in  "$OUT/nli_validation_pairs.csv" \
    --out "$OUT/nli_validation_pairs_blinded.csv"
T_BLIND=$(($(date +%s) - T0))

END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TOTAL=$(($(date +%s) - PIPELINE_START))

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "STAGES 1-8 COMPLETE"
echo "════════════════════════════════════════════════════════════════════════"
echo "  Started      : $START_TS"
echo "  Finished     : $END_TS"
echo "  Total elapsed: $((TOTAL / 3600))h $(((TOTAL % 3600) / 60))m $((TOTAL % 60))s"
echo ""
echo "Stage timings:"
printf "  %-32s %s\n" "Combine"             "${T_COMBINE}s"
printf "  %-32s %s\n" "Verify"              "${T_VERIFY}s"
printf "  %-32s %s\n" "Metric pipeline"     "$((T_METRIC / 3600))h $(((T_METRIC % 3600) / 60))m"
printf "  %-32s %s\n" "Omnibus"             "${T_OMNIBUS}s"
printf "  %-32s %s\n" "Pairwise (gated)"    "${T_PAIRWISE}s"
printf "  %-32s %s\n" "Pairwise (ungated)"  "${T_PAIRWISE_UNGATED}s"
printf "  %-32s %s\n" "NLI panel sampling"  "${T_SAMPLE_NLI}s"
printf "  %-32s %s\n" "BERTScore sampling"  "${T_SAMPLE_BERT}s"
printf "  %-32s %s\n" "Blind NLI export"    "${T_BLIND}s"

echo ""
echo "Outputs (in $OUT/):"
echo "  main_combined.csv                       (24,000 rows)"
echo "  main_combined_clean.csv                 (clean subset)"
echo "  metric_results.csv                      (1,200 cells × 4 pairwise metrics)"
echo "  metric_results_nli.csv                  (1,200 cells × 3 NLI aggregations)"
echo "  stat_omnibus.csv                        (21 rows; Friedman + Kendall W)"
echo "  stat_summary.csv / _ungated.csv         (descriptive)"
echo "  stat_pairs.csv                          (gated pairwise, PRIMARY)"
echo "  stat_pairs_ungated.csv                  (ungated sensitivity, Issue 4)"
echo "  nli_validation_pairs.csv                (60 pairs, audit trail)"
echo "  nli_validation_pairs_blinded.csv        (60 pairs, SHARE WITH JUDGES)"
echo "  bertscore_inspection_pairs.csv          (25 pairs, audit trail)"
echo "  bertscore_inspection_pairs_blinded.csv  (25 pairs, SHARE WITH JUDGES)"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "NEXT (manual): 3 judges score the panels via web UI"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  1. Share blinded CSVs with each judge in a fresh-context session"
echo "  2. Each judge produces:"
echo "     • An NLI scoring .txt (60 pairs × 1-5 score)"
echo "     • A BERTScore categorization .txt (25 pairs × A/B/C/D)"
echo "  3. Save 6 raw text files to panel_raw_main/:"
echo "       panel_raw_main/nli_claude.txt"
echo "       panel_raw_main/nli_gemini.txt"
echo "       panel_raw_main/nli_deepseek.txt"
echo "       panel_raw_main/bertscore_claude.txt"
echo "       panel_raw_main/bertscore_gemini.txt"
echo "       panel_raw_main/bertscore_deepseek.txt"
echo "  4. THEN run:  bash scripts/main_eval/run_main_panels.sh"
