#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation, RESUME-FROM-NLI variant (Stages 3-NLI … 8)
#
# Use this INSTEAD of run_main_eval.sh when the 4 non-NLI metrics are ALREADY
# computed (outputs/main_eval/metric_results.csv exists) and you only need to
# run the NLI step + statistics + panel sampling.
#
# What it does NOT do (vs run_main_eval.sh):
#   - Stage 1 combine_main_csvs.py   — SKIPPED (main_combined.csv already exists)
#   - Stage 2 verify_combined.py     — SKIPPED (replaced by a light existence check)
#   - Stage 3 BERTScore/ROUGE/MEDCON/numeric — NOT recomputed. metric_results.csv
#     is preserved untouched (the NLI step runs with --nli-only).
#
# What it DOES (in order):
#   3.  eval/metric_pipeline.py --nli-only  → metric_results_nli.csv   (LONG, ~2-3 days)
#       fp16 + full 190 pairs + per-cell resumable checkpoint
#       (metric_results_nli.partial.csv). Re-run this script to RESUME.
#   4.  eval/stat_omnibus.py        → stat_omnibus.csv
#   5.  eval/stat_pairwise.py       → stat_pairs.csv          (gated, PRIMARY)
#   5b. eval/stat_pairwise.py --no-gate → stat_pairs_ungated.csv (sensitivity)
#   6.  sample_nli_validation_pairs.py --scenario main → 60 pairs
#   7.  sample_bertscore_inspection_pairs.py --scenario main → 25 pairs
#   8.  export_blinded_pairs.py     → judge-ready blinded NLI CSV
#
# Resume-safe: the NLI step checkpoints per cell. If interrupted (Ctrl-C, crash,
# power loss), just re-run this script — it skips completed cells and continues.
# Stages 4-8 are cheap and simply overwrite their outputs on re-run.
#
# Run from project root:
#   bash scripts/main_eval/run_main_eval_from_nli.sh
# Optional env: ALT=two-sided  NO_GATE=1
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
LOG="logs/main_eval_from_nli_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
PIPELINE_START=$(date +%s)

echo "════════════════════════════════════════════════════════════════════════"
echo "MAIN EVALUATION — RESUME FROM NLI (Stages 3-NLI … 8)"
echo "════════════════════════════════════════════════════════════════════════"
echo "  Started        : $START_TS"
echo "  Log            : $LOG"
echo "  ALT            : $ALT"
echo "  NO_GATE        : ${NO_GATE_FLAG:-<gating enabled>}"
echo "  Expected total : ~2.2-3.2 days (NLI fp16+190, resumable) + ~minutes for stats"
echo ""

# ── Pre-flight: required inputs must already exist ───────────────────────────
echo "[pre-flight] checking required inputs ..."
if [[ ! -f "$OUT/main_combined.csv" ]]; then
    echo "ERROR: $OUT/main_combined.csv not found."
    echo "       Run Stage 1 first:  python scripts/main_eval/combine_main_csvs.py"
    echo "       (or use scripts/main_eval/run_main_eval.sh for the full pipeline)."
    exit 1
fi
if [[ ! -f "$OUT/metric_results.csv" ]]; then
    echo "ERROR: $OUT/metric_results.csv (the 4 non-NLI metrics) not found."
    echo "       This resume-from-NLI script assumes the 4 metrics are DONE."
    echo "       If they are not, use the full scripts/main_eval/run_main_eval.sh."
    exit 1
fi
echo "  [OK] main_combined.csv      ($(wc -l < "$OUT/main_combined.csv") rows)"
echo "  [OK] metric_results.csv     ($(wc -l < "$OUT/metric_results.csv") rows) — will NOT be recomputed"
if [[ -f "$OUT/metric_results_nli.partial.csv" ]]; then
    echo "  [INFO] resume checkpoint present: metric_results_nli.partial.csv "\
"($(( $(wc -l < "$OUT/metric_results_nli.partial.csv") - 1 )) cells done)"
fi
echo ""

# ── Stage 3 (NLI only) — LONG, resumable ─────────────────────────────────────
echo "[3/8] NLI sentence-matrix ONLY (fp16 + 190 pairs, resumable) — LONG ..."
T0=$(date +%s)
python scripts/eval/metric_pipeline.py \
    --nli-only \
    --generations "$OUT/main_combined.csv" \
    --out         "$OUT/metric_results.csv" \
    --out-nli     "$OUT/metric_results_nli.csv"
T_NLI=$(($(date +%s) - T0))

# ── Stage 4 — Friedman omnibus ───────────────────────────────────────────────
echo ""
echo "[4/8] Statistical omnibus (Friedman + Kendall's W) ..."
T0=$(date +%s)
python scripts/eval/stat_omnibus.py \
    --pairwise-metrics "$OUT/metric_results.csv" \
    --nli-metrics      "$OUT/metric_results_nli.csv" \
    --out              "$OUT/stat_omnibus.csv"
T_OMNIBUS=$(($(date +%s) - T0))

# ── Stage 5 — pairwise gated (PRIMARY) ───────────────────────────────────────
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

# ── Stage 5b — pairwise ungated (SENSITIVITY) ────────────────────────────────
echo ""
echo "[5b/8] Statistical pairwise (UN-GATED sensitivity) ..."
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

# ── Stage 6 — NLI panel sampling (main scenario) ─────────────────────────────
echo ""
echo "[6/8] Sampling 60 NLI panel pairs (--scenario main) ..."
T0=$(date +%s)
python scripts/validation/sample_nli_validation_pairs.py \
    --scenario    main \
    --nli-metrics "$OUT/metric_results_nli.csv" \
    --generations "$OUT/main_combined.csv" \
    --out         "$OUT/nli_validation_pairs.csv"
T_SAMPLE_NLI=$(($(date +%s) - T0))

# ── Stage 7 — BERTScore inspection sampling (main scenario) ──────────────────
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

# ── Stage 8 — blind NLI CSV for judges ───────────────────────────────────────
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
echo "STAGES 3-NLI … 8 COMPLETE"
echo "════════════════════════════════════════════════════════════════════════"
echo "  Started      : $START_TS"
echo "  Finished     : $END_TS"
echo "  Total elapsed: $((TOTAL / 3600))h $(((TOTAL % 3600) / 60))m $((TOTAL % 60))s"
echo ""
echo "Stage timings:"
printf "  %-32s %s\n" "NLI (fp16+190)"      "$((T_NLI / 3600))h $(((T_NLI % 3600) / 60))m"
printf "  %-32s %s\n" "Omnibus"             "${T_OMNIBUS}s"
printf "  %-32s %s\n" "Pairwise (gated)"    "${T_PAIRWISE}s"
printf "  %-32s %s\n" "Pairwise (ungated)"  "${T_PAIRWISE_UNGATED}s"
printf "  %-32s %s\n" "NLI panel sampling"  "${T_SAMPLE_NLI}s"
printf "  %-32s %s\n" "BERTScore sampling"  "${T_SAMPLE_BERT}s"
printf "  %-32s %s\n" "Blind NLI export"    "${T_BLIND}s"

echo ""
echo "Outputs (in $OUT/):"
echo "  metric_results.csv                      (4 metrics — PRESERVED, untouched)"
echo "  metric_results_nli.csv                  (3 NLI aggregations — NEW)"
echo "  stat_omnibus.csv / stat_pairs.csv / stat_pairs_ungated.csv / stat_summary*.csv"
echo "  nli_validation_pairs.csv + _blinded     (60 pairs)"
echo "  bertscore_inspection_pairs.csv + _blinded (25 pairs)"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "NEXT (manual): 3 judges score the panels via web UI"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  1. Share both blinded CSVs with each judge in a fresh-context session"
echo "  2. Save 6 raw text files to panel_raw_main/:"
echo "       nli_{claude,gemini,deepseek}.txt   bertscore_{claude,gemini,deepseek}.txt"
echo "  3. THEN run:  bash scripts/main_eval/run_main_panels.sh"
