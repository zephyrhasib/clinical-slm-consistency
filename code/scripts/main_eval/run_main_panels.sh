#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Main Panel Aggregation (Stages 9-11 of 11)
#
# Runs AFTER the 3 LLM judges have scored the 60 NLI pairs + 25 BERTScore
# pairs produced by stages 1-8 (run_main_eval.sh).
#
# Pre-condition: panel_raw_main/ contains the 6 judge text files:
#   nli_claude.txt        nli_gemini.txt        nli_deepseek.txt
#   bertscore_claude.txt  bertscore_gemini.txt  bertscore_deepseek.txt
#
# Stage map:
#   9. Parse raw judge text → 2 CSVs (180 NLI scores + 75 BERT)
#   10. Aggregate NLI panel: Spearman ρ × 3 aggregations + Krippendorff α
#       → reports the primary NLI metric
#   11. Aggregate BERTScore panel: A/B/C/D majority vote + Krippendorff α
#       → confirms BERTScore face validity at main scale
#
# Run from project root:
#   bash scripts/main_eval/run_main_panels.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd ~/Research
source venv_metrics/bin/activate

OUT=outputs/main_eval
RAW=panel_raw_main

# Check the 6 raw files exist before doing anything
for j in claude gemini deepseek; do
    for track in nli bertscore; do
        f="$RAW/${track}_${j}.txt"
        if [[ ! -f "$f" ]]; then
            echo "ERROR: $f not found."
            echo "       Have all 3 judges scored both panels?"
            echo "       Expected files in $RAW/:"
            echo "         nli_claude.txt / nli_gemini.txt / nli_deepseek.txt"
            echo "         bertscore_claude.txt / bertscore_gemini.txt / bertscore_deepseek.txt"
            exit 1
        fi
    done
done

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/main_panels_${TS}.log"
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
PIPELINE_START=$(date +%s)

echo "════════════════════════════════════════════════════════════════════════"
echo "MAIN PANEL AGGREGATION — Stages 9-11"
echo "════════════════════════════════════════════════════════════════════════"
echo "  Started        : $START_TS"
echo "  Log            : $LOG"
echo "  Expected total : ~30 seconds"
echo ""

# Stage 9 — parse 6 raw judge text files
echo "[9/11] Parsing 6 raw judge text files (--scenario main: 60 NLI + 25 BERT) ..."
T0=$(date +%s)
python scripts/validation/parse_panel_outputs.py \
    --scenario  main \
    --raw-dir   "$RAW" \
    --out-nli   "$OUT/nli_panel_scores.csv" \
    --out-bert  "$OUT/bertscore_panel_categories.csv"
T_PARSE=$(($(date +%s) - T0))

# Stage 10 — NLI aggregator
echo ""
echo "[10/11] Aggregating NLI panel (Spearman × 3 aggregations + Krippendorff α) ..."
T0=$(date +%s)
python scripts/validation/aggregate_panel_scores.py \
    --panel-scores   "$OUT/nli_panel_scores.csv" \
    --sampled-pairs  "$OUT/nli_validation_pairs.csv" \
    --nli-metrics    "$OUT/metric_results_nli.csv" \
    --out            "$OUT/nli_panel_validation_report.csv"
T_NLI=$(($(date +%s) - T0))

# Stage 11 — BERTScore aggregator
echo ""
echo "[11/11] Aggregating BERTScore panel (A/B/C/D majority vote + Krippendorff α) ..."
T0=$(date +%s)
python scripts/validation/aggregate_bertscore_categories.py \
    --panel-categories "$OUT/bertscore_panel_categories.csv" \
    --sampled-pairs    "$OUT/bertscore_inspection_pairs.csv" \
    --out              "$OUT/bertscore_validation_report.csv"
T_BERT=$(($(date +%s) - T0))

END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TOTAL=$(($(date +%s) - PIPELINE_START))

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "PANEL AGGREGATION COMPLETE"
echo "════════════════════════════════════════════════════════════════════════"
echo "  Started      : $START_TS"
echo "  Finished     : $END_TS"
echo "  Total elapsed: ${TOTAL}s"
echo ""
echo "Stage timings:"
printf "  %-28s %s\n" "Parse"               "${T_PARSE}s"
printf "  %-28s %s\n" "NLI aggregation"     "${T_NLI}s"
printf "  %-28s %s\n" "BERTScore aggreg."   "${T_BERT}s"
echo ""
echo "Outputs (in $OUT/):"
echo "  nli_panel_scores.csv                    (180 rows: 60 pairs × 3 judges)"
echo "  bertscore_panel_categories.csv          (75 rows: 25 pairs × 3 judges)"
echo "  nli_panel_validation_report.csv         (Spearman × 3 aggs + α; primary flag)"
echo "  bertscore_validation_report.csv         (A/B/C/D distribution + α)"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "MAIN EVALUATION FULLY COMPLETE"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "Open the NLI report to confirm (or update) the primary aggregation."
echo "Open the BERTScore report to confirm face-validity finding."
echo ""
echo "Then: interpretation and write-up."
