#!/bin/bash
# Pilot Evaluation Pipeline (post-audit, v2)
#
# Sequential testing protocol:
#   combine → verify → metric_pipeline → stat_omnibus (Friedman + Kendall W)
#   → stat_pairwise (Wilcoxon-pratt + BCa + Holm, gated on Friedman significance)
#
# All outputs land in outputs/pilot_eval/.
#
# Run from project root:
#   bash scripts/pilot_eval/run_pilot_eval.sh
#
# To override the directional one-sided alternative with two-sided (e.g. for
# a more conservative posture), set ALT=two-sided before invocation:
#   ALT=two-sided bash scripts/pilot_eval/run_pilot_eval.sh
#
# To bypass Friedman gating (run all pairs unconditionally), set NO_GATE=1:
#   NO_GATE=1 bash scripts/pilot_eval/run_pilot_eval.sh

set -euo pipefail

cd ~/Research
source venv_metrics/bin/activate

OUT=outputs/pilot_eval
mkdir -p "$OUT"

ALT="${ALT:-auto}"            # auto = directional one-sided; override via env
NO_GATE_FLAG=""
if [[ -n "${NO_GATE:-}" ]]; then
    NO_GATE_FLAG="--no-gate"
fi

START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
PIPELINE_START=$(date +%s)

echo "================================================================"
echo "PILOT EVALUATION PIPELINE  —  started $START_TS"
echo "================================================================"
echo "  ALT (Wilcoxon alternative) : $ALT"
echo "  NO_GATE flag               : ${NO_GATE_FLAG:-<gating enabled>}"

# Step 1 — combine the 12 pilot CSVs into one 960-row file
echo ""
echo "[1/5] Combining 12 pilot CSVs ..."
T0=$(date +%s)
python scripts/pilot_eval/combine_pilot_csvs.py
T_COMBINE=$(($(date +%s) - T0))

# Step 2 — pre-flight verification (validates UMLS + NLI load)
echo ""
echo "[2/5] Pre-flight verification ..."
T0=$(date +%s)
python scripts/pilot_eval/verify_combined.py
T_VERIFY=$(($(date +%s) - T0))

# Step 3 — full metric pipeline (~2 hours)
echo ""
echo "[3/5] Metric pipeline (long step, 2-3 hours) ..."
T0=$(date +%s)
python scripts/eval/metric_pipeline.py \
    --generations "$OUT/pilot_combined.csv" \
    --out         "$OUT/metric_results.csv" \
    --out-nli     "$OUT/metric_results_nli.csv"
T_METRIC=$(($(date +%s) - T0))

# Step 4 — Friedman omnibus (Issue 10)
echo ""
echo "[4/5] Statistical omnibus (Friedman + Kendall's W) ..."
T0=$(date +%s)
python scripts/eval/stat_omnibus.py \
    --pairwise-metrics "$OUT/metric_results.csv" \
    --nli-metrics      "$OUT/metric_results_nli.csv" \
    --out              "$OUT/stat_omnibus.csv"
T_OMNIBUS=$(($(date +%s) - T0))

# Step 5 — Pairwise (Wilcoxon-pratt + BCa + Holm), gated on Friedman significance
echo ""
echo "[5/5] Statistical pairwise (gated on Friedman) ..."
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

# Step 5b — Ungated pairwise (Issue 4 sensitivity analysis)
# Pre-specified counterpart to the gated primary. Reports Holm-corrected
# pairwise tests for ALL (model, metric) families regardless of Friedman
# significance. Addresses the Westfall et al. 2011 critique that omnibus
# gating reduces power. Convergent significance across both runs strengthens
# the substantive claim; divergent results are reported transparently.
echo ""
echo "[5b/5] Statistical pairwise (UN-GATED sensitivity analysis) ..."
T0=$(date +%s)
python scripts/eval/stat_pairwise.py \
    --pairwise-metrics "$OUT/metric_results.csv" \
    --nli-metrics      "$OUT/metric_results_nli.csv" \
    --omnibus          "$OUT/stat_omnibus.csv" \
    --summary          "$OUT/stat_summary_ungated.csv" \
    --pairs            "$OUT/stat_pairs_ungated.csv" \
    --alternative      "$ALT" \
    --no-gate

END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TOTAL=$(($(date +%s) - PIPELINE_START))

echo ""
echo "================================================================"
echo "PIPELINE COMPLETE"
echo "================================================================"
echo "  Started     : $START_TS"
echo "  Finished    : $END_TS"
echo "  Total elapsed: $((TOTAL / 60))m $((TOTAL % 60))s"
echo ""
echo "Stage timings:"
echo "  Combine            : ${T_COMBINE}s"
echo "  Verify             : ${T_VERIFY}s"
echo "  Metric             : $((T_METRIC / 60))m $((T_METRIC % 60))s"
echo "  Omnibus            : ${T_OMNIBUS}s"
echo "  Pairwise (gated)   : ${T_PAIRWISE}s"
echo "  Pairwise (ungated) : ${T_PAIRWISE_UNGATED}s"
echo ""
echo "Outputs:"
echo "  $OUT/pilot_combined.csv         (960 rows)"
echo "  $OUT/pilot_combined_clean.csv   (clean subset)"
echo "  $OUT/metric_results.csv         (240 cells × bertscore/rouge/medcon/numeric + n_skipped)"
echo "  $OUT/metric_results_nli.csv     (240 cells × NLI 3-aggregations + n_skipped_pairs)"
echo "  $OUT/stat_omnibus.csv           (Friedman + Kendall W per (model, metric))"
echo "  $OUT/stat_summary.csv           (per-condition descriptive, gated context)"
echo "  $OUT/stat_pairs.csv             (Wilcoxon-pratt + BCa + Holm, GATED primary)"
echo "  $OUT/stat_summary_ungated.csv   (per-condition descriptive, ungated)"
echo "  $OUT/stat_pairs_ungated.csv     (Holm without Friedman gate, SENSITIVITY)"
