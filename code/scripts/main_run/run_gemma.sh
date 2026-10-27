#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Main Run, Gemma-3-4B-it (all 4 quants)
#
# Runs 4 Gemma cells sequentially: Q6 → Q5 → Q4 → Q3.
# Estimated wallclock: ~14.7 hours total (4 × ~3.7 h per cell).
#
# Each cell is 2000 generations (100 patients × 20 seeds).
# Resume-safe: each python script skips (input_id, seed) pairs already written
# to its CSV, so this shell script can be safely Ctrl-C'd and re-run.
#
# Run from project root:
#     bash scripts/main_run/run_gemma.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd ~/Research
source venv/bin/activate

mkdir -p logs outputs/main_run
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/main_run_gemma_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BATCH_START=$(date +%s)

echo "================================================================"
echo "MAIN RUN — Gemma-3-4B-it (4 quants)"
echo "================================================================"
echo "  Started        : $START_TS"
echo "  Log            : $LOG"
echo "  Expected total : ~14.7 hours"
echo "  Order          : Q6_K → Q5_K_M → Q4_K_M → Q3_K_M"
echo "  Resume         : safe (Ctrl-C OK; partial saves on every generation)"
echo ""

SCRIPTS=(
    "scripts/main_run/gen_gemma_Q6_K_main.py"
    "scripts/main_run/gen_gemma_Q5_K_M_main.py"
    "scripts/main_run/gen_gemma_Q4_K_M_main.py"
    "scripts/main_run/gen_gemma_Q3_K_M_main.py"
)

declare -A TIMINGS
for SCRIPT in "${SCRIPTS[@]}"; do
    CELL=$(basename "$SCRIPT" .py)
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  STARTING: $CELL"
    echo "  Time     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "════════════════════════════════════════════════════════════════"
    T0=$(date +%s)
    python "$SCRIPT"
    T_CELL=$(($(date +%s) - T0))
    TIMINGS[$CELL]=$T_CELL
    echo ""
    echo "  FINISHED $CELL in $((T_CELL / 60))m $((T_CELL % 60))s"
done

END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TOTAL=$(($(date +%s) - BATCH_START))

echo ""
echo "================================================================"
echo "GEMMA BATCH COMPLETE"
echo "================================================================"
echo "  Started      : $START_TS"
echo "  Finished     : $END_TS"
echo "  Total elapsed: $((TOTAL / 3600))h $(((TOTAL % 3600) / 60))m $((TOTAL % 60))s"
echo ""
echo "Per-cell timings:"
for SCRIPT in "${SCRIPTS[@]}"; do
    CELL=$(basename "$SCRIPT" .py)
    T=${TIMINGS[$CELL]}
    printf "  %-32s  %2dh %2dm %2ds\n" "$CELL" $((T / 3600)) $(((T % 3600) / 60)) $((T % 60))
done

echo ""
echo "Outputs:"
ls -la outputs/main_run/gen_gemma_*_main.csv 2>/dev/null || echo "  (none — check log for errors)"
