#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Main Run, Qwen-MediCare-BD (all 4 quants)
#
# Runs 4 Qwen cells sequentially: Q6 → Q5 → Q4 → Q3.
# Estimated wallclock: ~8.3 hours total (4 × ~2.1 h per cell).
#
# Each cell is 2000 generations (100 patients × 20 seeds).
# Resume-safe: each python script skips (input_id, seed) pairs already written
# to its CSV, so this shell script can be safely Ctrl-C'd and re-run.
#
# Run from project root:
#     bash scripts/main_run/run_qwen.sh
# Or from anywhere (script cd's to project root itself).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd ~/Research
source venv/bin/activate

mkdir -p logs outputs/main_run
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/main_run_qwen_${TS}.log"

# Tee everything (stdout + stderr) into the log
exec > >(tee -a "$LOG") 2>&1

START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BATCH_START=$(date +%s)

echo "================================================================"
echo "MAIN RUN — Qwen-MediCare-BD (4 quants)"
echo "================================================================"
echo "  Started        : $START_TS"
echo "  Log            : $LOG"
echo "  Expected total : ~8.3 hours"
echo "  Order          : Q6_K → Q5_K_M → Q4_K_M → Q3_K_M"
echo "  Resume         : safe (Ctrl-C OK; partial saves on every generation)"
echo ""

# Order: Q6 (anchor) → Q5 → Q4 → Q3 (most quantization-degraded last)
SCRIPTS=(
    "scripts/main_run/gen_qwen_Q6_K_main.py"
    "scripts/main_run/gen_qwen_Q5_K_M_main.py"
    "scripts/main_run/gen_qwen_Q4_K_M_main.py"
    "scripts/main_run/gen_qwen_Q3_K_M_main.py"
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
echo "QWEN BATCH COMPLETE"
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
ls -la outputs/main_run/gen_qwen_*_main.csv 2>/dev/null || echo "  (none — check log for errors)"
