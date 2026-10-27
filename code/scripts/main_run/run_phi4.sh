#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Main Run, Phi-4-mini-instruct (all 4 quants)
#
# Runs 4 Phi-4 cells sequentially: Q6 → Q5 → Q4 → Q3.
# Estimated wallclock: ~14.8 hours total (4 × ~3.7 h per cell).
#
# Phi-4 is the highest-variance model in this study:
#   - REPEAT_PEN=1.40 (others 1.25) to suppress semantic loops
#   - N_PREDICT=768  (others 512) — Phi-4 generates longer outputs
#   - truncate_at_loop() at first 8-gram with ≥3 repeats
#   - Retry on word-count < 30
# Expect occasional [RETRY] markers in the log; that is normal Phi-4 behavior.
#
# Each cell is 2000 generations (100 patients × 20 seeds).
# Resume-safe: each python script skips (input_id, seed) pairs already written
# to its CSV, so this shell script can be safely Ctrl-C'd and re-run.
#
# Run from project root:
#     bash scripts/main_run/run_phi4.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd ~/Research
source venv/bin/activate

mkdir -p logs outputs/main_run
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/main_run_phi4_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BATCH_START=$(date +%s)

echo "================================================================"
echo "MAIN RUN — Phi-4-mini-instruct (4 quants)"
echo "================================================================"
echo "  Started        : $START_TS"
echo "  Log            : $LOG"
echo "  Expected total : ~14.8 hours"
echo "  Order          : Q6_K → Q5_K_M → Q4_K_M → Q3_K_M"
echo "  Phi-4 specials : REPEAT_PEN=1.40, N_PREDICT=768, truncate_at_loop, retry<30wc"
echo "  Resume         : safe (Ctrl-C OK; partial saves on every generation)"
echo ""

SCRIPTS=(
    "scripts/main_run/gen_phi4_Q6_K_main.py"
    "scripts/main_run/gen_phi4_Q5_K_M_main.py"
    "scripts/main_run/gen_phi4_Q4_K_M_main.py"
    "scripts/main_run/gen_phi4_Q3_K_M_main.py"
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
echo "PHI-4 BATCH COMPLETE"
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
ls -la outputs/main_run/gen_phi4_*_main.csv 2>/dev/null || echo "  (none — check log for errors)"
