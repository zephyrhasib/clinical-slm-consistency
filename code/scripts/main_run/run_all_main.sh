#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Main Run ORCHESTRATOR (all 12 cells, 24,000 generations)
#
# Runs the three per-model batches in optimal order:
#   1. Qwen   (~8.3 h)  — fastest, validates pipeline first
#   2. Gemma  (~14.7 h) — historically cleanest, low risk
#   3. Phi-4  (~14.8 h) — highest-variance, last
#
# Estimated total wallclock: ~38 hours.
#
# Resume-safe at multiple levels:
#   - Each python script: per-(input_id, seed) resume via CSV inspection
#   - This orchestrator: skips a model batch entirely if all 4 of its CSVs
#     are already 2001 lines (2000 generations + 1 header)
#
# Run from project root:
#     bash scripts/main_run/run_all_main.sh
#
# Recommended:
#     nohup bash scripts/main_run/run_all_main.sh > /dev/null 2>&1 &
#     # then follow the master log:
#     tail -f logs/main_run_master_*.log
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd ~/Research
mkdir -p logs outputs/main_run

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/main_run_master_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
MASTER_START=$(date +%s)

echo "════════════════════════════════════════════════════════════════════════"
echo "MAIN RUN MASTER ORCHESTRATOR"
echo "════════════════════════════════════════════════════════════════════════"
echo "  Started        : $START_TS"
echo "  Master log     : $LOG"
echo "  Expected total : ~38 hours (Qwen 8.3 + Gemma 14.7 + Phi-4 14.8)"
echo "  Order          : Qwen → Gemma → Phi-4"
echo ""
echo "Each per-model batch also writes its own log:"
echo "  logs/main_run_qwen_<TS>.log"
echo "  logs/main_run_gemma_<TS>.log"
echo "  logs/main_run_phi4_<TS>.log"
echo ""

# Helper: skip a model batch if all 4 of its CSVs are already complete
batch_already_complete() {
    local prefix="$1"   # e.g., "qwen_medicare_bd"
    local cells=(Q6_K Q5_K_M Q4_K_M Q3_K_M)
    for q in "${cells[@]}"; do
        local csv="outputs/main_run/gen_${prefix}_${q}_main.csv"
        if [[ ! -f "$csv" ]]; then return 1; fi
        # 2000 data rows + 1 header = 2001 lines
        local n=$(wc -l < "$csv")
        if [[ $n -lt 2001 ]]; then return 1; fi
    done
    return 0
}

run_batch() {
    local name="$1"
    local prefix="$2"
    local script="$3"
    local expected_hours="$4"

    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  BATCH: $name  (expected ~${expected_hours} h)"
    echo "  Time : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "════════════════════════════════════════════════════════════════════════"

    if batch_already_complete "$prefix"; then
        echo "  All 4 $name CSVs already complete (≥2001 lines each) — SKIPPING."
        return 0
    fi

    local T0=$(date +%s)
    bash "$script"
    local T=$(($(date +%s) - T0))
    echo ""
    echo "  $name BATCH DONE in $((T / 3600))h $(((T % 3600) / 60))m $((T % 60))s"
}

# ── Execute the three batches in order ──────────────────────────────────────
run_batch "QWEN"  "qwen_medicare_bd"     "scripts/main_run/run_qwen.sh"   "8.3"
run_batch "GEMMA" "gemma_3_4b_it"        "scripts/main_run/run_gemma.sh"  "14.7"
run_batch "PHI-4" "phi4_mini_instruct"   "scripts/main_run/run_phi4.sh"   "14.8"

END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TOTAL=$(($(date +%s) - MASTER_START))

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "MAIN RUN COMPLETE — 24,000 generations across 12 cells"
echo "════════════════════════════════════════════════════════════════════════"
echo "  Started      : $START_TS"
echo "  Finished     : $END_TS"
echo "  Total elapsed: $((TOTAL / 3600))h $(((TOTAL % 3600) / 60))m $((TOTAL % 60))s"
echo ""
echo "All output CSVs (expected 12 × 2000 rows):"
for csv in outputs/main_run/gen_*_main.csv; do
    n_rows=$(($(wc -l < "$csv") - 1))
    printf "  %-65s  %5d rows\n" "$csv" "$n_rows"
done
echo ""
echo "Next step: combine 12 CSVs and run the metric+statistical pipeline:"
echo "  bash scripts/main_eval/run_main_eval.sh   # (when written)"
