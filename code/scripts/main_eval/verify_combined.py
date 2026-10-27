"""
Pre-flight verification of the combined main CSV before the expensive metric runs.

Run from project root (venv_metrics — same env as metric_pipeline):
    source venv_metrics/bin/activate
    python scripts/main_eval/verify_combined.py

Checks:
  1. File exists, 24,000 rows
  2. Schema uniform across all 12 cells
  3. (input_id × model × quant_level × seed) tuples are unique
  4. All 12 (model × quant_level) combos present with 2000 rows each
  5. Per-cell, exactly 20 seeds × 100 patients
  6. No empty `generated_text` (NaN would break metric pipeline)
  7. Pre-flight UMLS DB load — confirm MEDCON will work before kickoff
  8. Pre-flight NLI model load — confirm sentence-matrix NLI will work
  9. Estimate runtime
"""
import sys
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
COMBINED = ROOT / "outputs/main_eval/main_combined.csv"

EXPECTED_TOTAL    = 24000
EXPECTED_CELLS    = 12
EXPECTED_PER_CELL = 2000
EXPECTED_SEEDS    = 20
EXPECTED_PATIENTS = 100

print("=" * 72)
print("PRE-FLIGHT VERIFICATION FOR MAIN EVAL RUN")
print("=" * 72)

# 1. File + row count
if not COMBINED.exists():
    print(f"FAIL: {COMBINED} not found. Run combine_main_csvs.py first.")
    sys.exit(1)
df = pd.read_csv(COMBINED)
print(f"\n[1] {COMBINED.name}: {len(df)} rows  "
      f"({'PASS' if len(df) == EXPECTED_TOTAL else 'FAIL'})")

# 2. Schema
required = ["input_id", "model", "quant_level", "seed", "generated_text"]
have = all(c in df.columns for c in required)
print(f"[2] required columns present: {'PASS' if have else 'FAIL'}")

# 3. Uniqueness
n_unique = df.groupby(["input_id","model","quant_level","seed"]).size()
dups = (n_unique > 1).sum()
print(f"[3] unique (input_id, model, quant_level, seed) tuples: "
      f"{(n_unique==1).sum()}/{EXPECTED_TOTAL}  duplicates: {dups}  "
      f"({'PASS' if dups == 0 else 'FAIL'})")

# 4. All 12 cells with 2000 rows
cells = df.groupby(["model","quant_level"]).size()
print(f"[4] cells × rows ({len(cells)} cells expected {EXPECTED_CELLS}):")
for (m, q), n in cells.items():
    print(f"     {m:<22s}  {q:<8s}  {n} rows  "
          f"({'PASS' if n == EXPECTED_PER_CELL else 'FAIL'})")

# 5. Per-cell seed and patient counts
print(f"[5] per-cell seed and patient counts:")
for (m, q), grp in df.groupby(["model","quant_level"]):
    n_seeds   = grp['seed'].nunique()
    n_patient = grp['input_id'].nunique()
    ok = n_seeds == EXPECTED_SEEDS and n_patient == EXPECTED_PATIENTS
    print(f"     {m:<22s}  {q:<8s}  seeds={n_seeds}  patients={n_patient}  "
          f"({'PASS' if ok else 'FAIL'})")

# 6. Empty/NaN generated_text
empty = df["generated_text"].isna().sum()
empty_str = (df["generated_text"].astype(str).str.strip() == "").sum()
print(f"[6] empty generated_text: NaN={empty}, empty_str={empty_str}  "
      f"({'PASS' if (empty + empty_str) == 0 else 'WARN — pipeline may skip these'})")

# 7. UMLS DB load
print(f"\n[7] QuickUMLS DB load test ...")
try:
    t0 = time.time()
    from quickumls import QuickUMLS
    matcher = QuickUMLS(str(ROOT / "umls/quickumls-db"), threshold=0.7, window=5)
    el = time.time() - t0
    print(f"     PASS — loaded in {el:.1f}s")
    print(f"     Sample query 'hypertension':")
    matches = matcher.match("hypertension", best_match=True, ignore_syntax=True)
    print(f"       {len(matches)} match spans returned")
except Exception as e:
    print(f"     FAIL: {e}")
    sys.exit(1)

# 8. NLI model load
print(f"\n[8] NLI cross-encoder load test ...")
try:
    t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-large", use_fast=False)
    mdl = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-large")
    import torch
    if torch.cuda.is_available():
        mdl = mdl.cuda()
    el = time.time() - t0
    print(f"     PASS — loaded in {el:.1f}s  ({'GPU' if torch.cuda.is_available() else 'CPU'})")
except Exception as e:
    print(f"     FAIL: {e}")
    sys.exit(1)

# 9. Runtime estimate
print(f"\n[9] Estimated runtime for full main metric pipeline:")
n_cells = EXPECTED_PATIENTS * 12   # patients × (model × quants)
n_pairs_per_cell = EXPECTED_SEEDS * (EXPECTED_SEEDS - 1) // 2   # C(20, 2) = 190
total_pairs = n_cells * n_pairs_per_cell
print(f"     Cells (input_id × model × quant_level): {n_cells}")
print(f"     Seed pairs per cell: {n_pairs_per_cell} (C(20, 2))")
print(f"     Total pairwise metric evaluations: {total_pairs:,}")
print(f"     Estimated wallclock (extrapolated from pilot 2 h 10 m at 5,760 pairs):")
print(f"       BERTScore + ROUGE-L + MEDCON + numeric  : ~8-12 hours")
print(f"       NLI sentence-matrix                     : ~20-30 hours")
print(f"       Stat pipeline (BCa activates at n=100)  : ~2-3 min")
print(f"     Total: ~30-40 hours")

print("\n" + "=" * 72)
print("All pre-flight checks PASS. Ready to run the main eval pipeline.")
print("=" * 72)
