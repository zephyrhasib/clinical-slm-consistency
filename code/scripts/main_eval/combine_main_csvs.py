"""
Combine all 12 main generation CSVs into a single 24,000-row file ready for
ingestion by metric_pipeline.py.

Input  : outputs/main_run/gen_*_main.csv  (12 files, 2000 rows each)
Output : outputs/main_eval/main_combined.csv  (24,000 rows)
         outputs/main_eval/main_combined_clean.csv (clean-only subset, for
                                                    sensitivity analysis later)

Schema preserved:
  input_id, original_aci_id, split, length_stratum,
  model, quant_level, seed,
  returncode, elapsed_seconds, word_count,
  is_anomalous, is_looped,
  generated_text

The metric pipeline uses only (input_id, model, quant_level, seed, generated_text)
but keeping the full schema enables sensitivity analysis without re-loading.

Run from project root (either venv is fine — pure pandas):
    python scripts/main_eval/combine_main_csvs.py
"""
import sys
from pathlib import Path
import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent.parent
SRC_DIR   = ROOT / "outputs/main_run"
OUT_DIR   = ROOT / "outputs/main_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL   = OUT_DIR / "main_combined.csv"
OUT_CLEAN = OUT_DIR / "main_combined_clean.csv"

EXPECTED_FILES = 12
EXPECTED_ROWS_PER_FILE = 2000
EXPECTED_TOTAL = EXPECTED_FILES * EXPECTED_ROWS_PER_FILE  # 24,000

# Find all 12 main CSVs
csv_files = sorted(SRC_DIR.glob("gen_*_main.csv"))
if len(csv_files) != EXPECTED_FILES:
    print(f"ERROR: expected {EXPECTED_FILES} main CSVs in {SRC_DIR}, found {len(csv_files)}")
    sys.exit(1)

print("=" * 72)
print(f"Combining {len(csv_files)} main CSVs")
print("=" * 72)
for f in csv_files:
    print(f"  {f.name}")

# Concat
dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    if len(df) != EXPECTED_ROWS_PER_FILE:
        print(f"  WARN: {f.name} has {len(df)} rows (expected {EXPECTED_ROWS_PER_FILE})")
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)

# Validate
print(f"\nTotal rows: {len(combined)} (expected {EXPECTED_TOTAL})")
print(f"Cells (model × quant): {combined.groupby(['model','quant_level']).ngroups} (expected 12)")
print(f"Patients per cell: {combined.groupby(['model','quant_level'])['input_id'].nunique().unique()}")
print(f"Seeds per (cell × patient): {combined.groupby(['input_id','model','quant_level'])['seed'].nunique().unique()}")
print(f"Columns: {list(combined.columns)}")

# Sanity: schema uniformity
required_cols = ["input_id", "original_aci_id", "split", "length_stratum",
                 "model", "quant_level", "seed", "returncode", "elapsed_seconds",
                 "word_count", "is_anomalous", "is_looped", "generated_text"]
missing = [c for c in required_cols if c not in combined.columns]
if missing:
    print(f"ERROR: missing columns: {missing}")
    sys.exit(1)

# Save the full combined CSV (used for PRIMARY analysis)
combined.to_csv(OUT_ALL, index=False)
print(f"\n✓ Wrote {OUT_ALL}  ({OUT_ALL.stat().st_size/1024/1024:.1f} MB)")

# Build the clean subset (for SENSITIVITY analysis)
# Drop rows where the model failed (anomalous OR looped)
clean = combined[~(combined["is_anomalous"] | combined["is_looped"])].copy()
clean.to_csv(OUT_CLEAN, index=False)
print(f"✓ Wrote {OUT_CLEAN}  ({len(clean)} clean rows, {len(combined)-len(clean)} dropped)")

# Show clean coverage per cell
print(f"\nClean-row coverage per cell (primary uses ALL 2000; sensitivity uses these):")
cov = clean.groupby(["model","quant_level"]).size().unstack(fill_value=0)
print(cov.to_string())
