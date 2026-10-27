"""
Combine all 12 pilot generation CSVs into a single 960-row file ready for
ingestion by metric_pipeline.py.

Input  : outputs/test_batch_1/gen_*_test.csv  (12 files)
Output : outputs/pilot_eval/pilot_combined.csv  (960 rows)
         outputs/pilot_eval/pilot_combined_clean.csv (clean-only subset, for
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
    python scripts/pilot_eval/combine_pilot_csvs.py
"""
import sys
from pathlib import Path
import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent.parent
SRC_DIR   = ROOT / "outputs/test_batch_1"
OUT_DIR   = ROOT / "outputs/pilot_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL   = OUT_DIR / "pilot_combined.csv"
OUT_CLEAN = OUT_DIR / "pilot_combined_clean.csv"

# Find all 12 pilot CSVs
csv_files = sorted(SRC_DIR.glob("gen_*_test.csv"))
if len(csv_files) != 12:
    print(f"ERROR: expected 12 pilot CSVs in {SRC_DIR}, found {len(csv_files)}")
    sys.exit(1)

print("=" * 72)
print(f"Combining {len(csv_files)} pilot CSVs")
print("=" * 72)
for f in csv_files:
    print(f"  {f.name}")

# Concat
dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    if len(df) != 80:
        print(f"  WARN: {f.name} has {len(df)} rows (expected 80)")
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)

# Validate
print(f"\nTotal rows: {len(combined)} (expected 960)")
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
print(f"\n✓ Wrote {OUT_ALL}  ({OUT_ALL.stat().st_size/1024:.0f} KB)")

# Build the clean subset (for SENSITIVITY analysis)
# Drop rows where the model failed (anomalous OR looped)
clean = combined[~(combined["is_anomalous"] | combined["is_looped"])].copy()
clean.to_csv(OUT_CLEAN, index=False)
print(f"✓ Wrote {OUT_CLEAN}  ({len(clean)} clean rows, {len(combined)-len(clean)} dropped)")

# Show clean coverage per cell
print(f"\nClean-row coverage per cell (primary uses ALL 80; sensitivity uses these):")
cov = clean.groupby(["model","quant_level"]).size().unstack(fill_value=0)
print(cov.to_string())
