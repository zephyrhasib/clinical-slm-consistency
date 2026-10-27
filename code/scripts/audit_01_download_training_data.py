"""
Step 1/4 of contamination audit — download Model A fine-tuning corpora.

Downloads both public datasets that Qwen-MediCare-BD was fine-tuned on,
saves them as CSVs in data/audit/ for indexing by audit_02.

Run from project root: python scripts/audit_01_download_training_data.py
"""
from datasets import load_dataset
import pandas as pd
from pathlib import Path

OUT_DIR = Path("data/audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("AUDIT 01 — Downloading Model A fine-tuning corpora")
print("=" * 60)

print("\n[1/2] Mahadih534/all-Bangladeshi-medicines ...")
ds1 = load_dataset("Mahadih534/all-Bangladeshi-medicines", split="train")
df1 = pd.DataFrame(ds1)
df1.to_csv(OUT_DIR / "corpus_bangladeshi_medicines.csv", index=False)
print(f"  Saved: {len(df1):>6} rows | columns: {list(df1.columns)}")

print("\n[2/2] GBaker/MedQA-USMLE-4-options ...")
ds2 = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
df2 = pd.DataFrame(ds2)
df2.to_csv(OUT_DIR / "corpus_medqa_usmle.csv", index=False)
print(f"  Saved: {len(df2):>6} rows | columns: {list(df2.columns)}")

print("\nDone. Inspect column names above before running audit_02.")
