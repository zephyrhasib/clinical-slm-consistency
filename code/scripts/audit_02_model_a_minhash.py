"""
Step 2/4 of contamination audit — MinHash LSH on Model A.

Method   : MinHash LSH, word-level shingles
Model    : Qwen-MediCare-BD (CBrootA/Qwen-MediCare-BD)
Corpora  : Mahadih534/all-Bangladeshi-medicines + GBaker/MedQA-USMLE-4-options
Queries  : 100 main ACI-Bench items
Output   : data/audit/model_a_minhash_audit.json

Pre-specified parameters (must NOT change once fixed):
  N_GRAM      = 13   (word shingles)
  JACCARD_THR = 0.8
  NUM_PERM    = 128

Run from project root: python scripts/audit_02_model_a_minhash.py
"""
import json
from pathlib import Path

import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

# ── pre-specified parameters (do not change once fixed) ──────────
N_GRAM      = 13
JACCARD_THR = 0.8
NUM_PERM    = 128
# ─────────────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parent.parent
AUDIT_DIR = ROOT / "data/audit"
STRAT_DIR = ROOT / "data/stratified"

ACIBENCH_COL = "dialogue"


def word_shingles(text: str, n: int) -> set:
    """Extract word-level n-gram shingles as a set of strings."""
    words = str(text).lower().split()
    return set(
        " ".join(words[i : i + n])
        for i in range(len(words) - n + 1)
    )


def make_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for shingle in word_shingles(text, N_GRAM):
        m.update(shingle.encode("utf-8"))
    return m


# ── Step 1: Collect all training text from both corpora ───────────────────────
print("=" * 60)
print("AUDIT 02 — Model A · MinHash LSH Contamination Audit")
print(f"N-gram: {N_GRAM}  |  Jaccard threshold: {JACCARD_THR}  |  Permutations: {NUM_PERM}")
print("=" * 60)

print("\n[Step 1] Loading fine-tuning corpora ...")
df1 = pd.read_csv(AUDIT_DIR / "corpus_bangladeshi_medicines.csv")
df2 = pd.read_csv(AUDIT_DIR / "corpus_medqa_usmle.csv")

train_entries = []   # list of (source_label, text_string)
for col in df1.columns:
    for val in df1[col].dropna().astype(str).tolist():
        if len(val.split()) >= N_GRAM:
            train_entries.append((f"bangladeshi_medicines::{col}", val))
for col in df2.columns:
    for val in df2[col].dropna().astype(str).tolist():
        if len(val.split()) >= N_GRAM:
            train_entries.append((f"medqa_usmle::{col}", val))

print(f"  bangladeshi_medicines: {len(df1):>6} rows, {len(df1.columns)} columns")
print(f"  medqa_usmle         : {len(df2):>6} rows, {len(df2.columns)} columns")
print(f"  Total indexable text entries (>= {N_GRAM} words): {len(train_entries):,}")

# ── Step 2: Build LSH index ───────────────────────────────────────────────────
print(f"\n[Step 2] Building MinHash LSH index ...")
lsh = MinHashLSH(threshold=JACCARD_THR, num_perm=NUM_PERM)

duplicate_count = 0
for i, (source, text) in enumerate(tqdm(train_entries, desc="Indexing")):
    m = make_minhash(text)
    key = f"{i}__{source[:40]}"
    try:
        lsh.insert(key, m)
    except ValueError:
        duplicate_count += 1

print(f"  Index built. Duplicate keys skipped: {duplicate_count}")

# ── Step 3: Query each main evaluation item ───────────────────────────────────
print(f"\n[Step 3] Querying 100 main evaluation items ...")
main_df = pd.read_csv(STRAT_DIR / "main_inputs.csv")

flagged     = []
not_flagged = []
short_skipped = []

for _, row in tqdm(main_df.iterrows(), total=len(main_df), desc="Auditing"):
    dialogue = str(row[ACIBENCH_COL])
    input_id = str(row["input_id"])
    if len(dialogue.split()) < N_GRAM:
        short_skipped.append(input_id)
        not_flagged.append(input_id)
        continue
    m       = make_minhash(dialogue)
    matches = lsh.query(m)
    if matches:
        flagged.append({
            "input_id":   input_id,
            "corpus_idx": int(row["corpus_idx"]),
            "n_matches":  len(matches),
            "match_keys": matches[:5],
        })
    else:
        not_flagged.append(input_id)

# ── Step 4: Save results ──────────────────────────────────────────────────────
results = {
    "model":  "Model A (Qwen-MediCare-BD)",
    "method": "MinHash LSH",
    "parameters": {
        "n_gram":             N_GRAM,
        "jaccard_threshold":  JACCARD_THR,
        "num_perm":           NUM_PERM,
        "shingle_type":       "word-level",
        "corpora_indexed":    [
            "Mahadih534/all-Bangladeshi-medicines",
            "GBaker/MedQA-USMLE-4-options",
        ],
        "fields_indexed":     "all columns in both corpora",
    },
    "corpus_stats": {
        "bangladeshi_medicines_rows": int(len(df1)),
        "medqa_usmle_rows":           int(len(df2)),
        "total_indexable_entries":    int(len(train_entries)),
        "duplicate_keys_skipped":     int(duplicate_count),
    },
    "evaluation_items":  int(len(main_df)),
    "flagged_count":     int(len(flagged)),
    "flagged_pct":       round(len(flagged) / len(main_df) * 100, 2),
    "flagged_items":     flagged,
    "clean_count":       int(len(not_flagged)),
    "short_skipped":     short_skipped,
}

out_path = AUDIT_DIR / "model_a_minhash_audit.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 60}")
print(f"Model A Audit Complete")
print(f"{'=' * 60}")
print(f"  Total items queried : {len(main_df)}")
print(f"  Flagged             : {len(flagged)} ({results['flagged_pct']}%)")
print(f"  Clean               : {len(not_flagged)}")
if len(flagged) == 0:
    print("  RESULT: CLEAN — no contamination detected.")
    print("  Proceed with all 100 main items.")
else:
    print(f"  RESULT: {len(flagged)} ITEMS FLAGGED")
    print(f"  Flagged input_ids: {[f['input_id'] for f in flagged]}")
print(f"\n  Saved: {out_path}")
