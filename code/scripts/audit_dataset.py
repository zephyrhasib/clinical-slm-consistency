"""
Comprehensive quality audit of the stratified ACI-Bench dataset.
Run from project root: python scripts/audit_dataset.py
"""
import pandas as pd
import json
import hashlib
import os
import re

print("="*70)
print("COMPREHENSIVE DATASET QUALITY AUDIT")
print("="*70)

full   = pd.read_csv("data/stratified/aci_bench_full.csv")
main   = pd.read_csv("data/stratified/full_records_main.csv")
pilot  = pd.read_csv("data/stratified/full_records_pilot.csv")
m_in   = pd.read_csv("data/stratified/main_inputs.csv")
p_in   = pd.read_csv("data/stratified/pilot_inputs.csv")
refs   = pd.read_csv("data/stratified/all_references.csv")
with open("data/stratified/stratification_metadata.json") as f:
    meta = json.load(f)

# 1. FILE SIZES & ROW COUNTS
print("\n1. FILE SIZES & ROW COUNTS")
print("-"*70)
for fname in ["aci_bench_full.csv", "full_records_main.csv", "full_records_pilot.csv",
              "main_inputs.csv", "pilot_inputs.csv", "all_references.csv",
              "stratification_metadata.json"]:
    path = f"data/stratified/{fname}"
    size_kb = os.path.getsize(path) / 1024
    if fname.endswith(".csv"):
        rows = len(pd.read_csv(path))
        print(f"  {fname:35s} {rows:4d} rows  {size_kb:7.1f} KB")
    else:
        print(f"  {fname:35s}            {size_kb:7.1f} KB")

# 2. COLUMN STRUCTURE
print("\n2. COLUMN STRUCTURE")
print("-"*70)
print(f"  aci_bench_full.csv     : {list(full.columns)}")
print(f"  full_records_main.csv  : {list(main.columns)}")
print(f"  main_inputs.csv        : {list(m_in.columns)}")
print(f"  all_references.csv     : {list(refs.columns)}")

# 3. LEAKAGE CHECK
print("\n3. REFERENCE NOTE LEAKAGE CHECK")
print("-"*70)
for fname, df in [("main_inputs.csv", m_in), ("pilot_inputs.csv", p_in)]:
    has_note = "note" in df.columns or "reference_note" in df.columns
    print(f"  {fname:25s} contains reference note: {'FAIL' if has_note else 'PASS (clean)'}")

# 4. SOURCE SPLIT DISTRIBUTION
print("\n4. SOURCE SPLIT REPRESENTATION")
print("-"*70)
print("  Main 100 split distribution:")
for split, n in main["split"].value_counts().sort_index().items():
    print(f"    {split:8s} : {n:3d} items")
print("  Pilot 20 split distribution:")
for split, n in pilot["split"].value_counts().sort_index().items():
    print(f"    {split:8s} : {n:3d} items")

# 5. LENGTH STRATUM x SPLIT CROSS-TAB
print("\n5. LENGTH STRATUM x SOURCE SPLIT (main 100)")
print("-"*70)
print(pd.crosstab(main["length_stratum"], main["split"], margins=True).to_string())
print("\n   LENGTH STRATUM x SOURCE SPLIT (pilot 20)")
print(pd.crosstab(pilot["length_stratum"], pilot["split"], margins=True).to_string())

# 6. WORD COUNT STATS BY STRATUM
print("\n6. WORD COUNT STATS BY STRATUM")
print("-"*70)
print("  MAIN - short stratum:")
print(main[main.length_stratum=="short"]["word_count"].describe().round(1).to_string())
print("\n  MAIN - long stratum:")
print(main[main.length_stratum=="long"]["word_count"].describe().round(1).to_string())
print("\n  PILOT - short stratum:")
print(pilot[pilot.length_stratum=="short"]["word_count"].describe().round(1).to_string())
print("\n  PILOT - long stratum:")
print(pilot[pilot.length_stratum=="long"]["word_count"].describe().round(1).to_string())

# 7. REFERENCE NOTE WORD COUNT STATS
print("\n7. REFERENCE NOTE WORD COUNT STATS")
print("-"*70)
refs["note_words"] = refs["reference_note"].str.split().str.len()
print("  Main + Pilot reference notes:")
print(refs["note_words"].describe().round(1).to_string())

# 8. SPEAKER TAG SANITY
print("\n8. SPEAKER TAG SANITY (main_inputs.csv)")
print("-"*70)
doctor_pat = re.compile(r"\[doctor\]")
patient_pat = re.compile(r"\[patient\]")
guest_pat = re.compile(r"\[patient_guest\]")
n_with_doc   = m_in["dialogue"].apply(lambda d: bool(doctor_pat.search(d))).sum()
n_with_pat   = m_in["dialogue"].apply(lambda d: bool(patient_pat.search(d))).sum()
n_with_guest = m_in["dialogue"].apply(lambda d: bool(guest_pat.search(d))).sum()
print(f"  Dialogues with [doctor] tag        : {n_with_doc}/100")
print(f"  Dialogues with [patient] tag       : {n_with_pat}/100")
print(f"  Dialogues with [patient_guest] tag : {n_with_guest}/100  (should be 0)")

# 9. DUPLICATE DIALOGUE CHECK
print("\n9. DUPLICATE DIALOGUE CHECK")
print("-"*70)
main_dup = m_in["dialogue"].duplicated().sum()
pilot_dup = p_in["dialogue"].duplicated().sum()
print(f"  Duplicate dialogues in main_inputs.csv  : {main_dup}")
print(f"  Duplicate dialogues in pilot_inputs.csv : {pilot_dup}")

# 10. SAMPLE INSPECTION
print("\n10. SAMPLE INSPECTION - MAIN-001 (first item)")
print("-"*70)
row = m_in.iloc[0]
print(f"  input_id        : {row['input_id']}")
print(f"  original_aci_id : {row['original_aci_id']}")
print(f"  split           : {row['split']}")
print(f"  word_count      : {row['word_count']}")
print(f"  length_stratum  : {row['length_stratum']}")
print(f"  dialogue (first 200 chars):")
print(f"    {row['dialogue'][:200]}...")

# 11. FILE HASHES
print("\n11. SHA-256 HASHES OF OUTPUT FILES")
print("-"*70)
hashes = {}
for fname in ["aci_bench_full.csv", "main_inputs.csv", "pilot_inputs.csv",
              "all_references.csv", "stratification_metadata.json"]:
    path = f"data/stratified/{fname}"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    hashes[fname] = digest
    print(f"  {digest}  {fname}")

with open("data/stratified/dataset_hashes.txt", "w") as f:
    f.write("ACI-Bench Dataset Preparation - Output Hashes\n")
    f.write("="*70 + "\n")
    f.write(f"Seed              : {meta['seed']}\n")
    f.write(f"Median word count : {meta['median_word_count']}\n")
    f.write(f"Total corpus      : {meta['total_corpus_size']} rows\n\n")
    for fname, h in hashes.items():
        f.write(f"{h}  {fname}\n")
print("\n  Saved: data/stratified/dataset_hashes.txt")

# 12. FINAL VERDICT
print("\n" + "="*70)
print("AUDIT VERDICT")
print("="*70)
checks = [
    ("Total corpus rows == 207",              len(full) == 207),
    ("Main count == 100",                     len(main) == 100),
    ("Pilot count == 20",                     len(pilot) == 20),
    ("Reference count == 120",                len(refs) == 120),
    ("No reference leak in main_inputs",      "note" not in m_in.columns),
    ("No reference leak in pilot_inputs",     "note" not in p_in.columns),
    ("Main 50/50 short/long balance",         (main.length_stratum=="short").sum() == 50 and (main.length_stratum=="long").sum() == 50),
    ("Pilot 10/10 short/long balance",        (pilot.length_stratum=="short").sum() == 10 and (pilot.length_stratum=="long").sum() == 10),
    ("Zero corpus_idx overlap main/pilot",    len(set(main.corpus_idx) & set(pilot.corpus_idx)) == 0),
    ("Zero input_id duplicates",              len(set(list(m_in.input_id)+list(p_in.input_id))) == 120),
    ("Zero [patient_guest] remaining",        n_with_guest == 0),
    ("All dialogues have [doctor] tag",       n_with_doc == 100),
    ("All dialogues have [patient] tag",      n_with_pat == 100),
    ("Zero duplicate dialogues (main)",       main_dup == 0),
    ("Zero duplicate dialogues (pilot)",      pilot_dup == 0),
]
for name, passed in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

n_pass = sum(1 for _, p in checks if p)
print(f"\n  Total: {n_pass}/{len(checks)} checks passed")
