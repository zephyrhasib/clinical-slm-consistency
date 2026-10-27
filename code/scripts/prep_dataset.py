"""
ACI-Bench dataset preparation
=============================
Run from project root:  python scripts/prep_dataset.py

INPUT
─────
  data/raw/train.csv                      67 rows
  data/raw/valid.csv                      20 rows
  data/raw/clinicalnlp_taskB_test1.csv    40 rows
  data/raw/clinicalnlp_taskC_test2.csv    40 rows
  data/raw/clef_taskC_test3.csv           40 rows

OUTPUT FILES  →  data/stratified/
────────────
Full-record files (all columns incl. reference note — debugging only):
  aci_bench_full.csv          207 rows  full corpus
  full_records_main.csv       100 rows
  full_records_pilot.csv       20 rows

Operational files (no reference note — feed the generation scripts):
  main_inputs.csv             100 rows  MAIN-001 … MAIN-100
  pilot_inputs.csv             20 rows  PILOT-001 … PILOT-020

Reference file:
  all_references.csv          120 rows  main + pilot reference notes

Metadata:
  stratification_metadata.json          written before any generation

ID SYSTEM
─────────
  MAIN-001  …  MAIN-100   (3-digit zero-padded, main study items)
  PILOT-001 …  PILOT-020  (3-digit zero-padded, disjoint from main)
  original_aci_id column preserves the ACI-bench source id (D2N001 etc.)

  Use pilot_inputs.csv for ALL pipeline and metric validation runs.
  Pilot items are outside the main 100 — validation on pilot items
  cannot contaminate the main analysis. No demo set.

DIALOGUE / NOTE FLATTENING
──────────────────────────
  Source dialogues are multi-line. Dialogue and reference note are
  flattened to a single line (newlines → single space, whitespace
  collapsed) so every CSV row is one self-contained encounter.

STRATIFICATION
──────────────
  Binary length stratification on whitespace word count.
  Model-agnostic: three study models use incompatible tokenizer
  families (Qwen2.5 BPE / Gemma-3 SentencePiece / Phi-4 200K BPE).
  Using any single tokenizer would bias the length split in favour
  of that model's subword segmentation. Whitespace word count is
  the only neutral measure that produces identical results across
  all three families.
  Split at median. 50 short + 50 long → main 100.
  Pilot: 10 short + 10 long from the remaining 107 NOT in main.
  Seed: 42 (fixed in advance).

CORPUS-LEVEL DIALOGUE FIXES
────────────────────────────
  Fix 1 — Missing [doctor] tag   : D2N131
  Fix 2 — [patient_guest] → [patient] : D2N076, D2N099, D2N100, D2N141, D2N185
  Flag  — D2N190 starts with [patient] (companion speaks first; no fix)
  Full rationale in apply_corpus_fixes() below.

FROZEN BEFORE GENERATION
────────────────────────
  Written and hash-recorded before any generation:
    data/stratified/aci_bench_full.csv
    data/stratified/main_inputs.csv
    data/stratified/pilot_inputs.csv
    data/stratified/stratification_metadata.json
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths (relative to project root) ────────────────────

CHALLENGE_DIR = Path("data/raw")
OUT_DIR       = Path("data/stratified")
SEED          = 42   # pre-specified stratification seed

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def flatten(text: str) -> str:
    """Collapse all whitespace (including newlines) to single spaces."""
    return " ".join(str(text).split())


# ── Corpus-level dialogue fixes ───────────────────────────────────────────────
#
# ─── Fix 1: Missing leading [doctor] tag ─────────────────────────────────────
# D2N131 (corpus_idx=130) starts with bare text "hi , susan , how are you ?"
# with no speaker tag. Inspection confirms this is a doctor utterance —
# addressed to the patient by name; the next tagged turn is [patient].
# ACI-bench requires every utterance to carry a speaker tag. Without the fix,
# the first utterance is speaker-unknown, which (a) breaks speaker-attribution
# pipelines and (b) may cause models to hallucinate a speaker identity.
# Correction: prepend "[doctor] " to the dialogue string.
#
# ─── Fix 2: [patient_guest] normalisation ────────────────────────────────────
# Five encounters use [patient_guest] (D2N076, D2N099, D2N100, D2N141, D2N185).
# D2N076 and D2N099 have NO [patient] tag at all. If left as-is:
#   (a) All three study models receive a three-speaker format they were not
#       instruction-tuned for — an out-of-distribution input.
#   (b) "[patient_guest]" appears literally in model outputs.
#   (c) The companion's speech is clinically equivalent to patient speech
#       for note-generation: symptoms, medications, and responses are
#       documented regardless of who physically speaks.
# Correction: replace all [patient_guest] with [patient].
# Two-speaker format restored; zero clinical content lost.
#
# ─── Flag: D2N190 ────────────────────────────────────────────────────────────
# Starts with [patient] not [doctor]. A companion speaks first.
# Internally consistent — no correction applied. Flagged for audit monitoring.

CORPUS_FIXES = {
    "D2N131": (
        "Missing leading [doctor] tag — prepend '[doctor] '",
        lambda d: "[doctor] " + d if not d.startswith("[doctor]") else d,
    ),
}

CORPUS_FLAGS = {
    "D2N190": (
        "Starts with [patient] not [doctor] — companion speaks first; "
        "no correction applied"
    ),
}


def apply_corpus_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply documented corpus-level fixes to the dialogue column."""

    # Fix 1: per-encounter tag fixes
    for aci_id, (desc, fn) in CORPUS_FIXES.items():
        mask = df["original_aci_id"] == aci_id
        if mask.any():
            original = df.loc[mask, "dialogue"].iloc[0]
            df.loc[mask, "dialogue"] = df.loc[mask, "dialogue"].apply(fn)
            fixed = df.loc[mask, "dialogue"].iloc[0]
            if original != fixed:
                print(f"  FIX [{aci_id}]: {desc}")

    # Fix 2: [patient_guest] → [patient] corpus-wide
    guest_mask     = df["dialogue"].str.contains(r"\[patient_guest\]", regex=True)
    guest_affected = df[guest_mask]
    if len(guest_affected) > 0:
        occurrence_count = sum(
            len(re.findall(r"\[patient_guest\]", d))
            for d in guest_affected["dialogue"]
        )
        affected_ids = ", ".join(guest_affected["original_aci_id"].tolist())
        df["dialogue"] = df["dialogue"].str.replace(
            r"\[patient_guest\]", "[patient]", regex=True
        )
        print(
            f"  FIX [patient_guest→patient]: {occurrence_count} occurrences "
            f"across {len(guest_affected)} encounters ({affected_ids})"
        )

    # Report flagged encounters (no fix applied)
    for aci_id, note in CORPUS_FLAGS.items():
        print(f"  FLAG [{aci_id}]: {note}")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Load and combine 5 source splits
# ═════════════════════════════════════════════════════════════════════════════

SPLITS = [
    ("train", "train.csv",                     67),
    ("valid", "valid.csv",                     20),
    ("test1", "clinicalnlp_taskB_test1.csv",   40),
    ("test2", "clinicalnlp_taskC_test2.csv",   40),
    ("test3", "clef_taskC_test3.csv",          40),
]

print("=" * 60)
print("STEP 1: Loading source splits")
print("=" * 60)

frames = []
for split_name, filename, expected in SPLITS:
    path = CHALLENGE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Source file not found: {path}\n"
            f"Verify files are in data/raw/ and rerun from project root."
        )
    df = pd.read_csv(path)

    # Normalise column names — handles all known ACI-bench split variants
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("src", "conversation", "text", "input", "dialogue"):
            col_map[c] = "dialogue"
        elif cl in ("tgt", "target", "reference", "output", "note"):
            col_map[c] = "note"
        elif cl == "encounter_id":
            col_map[c] = "id"
    if col_map:
        df = df.rename(columns=col_map)
        print(f"  {split_name}: renamed columns → {col_map}")

    df["split"] = split_name
    frames.append(df)
    status = (
        "OK" if len(df) == expected
        else f"WARNING: expected {expected}, got {len(df)}"
    )
    print(f"  {split_name:6s} | {len(df):3d} rows | {status}")

combined = pd.concat(frames, ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Validate combined corpus
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("STEP 2: Validation")
print(f"{'=' * 60}")

assert "dialogue" in combined.columns, (
    f"No 'dialogue' column found after renaming. "
    f"Actual columns: {combined.columns.tolist()}\n"
    f"Add your column name to the col_map block in STEP 1."
)
assert "note" in combined.columns, (
    f"No 'note' column found after renaming. "
    f"Actual columns: {combined.columns.tolist()}\n"
    f"Add your column name to the col_map block in STEP 1."
)

# Assign original_aci_id before corpus fixes (fixes look up by this column)
if "id" in combined.columns:
    combined = combined.rename(columns={"id": "original_aci_id"})
else:
    combined["original_aci_id"] = [
        f"{r['split']}_{i}" for i, r in combined.iterrows()
    ]

null_d  = combined["dialogue"].isna().sum()
empty_d = (combined["dialogue"].astype(str).str.strip() == "").sum()
print(f"  Total rows      : {len(combined)}")
print(f"  Null dialogues  : {null_d}")
print(f"  Empty dialogues : {empty_d}")
assert len(combined) == 207, f"Expected 207 rows, got {len(combined)}"
assert null_d + empty_d == 0, f"{null_d + empty_d} unusable dialogues — check source files"
print(f"  Row count       : PASS (207)")
print(f"  Usable rows     : PASS")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2.5: Corpus-level dialogue cleanup
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("STEP 2.5: Corpus-level dialogue cleanup")
print(f"{'=' * 60}")

combined = apply_corpus_fixes(combined)
print("  Cleanup complete.")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: Flatten, compute word count, assign stable corpus_idx
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("STEP 3: Flattening and word count")
print(f"{'=' * 60}")

combined = combined.reset_index(drop=True)
combined["corpus_idx"] = combined.index   # 0–206, stable reference key

combined["dialogue"] = combined["dialogue"].apply(flatten)
combined["note"]     = combined["note"].apply(flatten)

# Whitespace word count — model-agnostic stratification.
# token_count kept for schema consistency; equals word_count in this study.
combined["word_count"]  = combined["dialogue"].str.split().str.len()
combined["token_count"] = combined["word_count"]
tokenizer_used = "whitespace word count (model-agnostic, fixed in advance)"

print(f"  Stratification  : {tokenizer_used}")
print(f"\n  Dialogue word-count stats:")
print(combined["word_count"].describe().round(1).to_string())


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: Save full corpus
# ═════════════════════════════════════════════════════════════════════════════

full_cols = [
    "corpus_idx", "original_aci_id", "split",
    "word_count", "token_count", "dialogue", "note",
]
out_path = OUT_DIR / "aci_bench_full.csv"
combined[full_cols].to_csv(out_path, index=False)
print(f"\nSaved: {out_path}  ({len(combined)} rows)")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: Length stratification
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("STEP 5: Length stratification")
print(f"{'=' * 60}")

median_wc = combined["word_count"].median()
combined["length_stratum"] = (combined["word_count"] >= median_wc).map(
    {True: "long", False: "short"}
)

print(f"  Median word count : {median_wc:.0f}")
print(f"  Short items       : {(combined['length_stratum'] == 'short').sum()}")
print(f"  Long items        : {(combined['length_stratum'] == 'long').sum()}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6: Sample main 100 (50 short + 50 long)
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print(f"STEP 6: Sampling main 100  (seed={SEED})")
print(f"{'=' * 60}")

short_pool = combined[combined["length_stratum"] == "short"]
long_pool  = combined[combined["length_stratum"] == "long"]

short_main = short_pool.sample(n=min(50, len(short_pool)), random_state=SEED)
long_main  = long_pool.sample(n=min(50, len(long_pool)),   random_state=SEED)
main_100   = pd.concat([short_main, long_main]).reset_index(drop=True)

main_100["input_id"] = [f"MAIN-{i + 1:03d}" for i in range(len(main_100))]

print(f"  Short sampled : {len(short_main)}")
print(f"  Long sampled  : {len(long_main)}")
print(f"  Total main    : {len(main_100)}")
print(f"  ID range      : {main_100['input_id'].iloc[0]} … {main_100['input_id'].iloc[-1]}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7: Sample pilot 20 (disjoint from main by corpus_idx)
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print(f"STEP 7: Sampling pilot 20  (seed={SEED})")
print(f"{'=' * 60}")

main_corpus_idx = set(main_100["corpus_idx"])
pilot_pool      = combined[~combined["corpus_idx"].isin(main_corpus_idx)]

pilot_short = pilot_pool[pilot_pool["length_stratum"] == "short"].sample(
    n=min(10, len(pilot_pool[pilot_pool["length_stratum"] == "short"])),
    random_state=SEED,
)
pilot_long = pilot_pool[pilot_pool["length_stratum"] == "long"].sample(
    n=min(10, len(pilot_pool[pilot_pool["length_stratum"] == "long"])),
    random_state=SEED,
)
pilot_20 = pd.concat([pilot_short, pilot_long]).reset_index(drop=True)

pilot_20["input_id"] = [f"PILOT-{i + 1:03d}" for i in range(len(pilot_20))]

# Hard disjointness assertions
assert len(set(pilot_20["corpus_idx"]) & main_corpus_idx) == 0, (
    "FATAL: Pilot and main share corpus_idx values — check seed."
)
assert not any(
    pid in set(main_100["input_id"]) for pid in pilot_20["input_id"]
), "FATAL: PILOT-XXX ids collide with MAIN-XXX ids."

print(f"  Short sampled : {len(pilot_short)}")
print(f"  Long sampled  : {len(pilot_long)}")
print(f"  Total pilot   : {len(pilot_20)}")
print(f"  ID range      : {pilot_20['input_id'].iloc[0]} … {pilot_20['input_id'].iloc[-1]}")
print(f"  Disjoint from main (corpus_idx) : VERIFIED")
print(f"  Disjoint from main (input_id)   : VERIFIED")
print(f"\n  Use pilot_inputs.csv for ALL pipeline and metric validation.")
print(f"  Pilot items are outside the main 100 — safe to run freely.")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8: Save full-record files (include reference note — debugging only)
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("STEP 8: Saving full-record files (include note — debugging only)")
print(f"{'=' * 60}")

full_record_cols = [
    "input_id", "corpus_idx", "original_aci_id", "split",
    "word_count", "token_count", "length_stratum", "dialogue", "note",
]
main_100.to_csv(
    OUT_DIR / "full_records_main.csv",  index=False, columns=full_record_cols
)
pilot_20.to_csv(
    OUT_DIR / "full_records_pilot.csv", index=False, columns=full_record_cols
)
print(f"  full_records_main.csv    {len(main_100):3d} rows")
print(f"  full_records_pilot.csv   {len(pilot_20):3d} rows")
print(f"  WARNING: These files contain reference notes.")
print(f"  NEVER pass these to the generation script.")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 9: Save operational input files (NO note column)
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("STEP 9: Saving operational input files (no reference note)")
print(f"{'=' * 60}")

input_cols = [
    "input_id", "corpus_idx", "original_aci_id", "split",
    "word_count", "token_count", "length_stratum", "dialogue",
]
main_100.to_csv(OUT_DIR / "main_inputs.csv",  index=False, columns=input_cols)
pilot_20.to_csv(OUT_DIR / "pilot_inputs.csv", index=False, columns=input_cols)
print(f"  main_inputs.csv    {len(main_100):3d} rows  MAIN-001 … MAIN-100")
print(f"  pilot_inputs.csv   {len(pilot_20):3d} rows  PILOT-001 … PILOT-020")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 10: Save combined reference file (main + pilot)
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("STEP 10: Saving combined reference file")
print(f"{'=' * 60}")

all_refs = pd.concat(
    [
        main_100[["input_id", "note"]].rename(columns={"note": "reference_note"}),
        pilot_20[["input_id", "note"]].rename(columns={"note": "reference_note"}),
    ],
    ignore_index=True,
)
all_refs.to_csv(OUT_DIR / "all_references.csv", index=False)
print(
    f"  all_references.csv   {len(all_refs):3d} rows  "
    f"({len(main_100)} main + {len(pilot_20)} pilot)"
)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 11: Save stratification metadata
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("STEP 11: Saving stratification metadata")
print(f"{'=' * 60}")

metadata = {
    "seed": SEED,
    "stratification_method": tokenizer_used,
    "median_word_count": float(median_wc),
    "total_corpus_size": int(len(combined)),
    "main": {
        "n": int(len(main_100)),
        "n_short": int((main_100["length_stratum"] == "short").sum()),
        "n_long":  int((main_100["length_stratum"] == "long").sum()),
        "corpus_idx_list":      sorted(main_100["corpus_idx"].tolist()),
        "input_id_list":        main_100["input_id"].tolist(),
        "original_aci_id_list": main_100["original_aci_id"].tolist(),
    },
    "pilot": {
        "n": int(len(pilot_20)),
        "n_short": int((pilot_20["length_stratum"] == "short").sum()),
        "n_long":  int((pilot_20["length_stratum"] == "long").sum()),
        "corpus_idx_list":      sorted(pilot_20["corpus_idx"].tolist()),
        "input_id_list":        pilot_20["input_id"].tolist(),
        "original_aci_id_list": pilot_20["original_aci_id"].tolist(),
    },
    "corpus_fixes_applied": [
        "D2N131: prepended [doctor] tag",
        "[patient_guest] → [patient]: D2N076, D2N099, D2N100, D2N141, D2N185",
    ],
    "corpus_flags": [
        "D2N190: starts with [patient] (companion speaks first); no correction applied",
    ],
    "note": (
        "Only main_inputs.csv (seeds 0-19) is used for the main analysis. "
        "pilot_inputs.csv is used for pipeline validation only."
    ),
}

with open(OUT_DIR / "stratification_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  stratification_metadata.json written.")
print(f"  Freeze this file before starting any generation.")


# ═════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("VERIFICATION")
print(f"{'=' * 60}")

# 1. ID uniqueness across main + pilot
all_ids = list(main_100["input_id"]) + list(pilot_20["input_id"])
assert len(all_ids) == len(set(all_ids)), "FATAL: Duplicate input_ids across main and pilot."
print(f"  ID uniqueness (main+pilot)        : PASS  ({len(all_ids)} unique ids)")

# 2. Disjoint by corpus_idx
overlap_idx = set(main_100["corpus_idx"]) & set(pilot_20["corpus_idx"])
assert len(overlap_idx) == 0, f"FATAL: corpus_idx overlap: {overlap_idx}"
print(f"  Disjoint by corpus_idx            : PASS")

# 3. Reference file covers exactly main + pilot, no duplicates
ref_ids      = set(all_refs["input_id"])
expected_ref = set(main_100["input_id"]) | set(pilot_20["input_id"])
assert ref_ids == expected_ref, (
    f"FATAL: Reference id mismatch. Missing: {expected_ref - ref_ids}"
)
assert len(all_refs) == len(expected_ref), (
    f"FATAL: {len(all_refs)} rows but {len(expected_ref)} unique ids — duplicates present."
)
print(f"  Reference covers main+pilot       : PASS  ({len(all_refs)} rows, 0 duplicates)")

# 4. No note column in any input file
for fname in ("main_inputs.csv", "pilot_inputs.csv"):
    df_check = pd.read_csv(OUT_DIR / fname)
    assert "note" not in df_check.columns, (
        f"FATAL: 'note' column in {fname} — would leak reference into generation."
    )
print(f"  No note in input files            : PASS")

# 5. Single-line dialogues
for fname in ("main_inputs.csv", "pilot_inputs.csv"):
    df_check = pd.read_csv(OUT_DIR / fname)
    assert not df_check["dialogue"].str.contains("\n").any(), (
        f"FATAL: Multi-line dialogue in {fname}"
    )
print(f"  Single-line dialogues             : PASS")

# 6. Single-line reference notes
df_ref_check = pd.read_csv(OUT_DIR / "all_references.csv")
assert not df_ref_check["reference_note"].str.contains("\n").any(), (
    "FATAL: Multi-line reference_note in all_references.csv"
)
print(f"  Single-line reference notes       : PASS")

# 7. Main balance
assert (main_100["length_stratum"] == "short").sum() == 50, "FATAL: Main short != 50"
assert (main_100["length_stratum"] == "long").sum()  == 50, "FATAL: Main long != 50"
print(f"  Main balance (50 short / 50 long) : PASS")

# 8. Pilot balance
assert (pilot_20["length_stratum"] == "short").sum() == 10, "FATAL: Pilot short != 10"
assert (pilot_20["length_stratum"] == "long").sum()  == 10, "FATAL: Pilot long != 10"
print(f"  Pilot balance (10 short / 10 long): PASS")

# 9. token_count == word_count (whitespace stratification consistency)
assert (combined["token_count"] == combined["word_count"]).all(), (
    "FATAL: token_count diverges from word_count — check stratification method."
)
print(f"  token_count == word_count         : PASS  (whitespace stratification confirmed)")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("ALL FILES SAVED  →  data/stratified/")
print(f"{'=' * 60}")
print(f"  aci_bench_full.csv           207 rows  full corpus")
print(f"  full_records_main.csv        100 rows  (incl. reference note — debug only)")
print(f"  full_records_pilot.csv        20 rows  (incl. reference note — debug only)")
print(f"  main_inputs.csv              100 rows  MAIN-001 … MAIN-100  (no note)")
print(f"  pilot_inputs.csv              20 rows  PILOT-001 … PILOT-020 (no note)")
print(f"  all_references.csv           120 rows  main + pilot reference notes")
print(f"  stratification_metadata.json          → frozen before generation")
print(f"\n  Seed              : {SEED}")
print(f"  Median word count : {median_wc:.0f}")
print(f"  Stratification    : {tokenizer_used}")
print(f"\n  Files to freeze before generation:")
print(f"    [ ] data/stratified/aci_bench_full.csv")
print(f"    [ ] data/stratified/main_inputs.csv")
print(f"    [ ] data/stratified/pilot_inputs.csv")
print(f"    [ ] data/stratified/stratification_metadata.json")
print(f"\n  Pipeline validation: use pilot_inputs.csv only.")
print(f"  NEVER use main_inputs.csv for validation or test runs.")
print(f"\nData preparation complete.")
