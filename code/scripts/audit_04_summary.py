"""
Step 4/4 of contamination audit — aggregate results from all three audits
and print the final decision.

Run from project root: python scripts/audit_04_summary.py
"""
import json
from pathlib import Path

AUDIT_DIR = Path("data/audit")

AUDIT_FILES = [
    ("Model A (Qwen-MediCare-BD)",  "model_a_minhash_audit.json"),
    ("Model B (Gemma-3-4B-it)",     "model_b_probe_audit.json"),
    ("Model C (Phi-4-mini)",        "model_c_probe_audit.json"),
]

print("=" * 60)
print("CONTAMINATION AUDIT SUMMARY")
print("=" * 60)

all_flagged_ids = set()
any_missing     = False
per_model       = []

for model_name, fname in AUDIT_FILES:
    path = AUDIT_DIR / fname
    if not path.exists():
        print(f"  {model_name}: FILE NOT FOUND — {fname}")
        any_missing = True
        continue

    with open(path) as f:
        res = json.load(f)

    flagged_ids = [item["input_id"] for item in res.get("flagged_items", [])]
    all_flagged_ids.update(flagged_ids)

    status = "CLEAN" if res["flagged_count"] == 0 else f"FLAGGED {res['flagged_count']} ITEMS"
    err_count = res.get("error_count", 0)
    err_str   = f"  errors={err_count}" if err_count else ""
    print(f"  {model_name}: {status} ({res['flagged_pct']}%){err_str}")
    if flagged_ids:
        print(f"    Flagged: {flagged_ids}")

    per_model.append({
        "model":         model_name,
        "method":        res.get("method"),
        "flagged_count": res["flagged_count"],
        "flagged_ids":   flagged_ids,
        "error_count":   err_count,
    })

if any_missing:
    print("\nWARNING: Some audit files missing. Re-run missing scripts first.")

print(f"\n{'=' * 60}")
print(f"Union of flagged items across all three models: {sorted(all_flagged_ids)}")
print(f"Total unique flagged items: {len(all_flagged_ids)} / 100")
print(f"{'=' * 60}")

if len(all_flagged_ids) == 0:
    decision_text = """
DECISION: CLEAN
  No contamination detected across all three models.
  Proceed with all 100 main items in the main run.
  Sensitivity analysis: not required.

  Report:
  'A MinHash LSH contamination audit (13-gram word shingles,
   Jaccard >= 0.8, 128 permutations) of Model A's fine-tuning
   corpora against the 100 evaluation items returned zero flagged
   items. N-gram memorisation probes on Models B and C at
   temperature 0.0 (50-word prefix, 10-word verbatim match
   threshold) returned zero flagged items.'
"""
    print(decision_text)
else:
    clean_n = 100 - len(all_flagged_ids)
    decision_text = f"""
DECISION: FLAGGED ITEMS DETECTED
  {len(all_flagged_ids)} items flagged across all models.
  Run ALL primary analyses twice:
    (1) Full set  : 100 items (as specified)
    (2) Clean set : {clean_n} items (excluding flagged: {sorted(all_flagged_ids)})
  Report both sets of results.
  Record the flagged item list with the audit files.
"""
    print(decision_text)

# Save summary
summary = {
    "audit_files":          [fname for _, fname in AUDIT_FILES],
    "per_model":            per_model,
    "all_flagged_ids":      sorted(all_flagged_ids),
    "total_flagged":        len(all_flagged_ids),
    "decision":             "clean" if len(all_flagged_ids) == 0 else "flagged",
    "proceed_with_n_items": 100,
    "sensitivity_required": len(all_flagged_ids) > 0,
}
out_path = AUDIT_DIR / "audit_summary.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved: {out_path}")
print("Keep all 4 audit JSON files with the project record.")
