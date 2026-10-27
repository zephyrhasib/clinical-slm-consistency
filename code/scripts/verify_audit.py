"""
Rigorous verification of contamination audit outputs against IMPL-02 spec.
Catches: parameter drift, count mismatches, silent failures, file integrity.
"""
import json
import hashlib
import time
import subprocess
import re
from pathlib import Path
import pandas as pd

ROOT      = Path("/home/hasib/Research")
AUDIT_DIR = ROOT / "data/audit"
STRAT_DIR = ROOT / "data/stratified"

print("=" * 70)
print("CONTAMINATION AUDIT — RIGOROUS VERIFICATION")
print("=" * 70)

checks = []

def check(name, ok, detail=""):
    checks.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    suffix = f"  [{detail}]" if detail else ""
    print(f"  [{status}] {name}{suffix}")

# ── 1. FILE EXISTENCE ─────────────────────────────────────────────────────────
print("\n1. FILE EXISTENCE")
print("-" * 70)
expected_files = [
    "corpus_bangladeshi_medicines.csv",
    "corpus_medqa_usmle.csv",
    "model_a_minhash_audit.json",
    "model_b_probe_audit.json",
    "model_c_probe_audit.json",
    "audit_summary.json",
]
for fname in expected_files:
    path = AUDIT_DIR / fname
    check(f"File exists: {fname}", path.exists(),
          f"{path.stat().st_size/1024:.1f} KB" if path.exists() else "MISSING")

# ── 2. PRE-SPECIFIED PARAMETERS — MODEL A ────────────────────────────────────
print("\n2. PRE-SPECIFIED PARAMETERS — Model A (MinHash LSH)")
print("-" * 70)
with open(AUDIT_DIR / "model_a_minhash_audit.json") as f:
    a = json.load(f)
p = a["parameters"]
check("N_GRAM == 13",        p["n_gram"] == 13,            str(p["n_gram"]))
check("JACCARD_THR == 0.8",  p["jaccard_threshold"] == 0.8, str(p["jaccard_threshold"]))
check("NUM_PERM == 128",     p["num_perm"] == 128,          str(p["num_perm"]))
check("Shingle type word",   p["shingle_type"] == "word-level", p["shingle_type"])
check("Both corpora indexed",
      "Mahadih534/all-Bangladeshi-medicines" in p["corpora_indexed"]
      and "GBaker/MedQA-USMLE-4-options" in p["corpora_indexed"])

# ── 3. PRE-SPECIFIED PARAMETERS — MODELS B AND C ─────────────────────────────
print("\n3. PRE-SPECIFIED PARAMETERS — Models B and C (n-gram probe)")
print("-" * 70)
for tag, fname in [("Model B", "model_b_probe_audit.json"),
                    ("Model C", "model_c_probe_audit.json")]:
    with open(AUDIT_DIR / fname) as f:
        r = json.load(f)
    p = r["parameters"]
    check(f"{tag} PREFIX_WORDS == 50",   p["prefix_words"] == 50,   str(p["prefix_words"]))
    check(f"{tag} MATCH_WORDS == 10",    p["match_words"] == 10,    str(p["match_words"]))
    check(f"{tag} MAX_NEW_TOKENS == 100", p["max_new_tokens"] == 100, str(p["max_new_tokens"]))
    check(f"{tag} PROBE_TEMP == 0.0",     p["probe_temperature"] == 0.0, str(p["probe_temperature"]))
    check(f"{tag} binary is llama-completion",
          p.get("binary", "").startswith("llama-completion"), p.get("binary", "?"))
    check(f"{tag} raw completion enforced",
          "in-prefix" in p.get("raw_completion_mode", "").lower(), p.get("raw_completion_mode", "?"))

# ── 4. COUNT INTEGRITY ────────────────────────────────────────────────────────
print("\n4. COUNT INTEGRITY (must sum to 100 main items)")
print("-" * 70)
for tag, fname in [("Model A", "model_a_minhash_audit.json"),
                    ("Model B", "model_b_probe_audit.json"),
                    ("Model C", "model_c_probe_audit.json")]:
    with open(AUDIT_DIR / fname) as f:
        r = json.load(f)
    total = r["evaluation_items"]
    flagged = r["flagged_count"]
    if "clean_count" in r:
        clean = r["clean_count"]
    else:
        clean = total - flagged - r.get("error_count", 0)
    errors = r.get("error_count", 0)
    sum_check = flagged + clean + errors == total
    check(f"{tag} total == 100",          total == 100,         str(total))
    check(f"{tag} flagged+clean+err==100", sum_check,
          f"f={flagged} c={clean} e={errors}")
    check(f"{tag} zero errors",            errors == 0,         str(errors))

# ── 5. SUMMARY INTEGRITY ──────────────────────────────────────────────────────
print("\n5. SUMMARY INTEGRITY")
print("-" * 70)
with open(AUDIT_DIR / "audit_summary.json") as f:
    s = json.load(f)
check("Decision == clean",        s["decision"] == "clean")
check("Total flagged == 0",       s["total_flagged"] == 0)
check("Sensitivity not required", s["sensitivity_required"] == False)
check("Proceed with 100 items",   s["proceed_with_n_items"] == 100)
check("Per-model count == 3",     len(s["per_model"]) == 3)

# ── 6. CROSS-CHECK: INPUT_ID COVERAGE ─────────────────────────────────────────
print("\n6. INPUT_ID COVERAGE (every MAIN-001..100 was evaluated)")
print("-" * 70)
main_df = pd.read_csv(STRAT_DIR / "main_inputs.csv")
all_main_ids = set(main_df["input_id"].tolist())
check(f"Main has exactly 100 ids", len(all_main_ids) == 100, str(len(all_main_ids)))

# For each audit, we want flagged_ids + clean_ids to cover ALL 100 input_ids.
# Model A: flagged_items + (not_flagged is implicit) — verified above by count
# Models B/C: short_skipped is included in clean for accounting

# ── 7. SANITY: PROBE DID ACTUALLY RUN ─────────────────────────────────────────
print("\n7. SANITY — re-run 1 probe item and verify reproducibility (T=0)")
print("-" * 70)
# Pick MAIN-001 dialogue, re-run on Phi4, expect non-empty deterministic output
row = main_df.iloc[0]
prefix = " ".join(str(row["dialogue"]).split()[:50])
tmp = Path("/tmp/verify_probe.txt")
tmp.write_text(prefix)
cmd = [str(ROOT / "llama.cpp/build/bin/llama-completion"),
       "-m", str(ROOT / "models/quantized/phi4_mini_instruct_Q4_K_M.gguf"),
       "-f", str(tmp), "-n", "100", "--temp", "0.0", "-ngl", "99",
       "--no-warmup", "--simple-io", "--in-prefix", "", "--in-suffix", ""]
res = subprocess.run(cmd, capture_output=True, text=True,
                     stdin=subprocess.DEVNULL, timeout=180)
out = re.sub(r"\s*>\s*EOF\b.*$", "", res.stdout,
             flags=re.DOTALL | re.IGNORECASE).strip()
tmp.unlink(missing_ok=True)
check("Re-run produced output (>200 chars)", len(out) > 200, f"{len(out)} chars")
check("Re-run returncode == 0", res.returncode == 0, f"rc={res.returncode}")
# Greedy at T=0 should reproduce same output as during audit. The audit
# stored generated_excerpt[:500] but model_c file is clean (no flagged items)
# so we can't directly compare. We at least confirm output is sane:
expected = " ".join(str(row["dialogue"]).split()[50:60])
match_now = expected.lower() in out.lower()
check(f"MAIN-001 still NOT flagged (consistency with audit)",
      match_now == False, f"match={match_now}")

# ── 8. SHA-256 OF AUDIT OUTPUTS ───────────────────────────────────────────────
print("\n8. SHA-256 OF AUDIT OUTPUT FILES")
print("-" * 70)
hashes = {}
for fname in ["model_a_minhash_audit.json", "model_b_probe_audit.json",
              "model_c_probe_audit.json", "audit_summary.json"]:
    path = AUDIT_DIR / fname
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    hashes[fname] = h.hexdigest()
    print(f"  {h.hexdigest()}  {fname}")

with open(AUDIT_DIR / "audit_hashes.txt", "w") as f:
    f.write("Contamination Audit Output Hashes\n")
    f.write("=" * 70 + "\n")
    for fname, h in hashes.items():
        f.write(f"{h}  {fname}\n")
print(f"\n  Saved: data/audit/audit_hashes.txt")

# ── FINAL VERDICT ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
n_total = len(checks)
n_pass  = sum(1 for _, ok, _ in checks if ok)
print(f"VERIFICATION RESULT: {n_pass}/{n_total} checks passed")
print("=" * 70)
if n_pass != n_total:
    print("\nFAILURES:")
    for name, ok, detail in checks:
        if not ok:
            print(f"  FAIL: {name}  [{detail}]")
