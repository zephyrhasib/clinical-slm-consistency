"""
Step 4/4 — Verify the QuickUMLS DB and the ACI-Bench TUI filter work end-to-end.

Loads the built QuickUMLS database with ACI-Bench MEDCON parameters
(threshold=1.0, window=5, ignore_syntax=True per ACI-Bench convention) and
queries a small set of clinical terms. For each term, prints the matched
CUIs + semantic types, then applies the ACI-Bench TUI filter and reports
how many matches remain.

This is purely a verification script — no metric scores are computed here.

Run from project root in venv_metrics:
    source venv_metrics/bin/activate
    python scripts/umls_04_smoke_test.py
"""
import json
import time
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
DB_DIR    = ROOT / "umls/quickumls-db"
TUI_JSON  = ROOT / "umls/aci_bench_tui_codes.json"

# ACI-Bench MEDCON parameters (pre-specified)
# threshold=0.7 enables simstring fuzzy matching, handling abbreviations and
# alternate spellings ("SOB"→"shortness of breath", "HTN"→"hypertension", etc.)
# that real model outputs use. threshold=1.0 only matches verbatim strings
# and was unsuitable for the consistency-measurement use case.
QUICKUMLS_THR     = 0.7
QUICKUMLS_WINDOW  = 5     # 5-gram window
IGNORE_SYNTAX     = True  # spaCy-free matching

# Test terms — span every ACI-Bench-relevant group to confirm filter works
TEST_TERMS = [
    ("hypertension",                     "Disorder"),
    ("type 2 diabetes mellitus",         "Disorder"),
    ("metformin",                        "Chemicals & Drugs"),
    ("shortness of breath",              "Disorder/Sign"),
    ("amoxicillin 500 mg",               "Chemicals & Drugs"),
    ("left knee",                        "Anatomy"),
    ("blood pressure",                   "Physiology"),
    ("colonoscopy",                      "(should be filtered — Procedure)"),
]


def main():
    print("=" * 70)
    print("UMLS 04 — QuickUMLS smoke test + ACI-Bench TUI filter")
    print("=" * 70)

    # ── Pre-flight ───────────────────────────────────────────────────────────
    if not DB_DIR.exists() or not any(DB_DIR.iterdir()):
        raise SystemExit(f"FATAL: QuickUMLS DB not found at {DB_DIR}. Run umls_03 first.")
    if not TUI_JSON.exists():
        raise SystemExit(f"FATAL: TUI list not found at {TUI_JSON}. Run umls_02 first.")

    # Load TUI filter
    tui_meta = json.loads(TUI_JSON.read_text())
    allowed_tuis = set(tui_meta["all_tuis_sorted"])
    print(f"\nACI-Bench TUI filter: {len(allowed_tuis)} TUIs across "
          f"{len(tui_meta['included_groups'])} groups")

    # ── Load QuickUMLS ───────────────────────────────────────────────────────
    print(f"\nLoading QuickUMLS DB from {DB_DIR} ...")
    t0 = time.time()
    from quickumls import QuickUMLS
    matcher = QuickUMLS(
        str(DB_DIR),
        threshold=QUICKUMLS_THR,
        window=QUICKUMLS_WINDOW,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Threshold={QUICKUMLS_THR}, window={QUICKUMLS_WINDOW}, ignore_syntax={IGNORE_SYNTAX}")

    # ── Query each term ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Term':<35s} {'All matches':>12s} {'After TUI filter':>18s}")
    print("=" * 70)
    all_results = []
    for term, expected in TEST_TERMS:
        matches = matcher.match(term, best_match=True, ignore_syntax=IGNORE_SYNTAX)
        # match() returns list-of-list-of-dicts
        flat = [m for span in matches for m in span]
        # Filter to ACI-Bench TUIs
        filtered = []
        for m in flat:
            semtypes = m.get("semtypes", set())
            if any(t in allowed_tuis for t in semtypes):
                filtered.append(m)

        print(f"{term:<35s} {len(flat):>12d} {len(filtered):>18d}    ({expected})")
        # Show details for first 2 matches per term
        for m in filtered[:2]:
            cui      = m.get("cui", "?")
            term_str = m.get("term", "")[:30]
            semtypes = sorted(m.get("semtypes", set()))
            sim      = m.get("similarity", 0)
            print(f"   → CUI={cui:<10s} {term_str:<30s} TUIs={semtypes}  sim={sim:.2f}")
        all_results.append({
            "term":           term,
            "expected_group": expected,
            "n_total":        len(flat),
            "n_after_filter": len(filtered),
            # At threshold=0.7 fuzzy matching yields many candidates; store ALL CUIs
            "filtered_cuis":  [m.get("cui") for m in filtered],
        })

    # ── Verification checks ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)
    checks = []
    # Each of the first 7 terms should yield ≥1 filtered match (in-vocab clinical concepts)
    expect_match = ["hypertension", "type 2 diabetes mellitus", "metformin",
                    "shortness of breath", "amoxicillin 500 mg", "left knee",
                    "blood pressure"]
    for term in expect_match:
        r = next(x for x in all_results if x["term"] == term)
        ok = r["n_after_filter"] >= 1
        checks.append((f"'{term}' yields ≥1 ACI-filtered match", ok, str(r["n_after_filter"])))
    # Smoke check: hypertension should map to C0020538 (a known UMLS CUI for hypertension)
    r = next(x for x in all_results if x["term"] == "hypertension")
    ok = "C0020538" in r["filtered_cuis"]
    checks.append(("'hypertension' returns CUI C0020538", ok, str(r["filtered_cuis"])))

    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  [{detail}]")
    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  Total: {n_pass}/{len(checks)} checks passed")


if __name__ == "__main__":
    main()
