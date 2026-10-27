"""
Step 2/4 — Extract the canonical ACI-Bench TUI code list.

ACI-Bench's official evaluation methodology (evaluation/semantics.py) computes
MEDCON over a filtered subset of UMLS semantic types: 7 of the 15 SemGroups
listed in NLM's SemGroups_2018.txt are INCLUDED, 8 are EXCLUDED. This produces
the canonical "66 TUI codes" used by ACI-Bench for medical concept matching.

Source of truth (taken verbatim from aci-bench-main/evaluation/semantics.py):
  resources/semantic_types.txt  ← NLM SemGroups_2018.txt
  types_to_exclude=["Procedures","Organizations","Occupations","Living Beings",
                    "Objects","Geographic Areas","Concepts & Ideas",
                    "Activities & Behaviors"]
  types_included = the 7 remaining groups:
    Phenomena, Devices, Genes & Molecular Sequences, Disorders,
    Physiology, Anatomy, Chemicals & Drugs

This script reproduces that exact logic against the same source file
(data/aci-bench-main/resources/semantic_types.txt) and saves the resulting
TUI list to umls/aci_bench_tui_codes.json with full provenance.

Run from project root in venv_metrics:
    source venv_metrics/bin/activate
    python scripts/umls_02_extract_aci_tui.py
"""
import json
import hashlib
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

ROOT       = Path(__file__).resolve().parent.parent
ACI_SEMTXT = ROOT / "data/aci-bench-main/resources/semantic_types.txt"
ACI_SEMPY  = ROOT / "data/aci-bench-main/evaluation/semantics.py"
OUT_JSON   = ROOT / "umls/aci_bench_tui_codes.json"
OUT_TXT    = ROOT / "umls/aci_bench_tui_codes.txt"

# Exclusion list — exact copy from aci-bench/evaluation/semantics.py
TYPES_TO_EXCLUDE = [
    "Procedures",
    "Organizations",
    "Occupations",
    "Living Beings",
    "Objects",
    "Geographic Areas",
    "Concepts & Ideas",
    "Activities & Behaviors",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("UMLS 02 — Extract ACI-Bench TUI list (canonical 66-code subset)")
    print("=" * 70)

    if not ACI_SEMTXT.exists():
        raise SystemExit(f"FATAL: {ACI_SEMTXT} not found — copy aci-bench-main into data/")

    # Read SemGroups text — format: "ABBR|Group|TUI|Type Name" per line
    raw_lines = ACI_SEMTXT.read_text().splitlines()
    print(f"\nSource     : {ACI_SEMTXT.relative_to(ROOT)}")
    print(f"Total lines: {len(raw_lines)}")

    by_group = defaultdict(list)   # {Group : [(TUI, Type Name), ...]}
    included = []                  # [{group, tui, type_name}, ...]
    excluded_groups = set()

    for line in raw_lines:
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        abbr, group, tui, type_name = parts[0], parts[1], parts[2], parts[3]
        by_group[group].append((tui, type_name))
        if group in TYPES_TO_EXCLUDE:
            excluded_groups.add(group)
        else:
            included.append({"group": group, "abbr": abbr, "tui": tui, "type_name": type_name})

    # Order: group → list of TUIs (sorted by TUI)
    by_included_group = defaultdict(list)
    for item in included:
        by_included_group[item["group"]].append(item["tui"])
    for g in by_included_group:
        by_included_group[g].sort()

    all_tuis = sorted({item["tui"] for item in included})

    print(f"\nGroups in source file: {len(by_group)}")
    print(f"Groups excluded      : {sorted(excluded_groups)}  ({len(excluded_groups)})")
    print(f"Groups included      : {sorted(by_included_group.keys())}  ({len(by_included_group)})")
    print(f"\nTUI count per included group:")
    for g in sorted(by_included_group.keys()):
        print(f"  {g:30s} {len(by_included_group[g]):>3d}  ({', '.join(by_included_group[g])})")
    print(f"\nTotal included TUIs  : {len(all_tuis)}")

    # ── Save JSON with full provenance ───────────────────────────────────────
    record = {
        "generated_at":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_file":        str(ACI_SEMTXT.relative_to(ROOT)),
        "source_file_sha256": sha256_file(ACI_SEMTXT),
        "method_reference":   str(ACI_SEMPY.relative_to(ROOT)),
        "method_reference_sha256": sha256_file(ACI_SEMPY) if ACI_SEMPY.exists() else None,
        "exclusion_list":     TYPES_TO_EXCLUDE,
        "included_groups":    sorted(by_included_group.keys()),
        "n_included_tuis":    len(all_tuis),
        "tuis_by_group":      {g: by_included_group[g] for g in sorted(by_included_group.keys())},
        "all_tuis_sorted":    all_tuis,
        "included_details":   included,
    }
    OUT_JSON.write_text(json.dumps(record, indent=2))
    print(f"\nSaved JSON: {OUT_JSON}")

    # ── Save human-readable TXT ──────────────────────────────────────────────
    with open(OUT_TXT, "w") as f:
        f.write(f"ACI-Bench TUI Code List\n")
        f.write(f"Generated : {record['generated_at']}\n")
        f.write(f"Source    : {record['source_file']}  (SHA-256: {record['source_file_sha256']})\n")
        f.write(f"Method    : {record['method_reference']}\n")
        f.write(f"Total     : {len(all_tuis)} TUI codes across {len(by_included_group)} groups\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Excluded groups:\n")
        for g in TYPES_TO_EXCLUDE:
            f.write(f"  - {g}\n")
        f.write(f"\nIncluded groups + TUIs:\n")
        for g in sorted(by_included_group.keys()):
            f.write(f"\n  {g}  ({len(by_included_group[g])} TUIs)\n")
            for item in included:
                if item["group"] == g:
                    f.write(f"    {item['tui']}  {item['type_name']}\n")
    print(f"Saved TXT : {OUT_TXT}")

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()
