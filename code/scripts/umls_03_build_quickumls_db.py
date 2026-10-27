"""
Step 3/4 — Build the QuickUMLS database from extracted UMLS 2026AA META/.

QuickUMLS builds two indices:
  - simstring index (over surface forms)
  - cui-semtypes mapping (CUI → set of TUIs)

We use the `unqlite` backend (no system-level leveldb dependency required).
Language filter: English (ENG) only.

Pre-flight checks (script aborts if any fails):
  1. umls/META/MRCONSO.RRF exists and non-empty
  2. umls/META/MRSTY.RRF exists and non-empty
  3. quickumls_simstring + quickumls + leveldb stub all import (Python 3.12 patches applied)
  4. Target directory umls/quickumls-db/ either does not exist OR is empty

Run from project root in venv_metrics:
    source venv_metrics/bin/activate
    python scripts/umls_03_build_quickumls_db.py

Expected runtime: ~10 minutes on this hardware.
Expected DB size: ~1.5 GB.
"""
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT     = Path(__file__).resolve().parent.parent
META_DIR = ROOT / "umls/META"
DB_DIR   = ROOT / "umls/quickumls-db"

REQUIRED_FILES = ["MRCONSO.RRF", "MRSTY.RRF"]


def main():
    print("=" * 70)
    print("UMLS 03 — Build QuickUMLS DB (unqlite backend, ENG language)")
    print("=" * 70)

    # ── Pre-flight ───────────────────────────────────────────────────────────
    print("\n[Pre-flight checks]")
    for fname in REQUIRED_FILES:
        path = META_DIR / fname
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"  FAIL: missing or empty {path}")
        print(f"  [OK]  {fname}  ({path.stat().st_size/1024**2:.0f} MB)")

    # Confirm quickumls imports
    try:
        import quickumls          # noqa
        import quickumls_simstring  # noqa
        import leveldb            # noqa  (stub)
        print(f"  [OK]  quickumls / simstring / leveldb-stub all importable")
    except Exception as e:
        raise SystemExit(f"  FAIL importing quickumls stack: {e}\n  Apply Python 3.12 patches first.")

    # Refuse to overwrite a non-empty DB directory
    if DB_DIR.exists() and any(DB_DIR.iterdir()):
        raise SystemExit(
            f"  FAIL: {DB_DIR} already exists and is non-empty. "
            f"Delete it first if you want to rebuild."
        )
    DB_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build the DB by invoking the quickumls.install module ────────────────
    print(f"\n[Building QuickUMLS DB]")
    print(f"  META source : {META_DIR}")
    print(f"  DB target   : {DB_DIR}")
    print(f"  Language    : ENG")
    print(f"  Backend     : unqlite")
    print(f"\nStarting (this takes ~10 min) ...\n")

    t0  = time.time()
    cmd = [
        sys.executable, "-m", "quickumls.install",
        str(META_DIR),
        str(DB_DIR),
        "--language", "ENG",
        "--database-backend", "unqlite",
    ]
    res = subprocess.run(cmd)
    elapsed = time.time() - t0

    if res.returncode != 0:
        raise SystemExit(f"\nFAIL: quickumls.install exited with code {res.returncode}")
    print(f"\nDB built in {elapsed/60:.1f} min")

    # ── Inspect produced files + hash them ───────────────────────────────────
    print(f"\n[DB contents]")
    db_files = sorted(DB_DIR.iterdir())
    total_size = 0
    hashes = {}
    for p in db_files:
        if not p.is_file():
            continue
        size = p.stat().st_size
        total_size += size
        print(f"  {p.name:<35s}  {size/1024**2:>7.1f} MB")
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        hashes[p.name] = h.hexdigest()
    print(f"\n  TOTAL DB SIZE: {total_size/1024**3:.2f} GB")

    # ── Write build manifest ─────────────────────────────────────────────────
    manifest_path = DB_DIR / "build_manifest.txt"
    with open(manifest_path, "w") as f:
        f.write(f"QuickUMLS DB build manifest\n")
        f.write(f"Generated      : {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        f.write(f"UMLS release   : 2026AA\n")
        f.write(f"META source    : {META_DIR.relative_to(ROOT)}\n")
        f.write(f"Language       : ENG\n")
        f.write(f"Backend        : unqlite\n")
        f.write(f"Build duration : {elapsed/60:.1f} min\n")
        f.write(f"Total DB size  : {total_size:,} bytes ({total_size/1024**3:.2f} GB)\n")
        f.write("=" * 72 + "\n\nFile hashes (SHA-256):\n")
        for name, h in hashes.items():
            f.write(f"  {h}  {name}\n")
    print(f"\nManifest written: {manifest_path}")


if __name__ == "__main__":
    main()
