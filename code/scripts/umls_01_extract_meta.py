"""
Step 1/4 — Extract MRCONSO.RRF and MRSTY.RRF from the UMLS 2026AA .nlm archives.

The .nlm files are standard ZIP archives that contain gzipped RRF pieces inside
the `2026AA/META/` path. We extract only the two files QuickUMLS needs:

  MRCONSO.RRF  ← concept atoms (CUI, language, term, source, etc.)
  MRSTY.RRF    ← semantic type assignments (CUI → TUI)

MRCONSO is split into three parts (aa, ab, ac) inside the zip; we decompress
and concatenate them. MRSTY is a single gzipped file.

Inputs:
  umls/2026aa-1-meta.nlm      ← contains MRCONSO parts + MRSTY

Outputs:
  umls/META/MRCONSO.RRF       (~4 GB after concatenation)
  umls/META/MRSTY.RRF         (~50 MB)
  umls/META/extraction_hashes.txt   (SHA-256 of extracted files)

Run from project root in venv_metrics:
    source venv_metrics/bin/activate
    python scripts/umls_01_extract_meta.py
"""
import gzip
import hashlib
import shutil
import zipfile
from pathlib import Path
from datetime import datetime, timezone

ROOT     = Path(__file__).resolve().parent.parent
NLM_ZIP  = ROOT / "umls/2026aa-1-meta.nlm"
META_DIR = ROOT / "umls/META"

# Files needed inside the .nlm zip
CONSO_PARTS = [
    "2026AA/META/MRCONSO.RRF.aa.gz",
    "2026AA/META/MRCONSO.RRF.ab.gz",
    "2026AA/META/MRCONSO.RRF.ac.gz",
]
MRSTY_GZ = "2026AA/META/MRSTY.RRF.gz"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("UMLS 01 — Extract MRCONSO.RRF + MRSTY.RRF from 2026aa-1-meta.nlm")
    print("=" * 70)

    if not NLM_ZIP.exists():
        raise SystemExit(f"FATAL: .nlm archive not found at {NLM_ZIP}")
    META_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nSource archive: {NLM_ZIP.name}  ({NLM_ZIP.stat().st_size/1024**3:.2f} GB)")

    conso_out = META_DIR / "MRCONSO.RRF"
    mrsty_out = META_DIR / "MRSTY.RRF"

    # ── 1. Extract gz pieces from the zip into temp .gz files ────────────────
    print("\n[1/3] Extracting gzipped pieces from .nlm ...")
    tmp_gz_paths = []
    with zipfile.ZipFile(NLM_ZIP) as z:
        names = set(z.namelist())
        for member in CONSO_PARTS + [MRSTY_GZ]:
            if member not in names:
                raise SystemExit(f"FATAL: {member} not found inside {NLM_ZIP.name}")
            dest = META_DIR / Path(member).name
            if dest.exists():
                print(f"  {dest.name:>22s}  already extracted — skip")
            else:
                with z.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                print(f"  {dest.name:>22s}  extracted ({dest.stat().st_size/1024**2:.0f} MB)")
            tmp_gz_paths.append(dest)

    # ── 2. Decompress + concatenate MRCONSO parts in order ───────────────────
    print(f"\n[2/3] Decompressing + concatenating MRCONSO parts ...")
    if conso_out.exists():
        print(f"  {conso_out.name} already present ({conso_out.stat().st_size/1024**3:.2f} GB) — skip")
    else:
        with open(conso_out, "wb") as out_f:
            for part_name in CONSO_PARTS:
                gz_path = META_DIR / Path(part_name).name
                print(f"  decompressing {gz_path.name} ...", end="", flush=True)
                with gzip.open(gz_path, "rb") as in_f:
                    shutil.copyfileobj(in_f, out_f, length=8 * 1024 * 1024)
                print(" done")
        print(f"  Assembled: {conso_out.name}  ({conso_out.stat().st_size/1024**3:.2f} GB)")

    # ── 3. Decompress MRSTY ──────────────────────────────────────────────────
    print(f"\n[3/3] Decompressing MRSTY ...")
    if mrsty_out.exists():
        print(f"  {mrsty_out.name} already present ({mrsty_out.stat().st_size/1024**2:.0f} MB) — skip")
    else:
        gz_path = META_DIR / "MRSTY.RRF.gz"
        with gzip.open(gz_path, "rb") as in_f, open(mrsty_out, "wb") as out_f:
            shutil.copyfileobj(in_f, out_f, length=8 * 1024 * 1024)
        print(f"  Decompressed: {mrsty_out.name}  ({mrsty_out.stat().st_size/1024**2:.0f} MB)")

    # ── Hash the two final files ─────────────────────────────────────────────
    print(f"\nHashing final files ...")
    conso_sha = sha256_file(conso_out)
    mrsty_sha = sha256_file(mrsty_out)
    hashes_file = META_DIR / "extraction_hashes.txt"
    with open(hashes_file, "w") as f:
        f.write(f"UMLS 2026AA — META extraction hashes\n")
        f.write(f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        f.write(f"Source    : {NLM_ZIP.name} (SHA-256 of source not recomputed here)\n")
        f.write("=" * 72 + "\n")
        f.write(f"{conso_sha}  MRCONSO.RRF  ({conso_out.stat().st_size:,} bytes)\n")
        f.write(f"{mrsty_sha}  MRSTY.RRF    ({mrsty_out.stat().st_size:,} bytes)\n")
    print(f"  MRCONSO.RRF : {conso_sha}")
    print(f"  MRSTY.RRF   : {mrsty_sha}")
    print(f"  Written     : {hashes_file}")

    # ── Cleanup intermediate .gz files (optional but saves ~600 MB) ──────────
    print(f"\nRemoving intermediate .gz pieces ...")
    for p in tmp_gz_paths:
        p.unlink()
        print(f"  removed {p.name}")

    print(f"\n{'=' * 70}")
    print(f"Done. META/ ready for QuickUMLS DB build.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
