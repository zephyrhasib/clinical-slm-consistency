"""
quantize_all.py
Quantize all 3 F16 GGUFs into 4 levels each (12 artifacts total).

Quant levels : Q6_K, Q5_K_M, Q4_K_M, Q3_K_M
Source dir   : models/f16/
Output dir   : models/quantized/

Safety checks before any quantization:
  1. Verify llama.cpp git HEAD matches hashes/llama_cpp_commit.txt
  2. Verify SHA-256 of llama-quantize binary matches master_hashes.json
  3. Verify each F16 source SHA-256 matches master_hashes.json

After quantization:
  - Hashes written to hashes/quantized_hashes.txt  (standalone)
  - hashes/master_hashes.json updated with quantized section
  - hashes/master_hashes.txt regenerated in full

Run from project root with venv active:
    python scripts/quantize_all.py

Do NOT run on a different machine or llama.cpp commit without updating
hashes/llama_cpp_commit.txt and rebuilding — SHA-256 of quant outputs
must be bit-identical for reproducibility.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parent.parent
LLAMA_QUANTIZE  = ROOT / "llama.cpp/build/bin/llama-quantize"
COMMIT_FILE     = ROOT / "hashes/llama_cpp_commit.txt"
MASTER_JSON     = ROOT / "hashes/master_hashes.json"
MASTER_TXT      = ROOT / "hashes/master_hashes.txt"
QUANT_HASHES    = ROOT / "hashes/quantized_hashes.txt"
F16_DIR         = ROOT / "models/f16"
QUANT_DIR       = ROOT / "models/quantized"

# ── quant levels ───────────────────────────────────────────────────────────────
LEVELS = ["Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M"]

# ── models ─────────────────────────────────────────────────────────────────────
MODELS = [
    {
        "name":      "Qwen-MediCare-BD",
        "f16_file":  "qwen_medicare_bd_f16.gguf",
        "stem":      "qwen_medicare_bd",
    },
    {
        "name":      "Gemma-3-4B-it",
        "f16_file":  "gemma_3_4b_it_f16.gguf",
        "stem":      "gemma_3_4b_it",
    },
    {
        "name":      "Phi-4-mini-instruct",
        "f16_file":  "phi4_mini_instruct_f16.gguf",
        "stem":      "phi4_mini_instruct",
    },
]


# ── helpers ────────────────────────────────────────────────────────────────────
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def abort(msg: str):
    print(f"\n{'='*60}")
    print(f"ABORT: {msg}")
    print(f"{'='*60}")
    sys.exit(1)


# ── pre-flight checks ──────────────────────────────────────────────────────────
def preflight():
    print("=" * 60)
    print("PRE-FLIGHT CHECKS")
    print("=" * 60)

    # 1. llama-quantize binary exists
    if not LLAMA_QUANTIZE.exists():
        abort(f"llama-quantize not found at {LLAMA_QUANTIZE}\nBuild llama.cpp first.")

    # 2. commit file exists
    if not COMMIT_FILE.exists():
        abort(f"Commit record not found at {COMMIT_FILE}")

    # 3. master_hashes.json exists
    if not MASTER_JSON.exists():
        abort(f"Master hash file not found at {MASTER_JSON}\nRun hash registry script first.")

    with open(MASTER_JSON) as f:
        master = json.load(f)

    # 4. Verify llama.cpp git HEAD matches recorded commit
    expected_commit = COMMIT_FILE.read_text().strip()
    actual_commit   = subprocess.check_output(
        ["git", "-C", str(ROOT / "llama.cpp"), "rev-parse", "HEAD"],
        text=True
    ).strip()

    print(f"  llama.cpp commit (recorded) : {expected_commit}")
    print(f"  llama.cpp commit (git HEAD)  : {actual_commit}")

    if actual_commit != expected_commit:
        abort(
            f"llama.cpp commit mismatch!\n"
            f"  Recorded : {expected_commit}\n"
            f"  Actual   : {actual_commit}\n"
            f"Checkout the correct commit or update hashes/llama_cpp_commit.txt."
        )
    print("  [OK] llama.cpp commit verified")

    # 5. Verify llama-quantize binary SHA-256
    recorded_bin_sha = master["llama_cpp"]["binaries"]["llama-quantize"]
    actual_bin_sha   = sha256_of(LLAMA_QUANTIZE)

    print(f"  llama-quantize SHA-256 (recorded) : {recorded_bin_sha}")
    print(f"  llama-quantize SHA-256 (actual)   : {actual_bin_sha}")

    if actual_bin_sha != recorded_bin_sha:
        abort(
            f"llama-quantize binary SHA-256 mismatch!\n"
            f"  Recorded : {recorded_bin_sha}\n"
            f"  Actual   : {actual_bin_sha}\n"
            f"Rebuild llama.cpp at the pinned commit and update master_hashes.json."
        )
    print("  [OK] llama-quantize binary verified")

    # 6. Verify each F16 source against master_hashes.json
    print("\n  Verifying F16 source GGUFs...")
    for model in MODELS:
        f16_path = F16_DIR / model["f16_file"]
        if not f16_path.exists():
            abort(f"F16 source not found: {f16_path}")

        recorded_sha = master["models"]["f16"][model["f16_file"]]["sha256"]
        actual_sha   = sha256_of(f16_path)

        if actual_sha != recorded_sha:
            abort(
                f"F16 SHA-256 mismatch for {model['f16_file']}!\n"
                f"  Recorded : {recorded_sha}\n"
                f"  Actual   : {actual_sha}"
            )
        print(f"  [OK] {model['f16_file']}")

    print("\nAll pre-flight checks passed. Starting quantization.\n")
    return master


# ── quantize ───────────────────────────────────────────────────────────────────
def quantize(master: dict) -> list:
    QUANT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = []

    for model in MODELS:
        f16_path = F16_DIR / model["f16_file"]
        print(f"\n{'='*60}")
        print(f"Model: {model['name']}")
        print(f"{'='*60}")

        for level in LEVELS:
            out_name = f"{model['stem']}_{level}.gguf"
            out_path = QUANT_DIR / out_name

            if out_path.exists():
                print(f"\n  {level}: already exists — skipping quantization, re-hashing")
            else:
                print(f"\n  {level}: quantizing...")
                t0  = time.time()
                cmd = [str(LLAMA_QUANTIZE), str(f16_path), str(out_path), level]
                print(f"    cmd: {' '.join(cmd)}")

                result = subprocess.run(cmd, capture_output=True, text=True)
                elapsed = time.time() - t0

                if result.returncode != 0:
                    print(f"    FAILED (rc={result.returncode})")
                    print(f"    stderr:\n{result.stderr[-1500:]}")
                    continue

                print(f"    Done in {elapsed/60:.1f} min")

            sha      = sha256_of(out_path)
            size_gb  = round(out_path.stat().st_size / 1e9, 3)
            print(f"    {out_name}  |  {size_gb} GB  |  {sha}")

            artifacts.append({
                "model":        model["name"],
                "level":        level,
                "filename":     out_name,
                "size_gb":      size_gb,
                "sha256":       sha,
                "derived_from": model["f16_file"],
                "llama_cpp_commit": master["llama_cpp"]["commit"],
            })

    return artifacts


# ── write hashes ───────────────────────────────────────────────────────────────
def write_hashes(artifacts: list, master: dict):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Standalone quantized_hashes.txt
    with open(QUANT_HASHES, "w") as f:
        f.write(f"Quantized GGUF Hashes\n")
        f.write(f"Generated        : {timestamp}\n")
        f.write(f"llama.cpp tag    : {master['llama_cpp']['tag']}\n")
        f.write(f"llama.cpp commit : {master['llama_cpp']['commit']}\n")
        f.write(f"llama-quantize   : {master['llama_cpp']['binaries']['llama-quantize']}\n")
        f.write("=" * 72 + "\n\n")
        current_model = None
        for a in artifacts:
            if a["model"] != current_model:
                current_model = a["model"]
                f.write(f"{current_model}\n" + "-" * 40 + "\n")
            f.write(f"  {a['sha256']}  {a['filename']}  ({a['size_gb']} GB)\n")
    print(f"\nWritten: {QUANT_HASHES}")

    # 2. Update master_hashes.json
    master["models"]["quantized"] = {}
    for a in artifacts:
        master["models"]["quantized"][a["filename"]] = {
            "sha256":           a["sha256"],
            "size_gb":          a["size_gb"],
            "model":            a["model"],
            "level":            a["level"],
            "derived_from":     a["derived_from"],
            "llama_cpp_commit": a["llama_cpp_commit"],
        }
    master["generated_at"] = timestamp

    with open(MASTER_JSON, "w") as f:
        json.dump(master, f, indent=2)
    print(f"Updated: {MASTER_JSON}")

    # 3. Regenerate master_hashes.txt in full
    with open(MASTER_TXT, "w") as f:
        f.write(f"Master Hash Record\n")
        f.write(f"Generated : {timestamp}\n")
        f.write("=" * 72 + "\n\n")

        lc = master["llama_cpp"]
        f.write("LLAMA.CPP BUILD\n" + "-" * 72 + "\n")
        f.write(f"Tag         : {lc['tag']}\n")
        f.write(f"Commit      : {lc['commit']}\n")
        f.write(f"CUDA Arch   : {lc['cuda_arch']}\n")
        f.write(f"CMake Flags : {lc['cmake_flags']}\n\n")
        f.write("Binaries:\n")
        for name, sha in lc["binaries"].items():
            f.write(f"  {sha}  {name}\n")

        f.write("\nF16 GGUFs\n" + "-" * 72 + "\n")
        for fname, info in master["models"]["f16"].items():
            f.write(f"  {info['sha256']}  {fname}  ({info['size_gb']} GB)\n")

        f.write("\nQUANTIZED GGUFs\n" + "-" * 72 + "\n")
        current_model = None
        for fname, info in master["models"]["quantized"].items():
            if info["model"] != current_model:
                current_model = info["model"]
                f.write(f"\n  [{current_model}]\n")
            f.write(f"  {info['sha256']}  {fname}  ({info['size_gb']} GB)\n")

    print(f"Updated: {MASTER_TXT}")


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.chdir(ROOT)

    master    = preflight()
    artifacts = quantize(master)

    if not artifacts:
        abort("No artifacts produced — check errors above.")

    write_hashes(artifacts, master)

    print(f"\n{'='*60}")
    print(f"DONE — {len(artifacts)} artifacts quantized and hashed")
    print(f"{'='*60}")
    print(f"  hashes/quantized_hashes.txt  (standalone)")
    print(f"  hashes/master_hashes.json    (updated)")
    print(f"  hashes/master_hashes.txt     (regenerated)")
    print(f"\nExpected: 12 artifacts (3 models × 4 levels)")
    if len(artifacts) != 12:
        print(f"WARNING: got {len(artifacts)} — check for skipped models above.")
