"""
Step 3/4 of contamination audit — N-gram memorisation probes for Models B and C.

Method  : Greedy continuation from a 50-word prefix; flag if 10+ consecutive
          words after the prefix match the original ACI-Bench dialogue.
Models  : Gemma-3-4B-it (Q4_K_M), Phi-4-mini-instruct (Q4_K_M)
Binary  : llama-completion (NOT llama-cli)
Mode    : Raw completion (no chat template — --in-prefix '' --in-suffix '')

Pre-specified parameters (must NOT change once fixed):
  PREFIX_WORDS    = 50
  MATCH_WORDS     = 10
  MAX_NEW_TOKENS  = 100
  PROBE_TEMP      = 0.0   (greedy — maximises verbatim reproduction probability)

Output : data/audit/model_b_probe_audit.json
         data/audit/model_c_probe_audit.json

Run from project root: python scripts/audit_03_models_bc_probe.py
"""
import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── pre-specified parameters (do not change once fixed) ──────────
PREFIX_WORDS    = 50
MATCH_WORDS     = 10
MAX_NEW_TOKENS  = 100
PROBE_TEMP      = 0.0
# ─────────────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).resolve().parent.parent
CLI        = ROOT / "llama.cpp/build/bin/llama-completion"
AUDIT_DIR  = ROOT / "data/audit"
STRAT_DIR  = ROOT / "data/stratified"
PROMPT_TMP = Path("/tmp/audit_probe_prompt.txt")

ACIBENCH_COL = "dialogue"

MODELS = [
    {
        "model_id":   "model_b",
        "model_name": "Gemma-3-4B-it",
        "gguf":       ROOT / "models/quantized/gemma_3_4b_it_Q4_K_M.gguf",
    },
    {
        "model_id":   "model_c",
        "model_name": "Phi-4-mini-instruct",
        "gguf":       ROOT / "models/quantized/phi4_mini_instruct_Q4_K_M.gguf",
    },
]


def get_prefix(text: str, n_words: int) -> str:
    """First n_words words of text as a string."""
    return " ".join(str(text).split()[:n_words])


def get_expected_continuation(full_text: str, prefix: str, n_words: int) -> str:
    """N_words words immediately following prefix in the original text."""
    prefix_len = len(prefix.split())
    return " ".join(str(full_text).split()[prefix_len : prefix_len + n_words])


def check_verbatim_match(expected: str, generated: str) -> bool:
    """
    True if `expected` appears as case-insensitive substring in `generated`.
    `expected` is, by construction, the words AFTER the prefix in the original
    dialogue. The prefix is part of `generated` (echoed by llama-completion)
    but cannot contain `expected` — so substring match has no false positives
    from prompt echo.
    """
    if len(expected.split()) < MATCH_WORDS:
        return False
    return expected.lower() in generated.lower()


def clean_output(raw: str) -> str:
    """Strip llama-completion's interactive stubs from captured stdout."""
    out = raw
    out = re.sub(r"\s*>\s*EOF\b.*$", "", out, flags=re.DOTALL | re.IGNORECASE)
    out = re.sub(r"\[\s*Prompt:.*$",  "", out, flags=re.DOTALL)
    return out.strip()


def run_probe(gguf_path: Path, prompt: str) -> tuple[str, str]:
    """
    Run llama-completion in pure raw-completion mode.
    Returns (cleaned_stdout, error_marker_if_any).
    """
    PROMPT_TMP.write_text(prompt, encoding="utf-8")
    cmd = [
        str(CLI),
        "-m", str(gguf_path),
        "-f", str(PROMPT_TMP),
        "-n", str(MAX_NEW_TOKENS),
        "--temp", str(PROBE_TEMP),
        "-ngl", "99",
        "--no-warmup",
        "--simple-io",
        "--in-prefix", "",
        "--in-suffix", "",          # disables chat template (per llama.cpp docs)
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=180,
        )
        if result.returncode != 0:
            return "", f"rc={result.returncode}: {result.stderr[-300:]}"
        return clean_output(result.stdout), ""
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT"
    except Exception as e:
        return "", f"EXC: {e}"


# ── Main audit loop ───────────────────────────────────────────────────────────
print("=" * 60)
print("AUDIT 03 — Models B and C · N-gram Memorisation Probes")
print(f"Prefix: {PREFIX_WORDS} words  |  Match: {MATCH_WORDS} words  |  Temp: {PROBE_TEMP}  |  Max new: {MAX_NEW_TOKENS}")
print(f"Binary: {CLI.name}  |  Mode: raw completion (chat template disabled)")
print("=" * 60)

main_df = pd.read_csv(STRAT_DIR / "main_inputs.csv")
print(f"Loaded {len(main_df)} main items.")

for model_cfg in MODELS:
    print()
    print("-" * 60)
    print(f"Model: {model_cfg['model_name']}")
    print(f"GGUF : {model_cfg['gguf'].name}")
    print("-" * 60)

    if not model_cfg["gguf"].exists():
        print(f"  ABORT: GGUF not found at {model_cfg['gguf']}")
        continue

    flagged       = []
    not_flagged   = []
    errors        = []
    short_skipped = []
    t_start       = time.time()

    for _, row in tqdm(main_df.iterrows(), total=len(main_df), desc=model_cfg["model_id"]):
        full_dialogue = str(row[ACIBENCH_COL])
        input_id      = str(row["input_id"])
        prefix        = get_prefix(full_dialogue, PREFIX_WORDS)
        expected      = get_expected_continuation(full_dialogue, prefix, MATCH_WORDS)

        if len(expected.split()) < MATCH_WORDS:
            short_skipped.append(input_id)
            not_flagged.append(input_id)
            continue

        generated, err = run_probe(model_cfg["gguf"], prefix)

        if err:
            errors.append({"input_id": input_id, "error": err})
            continue

        if check_verbatim_match(expected, generated):
            flagged.append({
                "input_id":          input_id,
                "corpus_idx":        int(row["corpus_idx"]),
                "prefix_used":       prefix,
                "expected_words":    expected,
                "generated_excerpt": generated[:500],
            })
        else:
            not_flagged.append(input_id)

    elapsed = time.time() - t_start

    results = {
        "model_id":    model_cfg["model_id"],
        "model_name":  model_cfg["model_name"],
        "gguf_used":   model_cfg["gguf"].name,
        "method":      "n-gram memorisation probe",
        "parameters": {
            "prefix_words":          PREFIX_WORDS,
            "match_words":           MATCH_WORDS,
            "max_new_tokens":        MAX_NEW_TOKENS,
            "probe_temperature":     PROBE_TEMP,
            "binary":                "llama-completion",
            "raw_completion_mode":   "enforced via --in-prefix '' --in-suffix ''",
            "rationale_temperature": (
                "T=0.0 (greedy) maximises verbatim reproduction probability. "
                "Any memorisation not manifesting at T=0.0 is too weak to "
                "affect main run results at T=0.3."
            ),
        },
        "evaluation_items":  int(len(main_df)),
        "flagged_count":     int(len(flagged)),
        "flagged_pct":       round(len(flagged) / len(main_df) * 100, 2),
        "error_count":       int(len(errors)),
        "short_skipped":     short_skipped,
        "elapsed_seconds":   round(elapsed, 1),
        "flagged_items":     flagged,
        "errors":            errors,
    }

    out_path = AUDIT_DIR / f"{model_cfg['model_id']}_probe_audit.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Elapsed     : {elapsed/60:.1f} min")
    print(f"  Total items : {len(main_df)}")
    print(f"  Flagged     : {len(flagged)} ({results['flagged_pct']}%)")
    print(f"  Errors      : {len(errors)}")
    if len(flagged) == 0 and len(errors) == 0:
        print("  RESULT: CLEAN — no memorisation detected.")
    elif len(flagged) > 0:
        print(f"  RESULT: {len(flagged)} ITEMS FLAGGED")
        print(f"  Flagged input_ids: {[f['input_id'] for f in flagged]}")
    print(f"  Saved: {out_path}")

# Clean up temp prompt file
PROMPT_TMP.unlink(missing_ok=True)
print("\nModels B and C probe audit complete.")
