"""
Generation Test Script (Batch 1)
Model : Gemma-3-4B-it  |  Quant : Q6_K

Same v4 pipeline as gen_qwen_Q6_K_test.py, adapted for Gemma-3:
  - Gemma chat template (turn-based, not ChatML)
  - Extraction boundary uses Gemma role text 'model'
  - Gemma-specific turn-marker reverse prompts (no Qwen injection strings)
  - Anomaly detection list trimmed to general drift patterns (no Qwen-specific
    injection strings; we have not yet observed Gemma's failure patterns)

Scope:
  - 5 short + 5 long from pilot_inputs.csv
  - 4 seeds (0, 1, 2, 3)
  - Total: 40 generations

Generation parameters: identical to Qwen (pre-specified, applies to all 12 scripts).

Pre-flight (aborts on any mismatch):
  - llama.cpp git HEAD == hashes/llama_cpp_commit.txt
  - SHA-256 of llama-completion binary == master_hashes.json
  - SHA-256 of gemma_3_4b_it_Q5_K_M.gguf == master_hashes.json

Chat template — manually wrapped here. llama-completion is invoked with
--in-prefix "" --in-suffix "" so it does NOT auto-apply Gemma's embedded
template (Gemma applies its template by default — confirmed during the
contamination audit). We have full control.

Run from project root (main venv, NOT venv_metrics):
    source venv/bin/activate
    python scripts/gen_gemma_Q6_K_test.py
"""
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

# ─── identity ────────────────────────────────────────────────────────────────
MODEL_NAME   = "Gemma-3-4B-it"
QUANT_LEVEL  = "Q5_K_M"
GGUF_NAME    = "gemma_3_4b_it_Q5_K_M.gguf"
TEMPLATE     = "gemma3_turns"
SCRIPT_VER   = 8   # v8: add is_looped column (8-word n-gram repetition detect);
                   #     expand pilot scope 5+5 → 10+10 (full pilot_inputs.csv).

# ─── paths ───────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
LLAMA_BIN     = ROOT / "llama.cpp/build/bin/llama-completion"
MODEL_PATH    = ROOT / "models/quantized" / GGUF_NAME
COMMIT_FILE   = ROOT / "hashes/llama_cpp_commit.txt"
MASTER_JSON   = ROOT / "hashes/master_hashes.json"
PILOT_CSV     = ROOT / "data/stratified/pilot_inputs.csv"
OUT_DIR       = ROOT / "outputs/test_batch_1"
OUT_CSV       = OUT_DIR / f"gen_gemma_3_4b_it_{QUANT_LEVEL}_test.csv"
OUT_META      = OUT_DIR / f"gen_gemma_3_4b_it_{QUANT_LEVEL}_test_metadata.json"
PROMPT_FILE   = Path(f"/tmp/gen_gemma_3_4b_it_{QUANT_LEVEL}_prompt.txt")

# ─── scope ───────────────────────────────────────────────────────────────────
N_PILOT_PER_STRATUM = 10   # full pilot: 10 short + 10 long = 20 items
SEEDS               = [0, 1, 2, 3]

# ─── pre-specified generation parameters ────────────────────────────────────
TEMP          = 0.3
TOP_P         = 0.95
TOP_K         = 40
REPEAT_PEN    = 1.25
N_PREDICT     = 512
CTX_SIZE      = 8192
N_GPU_LAYERS  = 999

# ─── pre-specified zero-shot instruction (IDENTICAL across all 12 scripts) ──
# v6: reverted to v4 wording (v5 broke Qwen). Markdown/meta-prefix cleanup is
# now handled post-generation by strip_meta_and_markdown().
INSTRUCTION = (
    "Summarize the following doctor-patient conversation as a structured "
    "clinical note. Write in plain prose without section headers, markdown, "
    "asterisks, bullet points, numbered lists, or bold text."
)

# ─── stop strings (REVERSE_PROMPTS) — Gemma-specific ─────────────────────────
# Gemma-3 chat template uses <start_of_turn>...<end_of_turn> markers (not ChatML).
# NO Qwen-specific injection strings here — those were empirically observed in
# Qwen-MediCare-BD pilot data; Gemma has different weights and will produce
# (if any) different injection patterns. Per design principle:
# "only add strings you have empirically observed for THAT model".
REVERSE_PROMPTS = [
    # Gemma-3 turn markers (primary stops)
    "<end_of_turn>",
    "<start_of_turn>",
    # Dialogue artifact stops
    "[doctor]",
    "[patient]",
    "patient_guest",
    # AI disclaimer stops (general)
    "as an AI",
    "As an AI",
    "language model",
]

# ─── general anomaly flagging (v4 — pre-specified, identical to Qwen) ───────
# Patterns trimmed of Qwen-specific explicit injection strings. General drift
# patterns (date-cap, 2nd-person imperative, role-tag-at-start) retained
# unchanged — they apply to any model that exhibits these failure modes.

LOW_WORD_COUNT_THRESHOLD = 20
LOOP_NGRAM_SIZE          = 8   # pre-specified: an n-word phrase repeated 2+ times = loop

ANOMALY_PATTERNS_LC = [
    # general AI disclaimer drift (apply to all 12 scripts)
    r"as an (ai|artificial intelligence|language model)",
    r"i (cannot|can't|am unable to)",
    r"i am (an ai|a language model)",
    # generic 2nd-person imperative drift (any model can do this)
    r"\byou (are|see|have|will|can|should|might|need|must) ",
]

ANOMALY_PATTERNS_CS = [
    # date-then-capital drift: '1957-09-2Y', '9/2Y'
    r"\d[-/]\d{1,2}[A-Z]",
    # generic digit-immediately-followed-by-capitalised-word
    r"\d[A-Z][a-z]",
    # starts-with-role-tag drift
    r"^\s*\[(doctor|patient|patient_guest|physician|nurse|guest|assistant)\]",
]

_anomaly_lc = [re.compile(p) for p in ANOMALY_PATTERNS_LC]
_anomaly_cs = [re.compile(p) for p in ANOMALY_PATTERNS_CS]


def is_anomalous(text: str, word_count: int) -> bool:
    if not text:
        return True
    if word_count < LOW_WORD_COUNT_THRESHOLD:
        return True
    text_lower = text.lower()
    if any(rx.search(text_lower) for rx in _anomaly_lc):
        return True
    if any(rx.search(text) for rx in _anomaly_cs):
        return True
    return False


def has_loop(text: str, n: int = LOOP_NGRAM_SIZE) -> bool:
    """True if any n-word phrase appears 2+ times in text. See Qwen v8 docstring."""
    if not text:
        return False
    words = text.lower().split()
    if len(words) < 2 * n:
        return False
    seen = set()
    for i in range(len(words) - n + 1):
        phrase = " ".join(words[i:i + n])
        if phrase in seen:
            return True
        seen.add(phrase)
    return False


# ─── helpers ─────────────────────────────────────────────────────────────────
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def abort(msg: str):
    print(f"\n{'=' * 60}\nABORT: {msg}\n{'=' * 60}")
    sys.exit(1)


def flatten(text: str) -> str:
    """Collapse all whitespace (newlines, tabs) to single spaces."""
    return " ".join(str(text).split())


# v5 — pre-specified post-generation cleanup (identical to Qwen v5).
# Removes meta-prefixes and markdown formatting that the instruction tells
# the model to avoid but Gemma in particular emits anyway. Applied uniformly
# to ALL 12 scripts.

META_PREFIX_PATTERNS = [
    r"^\s*(?:Here['’]s|Here is)\s+(?:a |the )?(?:structured |brief |concise |comprehensive |detailed |formatted )?(?:clinical )?(?:summary|note|documentation)[^.:\n]*[.:\-—]\s*",
    r"^\s*(?:Below is|The following is|Following is)\s+[^.:\n]{0,80}[.:\-—]\s*",
    r"^\s*(?:Sure|Of course|Certainly|Okay|OK)[,.!]?\s+(?:here['’]?s|here is)\s+[^.:\n]{0,80}[.:\-—]\s*",
    r"^\s*(?:Clinical\s+)?Note\s*[:\-—]\s*",
    r"^\s*(?:Patient\s+)?Summary\s*[:\-—]\s*",
]
_meta_prefix_re = [re.compile(p, re.IGNORECASE) for p in META_PREFIX_PATTERNS]


def strip_meta_and_markdown(text: str) -> str:
    """Identical to Qwen v5 — see that file for full docstring."""
    if not text:
        return text
    for rx in _meta_prefix_re:
        new = rx.sub("", text, count=1)
        if new != text:
            text = new
            break
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    text = text.replace("**", "")  # orphan markers (started ** but didn't close)
    text = re.sub(r"(?m)^#+\s*",          "", text)
    text = re.sub(r"(?m)^\s*[-•]\s+",     "", text)
    text = re.sub(r"(?m)^\s*\*\s+",       "", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+",  "", text)
    return text.strip()


def build_prompt(dialogue: str) -> str:
    """Wrap the instruction + dialogue in Gemma-3 chat-turn format."""
    return (
        f"<start_of_turn>user\n"
        f"{INSTRUCTION}\n\n"
        f"Conversation:\n{dialogue}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def extract_completion(raw_stdout: str, prompt: str) -> str:
    """
    Strip the echoed prompt + trailing artifacts from llama-completion stdout.

    llama-completion echoes the prompt; special tokens (<start_of_turn>,
    <end_of_turn>) are typically stripped during echo. The boundary marker
    varies by Gemma's stdout formatting — observed forms include:
      '<start_of_turn>model\\n'  (special-token preserved)
      '\\nmodel\\n'              (special-token stripped, newline after)
      '\\nmodel '                (special-token stripped, space after — seen on Q6_K)

    Strategy: try the special-token form first; otherwise regex-split on
    '\\nmodel' followed by any whitespace, take everything after the LAST match.
    """
    text = raw_stdout
    # 1. Try special-token form
    if "<start_of_turn>model" in text:
        text = text.rsplit("<start_of_turn>model", 1)[-1].lstrip("\n ")
    else:
        # 2. Regex split on '\nmodel' + any whitespace (space, newline, tab)
        #    Take the LAST split — the boundary is the model-turn marker, which
        #    appears once in the echo of our prompt.
        parts = re.split(r"\nmodel[ \t\n]+", text)
        if len(parts) > 1:
            text = parts[-1]
    # Strip closing turn marker (matched by reverse-prompt)
    text = text.split("<end_of_turn>", 1)[0]
    # Strip llama-completion interactive stubs
    text = re.sub(r"\s*>\s*EOF\b.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\[\s*Prompt:.*$",  "", text, flags=re.DOTALL)
    return text.strip()


# ─── pre-flight ──────────────────────────────────────────────────────────────
def preflight() -> dict:
    print("=" * 60)
    print(f"PRE-FLIGHT — {MODEL_NAME} {QUANT_LEVEL} test")
    print("=" * 60)

    if not LLAMA_BIN.exists():     abort(f"binary not found: {LLAMA_BIN}")
    if not MODEL_PATH.exists():    abort(f"GGUF not found:   {MODEL_PATH}")
    if not COMMIT_FILE.exists():   abort(f"missing:          {COMMIT_FILE}")
    if not MASTER_JSON.exists():   abort(f"missing:          {MASTER_JSON}")
    if not PILOT_CSV.exists():     abort(f"missing:          {PILOT_CSV}")

    master = json.loads(MASTER_JSON.read_text())

    expected_commit = COMMIT_FILE.read_text().strip()
    actual_commit   = subprocess.check_output(
        ["git", "-C", str(ROOT / "llama.cpp"), "rev-parse", "HEAD"], text=True,
    ).strip()
    if actual_commit != expected_commit:
        abort(f"llama.cpp commit mismatch\n  recorded: {expected_commit}\n  actual:   {actual_commit}")
    print(f"  [OK] llama.cpp commit: {expected_commit}")

    recorded_bin_sha = master["llama_cpp"]["binaries"]["llama-completion"]
    actual_bin_sha   = sha256_of(LLAMA_BIN)
    if actual_bin_sha != recorded_bin_sha:
        abort(f"llama-completion SHA mismatch")
    print(f"  [OK] llama-completion SHA: {recorded_bin_sha}")

    recorded_gguf_sha = master["models"]["quantized"][GGUF_NAME]["sha256"]
    actual_gguf_sha   = sha256_of(MODEL_PATH)
    if actual_gguf_sha != recorded_gguf_sha:
        abort(f"GGUF SHA mismatch")
    print(f"  [OK] {GGUF_NAME} SHA: {recorded_gguf_sha}")

    return {
        "llama_cpp_commit":     expected_commit,
        "llama_completion_sha": recorded_bin_sha,
        "gguf_sha":             recorded_gguf_sha,
    }


# ─── generation ──────────────────────────────────────────────────────────────
def run_one(prompt: str, seed: int) -> tuple[str, float, int]:
    PROMPT_FILE.write_text(prompt, encoding="utf-8")
    cmd = [
        str(LLAMA_BIN),
        "-m", str(MODEL_PATH),
        "--file", str(PROMPT_FILE),
        "-c", str(CTX_SIZE),
        "-n", str(N_PREDICT),
        "--temp", str(TEMP),
        "--top-p", str(TOP_P),
        "--top-k", str(TOP_K),
        "--seed", str(seed),
        "--repeat-penalty", str(REPEAT_PEN),
        "-ngl", str(N_GPU_LAYERS),
        "--no-warmup",
        "--simple-io",
        "--in-prefix", "",          # disable Gemma's auto chat template
        "--in-suffix", "",
    ]
    for rp in REVERSE_PROMPTS:
        cmd.extend(["--reverse-prompt", rp])

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            stdin=subprocess.DEVNULL, timeout=300)
    elapsed = time.time() - t0
    return extract_completion(result.stdout, prompt), elapsed, result.returncode


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    meta_pre = preflight()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pilot_full = pd.read_csv(PILOT_CSV)
    short_items = pilot_full[pilot_full["length_stratum"] == "short"].head(N_PILOT_PER_STRATUM)
    long_items  = pilot_full[pilot_full["length_stratum"] == "long"].head(N_PILOT_PER_STRATUM)
    pilot_df    = pd.concat([short_items, long_items], ignore_index=True)
    assert len(pilot_df) == 2 * N_PILOT_PER_STRATUM

    print(f"\nPilot items     : {len(pilot_df)} "
          f"({N_PILOT_PER_STRATUM} short + {N_PILOT_PER_STRATUM} long)")
    print(f"  short IDs     : {short_items['input_id'].tolist()}")
    print(f"  long IDs      : {long_items['input_id'].tolist()}")
    print(f"Seeds           : {SEEDS}")
    print(f"Total to generate : {len(pilot_df) * len(SEEDS)}")

    # Resume support
    if OUT_CSV.exists():
        done_df   = pd.read_csv(OUT_CSV)
        done_keys = set(zip(done_df["input_id"], done_df["seed"]))
        print(f"Resuming — {len(done_df)} rows already done")
    else:
        done_df   = pd.DataFrame()
        done_keys = set()

    rows = []
    started = datetime.now(timezone.utc).isoformat()
    print()

    for _, item in pilot_df.iterrows():
        input_id = item["input_id"]
        dialogue = item["dialogue"]
        prompt   = build_prompt(dialogue)

        for seed in SEEDS:
            if (input_id, seed) in done_keys:
                print(f"  {input_id} seed={seed}: skip (already done)")
                continue

            print(f"  {input_id} seed={seed}: ", end="", flush=True)
            generated_raw, elapsed, rc = run_one(prompt, seed)
            # v8.1 extraction-leak guard (post-Q3 pilot, applied to all 12 scripts)
            if INSTRUCTION[:40].lower() in generated_raw.lower():
                generated_raw = ""
            generated_clean = strip_meta_and_markdown(generated_raw)
            generated_flat  = flatten(generated_clean)
            wc = len(generated_flat.split()) if generated_flat else 0
            anomalous = is_anomalous(generated_flat, wc)
            looped    = has_loop(generated_flat)
            status = "OK" if rc == 0 else f"rc={rc}"
            print(f"{elapsed:5.1f}s  {wc:4d} words  [{status}]")

            rows.append({
                "input_id":         input_id,
                "original_aci_id":  item["original_aci_id"],
                "split":            item["split"],
                "length_stratum":   item["length_stratum"],
                "model":            MODEL_NAME,
                "quant_level":      QUANT_LEVEL,
                "seed":             seed,
                "returncode":       rc,
                "elapsed_seconds":  round(elapsed, 2),
                "word_count":       wc,
                "is_anomalous":     anomalous,
                "is_looped":        looped,
                "generated_text":   generated_flat,
            })

            partial = pd.concat([done_df, pd.DataFrame(rows)], ignore_index=True) \
                      if len(done_df) else pd.DataFrame(rows)
            partial.to_csv(OUT_CSV, index=False)

    finished = datetime.now(timezone.utc).isoformat()

    metadata = {
        "model":                MODEL_NAME,
        "quant_level":          QUANT_LEVEL,
        "gguf_filename":        GGUF_NAME,
        "script":               Path(__file__).name,
        "script_version":       SCRIPT_VER,
        "chat_template":        TEMPLATE,
        "instruction":          INSTRUCTION,
        "generation_params": {
            "temperature":      TEMP,
            "top_p":            TOP_P,
            "top_k":            TOP_K,
            "repeat_penalty":   REPEAT_PEN,
            "n_predict":        N_PREDICT,
            "ctx_size":         CTX_SIZE,
            "n_gpu_layers":     N_GPU_LAYERS,
            "in_prefix":        "",
            "in_suffix":        "",
        },
        "reverse_prompts":      REVERSE_PROMPTS,
        "anomaly_patterns_lc":  ANOMALY_PATTERNS_LC,
        "anomaly_patterns_cs":  ANOMALY_PATTERNS_CS,
        "low_word_count_threshold": LOW_WORD_COUNT_THRESHOLD,
        "loop_ngram_size":      LOOP_NGRAM_SIZE,
        "meta_prefix_patterns": META_PREFIX_PATTERNS,
        "n_pilot_items":        len(pilot_df),
        "n_pilot_short":        N_PILOT_PER_STRATUM,
        "n_pilot_long":         N_PILOT_PER_STRATUM,
        "seeds":                SEEDS,
        "n_generations":        len(pilot_df) * len(SEEDS),
        "llama_cpp_commit":     meta_pre["llama_cpp_commit"],
        "llama_completion_sha": meta_pre["llama_completion_sha"],
        "gguf_sha":             meta_pre["gguf_sha"],
        "started_utc":          started,
        "finished_utc":         finished,
        "output_csv":           str(OUT_CSV.relative_to(ROOT)),
    }
    OUT_META.write_text(json.dumps(metadata, indent=2))

    total = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame(rows)
    print(f"\n{'=' * 60}\nDONE\n{'=' * 60}")
    print(f"  Output CSV  : {OUT_CSV}  ({len(total)} rows)")
    print(f"  Metadata    : {OUT_META}")
    print(f"  Word count stats:")
    print(total["word_count"].describe().round(1).to_string())
    print(f"  Anomalous: {total['is_anomalous'].sum()} / {len(total)}")

    PROMPT_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
