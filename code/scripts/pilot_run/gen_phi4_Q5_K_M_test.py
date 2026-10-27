"""
Generation Test Script (Batch 1)
Model : Phi-4-mini-instruct  |  Quant : Q6_K

Same v6 pipeline as gen_qwen_Q6_K_test.py / gen_gemma_Q6_K_test.py, adapted
for Phi-4 chat template:
  - Phi-4 turn format: <|user|>...<|end|><|assistant|>...
  - Extraction boundary uses Phi-4 role-text markers
  - Phi-4-specific turn-marker reverse prompts
  - Same INSTRUCTION (v6 — identical across all 12 scripts)
  - Same strip_meta_and_markdown() post-processing
  - Anomaly detection identical to Gemma (no model-specific injection patterns
    yet — only general drift patterns until Phi-4 pilot evidence emerges)

Scope:
  - 5 short + 5 long from pilot_inputs.csv
  - 4 seeds (0, 1, 2, 3)
  - Total: 40 generations

Pre-flight (aborts on any mismatch):
  - llama.cpp git HEAD == hashes/llama_cpp_commit.txt
  - SHA-256 of llama-completion binary == master_hashes.json
  - SHA-256 of phi4_mini_instruct_Q5_K_M.gguf == master_hashes.json

Chat template — manually wrapped, --in-prefix "" --in-suffix "" passed so the
GGUF's embedded template is not auto-applied. Phi-4 did not auto-apply its
template during the contamination audit, but we override defensively for
consistency with the Qwen and Gemma scripts.

Run from project root (main venv, NOT venv_metrics):
    source venv/bin/activate
    python scripts/gen_phi4_Q6_K_test.py
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
MODEL_NAME   = "Phi-4-mini-instruct"
QUANT_LEVEL  = "Q5_K_M"
GGUF_NAME    = "phi4_mini_instruct_Q5_K_M.gguf"
TEMPLATE     = "phi4_turns"
SCRIPT_VER   = 10  # v7: N_PREDICT 512→768 (Phi-4 specific); extraction-leak
                   #     detection (INSTRUCTION[:40] presence); low-wc retry.
                   # v8: add is_looped column (8-word n-gram repetition detect);
                   #     expand pilot scope 5+5 → 10+10 (full pilot_inputs.csv).
                   # v9: REPEAT_PEN 1.25 → 1.40 (Phi-4 specific). v8 pilot
                   #     showed 32/80 outputs with degenerate repetition at
                   #     1.25; the higher penalty is needed to suppress
                   #     semantic loops. Pre-specified as model-specific.
                   # v10: truncate_at_loop() — at first 8-gram appearing ≥3×,
                   #     truncate the output at the START of that 8-gram.
                   #     Phi-4-specific. Pre-specified. The truncated text
                   #     is the genuine clinical content before loop onset.
                   #     `is_looped` is computed on the PRE-truncation text
                   #     so the flag still reflects the model's behavior.

# ─── paths ───────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
LLAMA_BIN     = ROOT / "llama.cpp/build/bin/llama-completion"
MODEL_PATH    = ROOT / "models/quantized" / GGUF_NAME
COMMIT_FILE   = ROOT / "hashes/llama_cpp_commit.txt"
MASTER_JSON   = ROOT / "hashes/master_hashes.json"
PILOT_CSV     = ROOT / "data/stratified/pilot_inputs.csv"
OUT_DIR       = ROOT / "outputs/test_batch_1"
OUT_CSV       = OUT_DIR / f"gen_phi4_mini_instruct_{QUANT_LEVEL}_test.csv"
OUT_META      = OUT_DIR / f"gen_phi4_mini_instruct_{QUANT_LEVEL}_test_metadata.json"
PROMPT_FILE   = Path(f"/tmp/gen_phi4_mini_instruct_{QUANT_LEVEL}_prompt.txt")

# ─── scope ───────────────────────────────────────────────────────────────────
N_PILOT_PER_STRATUM = 10   # full pilot: 10 short + 10 long = 20 items
SEEDS               = [0, 1, 2, 3]

# ─── pre-specified generation parameters ────────────────────────────────────
TEMP          = 0.3
TOP_P         = 0.95
TOP_K         = 40
# v9: REPEAT_PEN bumped from 1.25 to 1.40 (Phi-4-only deviation; all other 11
# scripts retain 1.25). Pilot v8 at 1.25 showed 32/80 outputs entering
# degenerate repetition loops, with 8-grams like "impingement syndrome in his
# left shoulder causing referred" repeating 10× in single outputs. Higher
# penalty needed to suppress semantic loops. Pre-specified model-specific.
REPEAT_PEN    = 1.40
# v7: N_PREDICT bumped from 512 to 768 (Phi-4-only deviation; all other 11
# scripts retain 512). Pilot data at 512 showed 20/40 outputs hitting the cap
# and being truncated mid-sentence (425-464 words at ~1.2-1.4 words/token).
# 768 tokens gives 512-600 words headroom so Phi-4 completes via EOS. Must
# be documented as a model-specific parameter.
N_PREDICT     = 768
CTX_SIZE      = 8192
N_GPU_LAYERS  = 999

# Word-count threshold for retry. If extracted output is below this AFTER one
# run, re-run once (in case of transient subprocess/extraction issue). If
# still below, force is_anomalous=True.
MIN_WC_RETRY_THRESHOLD = 30

# ─── pre-specified zero-shot instruction (IDENTICAL across all 12 scripts) ──
INSTRUCTION = (
    "Summarize the following doctor-patient conversation as a structured "
    "clinical note. Write in plain prose without section headers, markdown, "
    "asterisks, bullet points, numbered lists, or bold text."
)

# ─── stop strings (REVERSE_PROMPTS) — Phi-4-specific ─────────────────────────
# Phi-4 chat template uses <|user|>...<|end|><|assistant|>...<|end|> markers.
# No Qwen-specific injection strings here — those were Qwen-MediCare-BD specific.
# Until we observe Phi-4 pilot evidence of injection patterns, we keep this set
# minimal (same as Gemma).
REVERSE_PROMPTS = [
    # Phi-4 turn markers (primary stops)
    "<|end|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    # Dialogue artifact stops
    "[doctor]",
    "[patient]",
    "patient_guest",
    # AI disclaimer stops (general)
    "as an AI",
    "As an AI",
    "language model",
]

# ─── general anomaly flagging (v4 — identical to Qwen/Gemma) ─────────────────
LOW_WORD_COUNT_THRESHOLD = 20
LOOP_NGRAM_SIZE          = 8   # pre-specified: an n-word phrase repeated 2+ times = loop
# v10 Phi-4-specific: when an 8-gram repeats ≥3 times (definitely degenerate,
# not legitimate clinical phrase repetition), truncate the output at the start
# of the first occurrence. Pre-specified.
LOOP_TRUNCATE_MIN_REPEATS = 3

ANOMALY_PATTERNS_LC = [
    r"as an (ai|artificial intelligence|language model)",
    r"i (cannot|can't|am unable to)",
    r"i am (an ai|a language model)",
    r"\byou (are|see|have|will|can|should|might|need|must) ",
]

ANOMALY_PATTERNS_CS = [
    r"\d[-/]\d{1,2}[A-Z]",
    r"\d[A-Z][a-z]",
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


def truncate_at_loop(text: str,
                      n: int = LOOP_NGRAM_SIZE,
                      min_repeats: int = LOOP_TRUNCATE_MIN_REPEATS) -> str:
    """
    Truncate the output at the START of the first n-gram that repeats
    min_repeats+ times. Returns the genuine clinical content generated
    BEFORE the model entered the loop.

    Pre-specified Phi-4-specific cleanup. Pilot data showed Phi-4 generates
    clean clinical content for 250-350 words then enters degenerate semantic
    repetition that runs to the n_predict cap. Truncating at loop onset
    preserves the usable clinical content and discards mechanically repeated
    text that contains no new clinical information.

    Parameters:
      n            : n-gram window (default = LOOP_NGRAM_SIZE = 8)
      min_repeats  : repetition count that triggers truncation (default = 3).
                     Conservative versus has_loop's 2 — 2 repeats might be
                     legitimate clinical phrase repetition; 3+ is definitely
                     degenerate.

    Returns the truncated text if a loop is detected; otherwise returns the
    input text unchanged.
    """
    if not text:
        return text
    words = text.split()
    if len(words) < n * min_repeats:
        return text
    seen = {}
    for i in range(len(words) - n + 1):
        ng = tuple(w.lower() for w in words[i:i + n])
        if ng in seen:
            seen[ng].append(i)
        else:
            seen[ng] = [i]
        if len(seen[ng]) >= min_repeats:
            # Loop confirmed — truncate at the START of the first occurrence
            first_pos = seen[ng][0]
            return " ".join(words[:first_pos]).strip()
    return text


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


# ─── strip_meta_and_markdown (v6 — identical to Qwen/Gemma) ──────────────────
META_PREFIX_PATTERNS = [
    r"^\s*(?:Here['’]s|Here is)\s+(?:a |the )?(?:structured |brief |concise |comprehensive |detailed |formatted )?(?:clinical )?(?:summary|note|documentation)[^.:\n]*[.:\-—]\s*",
    r"^\s*(?:Below is|The following is|Following is)\s+[^.:\n]{0,80}[.:\-—]\s*",
    r"^\s*(?:Sure|Of course|Certainly|Okay|OK)[,.!]?\s+(?:here['’]?s|here is)\s+[^.:\n]{0,80}[.:\-—]\s*",
    r"^\s*(?:Clinical\s+)?Note\s*[:\-—]\s*",
    r"^\s*(?:Patient\s+)?Summary\s*[:\-—]\s*",
]
_meta_prefix_re = [re.compile(p, re.IGNORECASE) for p in META_PREFIX_PATTERNS]


def strip_meta_and_markdown(text: str) -> str:
    """Identical to Qwen/Gemma v6 — see Qwen script for docstring."""
    if not text:
        return text
    for rx in _meta_prefix_re:
        new = rx.sub("", text, count=1)
        if new != text:
            text = new
            break
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    text = text.replace("**", "")
    text = re.sub(r"(?m)^#+\s*",          "", text)
    text = re.sub(r"(?m)^\s*[-•]\s+",     "", text)
    text = re.sub(r"(?m)^\s*\*\s+",       "", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+",  "", text)
    return text.strip()


# ─── prompt builder + extraction (Phi-4-specific) ────────────────────────────
def build_prompt(dialogue: str) -> str:
    """Wrap the instruction + dialogue in Phi-4 chat-turn format."""
    return (
        f"<|user|>\n"
        f"{INSTRUCTION}\n\n"
        f"Conversation:\n{dialogue}<|end|>\n"
        f"<|assistant|>\n"
    )


def extract_completion(raw_stdout: str, prompt: str) -> str:
    """
    Strip the echoed prompt + trailing artifacts from llama-completion stdout.

    Phi-4 strips ALL its special tokens (<|user|>, <|assistant|>, <|end|>) during
    stdout echo AND emits no role-text marker between the prompt's last token
    and the model's first generated token. There is literally no boundary
    string to split on — the echoed prompt flows directly into the completion.

    Strategy: build the "echoed prompt" by removing the special tokens we
    added during wrapping, then find where that text ends in stdout. Everything
    after is the completion. Use the last ~120 chars of the cleaned prompt as
    a unique anchor (long enough to be specific, short enough to skip minor
    echo artifacts like swapped quote characters).
    """
    text = raw_stdout

    # 1. Build the prompt's "echoed form" by stripping the special tokens we added
    echoed_prompt = prompt
    for tok in ("<|user|>\n", "<|assistant|>\n", "<|end|>\n", "<|system|>\n",
                "<|user|>", "<|assistant|>", "<|end|>", "<|system|>"):
        echoed_prompt = echoed_prompt.replace(tok, "")

    # 2. If a literal special-token marker survived in stdout, use it.
    if "<|assistant|>" in text:
        text = text.rsplit("<|assistant|>", 1)[-1].lstrip("\n ")
    else:
        # 3. Anchor on the LAST ~120 chars of the echoed prompt. Find this
        #    anchor in stdout and take everything after.
        anchor_len = min(120, len(echoed_prompt))
        anchor     = echoed_prompt[-anchor_len:].rstrip()
        if anchor:
            idx = text.rfind(anchor)
            if idx >= 0:
                text = text[idx + len(anchor):]
            else:
                # Last-resort fallback: assume echoed prompt is verbatim prefix
                # and split off its length. May leave a few residual chars from
                # the prompt at the start, but never drops actual content.
                if len(text) > len(echoed_prompt):
                    text = text[len(echoed_prompt):]

    # 4. Strip closing turn marker if present
    text = text.split("<|end|>", 1)[0]

    # 5. Strip llama-completion interactive stubs
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
        "--in-prefix", "",
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

    if OUT_CSV.exists():
        done_df   = pd.read_csv(OUT_CSV)
        done_keys = set(zip(done_df["input_id"], done_df["seed"]))
        print(f"Resuming — {len(done_df)} rows already done")
    else:
        done_df   = pd.DataFrame()
        done_keys = set()

    rows = []
    n_truncated_outputs = 0   # v10: count of outputs altered by truncate_at_loop()
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

            def _attempt(p, s):
                """Single generation attempt with extraction-leak detection
                and v10 loop-truncation. Returns the truncated/cleaned/flat
                text plus the PRE-truncation loop flag and truncation flag."""
                raw, el, retc = run_one(p, s)
                if INSTRUCTION[:40].lower() in raw.lower():
                    raw = ""
                # v10: compute is_looped on the PRE-truncation text. The flag
                # reflects the model's actual generation behavior; the text
                # we save is the usable clinical content before loop onset.
                loop_pre = has_loop(raw)
                # v10: truncate at first 8-gram with ≥3 repetitions.
                raw_trunc = truncate_at_loop(raw)
                truncated_flag = (raw_trunc != raw)
                cleaned = strip_meta_and_markdown(raw_trunc)
                flat = flatten(cleaned)
                wc_ = len(flat.split()) if flat else 0
                return flat, el, retc, wc_, loop_pre, truncated_flag

            generated_flat, elapsed, rc, wc, looped, truncated = _attempt(prompt, seed)
            retried = False
            if wc < MIN_WC_RETRY_THRESHOLD:
                retried = True
                print(f"[wc={wc}<{MIN_WC_RETRY_THRESHOLD},retry] ", end="", flush=True)
                f2, e2, r2, w2, lp2, tr2 = _attempt(prompt, seed)
                if w2 > wc:
                    generated_flat, elapsed, rc, wc = f2, e2, r2, w2
                    looped, truncated = lp2, tr2

            if truncated:
                n_truncated_outputs += 1

            anomalous = is_anomalous(generated_flat, wc)
            if retried and wc < MIN_WC_RETRY_THRESHOLD:
                anomalous = True

            status = "OK" if rc == 0 else f"rc={rc}"
            retry_tag = " [RETRY]" if retried else ""
            print(f"{elapsed:5.1f}s  {wc:4d} words  [{status}]{retry_tag}")

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
        "loop_truncate_min_repeats": LOOP_TRUNCATE_MIN_REPEATS,
        "n_truncated_outputs":  n_truncated_outputs,
        "min_wc_retry_threshold": MIN_WC_RETRY_THRESHOLD,
        "extraction_leak_check": "INSTRUCTION[:40] presence test post-extraction",
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
