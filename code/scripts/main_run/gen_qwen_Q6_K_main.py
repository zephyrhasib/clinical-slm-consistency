"""
Generation MAIN Run Script
Model : Qwen-MediCare-BD  |  Quant : Q6_K

Purpose: full main-run generation. All 12 cells together produce the
24,000-generation dataset (100 patients × 20 seeds × 3 models × 4 quants)
analyzed by the metric and statistical pipelines.

Scope:
  - 100 main items (50 short + 50 long from main_inputs.csv, MAIN-001..MAIN-100)
  - 20 seeds (range 0..19)
  - Total: 2000 generations per (model, quant) cell

Generation parameters (pre-specified, same as full run):
  temperature    = 0.3
  repeat_penalty = 1.25   (1.1/1.2 insufficient in earlier tests)
  n_predict      = 512
  ctx_size       = 8192   (max dialogue ~4200 tokens + prompt + buffer)
  n_gpu_layers   = 999    (full GPU offload)

Pre-flight (aborts on any mismatch):
  - llama.cpp git HEAD == hashes/llama_cpp_commit.txt
  - SHA-256 of llama-completion binary == master_hashes.json
  - SHA-256 of qwen_medicare_bd_Q6_K.gguf == master_hashes.json

Chat template: Qwen2 ChatML, manually wrapped here. llama-completion is
invoked with --in-prefix "" --in-suffix "" so it does NOT auto-apply the
embedded template — we have full control.

Run from project root (use main venv, NOT venv_metrics):
    source venv/bin/activate
    python scripts/main_run/gen_qwen_Q6_K_main.py
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
MODEL_NAME   = "Qwen-MediCare-BD"
QUANT_LEVEL  = "Q6_K"
GGUF_NAME    = "qwen_medicare_bd_Q6_K.gguf"
TEMPLATE     = "qwen2_chatml"
SCRIPT_VER   = 8   # v8: add is_looped column (8-word n-gram repetition detect);
                   #     expand pilot scope 5+5 → 10+10 (full pilot_inputs.csv).
# v2 (2026-05-23): new INSTRUCTION (no role priming, no section scaffolding);
#                  explicit --top-p / --top-k flags; injection-artifact reverse
#                  prompts; is_anomalous flagging column; 5-short+5-long pilot.
# v3 (2026-05-23): broadened is_anomalous — adds date-then-capital drift regex
#                  (\d[-/]\d[A-Z], \d[A-Z][a-z]), generic 2nd-person imperative
#                  regex (\byou (are|see|have|will|can|should|might) ), and a
#                  low-word-count flag (wc < 20). Pre-specified.
# v4 (2026-05-23): starts-with-role-tag anomaly pattern — flags outputs whose
#                  first non-whitespace token is [doctor]/[patient]/[physician]/
#                  [nurse]/[guest]/[assistant], catching dialogue-format drift
#                  that produces above-threshold word counts. Pre-specified.
# v5 (2026-05-23): stronger INSTRUCTION (no preamble, no markdown) + post-gen
#                  strip_meta_and_markdown(). REVERTED in v6 — Qwen interpreted
#                  the stronger "Output only the note" wording as 'respond
#                  minimally' and produced [doctor]/[patient] role-tag outputs
#                  for 22/40 generations. Gemma was unaffected.
# v6 (2026-05-23): INSTRUCTION reverted to v4 text (proven to anchor Qwen to
#                  clinical content). strip_meta_and_markdown() KEPT as a
#                  post-generation safety net for Gemma's meta-prefix and
#                  markdown formatting. The strip is a no-op for compliant
#                  outputs; transparently cleans the others.

# ─── paths ───────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
LLAMA_BIN     = ROOT / "llama.cpp/build/bin/llama-completion"
MODEL_PATH    = ROOT / "models/quantized" / GGUF_NAME
COMMIT_FILE   = ROOT / "hashes/llama_cpp_commit.txt"
MASTER_JSON   = ROOT / "hashes/master_hashes.json"
MAIN_CSV     = ROOT / "data/stratified/main_inputs.csv"
OUT_DIR       = ROOT / "outputs/main_run"
OUT_CSV       = OUT_DIR / f"gen_qwen_medicare_bd_{QUANT_LEVEL}_main.csv"
OUT_META      = OUT_DIR / f"gen_qwen_medicare_bd_{QUANT_LEVEL}_main_metadata.json"
PROMPT_FILE   = Path(f"/tmp/gen_{MODEL_NAME.lower()}_{QUANT_LEVEL}_prompt.txt")

# ─── scope ───────────────────────────────────────────────────────────────────
# Full main: all 50 short + all 50 long (MAIN-001..MAIN-100). 20 seeds per item.
# Total: 2000 generations per (model, quant) — frozen main dataset.
N_PER_STRATUM = 50
SEEDS               = list(range(20))

# ─── pre-specified generation parameters ────────────────────────────────────
TEMP          = 0.3
TOP_P         = 0.95   # pre-specified (now explicit in cmd for reproducibility)
TOP_K         = 40     # pre-specified (now explicit in cmd for reproducibility)
REPEAT_PEN    = 1.25
N_PREDICT     = 512
CTX_SIZE      = 8192
N_GPU_LAYERS  = 999

# ─── pre-specified zero-shot instruction (identical across all 12 scripts) ──
# v6: reverted to v4 wording. The v5 "Output only the note" wording broke
# Qwen-MediCare-BD (22/40 outputs became single role tags). The post-generation
# strip_meta_and_markdown() handles any preambles/markdown that slip through.
INSTRUCTION = (
    "Summarize the following doctor-patient conversation as a structured "
    "clinical note. Write in plain prose without section headers, markdown, "
    "asterisks, bullet points, numbered lists, or bold text."
)

# ─── stop strings (REVERSE_PROMPTS) ──────────────────────────────────────────
# Pre-specified Qwen-specific reverse prompts derived from pilot data.
# The injection-artifact group (last block) catches non-clinical strings that
# Qwen-MediCare-BD emits when it exits the documentation task at date-of-birth
# token boundaries. Reverse prompts halt generation BEFORE the injection
# propagates; output up to that point is valid clinical content.
REVERSE_PROMPTS = [
    # Qwen2 ChatML turn markers (primary stops)
    "<|im_end|>",
    "<|im_start|>",
    # Dialogue artifact stops
    "[doctor]",
    "[patient]",
    "patient_guest",
    # AI disclaimer stops
    "as an AI",
    "As an AI",
    "language model",
    # Injection artifact stops — Qwen-MediCare-BD specific (pilot-derived).
    # Indicates model exit from the clinical documentation task. Pre-specified.
    "You have accessed",
    "restricted section",
    "Please confirm your registration",
    "You are an assistant physician",
    "practicing at a busy medical office",
    "Review the following history",
    # v7 — additional date-token-triggered 2nd-person drift forms observed in
    # pilot at PILOT-001 ("1957-09-2You see a new male adult patient...").
    # Anomaly detector catches these post-hoc, but adding to reverse prompts
    # stops generation BEFORE the injection propagates downstream.
    "You see a new",
    "You see him",
    "You see her",
]

# ─── general anomaly flagging (passive — does not change output text) ────────
# Pre-specified v3 anomaly detection. Catches:
#   (a) explicit injection strings (web-text drift patterns observed in pilot)
#   (b) generic 2nd-person imperative drift ('you are seeing him', 'you have ...')
#   (c) date-then-capital tokenisation drift ('1957-09-2You', '9/2Youth')
#   (d) implausibly short outputs (< 20 words — model failed to start the task)
#
# Outputs flagged here are KEPT — the metric pipeline runs on all outputs.
# Sensitivity analysis re-runs excluding the flagged subset to confirm primary
# findings hold. Empty/missing text is treated as anomalous.

LOW_WORD_COUNT_THRESHOLD = 20
LOOP_NGRAM_SIZE          = 8   # pre-specified: an n-word phrase repeated 2+ times = loop

# Case-insensitive patterns (matched against lower-cased text)
ANOMALY_PATTERNS_LC = [
    # explicit injection strings
    r"you have accessed",
    r"restricted section",
    r"confirm your registration",
    r"you are an assistant physician",
    r"you are a (doctor|physician|nurse|medical)",
    r"as an (ai|artificial intelligence|language model)",
    r"i (cannot|can't|am unable to)",
    r"i am (an ai|a language model)",
    # generic 2nd-person imperative drift (catches 'you are seeing him', 'you see a new',
    # 'you have noticed', etc. — drift patterns observed in v2 pilot)
    r"\byou (are|see|have|will|can|should|might|need|must) ",
]

# Case-sensitive patterns (matched against original text — capitalisation matters)
ANOMALY_PATTERNS_CS = [
    # date-then-capital drift: '1957-09-2Y', '9/2Y'
    r"\d[-/]\d{1,2}[A-Z]",
    # generic digit-immediately-followed-by-capitalised-word ('2Youth', '2Yes')
    r"\d[A-Z][a-z]",
    # starts-with-role-tag drift: '[doctor] ...', '[patient] ...', '[physician] ...'
    # — model copied dialogue format instead of writing a summary. Anchored to
    # start (allowing leading whitespace) so legitimate mentions mid-note pass.
    r"^\s*\[(doctor|patient|patient_guest|physician|nurse|guest|assistant)\]",
]

_anomaly_lc = [re.compile(p) for p in ANOMALY_PATTERNS_LC]
_anomaly_cs = [re.compile(p) for p in ANOMALY_PATTERNS_CS]


def is_anomalous(text: str, word_count: int) -> bool:
    """Returns True if the output shows any drift / injection / failure pattern."""
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
    """Returns True if any n-word phrase appears 2+ times in text (case-insensitive).

    Pre-specified detector for semantic/textual repetition. Generic — applied
    to all 12 scripts. Does NOT count toward is_anomalous (separate column).
    Threshold n=8 chosen to avoid false positives on legitimate templated
    clinical phrases (which are typically <5 words) while catching semantic
    loops observed in pilot data.
    """
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


def build_prompt(dialogue: str) -> str:
    """Wrap the zero-shot instruction + dialogue in Qwen2 ChatML format."""
    return (
        f"<|im_start|>user\n"
        f"{INSTRUCTION}\n\n"
        f"Conversation:\n{dialogue}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def flatten(text: str) -> str:
    """Collapse all whitespace (including newlines, tabs) to single spaces.

    Same convention as data/stratified/all_references.csv (reference_note) and
    data/stratified/main_inputs.csv (dialogue) — single-line per CSV row.
    """
    return " ".join(str(text).split())


# v5 — pre-specified post-generation cleanup. Removes meta-prefixes and
# markdown formatting that the instruction tells the model to avoid but
# some models emit anyway. Applied uniformly to ALL 12 scripts so the metric
# pipeline sees content, not formatting variance. Pre-specified.

META_PREFIX_PATTERNS = [
    # "Here's a/the [adjective] clinical note [...]" + optional colon/dash
    r"^\s*(?:Here['’]s|Here is)\s+(?:a |the )?(?:structured |brief |concise |comprehensive |detailed |formatted )?(?:clinical )?(?:summary|note|documentation)[^.:\n]*[.:\-—]\s*",
    # "Below is/The following is" preambles
    r"^\s*(?:Below is|The following is|Following is)\s+[^.:\n]{0,80}[.:\-—]\s*",
    # "Sure/Of course/Certainly, here is..."
    r"^\s*(?:Sure|Of course|Certainly|Okay|OK)[,.!]?\s+(?:here['’]?s|here is)\s+[^.:\n]{0,80}[.:\-—]\s*",
    # Bare "Clinical note:" / "Note:" at start
    r"^\s*(?:Clinical\s+)?Note\s*[:\-—]\s*",
    # "Patient summary:" / "Summary:" at start
    r"^\s*(?:Patient\s+)?Summary\s*[:\-—]\s*",
]
_meta_prefix_re = [re.compile(p, re.IGNORECASE) for p in META_PREFIX_PATTERNS]


def strip_meta_and_markdown(text: str) -> str:
    """Remove meta-prefixes + markdown formatting from a generated note.

    Applied between extract_completion() and flatten(). Pre-specified.

    Operations (in order):
      1. Strip a single meta-prefix if present at the start.
      2. Strip markdown bold: **text** → text
      3. Strip markdown emphasis: *text* → text (paired, no nesting)
      4. Strip leading-line markdown headers: # / ## / ### → (removed)
      5. Strip leading-line bullets: -, *, • → (removed)
      6. Strip leading-line numbered lists: 1. / 2) → (removed)
      7. Strip leading-line "Key:" labels left over after markdown removal
         (e.g., "Patient: ", "HPI: ", "Plan: " at start of a line).
         These are tolerated within a line; only stripped if at line start.
    """
    if not text:
        return text
    # 1. meta-prefix (at most one)
    for rx in _meta_prefix_re:
        new = rx.sub("", text, count=1)
        if new != text:
            text = new
            break
    # 2. markdown bold/emphasis (handle ** before *)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    # paired single-* emphasis: require non-* on either side to avoid eating
    # legitimate text or partial **
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    # orphan markers (model started ** but didn't close it before halting)
    text = text.replace("**", "")
    # 3-6. line-start formatting
    text = re.sub(r"(?m)^#+\s*",                "", text)       # headers
    text = re.sub(r"(?m)^\s*[-•]\s+",          "",  text)       # dash/bullet
    text = re.sub(r"(?m)^\s*\*\s+",            "",  text)       # asterisk bullet (already de-bolded)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+",       "",  text)       # numbered lists
    return text.strip()


def extract_completion(raw_stdout: str, prompt: str) -> str:
    """
    Strip the echoed prompt and trailing artifacts from llama-completion stdout.

    llama-completion echoes the prompt by default, BUT special tokens
    (<|im_start|>, <|im_end|>) are stripped during the echo — only their
    surrounding literal text ("user", "assistant", etc.) appears in stdout.
    So the boundary is "\\nassistant\\n" in plain text, not the special-token form.

    After the generation, the model may emit <|im_end|> (which IS preserved
    in stdout because reverse-prompt matched on it), or hit max tokens.

    Then in interactive mode (reached by EOF on stdin) llama-completion
    writes `> \\nEOF by user`.
    """
    text = raw_stdout
    # Locate the boundary between echoed prompt and completion.
    # Try special-token form first (some configurations preserve them), then
    # fall back to plain literal text.
    if "<|im_start|>assistant\n" in text:
        text = text.rsplit("<|im_start|>assistant\n", 1)[-1]
    elif "\nassistant\n" in text:
        text = text.rsplit("\nassistant\n", 1)[-1]
    # Strip closing turn marker if present (was the reverse-prompt match)
    text = text.split("<|im_end|>", 1)[0]
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
    if not MAIN_CSV.exists():     abort(f"missing:          {MAIN_CSV}")

    master = json.loads(MASTER_JSON.read_text())

    # 1. llama.cpp commit
    expected_commit = COMMIT_FILE.read_text().strip()
    actual_commit   = subprocess.check_output(
        ["git", "-C", str(ROOT / "llama.cpp"), "rev-parse", "HEAD"], text=True,
    ).strip()
    if actual_commit != expected_commit:
        abort(f"llama.cpp commit mismatch\n  recorded: {expected_commit}\n  actual:   {actual_commit}")
    print(f"  [OK] llama.cpp commit: {expected_commit}")

    # 2. llama-completion binary
    recorded_bin_sha = master["llama_cpp"]["binaries"]["llama-completion"]
    actual_bin_sha   = sha256_of(LLAMA_BIN)
    if actual_bin_sha != recorded_bin_sha:
        abort(f"llama-completion SHA mismatch\n  recorded: {recorded_bin_sha}\n  actual:   {actual_bin_sha}")
    print(f"  [OK] llama-completion SHA: {recorded_bin_sha}")

    # 3. GGUF hash
    recorded_gguf_sha = master["models"]["quantized"][GGUF_NAME]["sha256"]
    actual_gguf_sha   = sha256_of(MODEL_PATH)
    if actual_gguf_sha != recorded_gguf_sha:
        abort(f"GGUF SHA mismatch\n  recorded: {recorded_gguf_sha}\n  actual:   {actual_gguf_sha}")
    print(f"  [OK] {GGUF_NAME} SHA: {recorded_gguf_sha}")

    return {
        "llama_cpp_commit":    expected_commit,
        "llama_completion_sha": recorded_bin_sha,
        "gguf_sha":             recorded_gguf_sha,
    }


# ─── generation ──────────────────────────────────────────────────────────────
def run_one(prompt: str, seed: int) -> tuple[str, float, int]:
    """Run llama-completion once; return (cleaned_output, elapsed_sec, return_code)."""
    PROMPT_FILE.write_text(prompt, encoding="utf-8")
    cmd = [
        str(LLAMA_BIN),
        "-m", str(MODEL_PATH),
        "--file", str(PROMPT_FILE),
        "-c", str(CTX_SIZE),
        "-n", str(N_PREDICT),
        "--temp", str(TEMP),
        "--top-p", str(TOP_P),      # pre-specified, now explicit
        "--top-k", str(TOP_K),      # pre-specified, now explicit
        "--seed", str(seed),
        "--repeat-penalty", str(REPEAT_PEN),
        "-ngl", str(N_GPU_LAYERS),
        "--no-warmup",
        "--simple-io",
        "--in-prefix", "",          # disable auto chat-template wrap
        "--in-suffix", "",
    ]
    for rp in REVERSE_PROMPTS:
        cmd.extend(["--reverse-prompt", rp])

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                            stdin=subprocess.DEVNULL, timeout=300)
    elapsed = time.time() - t0
    return extract_completion(result.stdout, prompt), elapsed, result.returncode


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    meta_pre = preflight()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    main_full = pd.read_csv(MAIN_CSV)
    short_items = main_full[main_full["length_stratum"] == "short"].head(N_PER_STRATUM)
    long_items  = main_full[main_full["length_stratum"] == "long"].head(N_PER_STRATUM)
    main_df    = pd.concat([short_items, long_items], ignore_index=True)
    assert len(main_df) == 2 * N_PER_STRATUM, \
        f"Expected {2 * N_PER_STRATUM} items, got {len(main_df)}"
    print(f"\nMain items      : {len(main_df)} "
          f"({N_PER_STRATUM} short + {N_PER_STRATUM} long)")
    print(f"  short IDs     : {short_items['input_id'].tolist()}")
    print(f"  long IDs      : {long_items['input_id'].tolist()}")
    print(f"Seeds           : {SEEDS}")
    print(f"Total to generate : {len(main_df) * len(SEEDS)}")

    # ── resume support ───────────────────────────────────────────────────────
    if OUT_CSV.exists():
        done_df   = pd.read_csv(OUT_CSV)
        done_keys = set(zip(done_df["input_id"], done_df["seed"]))
        print(f"Resuming — {len(done_df)} rows already done in {OUT_CSV.name}")
    else:
        done_df   = pd.DataFrame()
        done_keys = set()

    # ── generate ─────────────────────────────────────────────────────────────
    rows = []
    started = datetime.now(timezone.utc).isoformat()
    print()

    for _, item in main_df.iterrows():
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
            generated_flat  = flatten(generated_clean)        # single-line for CSV
            wc = len(generated_flat.split()) if generated_flat else 0
            status = "OK" if rc == 0 else f"rc={rc}"
            print(f"{elapsed:5.1f}s  {wc:4d} words  [{status}]")

            anomalous = is_anomalous(generated_flat, wc)
            looped    = has_loop(generated_flat)
            # Column order: provenance → condition → status → telemetry → flags → generated_text LAST
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

            # Incremental save after each item to make Ctrl-C safe
            partial = pd.concat([done_df, pd.DataFrame(rows)], ignore_index=True) \
                      if len(done_df) else pd.DataFrame(rows)
            partial.to_csv(OUT_CSV, index=False)

    finished = datetime.now(timezone.utc).isoformat()

    # ── write metadata sidecar ───────────────────────────────────────────────
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
        "n_main_items":        len(main_df),
        "n_main_short":        N_PER_STRATUM,
        "n_main_long":         N_PER_STRATUM,
        "seeds":                SEEDS,
        "n_generations":        len(main_df) * len(SEEDS),
        "llama_cpp_commit":     meta_pre["llama_cpp_commit"],
        "llama_completion_sha": meta_pre["llama_completion_sha"],
        "gguf_sha":             meta_pre["gguf_sha"],
        "started_utc":          started,
        "finished_utc":         finished,
        "output_csv":           str(OUT_CSV.relative_to(ROOT)),
    }
    OUT_META.write_text(json.dumps(metadata, indent=2))

    # ── summary ──────────────────────────────────────────────────────────────
    total = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame(rows)
    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"{'=' * 60}")
    print(f"  Output CSV  : {OUT_CSV}  ({len(total)} rows)")
    print(f"  Metadata    : {OUT_META}")
    print(f"  Word count stats:")
    print(total["word_count"].describe().round(1).to_string())

    PROMPT_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
