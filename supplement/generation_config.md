# Supplement — Generation Configuration

This document records the note-generation protocol exactly as executed. Every value is
taken verbatim from the project's generation scripts (one per model × quantization
cell, to be released with the full code on publication; the decoding parameters, chat templates, stop sequences, and
extraction rules are identical across the four quantization levels of each model,
except for the two Phi-4 deviations noted below). Generation binary:
`llama.cpp/build/bin/llama-completion` (raw completion mode, `--in-prefix ""
--in-suffix ""`, so the chat template below is applied manually rather than by the
runtime).

---

## 1. Zero-shot instruction (identical for all models and cells)

```
Summarize the following doctor-patient conversation as a structured clinical note.
Write in plain prose without section headers, markdown, asterisks, bullet points,
numbered lists, or bold text.
```

---

## 2. Native chat templates

Each model's own chat template is applied. `{INSTRUCTION}` is the string above;
`{dialogue}` is the encounter transcript. The templates are reproduced verbatim from
the local generation scripts.

**Qwen2 ChatML** (`gen_qwen_*`)
```
<|im_start|>user
{INSTRUCTION}

Conversation:
{dialogue}<|im_end|>
<|im_start|>assistant
```

**Gemma-3 turn markers** (`gen_gemma_*`)
```
<start_of_turn>user
{INSTRUCTION}

Conversation:
{dialogue}<end_of_turn>
<start_of_turn>model
```

**Phi-4 role markers** (`gen_phi4_*`)
```
<|user|>
{INSTRUCTION}

Conversation:
{dialogue}<|end|>
<|assistant|>
```

---

## 3. Decoding parameters

**All cells (default):**

| Parameter | Value |
|---|---|
| temperature | 0.3 |
| top-p | 0.95 |
| top-k | 40 |
| repetition penalty | 1.25 |
| max new tokens (`n_predict`) | 512 |
| context size | 8192 |
| GPU layers (`n_gpu_layers`) | 999 (full offload) |

**Phi-4-mini-instruct exceptions** (the only two per-model deviations; all other
parameters unchanged):

| Parameter | Phi-4 value | Default |
|---|---|---|
| repetition penalty | **1.40** | 1.25 |
| max new tokens (`n_predict`) | **768** | 512 |

Phi-4 required a stronger repetition penalty and a larger token budget because it was
prone to runaway repetition and to hitting the 512-token cap mid-note at the default
setting.

---

## 4. Reverse-prompt stop sequences (verbatim, all models)

Generation is halted (`--reverse-prompt`) whenever one of these strings appears, so the
note ends before drifting into dialogue-role or role-injection text. The clinical
content generated up to the stop is retained.

**Common to all three models:**
```
[doctor]
[patient]
patient_guest
as an AI
As an AI
language model
```

**Qwen-MediCare-BD** — ChatML turn markers plus **nine pilot-derived role-injection
stops** (the model occasionally exits the documentation task and drifts into
second-person, web-text-style injection, observed at date-of-birth token boundaries in
the pilot). The turn markers:
```
<|im_start|>
<|im_end|>
```
The nine role-injection stops, verbatim:
```
1. You have accessed
2. restricted section
3. Please confirm your registration
4. You are an assistant physician
5. practicing at a busy medical office
6. Review the following history
7. You see a new
8. You see him
9. You see her
```

**Gemma-3-4B-it** — turn markers only (no model-specific stops):
```
<start_of_turn>
<end_of_turn>
```

**Phi-4-mini-instruct** — turn markers only (no model-specific stops):
```
<|user|>
<|assistant|>
<|system|>
<|end|>
```

Principle: model-specific stops are added only for drift patterns empirically observed
in that specific model. Only Qwen-MediCare-BD required them.

---

## 5. Output extraction and cleanup

Applied in order to every generation:

1. **Prompt/echo stripping.** The echoed prompt is removed at the assistant-turn
   boundary, the closing turn marker is dropped, and any interactive-mode stubs are
   removed, leaving only the generated note.
2. **Preamble and markdown stripping.** A single leading meta-preamble (e.g.
   "Here's the clinical note:") is removed if present, and markdown formatting is
   stripped: bold (`**...**`), emphasis (`*...*`), leading `#` headers, `-`/`*`/`•`
   bullets, and numbered-list markers.
3. **Extraction-leak guard.** If the first 40 characters of the instruction appear in
   the output (a rare boundary-detection failure at low precision), the output is
   blanked.
4. **Whitespace flattening.** Newlines and tabs are collapsed to single spaces for
   one-line-per-row CSV storage.

**Phi-4 looping and truncation.** Phi-4-mini-instruct generates clean clinical content
for roughly 250-350 words and then enters degenerate semantic repetition that runs to
the token cap. For Phi-4 only, the output is truncated at the start of the first
eight-word phrase that recurs three or more times (definite degeneracy — a conservative
threshold; two repetitions may be legitimate clinical phrasing), preserving the genuine
clinical content generated before the loop and discarding the mechanical repetition. The
`is_looped` flag is computed on the raw output **before** truncation (an eight-word
phrase repeating two or more times), so the model-behavior signal is preserved even
though the stored text is truncated. The count of truncated outputs is recorded in the
per-cell metadata sidecar.

**Phi-4 short-output retry.** For Phi-4 only, if the extracted note is shorter than 30
words after cleanup, the generation is re-run once with the same seed and the longer of
the two outputs is kept; if it is still under 30 words it is marked anomalous.

---

## 6. Quality flags (recorded, not filtered)

Two independent boolean flags are recorded per generation. Flagged rows are **kept** in
the primary analysis; a clean subset (both flags false) is available as a sensitivity
stream.

**`is_anomalous`** — true if the output is empty, has fewer than 20 words, or matches any
drift/injection/refusal pattern. Patterns, verbatim:

Case-insensitive (all models):
```
as an (ai|artificial intelligence|language model)
i (cannot|can't|am unable to)
i am (an ai|a language model)
\byou (are|see|have|will|can|should|might|need|must) 
```
Qwen additionally (case-insensitive):
```
you have accessed
restricted section
confirm your registration
you are an assistant physician
you are a (doctor|physician|nurse|medical)
```
Case-sensitive (all models):
```
\d[-/]\d{1,2}[A-Z]                                          (date-then-capital drift, e.g. "1957-09-2You")
\d[A-Z][a-z]                                                (digit-then-capitalized-word)
^\s*\[(doctor|patient|patient_guest|physician|nurse|guest|assistant)\]   (starts as a dialogue role tag)
```

**`is_looped`** — true if any eight-word phrase appears two or more times (lowercased).
Independent of `is_anomalous`.

---

## 7. Sampling

| Item | Value |
|---|---|
| Generations per cell | 20 |
| Seeds | 0 through 19 (inclusive) |
| Encounters | 100 (50 short + 50 long) |
| Models × quantization levels | 3 × 4 = 12 cells |
| Total generations | 100 × 20 × 12 = 24,000 |

Run-to-run consistency is measured between the 20 seeds within each cell, giving
C(20, 2) = 190 seed pairs per cell.
