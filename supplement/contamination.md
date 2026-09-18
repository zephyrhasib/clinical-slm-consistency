# Supplement — Contamination Audit

This document records the pre-generation contamination audit exactly as the paper
cites it. Two complementary methods were used, matched to what could be inspected for
each model: a corpus-overlap test for the specialist model whose fine-tuning data is
public, and a behavioral memorization probe for the two general-purpose models whose
training data is not available. All 100 main-study encounters were screened.

All constants below are taken verbatim from the project's audit scripts (to be
released with the full code on publication). The audit decision (`clean`, 0/100) is
recorded in the project's internal decision log.

---

## 1. Qwen-MediCare-BD — MinHash corpus-overlap test

Qwen-MediCare-BD is fine-tuned from a publicly documented corpus, so the audit tests
directly for overlap between the benchmark inputs and that corpus. The model itself is
not run for this test.

| Parameter | Value |
|---|---|
| Method | MinHash + Locality-Sensitive Hashing (LSH) |
| Shingle size | 13-word shingles (word n-grams) |
| Number of permutations | 128 |
| Jaccard similarity threshold | 0.8 |
| Corpus indexed | 27,814 indexable entries from the public fine-tuning corpus |

**Corpus provenance.** The fine-tuning corpus comprises two public datasets — a
Bangladeshi-medicines corpus and a MedQA-USMLE four-option corpus — pooled and then
restricted to entries long enough to form at least one 13-word shingle, yielding the
27,814 indexable entries against which every benchmark encounter was compared.

**Procedure.** Each of the 100 benchmark encounters is shingled into 13-word word
n-grams, MinHash-signed with 128 permutations, and queried against the LSH index. An
encounter is flagged if any indexed corpus entry reaches an estimated Jaccard
similarity of 0.8 or above.

---

## 2. Gemma-3-4B-it and Phi-4-mini-instruct — behavioral memorization probe

Training data for these two general-purpose models is not public, so overlap cannot be
tested directly. Instead each model is probed for verbatim memorization: given the
opening of a source transcript, does the model reproduce the continuation word for
word?

| Parameter | Value |
|---|---|
| Method | n-gram verbatim-memorization probe |
| Prompt | first 50 words of the source transcript (prefix) |
| Decoding | greedy (temperature 0.0) |
| Continuation length | up to 100 new tokens |
| Flag rule | 10 or more consecutive generated words reproduce the source verbatim |
| Model precision probed | Q4_K_M quantization |
| Invocation | raw completion mode, no chat template applied |

**Procedure.** Each model is given the first 50 words of an encounter transcript and
allowed to continue greedily. If 10 or more consecutive generated words match the true
continuation exactly, the encounter is flagged as potentially memorized. The probe is
run at the Q4_K_M level (not full precision), the same family of quantized weights used
in the main study.

---

## 3. Result

| Model | Method | Encounters flagged |
|---|---|---|
| Qwen-MediCare-BD | MinHash LSH | 0 of 100 |
| Gemma-3-4B-it | memorization probe | 0 of 100 |
| Phi-4-mini-instruct | memorization probe | 0 of 100 |

**0 of 100 encounters were flagged for any model.** The audit decision was recorded as
`clean`, and the main study proceeded on all 100 encounters. No sensitivity exclusion
was required. The measured run-to-run agreement therefore reflects genuine model
behavior on unseen inputs rather than reproduction of memorized training text.
