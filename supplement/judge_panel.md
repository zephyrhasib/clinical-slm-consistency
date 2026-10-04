# Judge Panel: Protocol, Prompts, and Rubrics

A blinded panel of three large-language-model judges provides a coarse external check on the reference-free consistency metrics. The panel probes where the metrics diverge; it does not certify any metric as correct, and it is not a human evaluation. This document records how the panel was run and reproduces both prompts verbatim. The scores and validation reports are in the repository’s `data/judge_panel/` folder, and the panel's results are summarised in Section 6 of [`metrics.md`](metrics.md).

---

## 1. Judges

Three judges from three different model families, each used in its extended-reasoning mode through its standard web interface, in a fresh session with no conversation history:

| Judge | Mode |
|---|---|
| Claude Opus 4.7 | Adaptive Thinking |
| Gemini 3.1 Pro | Extended Thinking |
| DeepSeek | Expert mode with DeepThink enabled (underlying model version not recorded) |

## 2. Blinding

Judges received only a pair identifier, Note A, and Note B. Model names, quantization levels, encounter identifiers, random seeds, and all metric values were withheld.

## 3. Tracks and sampling

| | Track A: contradiction severity | Track B: semantic degradation |
|---|---|---|
| Question | How severely do the two notes contradict each other? | Why do the two notes differ? |
| Scale | Ordinal, 1 (consistent) to 5 (severely contradictory) | Nominal: A content disagreement, B format variation, C degeneracy, D mixed |
| Pairs | 60 (P01–P60) | 25 (BVAL-001 to BVAL-025) |
| Judgments | 180 (60 × 3 judges) | 75 (25 × 3 judges) |
| Sampling | Stratified: in each of the 12 model × quantization-level conditions, 5 encounters drawn at random (among encounters with a valid contradiction score), then one seed pair drawn at random from that encounter’s 190 pairs | The 5 model × encounter cells with the lowest BERTScore at Q3_K_M, at most 2 cells per model, 5 seed pairs from each |
| Random seed | 42 | 42 |
| Batches per judge | 6 batches of 10 pairs | 3 batches (10, 10, and 5 pairs) |
| Consensus | Median of the three scores | Majority vote; ties broken alphabetically (A > B > C > D) |
| Inter-judge agreement | Krippendorff’s α, ordinal | Krippendorff’s α, nominal |

Each batch prompt consisted of the rubric below followed by the pairs. Every judge scored every pair. The same rubrics were used, unchanged, in an earlier pilot panel of 36 Track A and 15 Track B pairs.

## 4. Analysis

- **Track A.** The median judge score for each pair is correlated, using Spearman’s ρ, with each of the three NLI contradiction aggregations (mean, q0.9, and max): at the pair level (n = 60), and as a sensitivity check at the condition level (n = 12). Pair-level p-values are approximate, because pairs are clustered within conditions.
- **Track B.** The distribution of consensus categories shows whether the lowest BERTScore values reflect genuine content disagreement or differences of format.

---

## 5. Track A prompt (verbatim)

```text
You are a senior clinical-documentation reviewer evaluating whether pairs
of clinical notes about the SAME patient encounter contradict each other.
Each pair was produced by a language model under different random seeds.
Your job is to score factual contradictions only, not stylistic differences
or omissions.

DEFINITIONS

A CONTRADICTION is a pair of statements that cannot both be true about the
same patient at the same encounter. Examples:
  - "Patient is a 32-year-old female" vs "Patient is a 65-year-old male"
  - "Positive patellar grind test" vs "Negative patellar grind test"
  - "Prescribed ibuprofen 400 mg" vs "Prescribed acetaminophen 500 mg"
  - "No prior surgeries" vs "Status post appendectomy in 2018"

An OMISSION is NOT a contradiction. If Note A mentions vital signs and Note
B does not, that is an omission. Score only the overlapping content.

STYLE, ORDERING, HEADINGS, VERBOSITY differences are NOT contradictions.

PARAPHRASES are NOT contradictions. "Patient reports knee pain for six
months" and "Six-month history of knee pain" agree.

SCALE (1 = perfectly consistent, 5 = severely contradictory)
  1 — No contradictions. All overlapping clinical facts agree.
  2 — Minor inconsistency on a non-clinical detail (name spelling, date
      phrasing). Clinical content fully agrees.
  3 — One contradiction on a moderate clinical detail (exam finding, dose,
      history element) but the overall diagnostic picture is the same.
  4 — Multiple contradictions, OR one contradiction on a key clinical fact
      (diagnosis, primary medication, major exam finding).
  5 — Severe contradictions. Notes describe materially different encounters
      (different chief complaint, demographics, incompatible diagnosis).

OUTPUT FORMAT — STRICT

For each pair, write:
  - Two to four short bullets listing the SPECIFIC contradictions you found
    (if any). Quote the conflicting phrases.
  - On the final line, write exactly:
        SCORE: <integer 1-5>

Do not explain the score after the SCORE line. Do not add commentary
between pairs. Do not skip pairs.

I will provide pairs in batches. Score every pair in this turn.
```

## 6. Track B prompt (verbatim)

```text
You are an expert clinical-documentation reviewer evaluating why two
AI-generated clinical notes about the SAME patient encounter DIFFER.

Each pair was generated by the same language model under different random
seeds at the same low-precision quantization level. Your job is to identify
the PRIMARY reason these two notes differ in their text.

CHOOSE ONE CATEGORY PER PAIR:

A = CONTENT DISAGREEMENT
    The notes assert different clinical facts about the patient.
    Examples:
      - Different chief complaints
      - Different exam findings (positive/negative)
      - Different medication names or doses
      - Different diagnoses
      - Different vital sign values
      - Different symptom durations

B = FORMAT VARIATION
    The notes describe the SAME clinical content but with different
    organizational structure.
    Examples:
      - One uses section headers, the other uses prose
      - Different order of presenting facts
      - One verbose with details, one concise
      - Same content, different sentence structure

C = DEGENERACY
    One or both notes contain text-generation pathologies.
    Examples:
      - N-gram repetition loops (a multi-word clinical phrase
        repeated 3+ times across the note)
      - Prompt-echo fragments where the note repeats or describes
        the instruction it was given rather than producing a note
      - Fragmentary or abandoned sentences
      - Empty or near-empty output (e.g., a single-sentence
        placeholder)
      - Off-topic content not about the patient encounter

D = MIXED
    The pair shows two or more of A, B, C in roughly equal measure.
    Cannot identify ONE primary reason.

GUIDANCE

- If the pair shows ANY content disagreement (even alongside format
  variation), pick A. Content disagreement is the most diagnostically
  important category.
- Pick B only when content fully agrees and only the organization differs.
- Pick C when text-generation pathologies overwhelm what could otherwise
  be a content or format judgment.
- Pick D only when no single reason dominates.

OUTPUT FORMAT — STRICT

For each pair, write:
  - A one-sentence justification quoting specific evidence from the texts.
  - On the final line, write exactly:
        CATEGORY: <letter A, B, C, or D>

Do not explain after the CATEGORY line. Do not skip pairs.

I will provide pairs in batches. Score every pair in this turn.
```
