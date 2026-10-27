"""
Metric pipeline (PAIRWISE CONSISTENCY between seed outputs).

What this pipeline measures: run-to-run consistency of the same model across
different random seeds on the same input. Reference notes are NOT used.

Metrics (all pairwise between seeds for each (input_id, model, quant_level)):
  Primary    : NLI sentence-matrix bidirectional contradiction   (lower = better)
               MEDCON F1 (concept-set agreement)                  (higher = better)
  Supporting : BERTScore F1                                       (higher = better)
               ROUGE-L F1                                         (higher = better)

For each (input_id, model, quant_level) cell:
  - All C(k, 2) seed pairs are scored
  - The mean across pairs is the cell-level scalar entering statistical analysis

Run in venv_metrics:
  source venv_metrics/bin/activate
  python scripts/metric_pipeline.py \\
      --generations data/demo_eval/synthetic_generations.csv \\
      --out         data/demo_eval/metric_results.csv \\
      --out-nli     data/demo_eval/metric_results_nli.csv
"""
import argparse
import csv
import json
import logging
import os
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Cosmetic only: silence the transformers tokenizer overflow/length WARNING
# that spams the terminal during the sentence-matrix loop. This changes NO
# tokenizer behavior (truncation/max_length unchanged) — it only raises the
# log level so the warning line is not printed. Verified: scores unchanged.
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize


# ─── BERTScore monkey-patch (R6.2) ───────────────────────────────────────────
# deberta-xlarge-mnli sets model_max_length=sys.maxsize which overflows the
# Rust tokenizer's C size_t. Clamp to 512 around each sent_encode call.
import bert_score.utils as _bsu
_orig_sent_encode = _bsu.sent_encode
def _patched_sent_encode(tokenizer, sent):
    saved = tokenizer.model_max_length
    tokenizer.model_max_length = 512
    result = _orig_sent_encode(tokenizer, sent)
    tokenizer.model_max_length = saved
    return result
_bsu.sent_encode = _patched_sent_encode

from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer as rs_module
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ─── Pre-specified configuration ────────────────────────────────────────────
NLI_MODEL          = "cross-encoder/nli-deberta-v3-large"
NLI_CONTRADICTION  = 0      # label index for 'contradiction'
NLI_BATCH_SIZE     = 32     # batch size for matrix forward passes

BERTSCORE_MODEL    = "microsoft/deberta-xlarge-mnli"
BERTSCORE_LAYERS   = 40
BERTSCORE_BATCH    = 4

# MEDCON / QuickUMLS — pre-specified (threshold corrected from 1.0 to 0.7)
QUICKUMLS_DB       = "umls/quickumls-db"
QUICKUMLS_TUI_JSON = "umls/aci_bench_tui_codes.json"
QUICKUMLS_THR      = 0.7
QUICKUMLS_WINDOW   = 5
QUICKUMLS_IGNORE_SYNTAX = True

# ─── Numerical-attribute extraction (Issue 3) ────────────────────────────────
# Pre-specified numerical-attribute extraction patterns.
# Six categories: drug+dose, frequency, duration, vitals, labs, age.
# Patterns are case-insensitive; text is lowercased before matching.
NUMERIC_PATTERNS = [
    # 1. Drug + dose + unit  (e.g., "metformin 500 mg", "lisinopril 10mg")
    r"(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|iu)\b",

    # 2. Frequency  ("3 times daily", "twice a day", "q12h", "BID", "TID")
    r"(\d+)\s*(times?\s+(?:daily|a\s+day|per\s+day)|qd|bid|tid|qid|q\d+h)\b",

    # 3. Duration  ("for 7 days", "x 14 days", "for 2 weeks")
    r"(?:for|x|×)\s*(\d+)\s*(days?|weeks?|months?|years?)\b",

    # 4. Vital signs  (BP, HR, RR, SpO2, temp, pulse)
    r"\b(bp|hr|rr|spo2|temp|temperature|blood\s+pressure|pulse|heart\s+rate)\s*(?::|of|is|was)?\s*(\d+(?:/\d+)?)",

    # 5. Common labs  (Hgb, HbA1c, glucose, creatinine, BUN, WBC, Na, K)
    r"\b(hemoglobin|hgb|hba1c|glucose|creatinine|bun|wbc|sodium|potassium|chloride|bicarbonate)\s*(?::|of|is|was)?\s*(\d+(?:\.\d+)?)",

    # 6. Patient age  ("32-year-old", "65 year old")
    r"\b(\d+)[\s-]+year[s\s-]+old\b",
]


DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"


# ═════════════════════════════════════════════════════════════════════════════
# NLI scorer  (sentence-matrix bidirectional contradiction, Laban et al. 2022)
# ═════════════════════════════════════════════════════════════════════════════
class NLIScorer:
    """Bidirectional sentence-matrix NLI contradiction (SummaC-ZS construction)."""

    def __init__(self):
        print(f"  Loading NLI model: {NLI_MODEL} on {DEVICE} ...")
        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL, use_fast=False)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        self.mdl.eval()
        if DEVICE == "cuda":
            self.mdl = self.mdl.cuda()
            # fp16: validated lossless (~1e-4 vs fp32) and ~2.9x faster at 190
            # pairs (see outputs/nli_proof/). The ONLY computational change.
            self.mdl = self.mdl.half()
        labels = self.mdl.config.id2label
        assert labels[NLI_CONTRADICTION].lower().startswith("contradict"), (
            f"NLI label[{NLI_CONTRADICTION}] is '{labels[NLI_CONTRADICTION]}' — expected 'contradiction'."
        )

    @torch.no_grad()
    def contradiction_prob(self, premise: str, hypothesis: str) -> float:
        """Single-pair contradiction probability (fallback / unit test)."""
        inputs = self.tok(
            premise, hypothesis,
            return_tensors="pt", truncation=True, padding=True, max_length=512,
        )
        if DEVICE == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        logits = self.mdl(**inputs).logits
        probs  = torch.softmax(logits, dim=-1)[0]
        return float(probs[NLI_CONTRADICTION].item())

    @torch.no_grad()
    def contradiction_prob_batch(self, premises: list, hypotheses: list,
                                 batch_size: int = NLI_BATCH_SIZE) -> list:
        """Batched contradiction probabilities — used by the sentence matrix."""
        all_probs = []
        for start in range(0, len(premises), batch_size):
            bp = premises[start:start + batch_size]
            bh = hypotheses[start:start + batch_size]
            inputs = self.tok(
                bp, bh,
                return_tensors="pt", truncation=True, padding=True, max_length=512,
            )
            if DEVICE == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            logits = self.mdl(**inputs).logits
            probs  = torch.softmax(logits, dim=-1)[:, NLI_CONTRADICTION]
            all_probs.extend(probs.cpu().tolist())
        return all_probs

    def bidirectional(self, a: str, b: str) -> float:
        """Document-level bidirectional contradiction — used only as <2-sentence fallback."""
        return (self.contradiction_prob(a, b) + self.contradiction_prob(b, a)) / 2.0

    def _all_aggregations_contradiction(self, sents_premise: list, sents_hypothesis: list) -> dict:
        """
        One-directional SummaC-ZS sentence-matrix aggregation.

        Builds the full N_premise × N_hypothesis contradiction matrix in
        ONE batched forward pass, then computes three row-aggregations
        (each row is one premise sentence vs. every hypothesis sentence):

          - max_mean  : mean over rows of (max over columns)   ← current production
          - mean_mean : mean over rows of (mean over columns)  ← whole-matrix mean
          - q90_mean  : mean over rows of (90th percentile over columns)

        All three are computed from the same matrix at no extra forward-pass
        cost. Returned as a dict so downstream code can pick its primary
        aggregation.
        """
        flat_p = [s_p for s_p in sents_premise for _ in sents_hypothesis]
        flat_h = [s_h for _ in sents_premise for s_h in sents_hypothesis]
        flat_scores = self.contradiction_prob_batch(flat_p, flat_h)

        n_h = len(sents_hypothesis)
        row_maxes  = []
        row_means  = []
        row_q90s   = []
        for i in range(len(sents_premise)):
            row = flat_scores[i * n_h : (i + 1) * n_h]
            row_maxes.append(max(row))
            row_means.append(sum(row) / len(row))
            # numpy.quantile with default linear interpolation
            row_q90s.append(float(np.quantile(row, 0.90)))
        return {
            "max_mean":  float(np.mean(row_maxes)),
            "mean_mean": float(np.mean(row_means)),
            "q90_mean":  float(np.mean(row_q90s)),
        }

    def sentence_matrix_contradiction(self, text_a: str, text_b: str) -> dict:
        """
        Bidirectional SummaC-ZS sentence-pair matrix contradiction.

        Returns a dict with THREE aggregations computed from the same
        matrices (no extra forward-pass cost):
          {"max_mean":  bidirectional max-then-mean,
           "mean_mean": bidirectional mean-then-mean (≡ matrix mean),
           "q90_mean":  bidirectional q90-then-mean}

        Algorithm (Laban et al. TACL 2022, applied bidirectionally for
        pairwise consistency measurement):

          1. Tokenize text_a and text_b into sentences via nltk.sent_tokenize.
          2. If either has fewer than 2 sentences, fall back to document-level
             bidirectional() — sentence splits would be noisy. The same scalar
             is replicated into all three keys (so downstream schema is stable).
          3. Direction A→B:
               matrix[i, j] = P(contradiction | premise = a_i, hypothesis = b_j)
               compute {max,mean,q90} per row, then mean over rows → score_AB[*]
          4. Direction B→A: same with premise/hypothesis swapped → score_BA[*]
          5. Return {k: (score_AB[k] + score_BA[k]) / 2 for k in 3 aggs}.

        Why three aggregations: row-max is sensitive to a single hostile
        sentence pair (catches local contradictions); row-mean is the
        whole-matrix average (smooth, dilutes single hits); row-q90 is a
        compromise (sensitive to the worst ~10 % of hypothesis sentences
        per premise sentence). Final pick is panel-validated.
        """
        sents_a = sent_tokenize(text_a)
        sents_b = sent_tokenize(text_b)
        if len(sents_a) < 2 or len(sents_b) < 2:
            doc_score = self.bidirectional(text_a, text_b)
            return {
                "max_mean":  doc_score,
                "mean_mean": doc_score,
                "q90_mean":  doc_score,
            }
        agg_ab = self._all_aggregations_contradiction(sents_a, sents_b)
        agg_ba = self._all_aggregations_contradiction(sents_b, sents_a)
        return {
            k: (agg_ab[k] + agg_ba[k]) / 2.0
            for k in ("max_mean", "mean_mean", "q90_mean")
        }


# ═════════════════════════════════════════════════════════════════════════════
# MEDCON scorer  (UMLS CUI set F1 between paired generations)
# ═════════════════════════════════════════════════════════════════════════════
class MEDCONScorer:
    """
    QuickUMLS-based pairwise medical-concept F1.

    Pipeline per text:
      1. matcher.match(text, best_match=True, ignore_syntax=True)
      2. Filter spans whose semtypes intersect the ACI-Bench 66-TUI set.
      3. Collect distinct CUIs.

    Pairwise F1 between texts A and B:
      precision = |A ∩ B| / |B|   recall = |A ∩ B| / |A|   F1 = 2pr/(p+r)
      Symmetric: F1(A, B) == F1(B, A). Higher F1 = more concept-set agreement.
    """

    def __init__(self, db_path: str, tui_json: str,
                 threshold: float = QUICKUMLS_THR, window: int = QUICKUMLS_WINDOW,
                 ignore_syntax: bool = QUICKUMLS_IGNORE_SYNTAX):
        print(f"  Loading QuickUMLS DB: {db_path} ...")
        from quickumls import QuickUMLS
        self.matcher = QuickUMLS(db_path, threshold=threshold, window=window)
        self.allowed_tuis  = set(json.loads(Path(tui_json).read_text())["all_tuis_sorted"])
        self.ignore_syntax = ignore_syntax
        self.threshold     = threshold
        self.window        = window
        print(f"  TUI filter: {len(self.allowed_tuis)} TUIs (ACI-Bench 66-code subset)")
        print(f"  Params    : threshold={threshold} window={window} ignore_syntax={ignore_syntax}")

    def extract_cuis(self, text: str) -> set:
        spans = self.matcher.match(text, best_match=True, ignore_syntax=self.ignore_syntax)
        cuis  = set()
        for span in spans:
            for m in span:
                semtypes = m.get("semtypes", set())
                if semtypes & self.allowed_tuis:
                    cuis.add(m["cui"])
        return cuis

    def f1(self, text_a: str, text_b: str) -> float:
        A = self.extract_cuis(text_a)
        B = self.extract_cuis(text_b)
        if not A and not B:
            return 1.0      # both empty → perfectly agreeing trivially
        if not A or not B:
            return 0.0
        inter = len(A & B)
        p = inter / len(B)
        r = inter / len(A)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Pairwise metric computations (one row per condition)
# ═════════════════════════════════════════════════════════════════════════════
def _is_empty(text) -> bool:
    """True if text is NaN/None/empty string. Used in all pair-scoring loops
    to skip pairs where either text is empty (Issue 8 — NaN guard for Gemma Q3
    empty outputs)."""
    return pd.isna(text) or not str(text).strip()


def compute_pairwise_bertscore_rouge(df_gen: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise BERTScore-F1 and ROUGE-L between every seed pair for
    each (input_id, model, quant_level) condition. Returns one row per
    condition with the mean across all C(k, 2) pairs.

    Issue 8: pairs where either text is NaN/empty are skipped (the bert-score
    and rouge-score libraries crash on NaN input). `n_pairs` reflects only the
    surviving pairs; `n_skipped_pairs` is tracked separately.
    """
    rouge = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
    rows  = []
    groups = df_gen.groupby(["input_id", "model", "quant_level"])

    for (input_id, model, quant), sub in tqdm(groups, desc="BERTScore+ROUGE pairs"):
        seeds = sorted(sub["seed"].tolist())
        seed_to_txt = {row["seed"]: row["generated_text"] for _, row in sub.iterrows()}

        all_pairs = list(combinations(seeds, 2))
        # Issue 8 — filter out pairs where either text is empty
        valid_pairs = [(i, j) for i, j in all_pairs
                       if not _is_empty(seed_to_txt[i]) and not _is_empty(seed_to_txt[j])]
        n_skipped = len(all_pairs) - len(valid_pairs)
        if len(valid_pairs) == 0:
            # All pairs invalid — emit an empty row so downstream knows the cell existed
            rows.append({
                "input_id": input_id, "model": model, "quant_level": quant,
                "n_pairs": 0, "n_skipped_pairs": n_skipped,
                "bertscore_f1_mean": float("nan"), "bertscore_f1_std": float("nan"),
                "rougeL_f1_mean":    float("nan"), "rougeL_f1_std":    float("nan"),
            })
            continue
        pairs = valid_pairs

        texts_a = [seed_to_txt[i] for i, j in pairs]
        texts_b = [seed_to_txt[j] for i, j in pairs]

        # BERTScore — batch all pairs in one call
        _, _, F1 = bert_score_fn(
            texts_a, texts_b,
            model_type=BERTSCORE_MODEL,
            num_layers=BERTSCORE_LAYERS,
            lang="en", verbose=False,
            batch_size=BERTSCORE_BATCH,
            rescale_with_baseline=False, idf=False,
            device=DEVICE,
        )
        bertscore_values = F1.cpu().numpy().tolist()

        # ROUGE-L — iterate (already cheap, ms per call)
        rougeL_values = [rouge.score(ta, tb)["rougeL"].fmeasure
                         for ta, tb in zip(texts_a, texts_b)]

        rows.append({
            "input_id":            input_id,
            "model":               model,
            "quant_level":         quant,
            "n_pairs":             len(pairs),
            "n_skipped_pairs":     n_skipped,
            "bertscore_f1_mean":   float(np.mean(bertscore_values)),
            "bertscore_f1_std":    float(np.std(bertscore_values, ddof=1)) if len(bertscore_values) > 1 else 0.0,
            "rougeL_f1_mean":      float(np.mean(rougeL_values)),
            "rougeL_f1_std":       float(np.std(rougeL_values, ddof=1)) if len(rougeL_values) > 1 else 0.0,
        })

    return pd.DataFrame(rows)


def compute_pairwise_medcon(df_gen: pd.DataFrame, medcon: MEDCONScorer) -> pd.DataFrame:
    """Pairwise MEDCON F1 between every seed pair, mean per condition.

    Issue 8: pairs where either text is NaN/empty are skipped. CUI extraction
    of empty text would return an empty set; the F1 logic already handles the
    both-empty (→1.0) and one-empty (→0.0) cases, but skipping empty *inputs*
    is cleaner since they reflect generation failures, not concept-set overlap.
    """
    rows = []
    groups = df_gen.groupby(["input_id", "model", "quant_level"])
    for (input_id, model, quant), sub in tqdm(groups, desc="MEDCON pairs"):
        seeds = sorted(sub["seed"].tolist())
        seed_to_txt = {row["seed"]: row["generated_text"] for _, row in sub.iterrows()}
        all_pairs = list(combinations(seeds, 2))
        # Issue 8 — skip pairs where either text is NaN/empty
        valid_pairs = [(i, j) for i, j in all_pairs
                       if not _is_empty(seed_to_txt[i]) and not _is_empty(seed_to_txt[j])]
        n_skipped = len(all_pairs) - len(valid_pairs)
        if len(valid_pairs) == 0:
            rows.append({
                "input_id": input_id, "model": model, "quant_level": quant,
                "n_pairs": 0, "n_skipped_pairs": n_skipped,
                "medcon_f1_mean": float("nan"), "medcon_f1_std": float("nan"),
                "mean_cui_count": float("nan"),
            })
            continue
        # Extract CUIs ONCE per seed for valid seeds (cache)
        seeds_needed = {s for pair in valid_pairs for s in pair}
        seed_to_cuis = {s: medcon.extract_cuis(seed_to_txt[s]) for s in seeds_needed}
        f1_values = []
        for i, j in valid_pairs:
            A = seed_to_cuis[i]; B = seed_to_cuis[j]
            if not A and not B:
                f1_values.append(1.0)
            elif not A or not B:
                f1_values.append(0.0)
            else:
                inter = len(A & B)
                p = inter / len(B); r = inter / len(A)
                f1_values.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
        rows.append({
            "input_id":         input_id,
            "model":            model,
            "quant_level":      quant,
            "n_pairs":          len(valid_pairs),
            "n_skipped_pairs":  n_skipped,
            "medcon_f1_mean":   float(np.mean(f1_values)),
            "medcon_f1_std":    float(np.std(f1_values, ddof=1)) if len(f1_values) > 1 else 0.0,
            "mean_cui_count":   float(np.mean([len(c) for c in seed_to_cuis.values()])),
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Numerical-attribute Jaccard (Issue 3 — addresses MEDCON's numerical blindness)
# ═════════════════════════════════════════════════════════════════════════════
def extract_numerical_facts(text) -> set:
    """
    Extract numerical-attribute triples from clinical text.

    Returns a set of (entity, value, unit) tuples. All strings lowercased
    and stripped. Returns empty set for empty/None input or no matches.

    Pre-specified patterns from NUMERIC_PATTERNS. NOT tuned per-model
    or per-quant — same patterns applied to all generations.
    """
    if _is_empty(text):
        return set()
    text_l = str(text).lower()
    facts = set()
    for pattern in NUMERIC_PATTERNS:
        for match in re.finditer(pattern, text_l):
            groups = match.groups()
            entity = groups[0].strip() if len(groups) > 0 and groups[0] else ""
            value  = groups[1].strip() if len(groups) > 1 and groups[1] else ""
            unit   = groups[2].strip() if len(groups) > 2 and groups[2] else ""
            facts.add((entity, value, unit))
    return facts


def compute_pairwise_numeric(df_gen: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Jaccard on numerical-attribute triples per cell.
    Symmetric. Higher = more consistent numerical attributes across seeds.

    Edge cases match MEDCON convention:
      - Both sets empty → 1.0 (trivially agreeing)
      - One set empty   → 0.0
      - Both populated  → |A ∩ B| / |A ∪ B|

    Issue 8 NaN guard: seeds with empty/NaN generated_text are dropped from
    the per-cell fact dictionary; pairs involving them are counted in
    `n_skipped_numeric`. If ALL pairs are skipped (every seed had empty text)
    the cell still emits a NaN row so downstream knows the cell existed.
    """
    rows = []
    groups = df_gen.groupby(["input_id", "model", "quant_level"])
    for (input_id, model, quant), sub in tqdm(groups, desc="Numeric Jaccard pairs"):
        seeds = sorted(sub["seed"].tolist())
        seed_to_facts = {
            row["seed"]: extract_numerical_facts(row["generated_text"])
            for _, row in sub.iterrows()
            if not _is_empty(row["generated_text"])
        }
        pairs = list(combinations(seeds, 2))
        if len(pairs) == 0:
            continue

        jaccard_values = []
        skipped = 0
        for i, j in pairs:
            if i not in seed_to_facts or j not in seed_to_facts:
                skipped += 1
                continue
            A, B = seed_to_facts[i], seed_to_facts[j]
            if not A and not B:
                jaccard_values.append(1.0)
            elif not A or not B:
                jaccard_values.append(0.0)
            else:
                jaccard_values.append(len(A & B) / len(A | B))

        if not jaccard_values:
            # All pairs had empty text on at least one side — emit NaN row
            rows.append({
                "input_id": input_id, "model": model, "quant_level": quant,
                "n_pairs": len(pairs), "n_skipped_numeric": skipped,
                "numeric_jaccard_mean": float("nan"),
                "numeric_jaccard_std":  float("nan"),
                "mean_fact_count":      0.0,
            })
            continue

        rows.append({
            "input_id":              input_id,
            "model":                 model,
            "quant_level":           quant,
            "n_pairs":               len(pairs),
            "n_skipped_numeric":     skipped,
            "numeric_jaccard_mean":  float(np.mean(jaccard_values)),
            "numeric_jaccard_std":   float(np.std(jaccard_values, ddof=1)) if len(jaccard_values) > 1 else 0.0,
            "mean_fact_count":       float(np.mean([len(f) for f in seed_to_facts.values()])),
        })

    return pd.DataFrame(rows)


NLI_COLS = ["input_id", "model", "quant_level", "n_pairs", "n_skipped_pairs",
            "nli_contradiction_max_mean",  "nli_contradiction_max_mean_std",
            "nli_contradiction_mean_mean", "nli_contradiction_mean_mean_std",
            "nli_contradiction_q90_mean",  "nli_contradiction_q90_mean_std"]


def compute_pairwise_nli(df_gen: pd.DataFrame, nli: NLIScorer,
                         checkpoint_path=None) -> pd.DataFrame:
    """
    Pairwise sentence-matrix bidirectional NLI contradiction between every
    seed pair. Each pair yields THREE aggregations (max_mean / mean_mean /
    q90_mean) from the same matrix; per-cell we report mean + sample std
    for each aggregation across the surviving pairs.

    Issue 8: pairs where either text is NaN/empty are skipped (sent_tokenize
    would crash on NaN).

    Checkpoint/resume (save behavior ONLY — never changes a computed value):
    if checkpoint_path is given, each completed cell's row is appended to that
    partial CSV (flush + os.fsync, header once). On restart, cells already in
    the partial are skipped and only missing cells are computed. The final
    DataFrame is assembled from the partial (all cells) and sorted by
    (input_id, model, quant_level) — identical order to a non-checkpointed
    groupby(sort=True) run.
    """
    def _mean(v): return float(np.mean(v))
    def _std(v):  return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

    # Resume: load already-completed cell keys from the partial file.
    completed = set()
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        prev = pd.read_csv(checkpoint_path)
        completed = set(zip(prev["input_id"], prev["model"], prev["quant_level"]))
        print(f"  Resume: {len(completed)} cells already in "
              f"{checkpoint_path}; skipping them.")

    # Open the partial in append mode; write the header exactly once.
    fh = writer = None
    if checkpoint_path is not None:
        new_file = not Path(checkpoint_path).exists()
        fh = open(checkpoint_path, "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=NLI_COLS)
        if new_file:
            writer.writeheader(); fh.flush(); os.fsync(fh.fileno())

    rows = []
    groups = df_gen.groupby(["input_id", "model", "quant_level"])
    print(f"  {len(groups)} (patient × model × quant) cells")
    for (input_id, model, quant), sub in tqdm(groups, desc="NLI sentence-matrix"):
        if (input_id, model, quant) in completed:
            continue
        seeds = sorted(sub["seed"].tolist())
        seed_to_txt = {row["seed"]: row["generated_text"] for _, row in sub.iterrows()}
        all_pairs = list(combinations(seeds, 2))
        valid_pairs = [(i, j) for i, j in all_pairs
                       if not _is_empty(seed_to_txt[i]) and not _is_empty(seed_to_txt[j])]
        n_skipped = len(all_pairs) - len(valid_pairs)
        if len(valid_pairs) == 0:
            # n_pairs=0 / NaN cell — still written as completed so resume
            # does not retry it (matches original NaN-emission behavior).
            row = {
                "input_id": input_id, "model": model, "quant_level": quant,
                "n_pairs": 0, "n_skipped_pairs": n_skipped,
                "nli_contradiction_max_mean":      float("nan"),
                "nli_contradiction_max_mean_std":  float("nan"),
                "nli_contradiction_mean_mean":     float("nan"),
                "nli_contradiction_mean_mean_std": float("nan"),
                "nli_contradiction_q90_mean":      float("nan"),
                "nli_contradiction_q90_mean_std":  float("nan"),
            }
        else:
            # Each contradictions[i] is a dict {max_mean, mean_mean, q90_mean}
            contradictions = [nli.sentence_matrix_contradiction(seed_to_txt[i], seed_to_txt[j])
                              for i, j in valid_pairs]
            max_vals  = [c["max_mean"]  for c in contradictions]
            mean_vals = [c["mean_mean"] for c in contradictions]
            q90_vals  = [c["q90_mean"]  for c in contradictions]
            row = {
                "input_id":                        input_id,
                "model":                           model,
                "quant_level":                     quant,
                "n_pairs":                         len(valid_pairs),
                "n_skipped_pairs":                 n_skipped,
                "nli_contradiction_max_mean":      _mean(max_vals),
                "nli_contradiction_max_mean_std":  _std(max_vals),
                "nli_contradiction_mean_mean":     _mean(mean_vals),
                "nli_contradiction_mean_mean_std": _std(mean_vals),
                "nli_contradiction_q90_mean":      _mean(q90_vals),
                "nli_contradiction_q90_mean_std":  _std(q90_vals),
            }
        rows.append(row)
        if writer is not None:
            writer.writerow(row); fh.flush(); os.fsync(fh.fileno())

    if fh is not None:
        fh.close()

    # Assemble final from the partial (all cells across resumes) or from rows.
    if checkpoint_path is not None:
        final = pd.read_csv(checkpoint_path)
    else:
        final = pd.DataFrame(rows, columns=NLI_COLS)
    final = final.sort_values(["input_id", "model", "quant_level"]).reset_index(drop=True)
    return final


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True,
                        help="long-format CSV: input_id, model, quant_level, seed, generated_text")
    parser.add_argument("--out",         required=True,
                        help="output CSV for pairwise BERTScore+ROUGE+MEDCON per condition")
    parser.add_argument("--out-nli",     default=None,
                        help="output CSV for pairwise NLI per condition (default: <out>_nli.csv)")
    parser.add_argument("--skip-medcon", action="store_true",
                        help="skip MEDCON (e.g. for environments without UMLS DB)")
    parser.add_argument("--nli-only", action="store_true",
                        help="compute ONLY the NLI step (steps 1-3 and the --out "
                             "metric_results.csv are skipped and NEVER touched). "
                             "Requires --out to already exist (the completed 4-metric file).")
    parser.add_argument("--nli-checkpoint", default=None,
                        help="partial CSV for per-cell NLI resume "
                             "(default: <out-nli>.partial.csv)")
    args = parser.parse_args()

    out_path     = Path(args.out)
    out_nli_path = Path(args.out_nli) if args.out_nli else out_path.with_name(out_path.stem + "_nli.csv")
    ckpt_path    = Path(args.nli_checkpoint) if args.nli_checkpoint \
                   else out_nli_path.with_name(out_nli_path.stem + ".partial.csv")

    df_gen = pd.read_csv(args.generations)

    print("=" * 70)
    print("METRIC PIPELINE (pairwise consistency between seeds)")
    print("=" * 70)
    print(f"Generations : {args.generations}  ({len(df_gen)} rows)")
    print(f"Device      : {DEVICE}")
    print(f"Patients    : {df_gen['input_id'].nunique()}")
    print(f"Conditions  : {df_gen.groupby(['model','quant_level']).ngroups}")
    print(f"Seeds/cell  : {sorted(df_gen.groupby(['input_id','model','quant_level'])['seed'].count().unique())}")
    print(f"NLI-only    : {args.nli_only}")

    if args.nli_only:
        # Steps 1-3 and metric_results.csv are NOT recomputed/touched. We only
        # confirm the 4-metric file already exists (for the downstream stats).
        if not out_path.exists():
            parser.error(f"--nli-only requires existing 4-metric file: {out_path}")
        print(f"\n[1-3] SKIPPED (--nli-only). 4-metric file preserved as-is: {out_path}")
        df_combined = None
    else:
        # 1. Pairwise BERTScore + ROUGE-L
        print("\n[1] BERTScore (deberta-xlarge-mnli, layer 40) + ROUGE-L pairwise ...")
        df_br = compute_pairwise_bertscore_rouge(df_gen)

        # 2. Pairwise MEDCON (unless skipped)
        if args.skip_medcon:
            print("\n[2] MEDCON skipped (--skip-medcon)")
            df_medcon = None
        else:
            print(f"\n[2] MEDCON (QuickUMLS, threshold={QUICKUMLS_THR}, window={QUICKUMLS_WINDOW}) pairwise ...")
            medcon = MEDCONScorer(QUICKUMLS_DB, QUICKUMLS_TUI_JSON)
            df_medcon = compute_pairwise_medcon(df_gen, medcon)

        # Merge BERTScore+ROUGE with MEDCON (if computed)
        if df_medcon is not None:
            df_combined = df_br.merge(
                df_medcon.drop(columns=["n_pairs"]),
                on=["input_id", "model", "quant_level"], how="left",
            )
        else:
            df_combined = df_br

        # 3. Numerical-attribute Jaccard (Issue 3 — pre-specified regex)
        print(f"\n[3] Numerical-attribute Jaccard (regex-based, pre-specified) ...")
        df_numeric = compute_pairwise_numeric(df_gen)
        df_combined = df_combined.merge(
            df_numeric.drop(columns=["n_pairs"]),
            on=["input_id", "model", "quant_level"], how="left",
        )

        df_combined.to_csv(out_path, index=False)
        print(f"\nSaved per-condition pairwise metrics: {out_path}")

    # 4. Pairwise NLI (sentence-matrix) — fp16, full 190 pairs, resumable.
    print(f"\n[4] NLI sentence-matrix (cross-encoder/nli-deberta-v3-large) pairwise ...")
    print(f"    checkpoint: {ckpt_path}")
    nli = NLIScorer()
    df_nli = compute_pairwise_nli(df_gen, nli, checkpoint_path=ckpt_path)
    df_nli.to_csv(out_nli_path, index=False)
    print(f"Saved pairwise NLI metric            : {out_nli_path}")

    # 4. Quick per-condition summary
    print("\n" + "=" * 70)
    print("Per-condition summary (means across seed pairs)")
    print("=" * 70)
    if df_combined is not None:
        metric_cols = [c for c in
                       ["bertscore_f1_mean", "rougeL_f1_mean", "medcon_f1_mean",
                        "numeric_jaccard_mean"]
                       if c in df_combined.columns]
        print("\nReference-free pairwise (higher = more consistent):")
        print(df_combined.groupby(["model", "quant_level"])[metric_cols].mean().round(4).to_string())
    print("\nNLI sentence-matrix contradiction (lower = more consistent):")
    print("  (showing all 3 aggregations: max_mean / mean_mean / q90_mean)")
    nli_cols = ["nli_contradiction_max_mean",
                "nli_contradiction_mean_mean",
                "nli_contradiction_q90_mean"]
    nli_cols_present = [c for c in nli_cols if c in df_nli.columns]
    print(df_nli.groupby(["model", "quant_level"])[nli_cols_present].mean().round(4).to_string())


if __name__ == "__main__":
    main()
