"""
Final verification of the entire metric + statistical + UMLS stack.
Runs every component, reports PASS/FAIL.

Validates the corrected pipeline:
  - QuickUMLS threshold = 0.7 (pre-specified; was 1.0 — exact-match)
  - Pairwise BERTScore + ROUGE-L (between seed outputs, NOT against reference)
  - Pairwise MEDCON F1 (concept-set agreement between seed CUI sets)
  - NLI sentence-matrix bidirectional contradiction (Laban et al. TACL 2022)

Run in venv_metrics:
    source venv_metrics/bin/activate
    python scripts/verify_metric_stack.py
"""
import inspect
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

checks = []
def check(name, ok, detail=""):
    checks.append((name, ok, detail))
    s = "PASS" if ok else "FAIL"
    print(f"  [{s}] {name}" + (f"  [{detail}]" if detail else ""))

print("=" * 72)
print("METRIC + STATISTICAL + UMLS STACK — FINAL VERIFICATION")
print("=" * 72)

# ── 1. Environment ───────────────────────────────────────────────────────────
print("\n[1] Environment")
import torch, transformers, tokenizers, huggingface_hub
import bert_score, rouge_score, scipy, statsmodels, numpy, pandas
check(f"venv is venv_metrics", "venv_metrics" in sys.prefix, sys.prefix)
check(f"torch == 2.11.0+cu128", torch.__version__ == "2.11.0+cu128", torch.__version__)
check(f"CUDA available", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no")
check(f"transformers == 4.35.2", transformers.__version__ == "4.35.2", transformers.__version__)
check(f"tokenizers == 0.15.2",  tokenizers.__version__ == "0.15.2", tokenizers.__version__)
check(f"huggingface_hub == 0.17.0", huggingface_hub.__version__ == "0.17.0", huggingface_hub.__version__)
check(f"bert_score installed", bert_score.__version__ in ("0.3.12", "0.3.13"), bert_score.__version__)
check(f"scipy installed",        scipy.__version__.startswith("1."), scipy.__version__)
check(f"statsmodels installed",  statsmodels.__version__.startswith("0."), statsmodels.__version__)

# ── 2. QuickUMLS imports (Python 3.12 patches applied) ───────────────────────
print("\n[2] QuickUMLS stack imports")
import importlib
for mod in ("leveldb", "quickumls_simstring.simstring", "quickumls", "spacy", "nltk"):
    try:
        importlib.import_module(mod)
        check(f"import {mod}", True)
    except Exception as e:
        check(f"import {mod}", False, str(e)[:100])

# ── 3. UMLS files on disk ────────────────────────────────────────────────────
print("\n[3] UMLS artifacts on disk")
files = {
    "umls/META/MRCONSO.RRF":               (2.0,  "GB"),
    "umls/META/MRSTY.RRF":                 (50,   "MB"),
    "umls/META/extraction_hashes.txt":     (0.0001, "KB"),
    "umls/aci_bench_tui_codes.json":       (0.01, "KB"),
    "umls/aci_bench_tui_codes.txt":        (0.01, "KB"),
    "umls/quickumls-db/language.flag":     (0.0,  "B"),
    "umls/quickumls-db/database_backend.flag": (0.0, "B"),
    "umls/quickumls-db/cui-semtypes.db/cui.unqlite":      (1.5, "GB"),
    "umls/quickumls-db/cui-semtypes.db/semtypes.unqlite": (0.4, "GB"),
    "umls/quickumls-db/umls-simstring.db/umls-terms.simstring": (1.0, "GB"),
    "umls/quickumls-db/build_manifest.txt": (0.0001, "KB"),
}
for fpath, (min_size, unit) in files.items():
    p = ROOT / fpath
    exists = p.exists()
    size = p.stat().st_size if exists else 0
    if unit == "GB":   min_b = min_size * 1024**3
    elif unit == "MB": min_b = min_size * 1024**2
    elif unit == "KB": min_b = min_size * 1024
    else:              min_b = 0
    check(f"{fpath}", exists and size >= min_b * 0.5,
          f"{size/(1024**3 if unit=='GB' else 1024**2 if unit=='MB' else 1024):.2f} {unit}")

# ── 4. ACI-Bench TUI list integrity ──────────────────────────────────────────
print("\n[4] ACI-Bench TUI list")
tui = json.loads((ROOT / "umls/aci_bench_tui_codes.json").read_text())
check("66 TUIs",           tui["n_included_tuis"] == 66, str(tui["n_included_tuis"]))
check("7 included groups", len(tui["included_groups"]) == 7, ",".join(tui["included_groups"]))
check("8 excluded groups", len(tui["exclusion_list"]) == 8, str(len(tui["exclusion_list"])))
expected_groups = {"Phenomena", "Devices", "Genes & Molecular Sequences",
                   "Disorders", "Physiology", "Anatomy", "Chemicals & Drugs"}
check("included groups match canonical",
      set(tui["included_groups"]) == expected_groups,
      "ok" if set(tui["included_groups"]) == expected_groups else "mismatch")
for t, name in [("T047", "Disease or Syndrome"),
                ("T121", "Pharmacologic Substance"),
                ("T023", "Body Part"),
                ("T184", "Sign or Symptom")]:
    check(f"TUI {t} ({name}) in list", t in tui["all_tuis_sorted"])
for t in ["T060", "T061", "T065"]:  # Procedures — excluded
    check(f"TUI {t} (Procedure) NOT in list", t not in tui["all_tuis_sorted"])

# ── 5. Pre-specified constants (corrected per fix-spec) ─────────────────────
print("\n[5] Pre-specified constants (post-fix)")
# Load metric_pipeline module to inspect
import importlib.util
spec = importlib.util.spec_from_file_location("metric_pipeline", ROOT / "scripts/eval/metric_pipeline.py")
mp   = importlib.util.module_from_spec(spec)
# Note: don't exec_module here because it would load heavy models. Inspect source instead.
src = (ROOT / "scripts/eval/metric_pipeline.py").read_text()
check("metric_pipeline: QUICKUMLS_THR == 0.7",       "QUICKUMLS_THR      = 0.7" in src, "in source")
check("metric_pipeline: QUICKUMLS_WINDOW == 5",      "QUICKUMLS_WINDOW   = 5" in src,   "in source")
check("metric_pipeline: NLI_CONTRADICTION == 0",     "NLI_CONTRADICTION  = 0" in src,   "in source")
check("metric_pipeline: BERTSCORE_LAYERS == 40",     "BERTSCORE_LAYERS   = 40" in src,  "in source")

src_smoke = (ROOT / "scripts/umls_04_smoke_test.py").read_text()
check("umls_04_smoke_test: QUICKUMLS_THR == 0.7",    "QUICKUMLS_THR     = 0.7" in src_smoke, "in source")

# ── 6. Pipeline structural fixes verified by inspection ──────────────────────
print("\n[6] Pipeline structural fixes")
check("metric_pipeline: compute_reference_based() REMOVED",
      "def compute_reference_based" not in src, "removed")
check("metric_pipeline: compute_pairwise_bertscore_rouge() ADDED",
      "def compute_pairwise_bertscore_rouge" in src, "added")
check("metric_pipeline: compute_pairwise_medcon() ADDED",
      "def compute_pairwise_medcon" in src, "added")
check("metric_pipeline: MEDCONScorer class ADDED",
      "class MEDCONScorer" in src, "added")
check("metric_pipeline: NLIScorer.sentence_matrix_contradiction ADDED",
      "def sentence_matrix_contradiction" in src, "added")
check("metric_pipeline: NLIScorer.contradiction_prob_batch ADDED",
      "def contradiction_prob_batch" in src, "added")
check("metric_pipeline: _all_aggregations_contradiction ADDED (Issue 1 — tri-aggregation)",
      "def _all_aggregations_contradiction" in src, "added")
check("metric_pipeline: numeric Jaccard ADDED (Issue 3 — MEDCON num-blindness fix)",
      "def compute_pairwise_numeric" in src and "NUMERIC_PATTERNS" in src, "added")
check("metric_pipeline: NLI uses sentence_matrix_contradiction (not bidirectional)",
      "nli.sentence_matrix_contradiction(" in src, "in compute_pairwise_nli")
check("metric_pipeline: SummaC removed from pipeline",
      "from summac" not in src and "summac_zs" not in src, "removed")
check("metric_pipeline: --references arg REMOVED",
      "--references" not in src, "removed")
check("metric_pipeline: nltk.sent_tokenize imported",
      "from nltk.tokenize import sent_tokenize" in src, "imported")

# stat pipeline (now split into stat_omnibus.py + stat_pairwise.py, both under scripts/eval/)
src_omn  = (ROOT / "scripts/eval/stat_omnibus.py").read_text()
src_pair = (ROOT / "scripts/eval/stat_pairwise.py").read_text()
check("stat_omnibus.py exists at scripts/eval/", True, "moved 2026-05-25")
check("stat_pairwise.py exists at scripts/eval/", True, "moved 2026-05-25")
check("stat: bertscore_f1_mean referenced (omnibus)",
      "bertscore_f1_mean" in src_omn, "column updated")
check("stat: rougeL_f1_mean referenced (omnibus)",
      "rougeL_f1_mean" in src_omn, "column updated")
check("stat: medcon_f1_mean in METRIC_DIRECTION (omnibus + pairwise)",
      "\"medcon_f1_mean\"" in src_omn and "\"medcon_f1_mean\"" in src_pair, "added")
check("stat: numeric_jaccard_mean in METRIC_DIRECTION (Issue 3)",
      "\"numeric_jaccard_mean\"" in src_omn and "\"numeric_jaccard_mean\"" in src_pair, "added")
check("stat: 3 NLI aggregations in METRIC_DIRECTION (Issue 1)",
      all(f"\"nli_contradiction_{a}_mean\"" in src_omn
          for a in ("max", "mean", "q90")), "all 3 present")
check("stat: summac_zs removed from METRIC_DIRECTION",
      "\"summac_zs\"" not in src_omn and "\"summac_zs\"" not in src_pair, "removed")
check("stat: legacy nli_contradiction_mean removed from METRIC_DIRECTION (Issue 1)",
      "\"nli_contradiction_mean\":" not in src_omn
      and "\"nli_contradiction_mean\":" not in src_pair, "removed")
check("stat: Wilcoxon zero_method='pratt' (audit fix 4)",
      "zero_method=\"pratt\"" in src_pair or "zero_method='pratt'" in src_pair, "fix applied")
check("stat: Holm-Bonferroni via statsmodels (audit fix 6)",
      "method=\"holm\"" in src_pair or "method='holm'" in src_pair, "fix applied")
check("stat: BCa percentile fallback BCA_MIN_N=30 (audit fix 2)",
      "BCA_MIN_N" in src_pair, "fix applied")

# ── 7. QuickUMLS live query at 0.7 ───────────────────────────────────────────
print("\n[7] QuickUMLS live query test at threshold=0.7 (loads ~5 GB DB)")
from quickumls import QuickUMLS
t0 = time.time()
matcher = QuickUMLS("umls/quickumls-db", threshold=0.7, window=5)
check(f"DB loads at threshold=0.7", True, f"{time.time()-t0:.1f}s")

allowed_tuis = set(tui["all_tuis_sorted"])
probes = [
    ("hypertension",                 "C0020538"),
    ("type 2 diabetes mellitus",     "C0011860"),
    ("metformin",                    "C0025598"),
    ("shortness of breath",          "C0013404"),
]
for term, expected_cui in probes:
    matches = matcher.match(term, best_match=True, ignore_syntax=True)
    flat = [m for span in matches for m in span]
    cuis_filtered = [m["cui"] for m in flat
                     if any(t in allowed_tuis for t in m.get("semtypes", set()))]
    check(f"'{term}' → CUI {expected_cui} present", expected_cui in cuis_filtered,
          f"{len(cuis_filtered)} CUIs returned")
# Procedure still filtered out at 0.7
flat = [m for span in matcher.match("colonoscopy", best_match=True, ignore_syntax=True) for m in span]
filtered = [m for m in flat if any(t in allowed_tuis for t in m.get("semtypes", set()))]
# At 0.7 fuzzy matching may pick up Medical Device (T074, e.g. "colonoscope") which IS in TUI list.
# This is correct behaviour — the filter is by TUI, not by intent.
check("'colonoscopy' filtered (Procedures excluded)",
      not any("T060" in m.get("semtypes", set())
              or "T061" in m.get("semtypes", set())
              or "T065" in m.get("semtypes", set()) for m in filtered),
      "no T060/T061/T065 in filtered output")

# ── 8. NLI cross-encoder load + score ────────────────────────────────────────
print("\n[8] NLI cross-encoder (deberta-v3-large)")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-large", use_fast=False)
mdl = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-large")
mdl.eval()
if torch.cuda.is_available():
    mdl = mdl.cuda()
labels = mdl.config.id2label
check("NLI label[0] == 'contradiction'",
      labels[0].lower().startswith("contradict"), labels[0])
with torch.no_grad():
    inputs = tok(["Patient has hypertension."], ["The patient is hypertensive."],
                 return_tensors="pt", truncation=True, padding=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    probs = torch.softmax(mdl(**inputs).logits, dim=-1)[0].cpu().numpy()
check("NLI: A↔B entailment > contradiction",
      probs[1] > probs[0],
      f"contradict={probs[0]:.4f} entail={probs[1]:.4f}")

# ── 9. NLI sentence-matrix behaviour ─────────────────────────────────────────
print("\n[9] NLI sentence-matrix behaviour (live)")
sys.path.insert(0, str(ROOT / "scripts"))
# Import the NLIScorer class (will reuse the already-loaded model)
# To avoid re-loading, do a minimal version inline.
import nltk
nltk.download("punkt", quiet=True); nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

@torch.no_grad()
def cprob_batch(premises, hypotheses, batch_size=32):
    out = []
    for s in range(0, len(premises), batch_size):
        inputs = tok(premises[s:s+batch_size], hypotheses[s:s+batch_size],
                     return_tensors="pt", truncation=True, padding=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        probs = torch.softmax(mdl(**inputs).logits, dim=-1)[:, 0]
        out.extend(probs.cpu().tolist())
    return out

def row_max_mean(sents_p, sents_h):
    flat_p = [s for s in sents_p for _ in sents_h]
    flat_h = [s for _ in sents_p for s in sents_h]
    flat_scores = cprob_batch(flat_p, flat_h)
    n_h = len(sents_h)
    return sum(max(flat_scores[i*n_h:(i+1)*n_h]) for i in range(len(sents_p))) / len(sents_p)

def smc(a, b):
    sa, sb = sent_tokenize(a), sent_tokenize(b)
    if len(sa) < 2 or len(sb) < 2:
        # fallback
        c1 = cprob_batch([a], [b])[0]
        c2 = cprob_batch([b], [a])[0]
        return (c1 + c2) / 2
    return (row_max_mean(sa, sb) + row_max_mean(sb, sa)) / 2

# Test 1: identical texts → contradiction near 0
text = "Patient is a 62-year-old man. He has hypertension. Blood pressure is 148 over 92. Start lisinopril 10 mg daily."
identical_score = smc(text, text)
check("Identical multi-sentence texts: NLI contradiction near 0",
      identical_score < 0.05, f"{identical_score:.4f}")

# Test 2: clear contradiction in 1 sentence among agreeing context
text_a = ("Patient is a 62-year-old man. He has hypertension. "
          "Blood pressure is 148 over 92. Start lisinopril 10 mg daily.")
text_b = ("Patient is a 62-year-old man. He has hypertension. "
          "Blood pressure is 148 over 92. Start metformin 500 mg daily.")
diff_score = smc(text_a, text_b)
check("Mostly-agreeing texts with 1 contradicting sentence: NLI > identical",
      diff_score > identical_score, f"{diff_score:.4f} vs identical {identical_score:.4f}")

# Test 3: short-text fallback path
short_a = "Patient has hypertension."
short_b = "Patient has hypertension."
short_score = smc(short_a, short_b)
check("Short-text (<2 sentences) fallback returns contradiction near 0 for identical input",
      short_score < 0.05, f"{short_score:.4f}")

# ── 10. BERTScore with monkey-patch ──────────────────────────────────────────
print("\n[10] BERTScore (deberta-xlarge-mnli) with overflow patch")
import bert_score.utils as _bsu
_orig = _bsu.sent_encode
def _patched(tokenizer, sent):
    saved = tokenizer.model_max_length
    tokenizer.model_max_length = 512
    r = _orig(tokenizer, sent)
    tokenizer.model_max_length = saved
    return r
_bsu.sent_encode = _patched
from bert_score import score as bert_score_fn
P, R, F = bert_score_fn(
    ["The patient has high blood pressure."],
    ["Patient suffers from hypertension."],
    model_type="microsoft/deberta-xlarge-mnli", num_layers=40,
    lang="en", verbose=False, batch_size=1)
check("BERTScore returns F1 in (0.5, 1.0)", 0.5 < F.item() < 1.0, f"F={F.item():.4f}")

# ── 11. ROUGE-L ──────────────────────────────────────────────────────────────
print("\n[11] ROUGE-L")
from rouge_score import rouge_scorer
sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
s = sc.score("The patient has high blood pressure.", "Patient suffers from hypertension.")
check("ROUGE-L returns F1 in [0, 1]", 0 <= s["rougeL"].fmeasure <= 1, f"F={s['rougeL'].fmeasure:.4f}")

# ── 12. Statistical functions ────────────────────────────────────────────────
print("\n[12] Statistical functions")
import numpy as np
from scipy import stats as scipy_stats
x = np.array([0.9, 0.85, 0.88, 0.92, 0.91])
y = np.array([0.77, 0.79, 0.82, 0.74, 0.78])
res = scipy_stats.wilcoxon(x, y)
check(f"scipy.stats.wilcoxon works", isinstance(res.pvalue, float), f"p={res.pvalue:.4f}")
rng = np.random.default_rng(42)
diffs = x - y
boot = np.array([np.median(diffs[rng.integers(0, len(diffs), len(diffs))]) for _ in range(1000)])
ci = np.percentile(boot, [2.5, 97.5])
check(f"Paired bootstrap works", ci[0] < ci[1], f"CI=[{ci[0]:.3f},{ci[1]:.3f}]")

# ── 13. Demo eval outputs (corrected pipeline) ───────────────────────────────
print("\n[13] Demo eval outputs from corrected pipeline")
import pandas as pd
mp_path  = ROOT / "data/demo_eval/metric_results.csv"
nli_path = ROOT / "data/demo_eval/metric_results_nli.csv"
for f in ["synthetic_generations.csv", "metric_results.csv",
          "metric_results_nli.csv", "stat_summary.csv", "stat_pairs.csv"]:
    p = ROOT / "data/demo_eval" / f
    check(f"data/demo_eval/{f}", p.exists())

# Validate columns in metric_results
if mp_path.exists():
    df_mp = pd.read_csv(mp_path)
    # Pairwise metrics (Issues 1 + 3 expanded schema)
    for col in ["bertscore_f1_mean", "rougeL_f1_mean", "medcon_f1_mean",
                "numeric_jaccard_mean", "n_pairs"]:
        check(f"metric_results.csv has column '{col}'", col in df_mp.columns)
    check("metric_results.csv has 10 rows (5 patients × 2 quants)",
          len(df_mp) == 10, str(len(df_mp)))
if nli_path.exists():
    df_nli = pd.read_csv(nli_path)
    # Issue 1 — three pre-specified NLI aggregations (legacy single col removed)
    for col in ["nli_contradiction_max_mean",
                "nli_contradiction_mean_mean",
                "nli_contradiction_q90_mean"]:
        check(f"metric_results_nli.csv has '{col}'", col in df_nli.columns)

# ── Verdict ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
n_pass = sum(1 for _, ok, _ in checks if ok)
n_total = len(checks)
print(f"VERIFICATION RESULT: {n_pass}/{n_total} checks passed")
print("=" * 72)
if n_pass != n_total:
    print("\nFAILURES:")
    for name, ok, detail in checks:
        if not ok:
            print(f"  FAIL: {name}  [{detail}]")
    sys.exit(1)
