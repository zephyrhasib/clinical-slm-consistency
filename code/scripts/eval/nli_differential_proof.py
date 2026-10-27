"""
Differential NLI proof — measure between-quant cancellation directly.

Standalone. Does NOT modify metric_pipeline.py or outputs/main_eval/.
Reads outputs/main_eval/main_combined.csv (read-only). Writes only to
outputs/nli_proof/. Reuses NLIScorer verbatim from metric_pipeline.py;
only precision (fp16 .half()) and pair-count (60-pair seed-42 subsample) vary.

Design: 2 Qwen patients present in all 4 quant levels (>=2 valid seeds each),
x 4 quants x 2 conditions (GOLD fp32+190, PROD fp16+60) = 16 NLI runs.
Then compares BETWEEN-QUANT differences (Q6-Q5, Q5-Q4, Q4-Q3, Q6-Q3) under
GOLD vs PROD -> diff_error = |PROD_diff - GOLD_diff|, the quantity the
hypotheses actually test.
"""
import sys, time, random
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

ROOT = Path("/home/hasib/Research")
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from metric_pipeline import NLIScorer, _is_empty   # verbatim reuse

SRC   = ROOT / "outputs/main_eval/main_combined.csv"
OUTDIR = ROOT / "outputs/nli_proof"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTDIR / "differential_results.csv"

MODEL  = "Qwen-MediCare-BD"
QUANTS = ["Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M"]
COMPARISONS = [("Q6_K","Q5_K_M"), ("Q5_K_M","Q4_K_M"), ("Q4_K_M","Q3_K_M"), ("Q6_K","Q3_K_M")]

ALL_190 = list(combinations(range(20), 2))
SUB60   = sorted(random.Random(42).sample(ALL_190, 60))
assert SUB60[:5] == [(0,2),(0,7),(0,8),(0,9),(0,12)], f"subsample mismatch: {SUB60[:5]}"

print("=" * 80)
print("DIFFERENTIAL NLI PROOF  (between-quant cancellation)")
print(f"  source (read-only): {SRC}")
print(f"  model             : {MODEL}")
print(f"  60-pair subsample : random.Random(42).sample(ALL_190,60); first5={SUB60[:5]}")
print("=" * 80)

df = pd.read_csv(SRC)
dm = df[df.model == MODEL]

# valid-seed count per (quant, patient)
def valid_seeds(sub):
    return [r["seed"] for _, r in sub.iterrows() if not _is_empty(r["generated_text"])]

# patients with >=2 valid seeds in ALL 4 quants
ok = None
for q in QUANTS:
    s = dm[dm.quant_level == q].groupby("input_id").apply(
        lambda g: sum(not _is_empty(t) for t in g["generated_text"]) >= 2)
    okq = set(s[s].index)
    ok = okq if ok is None else (ok & okq)
patients = sorted(ok)[:2]
print(f"  chosen patients   : {patients}\n")

def score_cell(nli, sub, pairs):
    """Mean of nli mean_mean over the given (valid) seed-pairs of one patient-cell."""
    seed_to_txt = {r["seed"]: r["generated_text"] for _, r in sub.iterrows()}
    valid = [(i, j) for (i, j) in pairs
             if i in seed_to_txt and j in seed_to_txt
             and not _is_empty(seed_to_txt[i]) and not _is_empty(seed_to_txt[j])]
    vals = [nli.sentence_matrix_contradiction(seed_to_txt[i], seed_to_txt[j])["mean_mean"]
            for (i, j) in valid]
    return float(np.mean(vals)), len(valid)

rows = []

def run_condition(nli, precision, pairs, npairs_label):
    for pid in patients:
        for q in QUANTS:
            sub = dm[(dm.quant_level == q) & (dm.input_id == pid)]
            t0 = time.time()
            mm, npv = score_cell(nli, sub, pairs)
            el = time.time() - t0
            rows.append({"input_id": pid, "model": MODEL, "quant_level": q,
                         "condition": ("GOLD" if precision=="fp32" else "PROD"),
                         "precision": precision, "n_pairs": npv,
                         "mean_mean": mm, "elapsed_seconds": round(el, 2)})
            print(f"  [{('GOLD fp32 190' if precision=='fp32' else 'PROD fp16 60 ')}] "
                  f"{pid} {q:<7}: mean_mean={mm:.6f}  np={npv}  {el/60:.2f} min")

# GOLD: fp32, all-190 valid pairs
nli = NLIScorer()
print("\n[GOLD] fp32, 190 pairs")
run_condition(nli, "fp32", ALL_190, 190)

# PROD: fp16, 60-pair subsample
print("\n[PROD] converting to fp16 (.half()) ...")
nli.mdl = nli.mdl.half()
print("[PROD] fp16, 60 pairs")
run_condition(nli, "fp16", SUB60, 60)

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print(f"\nWrote {OUT_CSV}  ({len(out)} rows)")

# ── between-quant differences under each recipe ──────────────────────────────
def cell(pid, q, cond):
    r = out[(out.input_id==pid)&(out.quant_level==q)&(out.condition==cond)]
    return float(r["mean_mean"].iloc[0])

print("\n" + "=" * 80)
print("BETWEEN-QUANT DIFFERENCES: GOLD vs PROD, and the error in the tested quantity")
print("=" * 80)
print(f"{'patient':<10}{'comparison':<16}{'GOLD_diff':>12}{'PROD_diff':>12}{'|diff_error|':>14}")
errs = []
for pid in patients:
    for (a, b) in COMPARISONS:
        gd = cell(pid, a, "GOLD") - cell(pid, b, "GOLD")
        pdd = cell(pid, a, "PROD") - cell(pid, b, "PROD")
        e = abs(pdd - gd)
        errs.append(e)
        print(f"{pid:<10}{a+'-'+b:<16}{gd:>12.6f}{pdd:>12.6f}{e:>14.6f}")

print("-" * 80)
print(f"  MEAN |diff_error| across {len(errs)} between-quant comparisons : {np.mean(errs):.6f}")
print(f"  (compare to per-cell |PROD-GOLD| mean_mean = 0.0074 from the 9-patient proof)")
print(f"  cancellation factor = 0.0074 / mean|diff_error| = "
      f"{0.0074/np.mean(errs):.1f}x" if np.mean(errs) > 0 else "n/a")
print("\n>>> DIFFERENTIAL PROOF done.")
