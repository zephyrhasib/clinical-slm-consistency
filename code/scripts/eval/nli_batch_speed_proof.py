"""
Direct fp16+190 speed + batch-size proof. Standalone, read-only on
outputs/main_eval/main_combined.csv, writes only outputs/nli_proof/.
Reuses NLIScorer verbatim; only precision (fp16) and BATCH SIZE vary.
Batch size only changes how forward passes are grouped -> results must be
bit-identical across batch sizes (asserted, with measured max drift).

3 cells (1 per model), each with a known fp32+190 GOLD anchor:
  Qwen  Q3_K_M MAIN-001  (fp32+190 GOLD = 741.98 s)
  Gemma Q3_K_M MAIN-001  (fp32+190 GOLD = 1084.96 s)
  Phi-4 Q4_K_M MAIN-001  (fp32+190 GOLD = 212.08 s)
Each cell: fp16, 190 pairs, batch in {32, 64, 128}.
"""
import sys, time
from pathlib import Path
from itertools import combinations
import numpy as np, pandas as pd, torch

ROOT = Path("/home/hasib/Research")
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import metric_pipeline as MP
from metric_pipeline import NLIScorer, _is_empty

SRC = ROOT / "outputs/main_eval/main_combined.csv"
OUTDIR = ROOT / "outputs/nli_proof"; OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTDIR / "batch_speed_results.csv"

# (model, quant, input_id, known fp32+190 GOLD seconds)
CELLS = [
    ("Qwen-MediCare-BD",    "Q3_K_M", "MAIN-001", 741.98),
    ("Gemma-3-4B-it",       "Q3_K_M", "MAIN-001", 1084.96),
    ("Phi-4-mini-instruct", "Q4_K_M", "MAIN-001", 212.08),
]
BATCH_SIZES = [32, 64, 128]
ALL_190 = list(combinations(range(20), 2))

print("=" * 84)
print("DIRECT fp16 + 190-pair SPEED & BATCH-SIZE PROOF")
print(f"  source (read-only): {SRC}")
print(f"  batch sizes tested: {BATCH_SIZES}  (results must be identical across them)")
print("=" * 84)

df = pd.read_csv(SRC)

def set_batch(bs):
    # contradiction_prob_batch(self, premises, hypotheses, batch_size=NLI_BATCH_SIZE)
    # patch the single default so the matrix call uses bs; math unchanged.
    NLIScorer.contradiction_prob_batch.__defaults__ = (bs,)

def score_cell(nli, sub):
    s2t = {r["seed"]: r["generated_text"] for _, r in sub.iterrows()}
    valid = [(i, j) for (i, j) in ALL_190
             if i in s2t and j in s2t and not _is_empty(s2t[i]) and not _is_empty(s2t[j])]
    vals = [nli.sentence_matrix_contradiction(s2t[i], s2t[j])["mean_mean"] for (i, j) in valid]
    return float(np.mean(vals)), len(valid)

print("\n  Loading NLI model (fp16) ...")
nli = NLIScorer()
nli.mdl = nli.mdl.half()

rows = []
for model, quant, pid, gold_fp32 in CELLS:
    sub = df[(df.model == model) & (df.quant_level == quant) & (df.input_id == pid)]
    print(f"\n--- {model} {quant} {pid}  (fp32+190 GOLD anchor = {gold_fp32:.1f}s) ---")
    cell_vals = {}
    for bs in BATCH_SIZES:
        set_batch(bs)
        torch.cuda.empty_cache()
        try:
            t0 = time.time()
            mm, npv = score_cell(nli, sub)
            el = time.time() - t0
            cell_vals[bs] = mm
            speedup = gold_fp32 / el if el > 0 else float("nan")
            print(f"  batch={bs:<4} mean_mean={mm:.8f}  np={npv}  {el:.1f}s  "
                  f"({el/60:.2f} min)  fp16+190 speedup vs fp32 GOLD = {speedup:.2f}x")
            rows.append({"model": model, "quant_level": quant, "input_id": pid,
                         "precision": "fp16", "n_pairs": npv, "batch_size": bs,
                         "mean_mean": mm, "elapsed_seconds": round(el, 2),
                         "fp32_190_gold_seconds": gold_fp32,
                         "fp16_speedup_vs_fp32_gold": round(speedup, 3)})
        except RuntimeError as e:
            msg = "OOM" if "out of memory" in str(e).lower() else f"ERR:{str(e)[:60]}"
            print(f"  batch={bs:<4} {msg}")
            rows.append({"model": model, "quant_level": quant, "input_id": pid,
                         "precision": "fp16", "n_pairs": 190, "batch_size": bs,
                         "mean_mean": float("nan"), "elapsed_seconds": float("nan"),
                         "fp32_190_gold_seconds": gold_fp32,
                         "fp16_speedup_vs_fp32_gold": float("nan")})
            torch.cuda.empty_cache()
    # identical-results check across batch sizes
    vv = [v for v in cell_vals.values() if v == v]
    if len(vv) >= 2:
        drift = max(vv) - min(vv)
        print(f"  >> max mean_mean drift across batch sizes = {drift:.2e}  "
              f"({'IDENTICAL (<1e-6)' if drift < 1e-6 else 'NEGLIGIBLE (<1e-4)' if drift < 1e-4 else 'CHECK'})")

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print(f"\nWrote {OUT_CSV}  ({len(out)} rows)")

# ── full-run projection from the measured fp16+190 best-batch rate per model ──
print("\n" + "=" * 84)
print("FULL-RUN PROJECTION (fp16+190), using each model's FASTEST measured batch")
print("  full study = 400 patient-cells per model = 1,200 total")
print("=" * 84)
total_h = 0.0
print(f"{'model':<22}{'best batch':>11}{'s/cell':>9}{'x400 cells':>13}")
for model, quant, pid, _ in CELLS:
    sub = out[(out.model == model) & out.elapsed_seconds.notna()]
    if len(sub):
        best = sub.loc[sub.elapsed_seconds.idxmin()]
        h = best.elapsed_seconds * 400 / 3600
        total_h += h
        print(f"{model:<22}{int(best.batch_size):>11}{best.elapsed_seconds:>9.1f}{h:>12.1f}h")
print("-" * 55)
print(f"{'TOTAL fp16+190 (best batch)':<42}{total_h:>12.1f}h  (~{total_h/24:.1f} days)")
print("\n  NOTE: these 3 anchor cells are Q3/Q4 (longer notes) -> upper-ish bound.")
print("  Qwen/Phi-4 Q6/Q5 cells are shorter, so the real full run is likely a bit faster.")
print("\n>>> BATCH-SPEED PROOF done.")
