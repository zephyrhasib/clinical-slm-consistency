"""
STANDALONE NLI subsample-validation proof.  READ-ONLY on generation data.

Does NOT modify or run the main metric pipeline. Reuses the production
NLIScorer VERBATIM (imported from metric_pipeline) so the matrix math is
byte-for-byte identical. The ONLY knobs this proof introduces are:
  (a) precision   : fp32 (default) vs fp16 (mdl.half())
  (b) pair count  : all C(20,2)=190  vs  a seed-42 random 60-pair subsample

Writes ONLY to outputs/nli_proof/. Never writes under outputs/main_eval/.

Stages:
  --stage time1   1 Qwen Q3 patient, GOLD (fp32, 190) only. Prints minutes. Exits.
  --stage full    3 Qwen Q3 + 1 Gemma Q3 + 1 Phi-4 Q4 patients.
                  Per patient: GOLD (fp32,190) + PROD (fp16,60).
                  Plus fp32+60 diagnostic on the 3 Qwen patients.
                  Writes results.csv, prints diffs + timing extrapolation.

Run (venv_metrics), plain redirect to a log (no tmux):
  source venv_metrics/bin/activate
  python scripts/eval/nli_subsample_proof.py --stage time1
  python scripts/eval/nli_subsample_proof.py --stage full \
      > outputs/nli_proof/proof_$(date +%Y%m%d_%H%M%S).log 2>&1
"""
import argparse, random, time, csv
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

# Reuse the PRODUCTION NLI scorer + helpers verbatim (byte-identical matrix math)
from metric_pipeline import NLIScorer, _is_empty, NLI_BATCH_SIZE  # noqa: E402

ROOT      = Path(__file__).resolve().parent.parent.parent
COMBINED  = ROOT / "outputs/main_eval/main_combined.csv"      # read-only source
OUT_DIR   = ROOT / "outputs/nli_proof"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS   = OUT_DIR / "results.csv"

FULL_SEEDS  = list(range(20))
ALL_190     = list(combinations(sorted(FULL_SEEDS), 2))        # identical order to pipeline
SUBSAMPLE_60 = sorted(random.Random(42).sample(ALL_190, 60))  # same selection every patient

TARGETS = [   # (model, quant_level, n_patients)
    ("Qwen-MediCare-BD",     "Q3_K_M", 3),
    ("Gemma-3-4B-it",        "Q3_K_M", 3),
    ("Phi-4-mini-instruct",  "Q4_K_M", 3),
]


def pick_patients(df, model, quant, n):
    """First n input_ids (ascending) in this cell that have >=2 valid seeds."""
    cell = df[(df["model"] == model) & (df["quant_level"] == quant)]
    if cell.empty:
        raise SystemExit(f"NO ROWS for ({model}, {quant}) in {COMBINED}")
    chosen = []
    for pid in sorted(cell["input_id"].unique()):
        sub = cell[cell["input_id"] == pid]
        txt = {r["seed"]: r["generated_text"] for _, r in sub.iterrows()}
        n_valid = sum(1 for s in sub["seed"] if not _is_empty(txt.get(s)))
        if n_valid >= 2:
            chosen.append(pid)
        if len(chosen) == n:
            break
    return chosen


def seed_text_map(df, model, quant, pid):
    sub = df[(df["model"] == model) & (df["quant_level"] == quant) & (df["input_id"] == pid)]
    return {r["seed"]: r["generated_text"] for _, r in sub.iterrows()}


def score_patient(nli, txt, candidate_pairs):
    """Per-patient NLI exactly as compute_pairwise_nli: mean over valid pairs of
    each aggregation. candidate_pairs is the universe to draw from (190 or the 60)."""
    valid = [(i, j) for i, j in candidate_pairs
             if not _is_empty(txt.get(i)) and not _is_empty(txt.get(j))]
    if not valid:
        return dict(mean_mean=float("nan"), q90_mean=float("nan"),
                    max_mean=float("nan"), n_pairs=0)
    contr = [nli.sentence_matrix_contradiction(txt[i], txt[j]) for i, j in valid]
    return dict(
        mean_mean=float(np.mean([c["mean_mean"] for c in contr])),
        q90_mean=float(np.mean([c["q90_mean"]  for c in contr])),
        max_mean=float(np.mean([c["max_mean"]  for c in contr])),
        n_pairs=len(valid),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["time1", "full"], required=True)
    args = ap.parse_args()

    print("=" * 80)
    print(f"NLI SUBSAMPLE PROOF  —  stage={args.stage}")
    print(f"  source (read-only): {COMBINED}")
    print(f"  batch size        : {NLI_BATCH_SIZE} (unchanged from production)")
    print(f"  60-pair subsample : random.Random(42).sample(ALL_190, 60), sorted; "
          f"first 5 = {SUBSAMPLE_60[:5]}")
    print("=" * 80)

    df = pd.read_csv(COMBINED)

    if args.stage == "time1":
        model, quant = "Qwen-MediCare-BD", "Q3_K_M"
        pid = pick_patients(df, model, quant, 1)[0]
        print(f"\nTIMING: GOLD (fp32, 190 pairs) on ONE patient: {model} {quant} {pid}")
        nli = NLIScorer()                       # fp32 (default)
        txt = seed_text_map(df, model, quant, pid)
        t0 = time.perf_counter()
        r = score_patient(nli, txt, ALL_190)
        el = time.perf_counter() - t0
        print(f"\n  patient        : {pid}")
        print(f"  valid pairs    : {r['n_pairs']} / 190")
        print(f"  mean_mean      : {r['mean_mean']:.6f}")
        print(f"  q90_mean       : {r['q90_mean']:.6f}")
        print(f"  max_mean       : {r['max_mean']:.6f}")
        print(f"  ELAPSED        : {el:.1f} s  =  {el/60:.2f} min")
        print(f"\n>>> ONE-PATIENT GOLD timing done. STOP for confirmation before full run.")
        return

    # ── full stage ───────────────────────────────────────────────────────────
    # Resolve patients per cell
    plan = []
    for model, quant, n in TARGETS:
        pts = pick_patients(df, model, quant, n)
        for pid in pts:
            plan.append((model, quant, pid))
        print(f"  {model} {quant}: {n} patient(s) -> {pts}")
    print()

    rows = []

    # PASS 1 — fp32 model: GOLD (190) for all; fp32+60 diagnostic for Qwen
    nli32 = NLIScorer()   # fp32
    for model, quant, pid in plan:
        txt = seed_text_map(df, model, quant, pid)
        t0 = time.perf_counter(); g = score_patient(nli32, txt, ALL_190); el = time.perf_counter() - t0
        rows.append(dict(input_id=pid, model=model, quant_level=quant, condition="GOLD",
                         precision="fp32", n_pairs=g["n_pairs"], mean_mean=g["mean_mean"],
                         q90_mean=g["q90_mean"], max_mean=g["max_mean"], elapsed_seconds=round(el, 2)))
        print(f"  [GOLD fp32 190] {model} {quant} {pid}: {el/60:.2f} min  (np={g['n_pairs']})")
        if model == "Qwen-MediCare-BD":
            t0 = time.perf_counter(); d = score_patient(nli32, txt, SUBSAMPLE_60); el = time.perf_counter() - t0
            rows.append(dict(input_id=pid, model=model, quant_level=quant, condition="DIAG_fp32_60",
                             precision="fp32", n_pairs=d["n_pairs"], mean_mean=d["mean_mean"],
                             q90_mean=d["q90_mean"], max_mean=d["max_mean"], elapsed_seconds=round(el, 2)))
            print(f"  [DIAG fp32 60 ] {model} {quant} {pid}: {el/60:.2f} min  (np={d['n_pairs']})")

    # PASS 2 — fp16 model: PROD (60) for all
    print("\n  Converting NLI model to fp16 (.half()) ...")
    nli16 = NLIScorer()
    nli16.mdl = nli16.mdl.half()
    for model, quant, pid in plan:
        txt = seed_text_map(df, model, quant, pid)
        t0 = time.perf_counter(); p = score_patient(nli16, txt, SUBSAMPLE_60); el = time.perf_counter() - t0
        rows.append(dict(input_id=pid, model=model, quant_level=quant, condition="PROD",
                         precision="fp16", n_pairs=p["n_pairs"], mean_mean=p["mean_mean"],
                         q90_mean=p["q90_mean"], max_mean=p["max_mean"], elapsed_seconds=round(el, 2)))
        print(f"  [PROD fp16 60 ] {model} {quant} {pid}: {el/60:.2f} min  (np={p['n_pairs']})")

    # write results.csv
    cols = ["input_id", "model", "quant_level", "condition", "precision",
            "n_pairs", "mean_mean", "q90_mean", "max_mean", "elapsed_seconds"]
    with open(RESULTS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nWrote {RESULTS}")

    # ── per-patient |PROD - GOLD| diffs ───────────────────────────────────────
    by = {(r["input_id"], r["condition"]): r for r in rows}
    print("\n" + "=" * 80)
    print("PER-PATIENT |PROD(fp16,60) - GOLD(fp32,190)|")
    print("=" * 80)
    print(f"{'patient':<12}{'model':<22}{'d_mean_mean':>13}{'d_q90_mean':>12}{'d_max_mean':>12}")
    per_model_diffs = {}
    for model, quant, pid in plan:
        g = by.get((pid, "GOLD")); p = by.get((pid, "PROD"))
        if g and p:
            dm = abs(p["mean_mean"] - g["mean_mean"])
            dq = abs(p["q90_mean"] - g["q90_mean"])
            dx = abs(p["max_mean"] - g["max_mean"])
            print(f"{pid:<12}{model:<22}{dm:>13.6f}{dq:>12.6f}{dx:>12.6f}")
            per_model_diffs.setdefault(model, []).append((dm, dq, dx))

    print("\n" + "-" * 80)
    print("PER-MODEL MEAN |PROD - GOLD|")
    print("-" * 80)
    print(f"{'model':<22}{'mean d_mean_mean':>18}{'mean d_q90_mean':>17}{'mean d_max_mean':>17}")
    for model, diffs in per_model_diffs.items():
        arr = np.array(diffs)
        print(f"{model:<22}{arr[:,0].mean():>18.6f}{arr[:,1].mean():>17.6f}{arr[:,2].mean():>17.6f}")
    alld = np.array([d for ds in per_model_diffs.values() for d in ds])
    print(f"{'ALL (9 patients)':<22}{alld[:,0].mean():>18.6f}{alld[:,1].mean():>17.6f}{alld[:,2].mean():>17.6f}")

    # ── timing extrapolation ──────────────────────────────────────────────────
    def mean_elapsed(cond, model=None):
        vals = [r["elapsed_seconds"] for r in rows
                if r["condition"] == cond and (model is None or r["model"] == model)]
        return float(np.mean(vals)) if vals else None

    FULL_CELLS = 1200
    print("\n" + "=" * 80)
    print("TIMING EXTRAPOLATION TO FULL STUDY (1,200 patient-cells)")
    print("=" * 80)
    print(f"{'recipe':<22}{'mean s/patient-cell':>22}{'x1200 = full hours':>22}")
    for label, cond in [("fp32+190 (GOLD)", "GOLD"), ("fp32+60 (diag)", "DIAG_fp32_60"),
                        ("fp16+60 (PROD)", "PROD")]:
        m = mean_elapsed(cond)
        if m is None:
            print(f"{label:<22}{'n/a':>22}{'n/a':>22}")
        else:
            print(f"{label:<22}{m:>22.1f}{m*FULL_CELLS/3600:>22.1f}")
    print(f"{'fp16+190':<22}{'n/a (not measured)':>22}{'':>22}")

    print("\nPer-MODEL s/patient-cell (each model = 400 patient-cells in full study):")
    print(f"{'model':<22}{'GOLD fp32+190':>16}{'PROD fp16+60':>16}")
    for model, quant, n in TARGETS:
        g = mean_elapsed("GOLD", model); p = mean_elapsed("PROD", model)
        print(f"{model:<22}{(g or 0):>16.1f}{(p or 0):>16.1f}")

    print("\nProjected FULL PROD (fp16+60) run time, three ways:")
    qwen_p = mean_elapsed("PROD", "Qwen-MediCare-BD")
    phi_p  = mean_elapsed("PROD", "Phi-4-mini-instruct")
    gem_p  = mean_elapsed("PROD", "Gemma-3-4B-it")
    if qwen_p:  print(f"  using Qwen Q3 rate   : {qwen_p*FULL_CELLS/3600:.1f} h  ({qwen_p:.1f} s/cell)")
    if phi_p:   print(f"  using Phi-4 Q4 rate  : {phi_p*FULL_CELLS/3600:.1f} h  ({phi_p:.1f} s/cell)  [slowest]")
    rates = [r for r in (qwen_p, gem_p, phi_p) if r]
    if rates:
        blended = float(np.mean(rates))
        print(f"  blended (3-cell mean): {blended*FULL_CELLS/3600:.1f} h  ({blended:.1f} s/cell)")

    print("\nSensitivity-analysis cost (NLI re-run on subsets):")
    print("  The clean-subset sensitivity (is_anomalous=F AND is_looped=F) re-runs NLI")
    print("  on the surviving pairs only. Clean-row fraction ~70% of pairs => roughly")
    print("  ~0.5-0.7x of a full PROD run per sensitivity stream (fewer valid pairs per")
    print("  cell). Gated vs ungated stat sensitivity adds NO NLI cost (stats only).")
    print("\n>>> FULL PROOF done. STOP. No pipeline modified, nothing committed.")


if __name__ == "__main__":
    main()
