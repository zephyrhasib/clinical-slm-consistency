"""
BERTScore face-validity sample selector (Issue 2).

Samples 15 generation pairs for human/LLM-panel face-validity inspection of
BERTScore F1 degradation at low quantization. The panel categorizes each
pair as content disagreement / format variation / degeneracy / mixed,
empirically validating that BERTScore drops at Q3 reflect clinical content
inconsistency rather than format artifacts or text-generation pathologies.

Sampling design (locked, deterministic):
  - Identify the 5 (model, input_id) cells with the LOWEST
    `bertscore_f1_mean` at Q3_K_M (where BERTScore declares "high
    inconsistency").
  - Stratify: no model contributes more than 2 cells.
  - From each selected cell, sample 3 random seed pairs from C(k, 2)
    candidates (k=4 seeds → 6 pairs).
  - Total: 5 cells × 3 pairs = 15 pairs.
  - All RNG flows from a single seed=42 master via
    np.random.default_rng(42), so the entire process is reproducible.

Outputs:
  --un-blinded  outputs/pilot_eval/bertscore_inspection_pairs.csv
                cols: pair_id, model, input_id, seed_a, seed_b,
                      cell_bertscore_mean, text_a, text_b
                (audit trail; do NOT share with judges)
  --blinded     outputs/pilot_eval/bertscore_inspection_pairs_blinded.csv
                cols: pair_id, text_a, text_b
                (judge-ready)

Run in venv_metrics:
  python scripts/validation/sample_bertscore_inspection_pairs.py \\
      --metric-results outputs/pilot_eval/metric_results.csv \\
      --generations    outputs/pilot_eval/pilot_combined.csv \\
      --un-blinded     outputs/pilot_eval/bertscore_inspection_pairs.csv \\
      --blinded        outputs/pilot_eval/bertscore_inspection_pairs_blinded.csv
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


SEED                 = 42
N_CELLS              = 5
MAX_CELLS_PER_MODEL  = 2
PAIR_PREFIX          = "BVAL"

# Pre-specified scenario → pairs per cell (12 cells × N pairs = total panel size)
SCENARIO_N_PER_CELL = {
    "pilot": 3,   # 5 cells × 3 pairs = 15 total
    "main":  5,   # 5 cells × 5 pairs = 25 total
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(SCENARIO_N_PER_CELL),
                        default="pilot",
                        help="pilot = 3/cell (15 total); main = 5/cell (25 total)")
    parser.add_argument("--metric-results", required=True,
                        help="metric_results.csv (per-cell metric CSV)")
    parser.add_argument("--generations", required=True,
                        help="long-format generations CSV (pilot or main)")
    parser.add_argument("--un-blinded", required=True,
                        help="output CSV with full metadata (audit trail)")
    parser.add_argument("--blinded", required=True,
                        help="output CSV with only pair_id/text_a/text_b (judge-ready)")
    args = parser.parse_args()

    n_pairs_per_cell = SCENARIO_N_PER_CELL[args.scenario]

    # Validate inputs
    for arg, path in [("--metric-results", args.metric_results),
                       ("--generations",   args.generations)]:
        if not Path(path).exists():
            parser.error(f"{arg} not found: {path}")

    df_metrics = pd.read_csv(args.metric_results)
    df_gen     = pd.read_csv(args.generations)

    if "bertscore_f1_mean" not in df_metrics.columns:
        parser.error("--metric-results is missing 'bertscore_f1_mean' column.")

    rng = np.random.default_rng(SEED)

    print("=" * 72)
    print("BERTSCORE INSPECTION SAMPLER (Issue 2)")
    print("=" * 72)
    print(f"  scenario                 : {args.scenario}")
    print(f"  master seed              : {SEED}")
    print(f"  target cells             : {N_CELLS}")
    print(f"  pairs per cell           : {n_pairs_per_cell}")
    print(f"  max cells per model      : {MAX_CELLS_PER_MODEL}")
    print(f"  expected total           : {N_CELLS} × {n_pairs_per_cell} = {N_CELLS * n_pairs_per_cell}")
    print(f"  selection criterion      : lowest bertscore_f1_mean at Q3_K_M")
    print(f"  metric rows              : {len(df_metrics)}")
    print(f"  generations rows         : {len(df_gen)}")

    # ── Step 1: find the lowest-BERTScore Q3 cells, stratified by model ──
    q3_cells = df_metrics[df_metrics["quant_level"] == "Q3_K_M"].copy()
    q3_cells = q3_cells.dropna(subset=["bertscore_f1_mean"])
    q3_cells = q3_cells.sort_values("bertscore_f1_mean", ascending=True)

    selected      = []
    model_counts  = {}
    for _, row in q3_cells.iterrows():
        model = row["model"]
        if model_counts.get(model, 0) >= MAX_CELLS_PER_MODEL:
            continue
        selected.append({
            "model":             model,
            "input_id":          row["input_id"],
            "bertscore_f1_mean": float(row["bertscore_f1_mean"]),
        })
        model_counts[model] = model_counts.get(model, 0) + 1
        if len(selected) >= N_CELLS:
            break

    print(f"\nSelected {len(selected)} cells (lowest Q3_K_M BERTScore, max "
          f"{MAX_CELLS_PER_MODEL}/model):")
    for s in selected:
        print(f"  {s['model']:<22} {s['input_id']:<12} "
              f"BERTScore={s['bertscore_f1_mean']:.4f}")

    if len(selected) < N_CELLS:
        print(f"  WARN: only {len(selected)} cells found "
              f"(requested {N_CELLS}) — stratification constraint may be too tight.")

    # ── Step 2: sample 3 random seed pairs per cell ──
    out_rows = []
    pair_id_counter = 1

    for cell in selected:
        cell_gens = df_gen[
            (df_gen["model"]       == cell["model"]) &
            (df_gen["input_id"]    == cell["input_id"]) &
            (df_gen["quant_level"] == "Q3_K_M")
        ]
        seeds     = sorted(cell_gens["seed"].tolist())
        all_pairs = list(combinations(seeds, 2))

        if len(all_pairs) == 0:
            print(f"  WARN: 0 seeds for {cell['model']} {cell['input_id']} — skipping.")
            continue

        if len(all_pairs) <= n_pairs_per_cell:
            sampled_pairs = all_pairs
            if len(all_pairs) < n_pairs_per_cell:
                print(f"  WARN: only {len(all_pairs)} pairs available "
                      f"for {cell['model']} {cell['input_id']} "
                      f"(requested {n_pairs_per_cell}); taking all.")
        else:
            idx = rng.choice(len(all_pairs), size=n_pairs_per_cell, replace=False)
            sampled_pairs = [all_pairs[i] for i in idx]

        for a, b in sampled_pairs:
            text_a = cell_gens[cell_gens["seed"] == a]["generated_text"].iloc[0]
            text_b = cell_gens[cell_gens["seed"] == b]["generated_text"].iloc[0]
            out_rows.append({
                "pair_id":             f"{PAIR_PREFIX}-{pair_id_counter:03d}",
                "model":               cell["model"],
                "input_id":            cell["input_id"],
                "seed_a":              int(a),
                "seed_b":              int(b),
                "cell_bertscore_mean": cell["bertscore_f1_mean"],
                "text_a":              text_a,
                "text_b":              text_b,
            })
            pair_id_counter += 1

    out_df = pd.DataFrame(out_rows)

    # Un-blinded (full audit trail)
    out_df.to_csv(args.un_blinded, index=False)
    print(f"\nWrote {len(out_df)} pairs → {args.un_blinded}  (full metadata)")

    # Blinded (for judges)
    blinded = out_df[["pair_id", "text_a", "text_b"]].copy()
    blinded.to_csv(args.blinded, index=False)
    print(f"Wrote blinded version  → {args.blinded}  (pair_id, text_a, text_b only)")

    print("\nBreakdown by model:")
    print(out_df.groupby("model").size().rename("n_pairs").to_string())


if __name__ == "__main__":
    main()
