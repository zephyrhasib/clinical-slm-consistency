"""
NLI panel-validation sample selector.

Pulls seed-pairs from the pilot/main dataset for human-panel contradiction
scoring. The panel scores feed `aggregate_panel_scores.py`, which
reports how each of three NLI aggregations (max_mean / mean_mean /
q90_mean) agrees with the panel.

Two scenarios are supported (pre-specified):
  --scenario pilot  →  3 pairs per (model, quant_level) cell  →  36 total
  --scenario main   →  5 pairs per (model, quant_level) cell  →  60 total
(default: main)

Sampling design (locked, deterministic):
  - Stratified across (model, quant_level): N pairs per condition.
  - UNIFORM random sampling within each condition (no bias toward
    high-disagreement cells). Bias was used in an earlier revision but
    removed to give the panel an unbiased view of the metric's behavior
    across the full disagreement spectrum.
  - Inside each chosen cell, one random seed-pair is drawn from C(k, 2)
    candidates.
  - All RNG goes through a single seed=42 master via
    `np.random.default_rng(42)`, so the entire process is reproducible.

Output (CSV):
  pair_id, input_id, model, quant_level, seed_a, seed_b,
  text_a, text_b, cell_nli_max_mean

  (Use `export_blinded_pairs.py` afterwards to strip metadata columns
  before sharing the file with judges.)

Run in venv_metrics (or any env with numpy+pandas):
  # Pilot panel (36 pairs)
  python scripts/validation/sample_nli_validation_pairs.py \\
      --scenario    pilot \\
      --nli-metrics outputs/pilot_eval/metric_results_nli.csv \\
      --generations outputs/pilot_eval/pilot_combined.csv \\
      --out         outputs/pilot_eval/nli_validation_pairs.csv

  # Main panel (60 pairs)
  python scripts/validation/sample_nli_validation_pairs.py \\
      --scenario    main \\
      --nli-metrics outputs/main_eval/metric_results_nli.csv \\
      --generations outputs/main_eval/main_combined.csv \\
      --out         outputs/main_eval/nli_validation_pairs.csv
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42

# Pre-specified scenario → pairs per (model, quant_level) condition.
# 12 conditions × N = total panel size.
SCENARIO_N_PER_CELL = {
    "pilot": 3,   # pilot dataset, smaller panel
    "main":  5,   # main dataset, fuller panel
}

# The "production" NLI aggregation column we report alongside each pair
# for downstream auditing. NOT used as a sampling bias any more.
REFERENCE_COLUMN_NEW = "nli_contradiction_max_mean"
REFERENCE_COLUMN_OLD = "nli_contradiction_mean"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(SCENARIO_N_PER_CELL),
                        default="main",
                        help="pilot = 3/cell (36 total); main = 5/cell (60 total)")
    parser.add_argument("--nli-metrics", required=True,
                        help="metric_results_nli.csv (per-cell NLI aggregations)")
    parser.add_argument("--generations", required=True,
                        help="long-format generations CSV (pilot_combined / main_combined)")
    parser.add_argument("--out",         required=True,
                        help="output CSV of sampled pairs (with full metadata)")
    args = parser.parse_args()

    n_per_cell = SCENARIO_N_PER_CELL[args.scenario]

    # Validate input paths
    for arg, path in [("--nli-metrics", args.nli_metrics),
                       ("--generations", args.generations)]:
        if not Path(path).exists():
            parser.error(f"{arg} not found: {path}")

    df_nli = pd.read_csv(args.nli_metrics)
    df_gen = pd.read_csv(args.generations)

    # Reference column — try new schema first, fall back to legacy.
    # We only USE this as a value to carry through to the output (for
    # downstream auditing); it does NOT bias the sampling any more.
    if REFERENCE_COLUMN_NEW in df_nli.columns:
        ref_col = REFERENCE_COLUMN_NEW
    elif REFERENCE_COLUMN_OLD in df_nli.columns:
        ref_col = REFERENCE_COLUMN_OLD
        print(f"  NOTE: '{REFERENCE_COLUMN_NEW}' not present — using "
              f"legacy column '{REFERENCE_COLUMN_OLD}' (semantically equivalent).")
    else:
        parser.error(
            f"--nli-metrics has neither '{REFERENCE_COLUMN_NEW}' nor "
            f"'{REFERENCE_COLUMN_OLD}'. Re-run metric_pipeline.py first."
        )

    rng = np.random.default_rng(SEED)

    print("=" * 72)
    print("NLI VALIDATION SAMPLER")
    print("=" * 72)
    print(f"  scenario          : {args.scenario}")
    print(f"  pairs / condition : {n_per_cell}")
    print(f"  expected total    : {n_per_cell} × 12 = {n_per_cell * 12}")
    print(f"  master seed       : {SEED}")
    print(f"  sampling          : UNIFORM (no NLI-score bias)")
    print(f"  reference column  : {ref_col}  (carried through for audit only)")
    print(f"  NLI rows          : {len(df_nli)}")
    print(f"  generations rows  : {len(df_gen)}")

    sampled = []
    for (model, quant), sub in df_nli.groupby(["model", "quant_level"]):
        # Need at least one valid cell with non-NaN reference (proxy for
        # "the cell exists and had at least one valid pair").
        sub_valid = sub.dropna(subset=[ref_col])
        if sub_valid.empty:
            print(f"  WARN: {model} / {quant} has 0 non-NaN cells — skipping")
            continue
        if len(sub_valid) < n_per_cell:
            print(f"  WARN: {model} / {quant} has only {len(sub_valid)} cells "
                  f"(< {n_per_cell} requested); taking all.")

        # Deterministic child seed per condition
        cond_seed = int(rng.integers(0, 2 ** 31 - 1))
        chosen_cells = sub_valid.sample(
            n=min(n_per_cell, len(sub_valid)),
            random_state=cond_seed,
        )

        for _, cell in chosen_cells.iterrows():
            cell_gens = df_gen[
                (df_gen["input_id"]    == cell["input_id"]) &
                (df_gen["model"]       == model) &
                (df_gen["quant_level"] == quant)
            ]
            seeds = sorted(cell_gens["seed"].tolist())
            if len(seeds) < 2:
                continue
            all_pairs = list(combinations(seeds, 2))
            idx       = int(rng.integers(0, len(all_pairs)))
            seed_a, seed_b = all_pairs[idx]
            text_a = cell_gens[cell_gens["seed"] == seed_a]["generated_text"].iloc[0]
            text_b = cell_gens[cell_gens["seed"] == seed_b]["generated_text"].iloc[0]
            sampled.append({
                "input_id":          cell["input_id"],
                "model":             model,
                "quant_level":       quant,
                "seed_a":            int(seed_a),
                "seed_b":            int(seed_b),
                "text_a":            text_a,
                "text_b":            text_b,
                # Carry the production cell-level score for reference / audit
                "cell_nli_max_mean": float(cell[ref_col]),
            })

    df_sampled = pd.DataFrame(sampled).reset_index(drop=True)
    df_sampled.insert(0, "pair_id", [f"P{i + 1:02d}" for i in range(len(df_sampled))])
    df_sampled.to_csv(args.out, index=False)

    print(f"\nWrote {len(df_sampled)} pairs → {args.out}")
    print("\nBreakdown by (model, quant_level):")
    print(df_sampled.groupby(["model", "quant_level"]).size().rename("n").to_string())
    print("\nNext step: run scripts/validation/export_blinded_pairs.py to "
          "produce the judge-ready CSV (text_a/text_b only).")


if __name__ == "__main__":
    main()
