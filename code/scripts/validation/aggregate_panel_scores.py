"""
NLI panel-score aggregator (Track A convergent validity).

Reads human-judge scores from `nli_panel_scores.csv` and produces:
  1. Per-pair consensus = median across judges (robust to outliers).
  2. Inter-judge reliability = Krippendorff's α (ordinal level of measurement).
  3. PRIMARY analysis · Spearman ρ between per-pair consensus and each of
     the 3 cell-level NLI aggregations (`max_mean`, `mean_mean`, `q90_mean`),
     computed at PAIR LEVEL (n = 36 pilot, n = 60 main).
     Pairs are nested within (input_id, model, quant_level) cells;
     p-values are approximate under within-cell clustering.
  4. SENSITIVITY analysis · Same Spearman ρ computed at CELL LEVEL
     (n = 12). Pair-level consensus and pair-level NLI values are first
     averaged per (input_id, model, quant_level) cell. Reported as
     sensitivity check on aggregation selection only; significance test
     at n = 12 is underpowered by design and treated as descriptive.
  5. The PRIMARY flag marks the aggregation with the highest
     PAIR-LEVEL Spearman ρ (ties broken by lower p-value, then
     alphabetical). Cell-level ranking reported for transparency.

Expected panel-score CSV schema:
  pair_id, judge_id, score
    - pair_id  : matches IDs from sample_nli_validation_pairs.py
                 (P01..P36 pilot, P01..P60 main)
    - judge_id : string/int identifier
    - score    : ordinal contradiction score (1..5 Likert)

Panel size scenarios (locked by sample_nli_validation_pairs.py --scenario):
  pilot : 36 pairs (3 per cell × 12 cells)
  main  : 60 pairs (5 per cell × 12 cells)

Output CSV columns (one row per analysis × aggregation, 6 rows total):
  analysis_level         : "pair" or "cell"
  aggregation            : nli_contradiction_{max,mean,q90}_mean
  n                      : 36 or 60 (pair) ; 12 (cell)
  spearman_rho           : Spearman correlation coefficient
  spearman_p             : Spearman p-value (approximate; see notes above)
  krippendorff_alpha     : ordinal α (same for all rows)
  is_primary             : True for highest-ρ PAIR-LEVEL row only

Run in venv_metrics:
  python scripts/validation/aggregate_panel_scores.py \\
      --panel-scores   outputs/pilot_eval/nli_panel_scores.csv \\
      --sampled-pairs  outputs/pilot_eval/nli_validation_pairs.csv \\
      --nli-metrics    outputs/pilot_eval/metric_results_nli.csv \\
      --out            outputs/pilot_eval/nli_panel_validation_report.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import krippendorff
    HAVE_KRIPP = True
except ImportError:
    krippendorff = None
    HAVE_KRIPP = False


AGGREGATIONS = [
    "nli_contradiction_max_mean",
    "nli_contradiction_mean_mean",
    "nli_contradiction_q90_mean",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-scores",  required=True)
    parser.add_argument("--sampled-pairs", required=True)
    parser.add_argument("--nli-metrics",   required=True)
    parser.add_argument("--out",           required=True)
    args = parser.parse_args()

    for arg, path in [("--panel-scores", args.panel_scores),
                      ("--sampled-pairs", args.sampled_pairs),
                      ("--nli-metrics", args.nli_metrics)]:
        if not Path(path).exists():
            parser.error(f"{arg} not found: {path}")

    df_panel = pd.read_csv(args.panel_scores)
    df_pairs = pd.read_csv(args.sampled_pairs)
    df_nli   = pd.read_csv(args.nli_metrics)

    print("=" * 72)
    print("NLI PANEL-SCORE AGGREGATOR (pair-level primary + cell-level sensitivity)")
    print("=" * 72)
    print(f"  panel rows    : {len(df_panel)}")
    print(f"  sampled pairs : {len(df_pairs)}")
    print(f"  pairs scored  : {df_panel['pair_id'].nunique()}")
    print(f"  judges        : {df_panel['judge_id'].nunique()}")
    print(f"  krippendorff  : {'installed' if HAVE_KRIPP else 'NOT installed (alpha skipped)'}")

    # 1. Per-pair consensus = median across judges
    consensus = (df_panel.groupby("pair_id")["score"]
                          .median()
                          .rename("consensus")
                          .reset_index())
    print(f"\n[1/5] Pair-level consensus computed for {len(consensus)} pairs.")

    # 2. Krippendorff alpha (ordinal)
    if HAVE_KRIPP:
        wide = df_panel.pivot(index="judge_id", columns="pair_id", values="score")
        alpha = float(krippendorff.alpha(
            reliability_data=wide.values,
            level_of_measurement="ordinal",
        ))
    else:
        alpha = float("nan")
    print(f"[2/5] Krippendorff alpha (ordinal) = {alpha:.4f}")

    # 3. Join consensus + cell-level NLI scalars onto each pair
    pairs_x = df_pairs.merge(
        df_nli[["input_id", "model", "quant_level"] + AGGREGATIONS],
        on=["input_id", "model", "quant_level"], how="left",
    )
    pairs_x = pairs_x.merge(consensus, on="pair_id", how="left")
    print(f"[3/5] Joined for {len(pairs_x)} pair-level rows.")

    # 4. PAIR-LEVEL Spearman (primary)
    pair_rows = []
    for agg in AGGREGATIONS:
        sub = pairs_x.dropna(subset=["consensus", agg])
        if len(sub) < 3:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = spearmanr(sub["consensus"], sub[agg])
        pair_rows.append({
            "analysis_level":           "pair",
            "aggregation":              agg,
            "n":                        int(len(sub)),
            "spearman_rho":             float(rho),
            "spearman_p":               float(p),
            "krippendorff_alpha":       alpha,
            "is_primary": False,
        })
    print(f"[4/5] Pair-level Spearman computed (n = {pair_rows[0]['n']}).")

    # 5. CELL-LEVEL Spearman (sensitivity)
    # Aggregate pair consensus and NLI per cell, then correlate.
    cell_agg_cols = AGGREGATIONS + ["consensus"]
    cell_means = (pairs_x.groupby(["model", "quant_level"])[cell_agg_cols]
                          .mean()
                          .reset_index())
    cell_rows = []
    for agg in AGGREGATIONS:
        sub = cell_means.dropna(subset=["consensus", agg])
        if len(sub) < 3:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = spearmanr(sub["consensus"], sub[agg])
        cell_rows.append({
            "analysis_level":           "cell",
            "aggregation":              agg,
            "n":                        int(len(sub)),
            "spearman_rho":             float(rho),
            "spearman_p":               float(p),
            "krippendorff_alpha":       alpha,
            "is_primary": False,
        })
    print(f"[5/5] Cell-level Spearman computed (n = {cell_rows[0]['n']}).")

    # Combine and mark primary by PAIR-LEVEL ranking
    df_out = pd.DataFrame(pair_rows + cell_rows)

    pair_only = df_out[df_out["analysis_level"] == "pair"].copy()
    if pair_only["spearman_rho"].notna().any():
        # Tie-break: highest rho, then lowest p, then alphabetical aggregation name
        pair_only_sorted = pair_only.sort_values(
            by=["spearman_rho", "spearman_p", "aggregation"],
            ascending=[False, True, True],
        )
        primary_agg = pair_only_sorted.iloc[0]["aggregation"]
        df_out.loc[
            (df_out["analysis_level"] == "pair") &
            (df_out["aggregation"] == primary_agg),
            "is_primary",
        ] = True
    else:
        primary_agg = "<UNDEFINED — all pair-level correlations NaN>"

    df_out = df_out.sort_values(
        by=["analysis_level", "spearman_rho"],
        ascending=[True, False],
        na_position="last",
    )

    df_out.to_csv(args.out, index=False)
    print(f"\nWrote validation report → {args.out}")
    print("\nFull results:")
    print(df_out[["analysis_level", "aggregation", "n",
                  "spearman_rho", "spearman_p"]]
          .round(4)
          .to_string(index=False))
    print(f"\nPRIMARY (pair-level): {primary_agg}")


if __name__ == "__main__":
    main()
