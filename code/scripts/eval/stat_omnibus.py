"""
Statistical Omnibus Pipeline.

Friedman test + Kendall's W across all 4 quant levels, per (model, metric).
Runs FIRST in the statistical sequence. Pairwise comparisons (stat_pairwise.py)
are gated on Friedman significance per the audit's standard sequential testing
protocol.

Inputs:
  --pairwise-metrics : per-cell pairwise metric CSV
                       cols: input_id, model, quant_level, n_pairs,
                             bertscore_f1_mean, rougeL_f1_mean, medcon_f1_mean
  --nli-metrics      : per-cell pairwise NLI CSV
                       cols: input_id, model, quant_level, n_pairs,
                             nli_contradiction_mean

Output:
  --out              : stat_omnibus.csv
                       cols: model, metric, n_patients, k_conditions,
                             friedman_chi2, friedman_p,
                             kendall_w, friedman_significant_05

Run in venv_metrics:
  python scripts/stat_omnibus.py \\
      --pairwise-metrics outputs/pilot_eval/metric_results.csv \\
      --nli-metrics      outputs/pilot_eval/metric_results_nli.csv \\
      --out              outputs/pilot_eval/stat_omnibus.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# Locked quant ordering (highest precision → lowest)
QUANT_ORDER = ["Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M"]

ALPHA = 0.05


# Pre-specified metric direction (True = higher is better / more consistent)
METRIC_DIRECTION = {
    "bertscore_f1_mean":           True,
    "rougeL_f1_mean":              True,
    "medcon_f1_mean":              True,
    "numeric_jaccard_mean":        True,   # higher = more consistent (Issue 3)
    "nli_contradiction_max_mean":  False,
    "nli_contradiction_mean_mean": False,
    "nli_contradiction_q90_mean":  False,
}


def kendalls_w(chi2: float, n: int, k: int) -> float:
    """Kendall's W coefficient of concordance from Friedman chi-squared.

    W = chi2 / (n * (k - 1))

    Where:
      n = number of patients (subjects)
      k = number of conditions (quant levels)

    Range: 0 (no agreement) to 1 (perfect agreement).
    """
    if n == 0 or k <= 1:
        return float("nan")
    return chi2 / (n * (k - 1))


def run_friedman_for_metric(df_long: pd.DataFrame, metric: str) -> list[dict]:
    """For each model, run Friedman across all available quant levels."""
    out = []
    for model, sub in df_long.groupby("model"):
        # Pivot: rows = patients, cols = quant levels
        wide = sub.pivot_table(index="input_id", columns="quant_level", values="value")
        # Keep only quant levels actually present, in pre-specified order
        quants_present = [q for q in QUANT_ORDER if q in wide.columns]
        if len(quants_present) < 2:
            continue
        wide = wide[quants_present].dropna(how="any")
        n = len(wide)
        k = len(quants_present)
        if n < 3:
            out.append({
                "model":                model,
                "metric":               metric,
                "n_patients":           int(n),
                "k_conditions":         int(k),
                "friedman_chi2":        float("nan"),
                "friedman_p":           float("nan"),
                "kendall_w":            float("nan"),
                "friedman_significant_05": False,
                "note":                 "n<3 — Friedman not computed",
            })
            continue
        # Pass each quant column as a separate sample
        columns_data = [wide[q].to_numpy() for q in quants_present]
        chi2, p = stats.friedmanchisquare(*columns_data)
        w = kendalls_w(chi2, n, k)
        out.append({
            "model":                model,
            "metric":               metric,
            "n_patients":           int(n),
            "k_conditions":         int(k),
            "friedman_chi2":        float(chi2),
            "friedman_p":           float(p),
            "kendall_w":            float(w),
            "friedman_significant_05": bool(p < ALPHA),
            "note":                 "",
        })
    return out


def aggregate_metrics(df_pairwise: pd.DataFrame, df_nli: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the per-cell metric CSVs into long format with one row per
    (input_id, model, quant_level, metric, value).
    """
    pieces = []
    for metric in ["bertscore_f1_mean", "rougeL_f1_mean", "medcon_f1_mean",
                   "numeric_jaccard_mean"]:
        if metric in df_pairwise.columns:
            p = df_pairwise[["input_id", "model", "quant_level", metric]].copy()
            p.columns = ["input_id", "model", "quant_level", "value"]
            p["metric"] = metric
            pieces.append(p)
    for nli_col in ("nli_contradiction_max_mean",
                    "nli_contradiction_mean_mean",
                    "nli_contradiction_q90_mean"):
        if nli_col in df_nli.columns:
            n = df_nli[["input_id", "model", "quant_level", nli_col]].copy()
            n.columns = ["input_id", "model", "quant_level", "value"]
            n["metric"] = nli_col
            pieces.append(n)
    return pd.concat(pieces, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise-metrics", required=True)
    parser.add_argument("--nli-metrics",      required=True)
    parser.add_argument("--out",              required=True)
    args = parser.parse_args()

    # Issue 11 — validate inputs early
    for arg, path in [("--pairwise-metrics", args.pairwise_metrics),
                       ("--nli-metrics",      args.nli_metrics)]:
        if not Path(path).exists():
            parser.error(f"{arg} not found: {path}")

    df_pw  = pd.read_csv(args.pairwise_metrics)
    df_nli = pd.read_csv(args.nli_metrics)

    print("=" * 72)
    print("STATISTICAL OMNIBUS — Friedman + Kendall's W")
    print("=" * 72)
    print(f"  pairwise rows : {len(df_pw)}")
    print(f"  nli rows      : {len(df_nli)}")
    print(f"  quant order   : {QUANT_ORDER}")
    print(f"  alpha         : {ALPHA}")

    # Long format with all metrics
    df_long = aggregate_metrics(df_pw, df_nli)

    # Run Friedman per (model, metric)
    rows = []
    for metric in METRIC_DIRECTION:
        sub = df_long[df_long["metric"] == metric]
        if len(sub) == 0:
            continue
        rows.extend(run_friedman_for_metric(sub, metric))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    # Pretty print
    print("\nResults:")
    print(out_df.round(4).to_string(index=False))

    n_sig   = int(out_df["friedman_significant_05"].sum())
    n_total = len(out_df)
    print(f"\nFriedman-significant (model, metric) combos: {n_sig}/{n_total}")
    print(f"\nSaved: {args.out}")
    print("\nNext step: stat_pairwise.py — gated on Friedman significance.")


if __name__ == "__main__":
    main()
