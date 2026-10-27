"""
BERTScore face-validity panel aggregator (Issue 2).

Reads LLM-panel category votes from `bertscore_panel_categories.csv` and
produces:
  1. Per-pair majority-vote category (ties broken alphabetically: A > B > C > D).
  2. Distribution of pairs across the 4 categories (n + fraction).
  3. Inter-judge reliability via Krippendorff's α (nominal level — the
     4 categories are unordered).
  4. A summary sentence ready to paste into a methods section.

Expected panel CSV schema (manually populated after panel scoring):
  pair_id, judge_id, category
    - pair_id : BVAL-001 … BVAL-015
    - judge_id: any string/int identifier (e.g. "claude", "gemini", "deepseek")
    - category: A (content disagreement) / B (format variation) /
                C (degeneracy) / D (mixed)

Run in venv_metrics:
  pip install krippendorff
  python scripts/validation/aggregate_bertscore_categories.py \\
      --panel-categories outputs/pilot_eval/bertscore_panel_categories.csv \\
      --sampled-pairs    outputs/pilot_eval/bertscore_inspection_pairs.csv \\
      --out              outputs/pilot_eval/bertscore_validation_report.csv
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    import krippendorff
    HAVE_KRIPP = True
except ImportError:
    krippendorff = None
    HAVE_KRIPP = False


CATEGORIES = {
    "A": "content_disagreement",
    "B": "format_variation",
    "C": "degeneracy",
    "D": "mixed",
}
# Integer encoding for Krippendorff's α (nominal level)
CAT_TO_INT = {"A": 0, "B": 1, "C": 2, "D": 3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-categories", required=True,
                        help="CSV: pair_id, judge_id, category")
    parser.add_argument("--sampled-pairs", required=True,
                        help="CSV from sample_bertscore_inspection_pairs.py")
    parser.add_argument("--out", required=True,
                        help="output validation report CSV")
    args = parser.parse_args()

    for arg, path in [("--panel-categories", args.panel_categories),
                       ("--sampled-pairs",   args.sampled_pairs)]:
        if not Path(path).exists():
            parser.error(f"{arg} not found: {path}")

    df_cats  = pd.read_csv(args.panel_categories)
    df_pairs = pd.read_csv(args.sampled_pairs)

    # Validate category values
    bad = set(df_cats["category"].unique()) - set(CATEGORIES.keys())
    if bad:
        parser.error(f"Unknown category values in panel CSV: {bad}. "
                     f"Allowed: {sorted(CATEGORIES.keys())}")

    print("=" * 72)
    print("BERTSCORE PANEL AGGREGATOR (Issue 2)")
    print("=" * 72)
    print(f"  panel rows    : {len(df_cats)}")
    print(f"  sampled pairs : {len(df_pairs)}")
    print(f"  pairs scored  : {df_cats['pair_id'].nunique()}")
    print(f"  judges        : {df_cats['judge_id'].nunique()}")
    print(f"  krippendorff  : {'installed' if HAVE_KRIPP else 'NOT installed (α skipped)'}")

    # ── 1. Per-pair majority vote ────────────────────────────────────────────
    consensus_rows = []
    for pair_id, grp in df_cats.groupby("pair_id"):
        votes      = grp["category"].tolist()
        counter    = Counter(votes)
        max_count  = max(counter.values())
        # Tie-break: alphabetical preference (A wins over B over C over D)
        candidates = sorted([c for c, n in counter.items() if n == max_count])
        consensus  = candidates[0]
        consensus_rows.append({
            "pair_id":   pair_id,
            "consensus": consensus,
            "votes":     ",".join(votes),
            "n_judges":  len(votes),
        })
    consensus_df = pd.DataFrame(consensus_rows)
    n_total      = len(consensus_df)
    print(f"\n[1/4] Consensus computed for {n_total} pairs "
          f"(majority vote, alphabetical tie-break).")

    # ── 2. Distribution across 4 categories ──────────────────────────────────
    dist = consensus_df["consensus"].value_counts().to_dict()
    print(f"\n[2/4] Consensus category distribution (n={n_total}):")
    for cat_letter, cat_name in CATEGORIES.items():
        count = dist.get(cat_letter, 0)
        frac  = count / n_total if n_total > 0 else 0.0
        print(f"  {cat_letter} ({cat_name:<22}): {count:>2} ({frac:.1%})")

    # ── 3. Krippendorff α (nominal) ──────────────────────────────────────────
    if HAVE_KRIPP:
        judges    = sorted(df_cats["judge_id"].unique())
        pair_ids  = sorted(df_cats["pair_id"].unique())
        matrix = []
        for judge in judges:
            row = []
            jdf = df_cats[df_cats["judge_id"] == judge].set_index("pair_id")
            for pid in pair_ids:
                if pid in jdf.index:
                    row.append(CAT_TO_INT[jdf.loc[pid, "category"]])
                else:
                    row.append(None)
            matrix.append(row)
        alpha = float(krippendorff.alpha(
            reliability_data=matrix,
            level_of_measurement="nominal",
        ))
    else:
        alpha = float("nan")
    print(f"\n[3/4] Inter-judge Krippendorff α (nominal) = {alpha:.4f}")

    # ── 4. Report CSV ────────────────────────────────────────────────────────
    report_rows = []
    for cat_letter, cat_name in CATEGORIES.items():
        count = dist.get(cat_letter, 0)
        report_rows.append({
            "category_letter":    cat_letter,
            "category_name":      cat_name,
            "n_pairs":            count,
            "fraction":           count / n_total if n_total > 0 else 0.0,
            "n_total":            n_total,
            "krippendorff_alpha": alpha,
        })
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(args.out, index=False)
    print(f"\n[4/4] Wrote report → {args.out}")

    # ── Summary sentence ────────────────────────────────────────────
    n_content = dist.get("A", 0)
    n_format  = dist.get("B", 0)
    n_degen   = dist.get("C", 0)
    n_mixed   = dist.get("D", 0)
    alpha_str = f"{alpha:.3f}" if HAVE_KRIPP else "[run with krippendorff installed]"

    print("\n" + "=" * 72)
    print("Summary sentence (for a methods section):")
    print("=" * 72)
    print(f"""
  We assessed BERTScore F1 face validity by manual inspection of {n_total}
  high-degradation generation pairs sampled from the 5 cells with lowest
  Q3_K_M BERTScore F1 (balanced across models, max 2 per model). A 3-judge
  LLM panel categorized each pair as content disagreement, format variation,
  degeneracy, or mixed. Inter-judge Krippendorff α (nominal) = {alpha_str}.
  By majority vote: {n_content}/{n_total} pairs were classified as content
  disagreement, {n_format}/{n_total} as format variation, {n_degen}/{n_total} as
  degeneracy, {n_mixed}/{n_total} as mixed.
""")


if __name__ == "__main__":
    main()
