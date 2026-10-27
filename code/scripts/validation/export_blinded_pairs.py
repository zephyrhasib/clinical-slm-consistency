"""
Blinded NLI panel CSV exporter.

Strips identifying metadata from the sampled-pairs CSV so each judge sees
ONLY (pair_id, text_a, text_b). This prevents the judge from anchoring on
model name, quantization level, or the production NLI score.

The full-metadata CSV produced by `sample_nli_validation_pairs.py` stays
around as the audit trail — it is used later by `aggregate_panel_scores.py`
to join panel consensus back with the production NLI aggregations.

Output schema (CSV, 3 columns only):
  pair_id, text_a, text_b

Run in venv_metrics (or any env with pandas):
  python scripts/validation/export_blinded_pairs.py \\
      --in  outputs/pilot_eval/nli_validation_pairs.csv \\
      --out outputs/pilot_eval/nli_validation_pairs_blinded.csv
"""
import argparse
from pathlib import Path

import pandas as pd


KEEP_COLUMNS = ["pair_id", "text_a", "text_b"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",  dest="in_path",  required=True,
                        help="sampled-pairs CSV from sample_nli_validation_pairs.py")
    parser.add_argument("--out", required=True,
                        help="blinded judge-ready CSV (pair_id, text_a, text_b)")
    args = parser.parse_args()

    if not Path(args.in_path).exists():
        parser.error(f"--in not found: {args.in_path}")

    df = pd.read_csv(args.in_path)
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        parser.error(f"input is missing required columns: {missing}")

    df_blind = df[KEEP_COLUMNS].copy()
    df_blind.to_csv(args.out, index=False)

    print("=" * 72)
    print("BLINDED EXPORT")
    print("=" * 72)
    print(f"  input            : {args.in_path}  ({len(df)} rows, "
          f"{len(df.columns)} columns)")
    print(f"  output           : {args.out}  ({len(df_blind)} rows, "
          f"{len(df_blind.columns)} columns)")
    print(f"  dropped columns  : "
          f"{[c for c in df.columns if c not in KEEP_COLUMNS]}")
    print(f"  kept columns     : {KEEP_COLUMNS}")
    print("\nThis file is what you share with each judge.")
    print("Keep the un-blinded file as the audit trail — "
          "`aggregate_panel_scores.py` needs it to join back to model/quant/cell.")


if __name__ == "__main__":
    main()
