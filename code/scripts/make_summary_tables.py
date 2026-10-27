"""
Write the two summary table CSVs from the frozen evaluation outputs (read-only on them).

  outputs/table3_per_cell.csv  -> per-condition means table
      per-condition means: mean over the 100 encounter cells of each (model, level)
      of the five primary metrics, rounded to 3 decimals.
  outputs/table4_effects.csv   -> Q6_K -> Q3_K_M effect-size table
      Q6_K -> Q3_K_M oriented rank-biserial effect (positive = more consistent at Q6_K):
      effect_r_rb as stored for higher-is-better metrics, sign-flipped for NLI contradiction;
      "✓" if Holm-significant (significant_05), else "n.s.".

Inputs : outputs/main_eval/metric_results.csv, metric_results_nli.csv, stat_pairs.csv
Run    : venv_metrics/bin/python scripts/make_summary_tables.py [--out-dir outputs]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs/main_eval"
MODELS = ["Qwen-MediCare-BD", "Gemma-3-4B-it", "Phi-4-mini-instruct"]
SHORT = {"Qwen-MediCare-BD": "Qwen", "Gemma-3-4B-it": "Gemma", "Phi-4-mini-instruct": "Phi-4"}
LEVELS = ["Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M"]
METRICS = [("bertscore_f1_mean", "BERTScore"), ("rougeL_f1_mean", "ROUGE-L"), ("medcon_f1_mean", "Concept-set F1"),
           ("numeric_jaccard_mean", "Numeric Jaccard"), ("nli_contradiction_mean_mean", "NLI")]
EFFECT_LABEL = {"nli_contradiction_mean_mean": "NLI contradiction"}


def table3():
    mr = pd.read_csv(SRC / "metric_results.csv")
    nli = pd.read_csv(SRC / "metric_results_nli.csv")
    df = mr.merge(nli, on=["input_id", "model", "quant_level"])
    cols = [m for m, _ in METRICS]
    means = df.groupby(["model", "quant_level"])[cols].mean().round(3)
    rows = []
    for m in MODELS:
        for q in LEVELS:
            r = {"Model": m, "Quant": q}
            for c, lab in METRICS:
                r[lab] = means.loc[(m, q), c]
            rows.append(r)
    return pd.DataFrame(rows)


def table4():
    sp = pd.read_csv(SRC / "stat_pairs.csv")
    q = sp[(sp.quant_a == "Q6_K") & (sp.quant_b == "Q3_K_M")].copy()
    q["oriented"] = np.where(q.higher_is_better, q.effect_r_rb, -q.effect_r_rb)
    rows = []
    for c, lab in METRICS:
        r = {"Metric": EFFECT_LABEL.get(c, lab)}
        for m in MODELS:
            x = q[(q.model == m) & (q.metric == c)].iloc[0]
            r[SHORT[m]] = f"{x.oriented:+.2f} " + ("✓" if bool(x.significant_05) else "n.s.")
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table3().to_csv(out / "table3_per_cell.csv", index=False)
    table4().to_csv(out / "table4_effects.csv", index=False)
    print(f"wrote {out / 'table3_per_cell.csv'} and {out / 'table4_effects.csv'}")
