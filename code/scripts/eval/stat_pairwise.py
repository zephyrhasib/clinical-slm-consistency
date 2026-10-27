"""
Statistical Pairwise Pipeline.

Pairwise Wilcoxon signed-rank (zero_method='pratt', directional one-sided)
+ BCa bootstrap with master-seed-derived child rngs
+ Holm-Bonferroni multiple-comparison correction
+ BCa fallback to percentile bootstrap when n < BCA_MIN_N
+ z0 ties-corrected per Efron & Tibshirani (1993)
+ ci_excludes_zero flag for CI-significance disagreement detection.

Pre-specified statistical methodology:
  - Wilcoxon zero_method = "pratt" (keeps zero-diff patients, drops their ranks)
  - alternative          = "auto" (directional, per METRIC_DIRECTION) by default
  - multiple comparisons = Holm-Bonferroni (uniformly more powerful than Bonferroni)
  - BCa bootstrap        = 10,000 iterations
  - master rng seed      = 42, child seeds derived deterministically per (model, metric, a, b)
  - BCa fallback         = percentile bootstrap when n < BCA_MIN_N=30
  - z0 ties correction   = strict-less + 0.5 × ties
  - omnibus gating       = pairs run only when Friedman p < 0.05 (unless --no-gate)

Inputs:
  --pairwise-metrics : per-cell pairwise metric CSV
  --nli-metrics      : per-cell pairwise NLI CSV
  --omnibus          : stat_omnibus.csv (Friedman results, for gating)

Outputs:
  --pairs            : stat_pairs.csv (one row per Friedman-significant pair)
  --summary          : stat_summary.csv (descriptive per-condition)

CLI:
  --alternative {auto, two-sided, greater, less}  — directional test (default auto)
  --no-gate                                       — bypass Friedman gating
"""
import argparse
import hashlib
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ─── Locked configuration ────────────────────────────────────────────────────
SEED        = 42
N_BOOTSTRAP = 10_000
BCA_MIN_N   = 30        # below this, fall back to percentile bootstrap
ALPHA       = 0.05

# Pre-specified quant ordering: highest precision first → lowest precision last.
# A pair (a, b) drawn from combinations(QUANT_ORDER, 2) always has a being
# higher-precision than b. The directional alternative builds on this invariant.
QUANT_ORDER = ["Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M"]

# Pre-specified metric direction
#   True  : higher value = more consistent (BERTScore, ROUGE-L, MEDCON F1)
#   False : lower value  = more consistent (NLI contradiction)
METRIC_DIRECTION = {
    "bertscore_f1_mean":           True,
    "rougeL_f1_mean":              True,
    "medcon_f1_mean":              True,
    "numeric_jaccard_mean":        True,   # higher = more consistent (Issue 3)
    "nli_contradiction_max_mean":  False,
    "nli_contradiction_mean_mean": False,
    "nli_contradiction_q90_mean":  False,
}


# ─── Deterministic per-pair seed derivation (fixes Issue 1) ──────────────────
def derive_pair_seed(master_seed: int, *parts) -> int:
    """SHA-256-based derivation of a 32-bit child seed from master + parts.
    Reproducible, independent per pair, no spawn-API requirements."""
    h = hashlib.sha256()
    h.update(str(master_seed).encode())
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode())
    return int.from_bytes(h.digest()[:4], "big")


# ─── BCa CI with z0 ties correction + small-n fallback ───────────────────────
def _bca_ci(theta_hat: float, theta_boot: np.ndarray, jackknife: np.ndarray,
            alpha: float = ALPHA) -> tuple[float, float]:
    """BCa CI per Efron & Tibshirani (1993) with the standard tie correction
    on the bias term z0 (Issue 3)."""
    n_boot = len(theta_boot)
    # Issue 3: tie correction — strict-less + 0.5 * equal
    z0_num   = float(np.sum(theta_boot < theta_hat) + 0.5 * np.sum(theta_boot == theta_hat))
    z0_frac  = z0_num / n_boot
    # Clamp to avoid infinite ppf at the extremes
    eps      = 1.0 / (2.0 * n_boot)
    z0_frac  = min(max(z0_frac, eps), 1.0 - eps)
    z0       = stats.norm.ppf(z0_frac)
    # Acceleration from jackknife
    jack_mean = jackknife.mean()
    num = float(np.sum((jack_mean - jackknife) ** 3))
    den = 6.0 * (float(np.sum((jack_mean - jackknife) ** 2)) ** 1.5)
    a   = num / den if den != 0 else 0.0
    # Adjusted percentiles
    z_lo, z_hi = stats.norm.ppf(alpha / 2.0), stats.norm.ppf(1.0 - alpha / 2.0)
    alpha_lo   = stats.norm.cdf(z0 + (z0 + z_lo) / (1.0 - a * (z0 + z_lo)))
    alpha_hi   = stats.norm.cdf(z0 + (z0 + z_hi) / (1.0 - a * (z0 + z_hi)))
    lo = float(np.percentile(theta_boot, 100.0 * alpha_lo))
    hi = float(np.percentile(theta_boot, 100.0 * alpha_hi))
    return lo, hi


def bootstrap_diff_median(x: np.ndarray, y: np.ndarray, rng_seed: int,
                          n_boot: int = N_BOOTSTRAP) -> tuple[float, float, float, str]:
    """Paired bootstrap of median(x - y). Returns (theta_hat, lo, hi, ci_method).

    Issue 2: when n < BCA_MIN_N, fall back to percentile bootstrap (BCa
    corrections require n ≥ ~30 to be reliable).
    """
    assert x.shape == y.shape
    rng = np.random.default_rng(rng_seed)
    n = len(x)
    diffs = x - y
    theta_hat = float(np.median(diffs))

    theta_boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        theta_boot[i] = np.median(diffs[idx])

    if n < BCA_MIN_N:
        # Issue 2 — percentile bootstrap fallback at small n
        lo = float(np.percentile(theta_boot, 100.0 * (ALPHA / 2.0)))
        hi = float(np.percentile(theta_boot, 100.0 * (1.0 - ALPHA / 2.0)))
        return theta_hat, lo, hi, "percentile"

    # n ≥ 30 → BCa
    jackknife = np.array([np.median(np.delete(diffs, i)) for i in range(n)])
    lo, hi = _bca_ci(theta_hat, theta_boot, jackknife, alpha=ALPHA)
    return theta_hat, lo, hi, "bca"


# ─── Wilcoxon signed-rank with pratt + directional alternative ───────────────
def wilcoxon_pair(x: np.ndarray, y: np.ndarray, alternative: str) -> dict:
    """Wilcoxon signed-rank with zero_method='pratt' (Issue 4).
    `alternative` ∈ {'two-sided', 'greater', 'less'}.
    Returns W, p, rank-biserial effect size, and n_nonzero."""
    if np.all(x == y):
        return {"W": float("nan"), "p_value": 1.0, "r_rb": 0.0, "n_nonzero": 0}
    res = stats.wilcoxon(x, y, zero_method="pratt", correction=False,
                          alternative=alternative)
    diffs = x - y
    # For rank-biserial, drop zeros per Pratt convention
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return {"W": float(res.statistic), "p_value": float(res.pvalue),
                "r_rb": 0.0, "n_nonzero": 0}
    abs_ranks = stats.rankdata(np.abs(nz))
    pos = float(abs_ranks[nz > 0].sum())
    neg = float(abs_ranks[nz < 0].sum())
    r_rb = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0
    return {
        "W":         float(res.statistic),
        "p_value":   float(res.pvalue),
        "r_rb":      float(r_rb),
        "n_nonzero": int(len(nz)),
    }


# ─── Direction picker (Issue 5) ──────────────────────────────────────────────
def resolve_alternative(cli_choice: str, metric: str) -> str:
    """Map CLI choice to scipy.wilcoxon `alternative` value.

    'auto' uses pre-specified METRIC_DIRECTION:
      - higher_is_better metrics (BERTScore, ROUGE-L, MEDCON F1):
          We test (higher-precision) vs (lower-precision). Hypothesis: higher
          precision is more consistent → x_a > x_b → 'greater'.
      - lower_is_better metric (NLI contradiction):
          We test (higher-precision) vs (lower-precision). Hypothesis: higher
          precision is more consistent → x_a < x_b → 'less'.
    """
    if cli_choice in ("two-sided", "greater", "less"):
        return cli_choice
    # cli_choice == "auto"
    return "greater" if METRIC_DIRECTION[metric] else "less"


# ─── Pairwise driver ─────────────────────────────────────────────────────────
def run_pairs(df_long: pd.DataFrame, metric: str,
              alternative_cli: str, omnibus_lookup: dict | None) -> pd.DataFrame:
    rows = []
    df_metric = df_long[df_long["metric"] == metric]
    for model, sub in df_metric.groupby("model"):
        # Gate on Friedman omnibus significance (Issue 10)
        if omnibus_lookup is not None:
            o_key = (model, metric)
            if o_key not in omnibus_lookup:
                continue
            if not omnibus_lookup[o_key]["friedman_significant_05"]:
                continue

        # Pivot wide: rows = patients, cols = quant levels
        wide = sub.pivot_table(index="input_id", columns="quant_level", values="value").dropna()
        if wide.shape[0] < 3:
            continue

        # Issue 7 — explicit QUANT_ORDER
        quants = [q for q in QUANT_ORDER if q in wide.columns]
        alt = resolve_alternative(alternative_cli, metric)

        pair_rows = []
        for a, b in combinations(quants, 2):
            x = wide[a].to_numpy()
            y = wide[b].to_numpy()
            w = wilcoxon_pair(x, y, alternative=alt)
            seed = derive_pair_seed(SEED, model, metric, a, b)
            md, lo, hi, ci_method = bootstrap_diff_median(x, y, rng_seed=seed)
            # Issue 9 — does the CI exclude zero?
            ci_excludes_zero = bool((lo > 0) or (hi < 0))
            pair_rows.append({
                "model":                 model,
                "metric":                metric,
                "quant_a":               a,
                "quant_b":               b,
                "n_patients":            int(wide.shape[0]),
                "median_a":              float(np.median(x)),
                "median_b":              float(np.median(y)),
                "median_diff_a_minus_b": md,
                "bca_ci_lo":             lo,
                "bca_ci_hi":             hi,
                "ci_method":             ci_method,
                "ci_excludes_zero":      ci_excludes_zero,
                "wilcoxon_W":            w["W"],
                "wilcoxon_p":            w["p_value"],
                "alternative":           alt,
                "effect_r_rb":           w["r_rb"],
                "n_nonzero_diffs":       w["n_nonzero"],
                "higher_is_better":      METRIC_DIRECTION[metric],
                # Issue 4 — reflect whether gating was actually applied
                # (False under --no-gate sensitivity run; True for primary run)
                "omnibus_gated":         omnibus_lookup is not None,
            })

        # Issue 6 — Holm-Bonferroni within (model, metric)
        if pair_rows:
            pvals = [r["wilcoxon_p"] for r in pair_rows]
            _, p_adj, _, _ = multipletests(pvals, alpha=ALPHA, method="holm")
            for r, p_h in zip(pair_rows, p_adj):
                r["p_holm"]         = float(p_h)
                r["significant_05"] = bool(p_h < ALPHA)
                r["sig_ci_agreement"] = bool(r["significant_05"] == r["ci_excludes_zero"])
        rows.extend(pair_rows)

    return pd.DataFrame(rows)


# ─── Descriptive summary ─────────────────────────────────────────────────────
def summary_table(df_pw: pd.DataFrame, df_nli: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for metric in ["bertscore_f1_mean", "rougeL_f1_mean", "medcon_f1_mean",
                   "numeric_jaccard_mean"]:
        if metric in df_pw.columns:
            g = df_pw.groupby(["model", "quant_level"])[metric].agg(
                ["count", "mean", "median", "std"]).reset_index()
            g["metric"] = metric
            pieces.append(g)
    for nli_col in ("nli_contradiction_max_mean",
                    "nli_contradiction_mean_mean",
                    "nli_contradiction_q90_mean"):
        if nli_col in df_nli.columns:
            g = df_nli.groupby(["model", "quant_level"])[nli_col].agg(
                ["count", "mean", "median", "std"]).reset_index()
            g["metric"] = nli_col
            pieces.append(g)
    out = pd.concat(pieces, ignore_index=True)
    return out[["model", "quant_level", "metric", "count", "mean", "median", "std"]]


# ─── Aggregate into long format ──────────────────────────────────────────────
def to_long(df_pw: pd.DataFrame, df_nli: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for metric in ["bertscore_f1_mean", "rougeL_f1_mean", "medcon_f1_mean",
                   "numeric_jaccard_mean"]:
        if metric in df_pw.columns:
            p = df_pw[["input_id", "model", "quant_level", metric]].copy()
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
    parser.add_argument("--omnibus",          required=False, default=None,
                        help="stat_omnibus.csv — when provided, gates pairs on Friedman significance")
    parser.add_argument("--summary",          required=True)
    parser.add_argument("--pairs",            required=True)
    parser.add_argument("--alternative",      choices=["auto", "two-sided", "greater", "less"],
                        default="auto",
                        help="Wilcoxon alternative. 'auto' = directional per METRIC_DIRECTION (pre-specified).")
    parser.add_argument("--no-gate",          action="store_true",
                        help="Bypass Friedman gating; run pairs for every (model, metric).")
    args = parser.parse_args()

    # Issue 11 — validate inputs early
    for arg, path in [("--pairwise-metrics", args.pairwise_metrics),
                       ("--nli-metrics",      args.nli_metrics)]:
        if not Path(path).exists():
            parser.error(f"{arg} not found: {path}")
    if args.omnibus and not Path(args.omnibus).exists():
        parser.error(f"--omnibus not found: {args.omnibus}")

    df_pw  = pd.read_csv(args.pairwise_metrics)
    df_nli = pd.read_csv(args.nli_metrics)

    print("=" * 72)
    print("STATISTICAL PAIRWISE — Wilcoxon-pratt + BCa + Holm")
    print("=" * 72)
    print(f"  pairwise rows : {len(df_pw)}")
    print(f"  nli rows      : {len(df_nli)}")
    print(f"  alternative   : {args.alternative}")
    print(f"  bootstrap     : {N_BOOTSTRAP} iters, master seed {SEED}, child seeds derived per pair")
    print(f"  BCa fallback  : percentile-bootstrap when n < {BCA_MIN_N}")
    print(f"  multi-test    : Holm-Bonferroni")
    print(f"  alpha         : {ALPHA}")

    # Omnibus lookup
    omnibus_lookup = None
    if args.omnibus and not args.no_gate:
        df_omn = pd.read_csv(args.omnibus)
        omnibus_lookup = {(r["model"], r["metric"]): r.to_dict() for _, r in df_omn.iterrows()}
        n_sig = sum(1 for v in omnibus_lookup.values() if v["friedman_significant_05"])
        print(f"  omnibus       : gating on {n_sig}/{len(omnibus_lookup)} Friedman-significant (model, metric) cells")
    elif args.no_gate:
        print(f"  omnibus       : --no-gate set; running all pairs unconditionally")
    else:
        print(f"  omnibus       : no --omnibus passed; running all pairs unconditionally")

    # 1. Descriptive summary (always)
    t0 = time.perf_counter()
    summ = summary_table(df_pw, df_nli)
    summ.to_csv(args.summary, index=False)
    print(f"\n[1/2] Descriptive summary: {args.summary}  ({time.perf_counter()-t0:.1f}s)")

    # 2. Pairwise comparisons
    t0 = time.perf_counter()
    df_long = to_long(df_pw, df_nli)
    all_pairs = []
    for metric in METRIC_DIRECTION:
        all_pairs.append(run_pairs(df_long, metric, args.alternative, omnibus_lookup))
    pairs_df = pd.concat([p for p in all_pairs if not p.empty], ignore_index=True) \
               if any(not p.empty for p in all_pairs) else pd.DataFrame()
    pairs_df.to_csv(args.pairs, index=False)
    print(f"[2/2] Pairwise comparisons: {args.pairs}  ({time.perf_counter()-t0:.1f}s)")

    # Headline
    if len(pairs_df) > 0:
        n_sig         = int(pairs_df["significant_05"].sum())
        n_total       = len(pairs_df)
        n_ci_excludes = int(pairs_df["ci_excludes_zero"].sum())
        n_disagree    = int((~pairs_df["sig_ci_agreement"]).sum())
        print(f"\nResults across all retained pairs:")
        print(f"  total pairs              : {n_total}")
        print(f"  Holm-significant         : {n_sig}")
        print(f"  CI excludes zero         : {n_ci_excludes}")
        print(f"  Holm vs CI disagreement  : {n_disagree}  (manual review if > 0)")
        print(f"\n{pairs_df[['model','metric','quant_a','quant_b','median_diff_a_minus_b','bca_ci_lo','bca_ci_hi','p_holm','significant_05']].round(4).to_string(index=False)}")


if __name__ == "__main__":
    main()
