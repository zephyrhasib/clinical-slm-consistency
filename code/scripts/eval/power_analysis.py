"""
Simulation-based power analysis for the statistical pipeline.

Simulates n=100 paired observations under a range of effect sizes,
applies the exact analysis pipeline (Wilcoxon signed-rank with Pratt
zero-handling + one-sided directional alternative + Holm-Bonferroni
correction across 6 pairs per family), and estimates rejection rate
(power) per effect size.

Outputs:
    outputs/pilot_eval/power_analysis.csv  — power per effect size
    outputs/pilot_eval/power_curve.png      — plot

Estimates the minimum detectable effect size at 80% power.

Run once before main generation.

Methods section text:
    "Statistical power was assessed via simulation. We generated
    paired observations at n=100 with effect sizes ranging 0.0-0.7
    (standardized median difference). For each effect size, we ran
    2,000 simulations applying the exact analysis pipeline (Wilcoxon
    signed-rank with Pratt zero-handling, one-sided directional
    alternative, Holm-Bonferroni correction across 6 pairs per
    family, alpha=0.05). Our design has 80% power to detect effect
    sizes delta >= [X]. Smaller effects may produce null results
    not due to absence of effect but due to insufficient power."

Run from project root in venv_metrics:
    python scripts/power_analysis.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# ─── Pre-specified design parameters ────────────────────────────────────────
N_PATIENTS         = 100   # main-run sample size per (model, metric) family
N_PAIRS_PER_FAMILY = 6     # C(4,2) quant pairs per family
ALPHA              = 0.05  # significance level
N_SIMS_DEFAULT     = 2000  # simulations per effect size (pre-specified)
SEED               = 42    # master RNG seed

# Effect sizes tested (standardized median difference, Cliff's δ analogue)
EFFECT_SIZES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70]


def simulate_one_family(effect_size: float, rng, n_pairs: int = N_PAIRS_PER_FAMILY) -> bool:
    """
    Simulate one (model, metric) family under a given effect size.

    Returns True if at least one of the 6 pairwise tests survives
    Holm-Bonferroni correction at α=0.05 (i.e., the family rejects
    the null at the family level).

    Each pair is generated as a paired difference between two normal
    samples: the "higher-precision" condition centered at 0 and the
    "lower-precision" condition centered at -effect_size. The
    Wilcoxon test asks whether the high-precision distribution
    stochastically dominates the low-precision distribution.
    """
    p_values = []
    for _ in range(n_pairs):
        # Higher-precision condition (Q6 baseline)
        q_high = rng.normal(loc=0.0, scale=1.0, size=N_PATIENTS)
        # Lower-precision condition (degraded by effect_size)
        q_low  = rng.normal(loc=-effect_size, scale=1.0, size=N_PATIENTS)

        try:
            res = stats.wilcoxon(
                q_high, q_low,
                zero_method="pratt",
                alternative="greater",  # one-sided directional
                correction=False,
            )
            p_values.append(res.pvalue)
        except ValueError:
            # All-zero diffs case — return non-significant
            p_values.append(1.0)

    # Holm-Bonferroni correction across the 6 pairs
    _, p_corrected, _, _ = multipletests(p_values, alpha=ALPHA, method="holm")

    # Family rejects if at least one Holm-corrected p < 0.05
    return any(p < ALPHA for p in p_corrected)


def run_power_analysis(n_sims: int):
    """Run the full power simulation across all effect sizes."""
    results = []

    print("=" * 72)
    print("POWER ANALYSIS — Simulation-Based")
    print("=" * 72)
    print(f"  n_patients            : {N_PATIENTS}")
    print(f"  n_pairs_per_family    : {N_PAIRS_PER_FAMILY}")
    print(f"  alpha                 : {ALPHA}")
    print(f"  n_simulations         : {n_sims}")
    print(f"  master_seed           : {SEED}")
    print(f"  effect_sizes          : {EFFECT_SIZES}")
    print(f"  test                  : Wilcoxon signed-rank (Pratt, one-sided greater)")
    print(f"  correction            : Holm-Bonferroni across 6 pairs per family")
    print()

    for eff in EFFECT_SIZES:
        # Spawn a child rng for this effect size for reproducibility
        child_seed = hash((SEED, eff)) & 0xFFFFFFFF
        child_rng  = np.random.default_rng(child_seed)

        n_rejections = sum(
            simulate_one_family(eff, child_rng)
            for _ in tqdm(range(n_sims), desc=f"  effect δ={eff:.2f}")
        )
        power = n_rejections / n_sims
        # Wilson-ish 95% CI on the proportion
        se    = float(np.sqrt(power * (1 - power) / n_sims))
        ci_lo = max(0.0, power - 1.96 * se)
        ci_hi = min(1.0, power + 1.96 * se)

        results.append({
            "effect_size":  eff,
            "power":        power,
            "ci_lo":        ci_lo,
            "ci_hi":        ci_hi,
            "n_rejections": n_rejections,
            "n_sims":       n_sims,
        })
        print(f"  δ={eff:.2f} → power = {power:.3f}  [95% CI: {ci_lo:.3f}, {ci_hi:.3f}]")

    return pd.DataFrame(results)


def find_minimum_detectable_effect(results_df: pd.DataFrame, target_power: float = 0.80) -> float:
    """
    Find the smallest effect size where power >= target_power.
    Uses linear interpolation between simulated points for precision.
    """
    above = results_df[results_df["power"] >= target_power]
    if len(above) == 0:
        return float("nan")

    # Find the smallest effect size at or above target
    min_above_idx = above["effect_size"].idxmin()
    min_above_eff = results_df.loc[min_above_idx, "effect_size"]
    min_above_pow = results_df.loc[min_above_idx, "power"]

    # Linear interpolation: find the effect size just below where power crossed 80%
    below = results_df[results_df["power"] < target_power]
    if len(below) == 0:
        return min_above_eff  # all effects achieve target

    max_below_idx = below["effect_size"].idxmax()
    max_below_eff = results_df.loc[max_below_idx, "effect_size"]
    max_below_pow = results_df.loc[max_below_idx, "power"]

    if max_below_eff >= min_above_eff:
        return min_above_eff

    slope        = (min_above_pow - max_below_pow) / (min_above_eff - max_below_eff)
    interpolated = max_below_eff + (target_power - max_below_pow) / slope
    return float(interpolated)


def plot_power_curve(results_df: pd.DataFrame, min_detectable: float,
                     output_path: Path, n_sims: int):
    """Plot power vs effect size with 80% target line and minimum-detectable annotation."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(results_df["effect_size"], results_df["power"],
            marker="o", color="#2E86AB", linewidth=2, markersize=8,
            label="Empirical power")

    ax.fill_between(results_df["effect_size"],
                    results_df["ci_lo"], results_df["ci_hi"],
                    color="#2E86AB", alpha=0.2, label="95% CI")

    ax.axhline(0.80, color="red", linestyle="--", linewidth=1.5,
               label="80% power threshold")

    if not np.isnan(min_detectable):
        ax.axvline(min_detectable, color="green", linestyle=":", linewidth=1.5,
                   label=f"Min. detectable δ ≈ {min_detectable:.3f}")

    ax.set_xlabel("Effect size (standardized median difference)", fontsize=12)
    ax.set_ylabel("Power (rejection rate at α_holm = 0.05/6)", fontsize=12)
    ax.set_title(f"Statistical Power Curve — Wilcoxon-pratt + Holm-Bonferroni\n"
                 f"n={N_PATIENTS} paired patients, {N_PAIRS_PER_FAMILY} pairs/family, "
                 f"{n_sims} simulations per point",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, max(EFFECT_SIZES) + 0.02)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv",
                        default="outputs/pilot_eval/power_analysis.csv")
    parser.add_argument("--out-png",
                        default="outputs/pilot_eval/power_curve.png")
    parser.add_argument("--n-sims", type=int, default=N_SIMS_DEFAULT,
                        help=f"simulations per effect size (default: {N_SIMS_DEFAULT}, "
                             "use 100 for fast smoke-test)")
    args = parser.parse_args()

    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Run the simulation
    results_df = run_power_analysis(n_sims=args.n_sims)

    # Find minimum detectable effect
    min_detectable = find_minimum_detectable_effect(results_df, target_power=0.80)

    # Save results
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Plot
    plot_power_curve(results_df, min_detectable, out_png, n_sims=args.n_sims)

    # Summary
    print("\n" + "=" * 72)
    print("POWER ANALYSIS SUMMARY")
    print("=" * 72)
    print(f"\n  Design: n={N_PATIENTS} patients × {N_PAIRS_PER_FAMILY} pairs × Holm-Bonferroni")
    if not np.isnan(min_detectable):
        print(f"\n  Minimum detectable effect at 80% power: δ ≈ {min_detectable:.3f}")
        print(f"\n  Power statement:")
        print(f"    'With n={N_PATIENTS} paired observations and α_holm=0.0083,'")
        print(f"    'our design has 80% power to detect Cliff's δ >= {min_detectable:.3f}.'")
    else:
        print(f"\n  Warning: 80% power not reached within tested effect-size range [0.0, 0.7]")
        print(f"  Maximum power achieved: {results_df['power'].max():.3f} "
              f"at δ={results_df.loc[results_df['power'].idxmax(), 'effect_size']:.2f}")
        print(f"  Recommendation: expand EFFECT_SIZES list and re-run.")


if __name__ == "__main__":
    main()
