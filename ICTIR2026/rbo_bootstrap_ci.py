"""
Bootstrap confidence intervals for mean RBO scores.
Data sourced from Table 2 of "Explainable Legal Similarity: From Embeddings to Obligations."

Usage:
    python bootstrap_rbo_ci.py

To swap in your full results (including omitted model configs from the repo),
replace the values in RBO_SCORES with your complete per-clause arrays.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Data: per-clause RBO scores used in Table 2 (rows = T1..T10, cols = methods)
# ---------------------------------------------------------------------------
RBO_SCORES = {
    "O": [0.563, 0.401, 0.75, 0.555, 0.523, 0.395, 0.426, 0.45, 0.541, 0.573],
    "G": [0.492, 0.295, 0.629, 0.874, 0.523, 0.413, 0.42, 0.484, 0.503, 0.581], 
    "1P(5.2)": [0.392, 0.332, 0.757, 0.543, 0.543, 0.341, 0.642, 0.729, 0.543, 0.444],
    "2P(4.1/5.2)": [0.56,0.617,0.748,0.43, 0.481, 0.491, 0.456, 0.369, 0.547, 0.645],
    "2P(5.2/4.1)": [0.479,0.46,0.693, 0.66, 0.547, 0.345, 0.306, 0.459, 0.503, 0.513],
    "2P(5.2)": [0.603, 0.36, 0.445, 0.555, 0.282, 0.365, 0.348, 0.449, 0.706,0.613],
    "L(5.2)": [0.729,0.442,1.0,0.588,0.433,0.497,0.396,0.297,0.55,0.576],
}

CLAUSE_TYPES = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]


def bootstrap_mean_ci(scores, n_resamples=10_000, ci=95, seed=42):
    """
    Non-parametric bootstrap confidence interval for the mean.

    Parameters
    ----------
    scores      : array-like of observed values
    n_resamples : number of bootstrap iterations
    ci          : confidence level (%)
    seed        : RNG seed for reproducibility

    Returns
    -------
    mean, lower_bound, upper_bound
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(scores)
    observed_mean = arr.mean()

    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_resamples)
    ])

    alpha = (100 - ci) / 2
    lo = np.percentile(boot_means, alpha)
    hi = np.percentile(boot_means, 100 - alpha)
    return observed_mean, lo, hi


if __name__ == "__main__":
    N_RESAMPLES = 10_000
    CI = 95
    results = {}
    for method, scores in RBO_SCORES.items():
        mean, lo, hi = bootstrap_mean_ci(scores, n_resamples=N_RESAMPLES, ci=CI)
        sd = np.std(scores, ddof=1)
        results[method] = (mean, lo, hi, sd)


    # LaTeX table printing
    print(r"\begin{tabular}{l|cccc}")
    print(r"Method &   Lower & Upper \\")
    print(r"\midrule")
    for method, (mean, lo, hi, sd) in results.items():
        print(rf"{method} & {lo:.3f} & {hi:.3f}\\")
    print(r"\end{tabular}")
