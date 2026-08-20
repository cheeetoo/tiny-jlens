"""Uncertainty helpers for the experiment summaries.

wilson(k, n)          -> (lo, hi) 95% Wilson score interval for a proportion.
sign_test(wins, n)    -> two-sided exact binomial p for paired win counts
                         (ties excluded by the caller).
cluster_boot(vals_by_cluster, n_boot) -> (mean, lo, hi) 95% percentile
                         bootstrap over clusters, for trial-level outcomes
                         that are correlated within a cluster (same country,
                         category, passage...).
"""

from __future__ import annotations

import math
import random


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fmt_prop(k: int, n: int) -> str:
    if n == 0:
        return "n/a (n=0)"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100*k/n:.0f}% [{100*lo:.0f}, {100*hi:.0f}]"


def sign_test(wins: int, n: int) -> float:
    """Two-sided exact binomial test of wins ~ Binom(n, 0.5), computed in
    log space so large n doesn't overflow."""
    if n == 0:
        return float("nan")
    def logpmf(i):
        return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                + n * math.log(0.5))
    lp_obs = logpmf(wins)
    p = sum(math.exp(lp) for i in range(n + 1)
            if (lp := logpmf(i)) <= lp_obs + 1e-9)
    return min(1.0, p)


def cluster_boot(vals_by_cluster: dict, n_boot: int = 4000, seed: int = 0):
    """95% CI for the mean of trial-level values, resampling clusters."""
    keys = list(vals_by_cluster)
    if not keys:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    flat = [v for vs in vals_by_cluster.values() for v in vs]
    mean = sum(flat) / len(flat)
    stats = []
    for _ in range(n_boot):
        sample = [rng.choice(keys) for _ in keys]
        vs = [v for k in sample for v in vals_by_cluster[k]]
        if vs:
            stats.append(sum(vs) / len(vs))
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[min(len(stats) - 1, int(0.975 * len(stats)))]
    return mean, lo, hi
