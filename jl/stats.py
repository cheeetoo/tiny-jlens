"""The small statistics the criterion summaries share.  No dependencies beyond the stdlib."""
from __future__ import annotations

from math import erfc


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(rate, lo, hi) -- Wilson score interval for k successes out of n."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def median(xs):
    """Upper median of a sequence (nan if empty)."""
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def sign_test(pairs) -> tuple[float, int, float]:
    """pairs of (a, b): fraction with a < b, n, and a two-sided p (normal approx, ties dropped)."""
    d = [(a, b) for a, b in pairs if a != b]
    n = len(d)
    k = sum(a < b for a, b in d)
    if n == 0:
        return float("nan"), n, float("nan")
    z = (k - n / 2) / (n ** 0.5 / 2)
    return k / n, n, erfc(abs(z) / 2 ** 0.5)


def spearman(xs, ys) -> float:
    """Spearman rank correlation, no tie correction (the d^2 formula)."""
    if len(xs) < 2:
        return float("nan")

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for i, idx in enumerate(order):
            r[idx] = i
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n * n - 1))
