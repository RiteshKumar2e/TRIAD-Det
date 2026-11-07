"""Uncertainty estimates and tests."""
from __future__ import annotations

import numpy as np

from triad.conformal import aurc, clopper_pearson, risk_coverage_curve  # noqa: F401


def bootstrap_ci(values_fn, n, n_boot, seed, level=0.95):
    """Percentile bootstrap over images. values_fn(index) -> float."""
    rng = np.random.default_rng(seed)
    stats = [values_fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    stats = [s for s in stats if s is not None and np.isfinite(s)]
    if not stats:
        return (None, None)
    tail = (1 - level) / 2
    return (float(np.quantile(stats, tail)), float(np.quantile(stats, 1 - tail)))


def paired_bootstrap_aurc(u_a, u_b, loss, n_boot, seed):
    """Two-sided paired bootstrap test of AURC(a) - AURC(b) on the same images."""
    u_a, u_b, loss = map(np.asarray, (u_a, u_b, loss))
    rng = np.random.default_rng(seed)
    observed = aurc(u_a, loss) - aurc(u_b, loss)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(loss), len(loss))
        diffs.append(aurc(u_a[i], loss[i]) - aurc(u_b[i], loss[i]))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(observed), "ci95": (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))),
            "p_value": float(min(p, 1.0))}


def holm(p_values):
    """Holm-Bonferroni adjusted p-values, same order as input."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def mean_std(values):
    v = np.array([x for x in values if x is not None], float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)) if len(v) else (None, None, 0)
