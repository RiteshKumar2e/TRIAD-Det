"""Selective conformal risk control, independent of any detector.

Inputs are two arrays per image: an uncertainty score u (lower is safer) and a
binary loss (1 = a defect was missed). An image is auto-accepted when u <= lam.

The controlled quantity is the risk among accepted images,
    R(lam) = P(loss = 1 | u <= lam).
This is not monotone in lam, so plain conformal risk control does not apply.
Thresholds are chosen with Learn-then-Test fixed-sequence testing: H_lam says
R(lam) > alpha, tested with an exact binomial p-value on the accepted calibration
images. Thresholds are walked from strict to lenient, stopping at the first failure.

If calibration and future images are i.i.d. from one distribution, then with
probability at least 1 - delta over calibration, R(lam_hat) <= alpha. This says
nothing about a different distribution, such as another dataset.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import beta, binom


def clopper_pearson(k: int, n: int, confidence: float = 0.95):
    if n == 0:
        return (0.0, 1.0)
    tail = (1 - confidence) / 2
    low = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    return (low, high)


def min_accepted(alpha: float, delta: float) -> int:
    """Smallest accepted count that can pass the test, reached with zero losses."""
    return math.ceil(math.log(delta) / math.log(1 - alpha))


def threshold_grid(reference_u, reference_loss, n_calib: int, alpha: float, delta: float, size: int) -> np.ndarray:
    """Ascending fixed sequence, built from validation data and never from calibration.

    Starting at the strictest candidate would need near-zero losses in a tiny accepted
    set, and one early loss would end the walk. The sequence therefore starts at the
    candidate whose projected calibration p-value is lowest. Using independent data
    to order the hypotheses leaves the fixed-sequence guarantee intact.
    """
    u, loss = np.asarray(reference_u, float), np.asarray(reference_loss, int)
    candidates = np.unique(np.quantile(u, np.linspace(0.0, 1.0, size)))
    projected = []
    for lam in candidates:
        accepted = u <= lam
        risk = (loss[accepted].sum() + 1) / (accepted.sum() + 2)   # smoothed: no false zero
        n = round(accepted.mean() * n_calib)
        projected.append(binom.cdf(round(risk * n), n, alpha) if n else 1.0)
    return candidates[int(np.argmin(projected)):]


@dataclass
class Calibration:
    alpha: float
    delta: float
    threshold: float | None
    certified: bool
    n_calib: int
    n_accepted: int
    n_losses: int
    p_value: float | None
    tests_run: int


def calibrate(u, loss, grid, alpha: float, delta: float) -> Calibration:
    u, loss = np.asarray(u, float), np.asarray(loss, int)
    chosen, tests, n_acc, n_loss, p_chosen = None, 0, 0, 0, None
    for lam in grid:
        accepted = u <= lam
        n, k = int(accepted.sum()), int(loss[accepted].sum())
        tests += 1
        p_value = float(binom.cdf(k, n, alpha)) if n else 1.0
        if p_value > delta:
            break
        chosen, n_acc, n_loss, p_chosen = float(lam), n, k, p_value
    return Calibration(alpha, delta, chosen, chosen is not None, len(u), n_acc, n_loss, p_chosen, tests)


def apply_threshold(u, loss, threshold) -> dict:
    """Risk, coverage and review rate on held-out images. No threshold means review all."""
    u, loss = np.asarray(u, float), np.asarray(loss, int)
    accepted = np.zeros(len(u), bool) if threshold is None else u <= threshold
    n, k = int(accepted.sum()), int(loss[accepted].sum())
    return {
        "n_images": len(u),
        "n_accepted": n,
        "n_missed_among_accepted": k,
        "acceptance_rate": n / len(u) if len(u) else 0.0,
        "manual_review_rate": 1 - n / len(u) if len(u) else 1.0,
        "risk_among_accepted": k / n if n else None,
        "risk_ci95": clopper_pearson(k, n),
    }


def risk_coverage_curve(u, loss) -> dict:
    """Empirical curve. Tied scores share one point, since they cannot be separated."""
    u, loss = np.asarray(u, float), np.asarray(loss, float)
    order = np.argsort(u, kind="stable")
    sorted_u, cumulative = u[order], np.cumsum(loss[order])
    last_of_tie = np.r_[sorted_u[1:] != sorted_u[:-1], True] if len(u) else np.array([], bool)
    index = np.flatnonzero(last_of_tie)
    return {
        "threshold": sorted_u[index],
        "coverage": (index + 1) / max(len(u), 1),
        "risk": cumulative[index] / (index + 1),
    }


def aurc(u, loss) -> float:
    """Area under the risk-coverage curve; lower means the score ranks errors better."""
    curve = risk_coverage_curve(u, loss)
    if len(curve["coverage"]) == 0:
        return float("nan")
    widths = np.diff(np.r_[0.0, curve["coverage"]])
    return float((curve["risk"] * widths).sum())


def bootstrap_threshold(u, loss, grid, alpha, delta, n_boot, seed) -> dict:
    """How much the selected threshold moves when calibration is resampled."""
    u, loss = np.asarray(u, float), np.asarray(loss, int)
    rng = np.random.default_rng(seed)
    values, uncertified = [], 0
    for _ in range(n_boot):
        index = rng.integers(0, len(u), len(u))
        result = calibrate(u[index], loss[index], grid, alpha, delta)
        if result.threshold is None:
            uncertified += 1
        else:
            values.append(result.threshold)
    return {
        "n_boot": n_boot,
        "uncertified_fraction": uncertified / n_boot if n_boot else None,
        "threshold_p05": float(np.quantile(values, 0.05)) if values else None,
        "threshold_median": float(np.median(values)) if values else None,
        "threshold_p95": float(np.quantile(values, 0.95)) if values else None,
    }


def as_dict(calibration: Calibration) -> dict:
    return asdict(calibration)
