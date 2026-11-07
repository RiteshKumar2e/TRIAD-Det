"""Calibration of the acceptance threshold lambda (AUTO-ACCEPT iff u <= lambda).

Controlled quantity (selective risk):  R(lam) = E[ L | u <= lam ].
R is not monotone in lam, so Conformal Risk Control (Angelopoulos et al., ICLR 2024) does not
apply to it directly. We use Learn-then-Test (Angelopoulos et al., AOAS 2025):

  H_lam : R(lam) > alpha.  Valid p-value from the n_lam accepted calibration images:
     binary loss (L_any)       p = BinomCDF(k; n_lam, alpha)                          [exact]
     bounded loss (L_fraction) p = Hoeffding-Bentkus p-value (Bates et al. JACM 2021)   [valid, conservative]
  Fixed-sequence testing over lam_1 < lam_2 < ... (order fixed from VAL data only), stop at the
  first p > delta; lam_hat = last rejected. FWER <= delta, hence

     P_calib( R(lam_hat) <= alpha ) >= 1 - delta,

  provided calibration and test images are i.i.d. from one distribution. Conditional on
  acceptance count n_lam, accepted calibration losses are i.i.d. draws from L | u <= lam, so
  the p-values are valid. If no lam is rejected, lam_hat = None: every image goes to review.

Also implemented, for comparison only:
  crc_joint  Conformal Risk Control of the JOINT risk E[L * 1{u <= lam}] (monotone in lam).
             Guarantee is in expectation over calibration, and the quantity differs from R(lam).
  naive      largest lam whose empirical calibration selective risk is <= alpha. No guarantee.
  fixed_conf accept if top confidence >= 0.5 (u_conf <= 0.5). No calibration, no guarantee.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import binom

from triad.conformal import clopper_pearson, threshold_grid  # noqa: F401  (grid ordering reused)


def _h1(a, b):
    a = min(max(a, 1e-12), 1 - 1e-12)
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def p_binomial(losses, alpha):
    n = len(losses)
    return float(binom.cdf(int(np.sum(losses)), n, alpha)) if n else 1.0


def p_hoeffding_bentkus(losses, alpha):
    """Valid p-value for H: E[L] > alpha with L in [0, 1] (LTT, Prop. / RCPS Sec. 3)."""
    n = len(losses)
    if n == 0:
        return 1.0
    r = float(np.mean(losses))
    hoeffding = math.exp(-n * _h1(min(r, alpha), alpha))
    bentkus = math.e * float(binom.cdf(math.ceil(n * r), n, alpha))
    return min(hoeffding, bentkus, 1.0)


P_VALUES = {"binomial": p_binomial, "hb": p_hoeffding_bentkus}


@dataclass
class Result:
    method: str
    alpha: float
    delta: float
    threshold: float | None
    certified: bool
    n_calib: int
    n_accepted: int
    calib_risk: float | None
    p_value: float | None
    tests_run: int
    guarantee: str


def ltt(u, loss, grid, alpha, delta, pvalue="binomial"):
    u, loss = np.asarray(u, float), np.asarray(loss, float)
    if pvalue == "binomial" and not np.all((loss == 0) | (loss == 1)):
        raise ValueError("exact binomial p-value requires a binary loss; use pvalue='hb'")
    chosen, n_acc, risk, p_chosen, tests = None, 0, None, None, 0
    for lam in grid:
        accepted = u <= lam
        tests += 1
        p = P_VALUES[pvalue](loss[accepted], alpha)
        if p > delta:
            break
        chosen, n_acc, p_chosen = float(lam), int(accepted.sum()), p
        risk = float(loss[accepted].mean())
    return Result(f"LTT-{pvalue}", alpha, delta, chosen, chosen is not None, len(u), n_acc, risk, p_chosen, tests,
                  "P(selective risk <= alpha) >= 1 - delta, i.i.d. calib/test")


def crc_joint(u, loss, alpha):
    """Largest lam with (sum_i L_i 1{u_i<=lam} + 1) / (n + 1) <= alpha (loss bound B = 1)."""
    u, loss = np.asarray(u, float), np.asarray(loss, float)
    n = len(u)
    order = np.argsort(u, kind="stable")
    su, cum = u[order], np.cumsum(loss[order])
    chosen = None
    for i in range(n):
        if i + 1 < n and su[i + 1] == su[i]:
            continue
        if (cum[i] + 1) / (n + 1) <= alpha:
            chosen = float(su[i])
        else:
            break
    acc = u <= chosen if chosen is not None else np.zeros(n, bool)
    return Result("CRC-joint", alpha, float("nan"), chosen, chosen is not None, n, int(acc.sum()),
                  float(loss[acc].mean()) if acc.any() else None, None, n,
                  "E[L * 1{accept}] <= alpha in expectation (joint, not selective, risk)")


def naive(u, loss, alpha):
    u, loss = np.asarray(u, float), np.asarray(loss, float)
    best = None
    for lam in np.unique(u):
        acc = u <= lam
        if loss[acc].mean() <= alpha:
            best = float(lam)
    acc = u <= best if best is not None else np.zeros(len(u), bool)
    return Result("naive-empirical", alpha, float("nan"), best, best is not None, len(u), int(acc.sum()),
                  float(loss[acc].mean()) if acc.any() else None, None, len(np.unique(u)), "none")


def fixed_confidence(u_conf, loss, alpha, cut=0.5):
    acc = np.asarray(u_conf) <= cut
    loss = np.asarray(loss, float)
    return Result("fixed-conf-0.5", alpha, float("nan"), cut, True, len(loss), int(acc.sum()),
                  float(loss[acc].mean()) if acc.any() else None, None, 0, "none")


def evaluate(u, loss, threshold):
    """Held-out selective metrics. No threshold = everything to MANUAL-REVIEW."""
    u, loss = np.asarray(u, float), np.asarray(loss, float)
    acc = np.zeros(len(u), bool) if threshold is None else u <= threshold
    n, k = int(acc.sum()), float(loss[acc].sum())
    binary = bool(np.all((loss == 0) | (loss == 1)))
    return {
        "n_test": len(u), "n_accepted": n,
        "coverage": n / len(u) if len(u) else 0.0,
        "manual_review_rate": 1 - n / len(u) if len(u) else 1.0,
        "risk_accepted": k / n if n else None,
        "risk_accepted_ci95": clopper_pearson(int(k), n) if (binary and n) else None,
        "risk_all": float(loss.mean()) if len(loss) else None,
        "joint_risk": k / len(u) if len(u) else None,
    }


def as_dict(result):
    return asdict(result)
