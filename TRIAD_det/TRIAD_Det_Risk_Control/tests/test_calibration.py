"""Checks of the calibration code on synthetic data with KNOWN true selective risk.

Data model: u ~ U(0,1); P(loss = 1 | u) = 0.02 + 0.6 u^2. The true selective risk
R(lam) = E[loss | u <= lam] = 0.02 + 0.2 lam^2 is known in closed form, so violations of
R(lam_hat) <= alpha can be counted exactly (no finite-test-set noise).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import calibration as cal  # noqa: E402


def true_risk(lam):
    return 0.02 + 0.2 * lam ** 2


def draw(rng, n):
    u = rng.uniform(0, 1, n)
    return u, (rng.uniform(0, 1, n) < 0.02 + 0.6 * u ** 2).astype(float)


def _violation_rates(n_cal=300, alpha=0.10, delta=0.10, trials=400, seed=0):
    rng = np.random.default_rng(seed)
    u_val, l_val = draw(rng, 500)
    grid = cal.threshold_grid(u_val, l_val.astype(int), n_cal, alpha, delta, 100)
    v_ltt = v_hb = v_naive = 0
    for _ in range(trials):
        u, l = draw(rng, n_cal)
        for res, key in ((cal.ltt(u, l, grid, alpha, delta, "binomial"), "ltt"),
                         (cal.ltt(u, l, grid, alpha, delta, "hb"), "hb"),
                         (cal.naive(u, l, alpha), "naive")):
            if res.threshold is not None and true_risk(res.threshold) > alpha:
                if key == "ltt":
                    v_ltt += 1
                elif key == "hb":
                    v_hb += 1
                else:
                    v_naive += 1
    return v_ltt / trials, v_hb / trials, v_naive / trials


def test_ltt_controls_selective_risk():
    ltt, hb, naive = _violation_rates()
    assert ltt <= 0.10 + 0.04, ltt          # delta plus Monte-Carlo slack
    assert hb <= 0.10 + 0.04, hb
    assert naive > ltt, (naive, ltt)        # uncorrected selection violates more often


def test_binomial_rejects_fractional_loss():
    try:
        cal.ltt(np.array([0.1, 0.2]), np.array([0.5, 0.0]), np.array([0.5]), 0.1, 0.1, "binomial")
    except ValueError:
        return
    raise AssertionError("binomial p-value must refuse non-binary losses")


def test_crc_joint_expected_risk():
    rng = np.random.default_rng(1)
    alpha, joint = 0.05, []
    for _ in range(400):
        u, l = draw(rng, 200)
        res = cal.crc_joint(u, l, alpha)
        lam = res.threshold if res.threshold is not None else -1
        # true joint risk E[L 1{u<=lam}] = integral_0^lam (0.02 + 0.6 s^2) ds
        joint.append(0.02 * max(lam, 0) + 0.2 * max(lam, 0) ** 3)
    assert np.mean(joint) <= alpha + 0.005


def test_no_threshold_means_review_all():
    ev = cal.evaluate(np.array([0.1, 0.2]), np.array([0.0, 1.0]), None)
    assert ev["coverage"] == 0 and ev["manual_review_rate"] == 1
