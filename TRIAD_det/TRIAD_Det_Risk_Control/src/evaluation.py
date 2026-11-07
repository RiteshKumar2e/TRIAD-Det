"""Detection metrics, operating point selection and the selective-risk evaluation loop.

Kept separate on purpose:
  detection performance   mAP50, mAP50-95, precision, recall, F1, missed-defect rate (box level)
  image-level risk        any-miss risk, mean missed fraction (all test images, no selection)
  selective coverage      AUTO-ACCEPT coverage, MANUAL-REVIEW rate
  risk among accepted     the quantity the calibration controls
"""
from __future__ import annotations

import numpy as np

from . import calibration as cal
from .detection_matching import counts, map_metrics
from .risk_losses import loss_array
from .risk_scores import StatsScore, base_scores
from .statistics import aurc


def choose_tau(val, iou, grid, class_aware=True):
    """Box confidence threshold with best validation F1. Fixed before calibration."""
    return float(max(grid, key=lambda t: counts(val, t, iou, class_aware)["f1"]))


def detection_metrics(records, tau, iou, classes=None, class_aware=True):
    index = {c: i for i, c in enumerate(classes)} if (classes and class_aware) else None
    m = map_metrics(records, index)
    c = counts(records, tau, iou, class_aware)
    n_gt = c["tp"] + c["fn"]
    any_l, _ = loss_array(records, "any", tau, iou, class_aware)
    frac_l, _ = loss_array(records, "fraction", tau, iou, class_aware)
    return {"n_images": len(records), "n_gt": n_gt, "mAP50": m["AP50"], "mAP50_95": m["mAP"], "AP75": m["AP75"],
            "tau": tau, "match_iou": iou, "class_aware": class_aware,
            "precision": c["precision"], "recall": c["recall"], "f1": c["f1"],
            "missed_defect_rate": c["fn"] / n_gt if n_gt else None,
            "any_miss_image_risk": float(any_l.mean()) if len(any_l) else None,
            "mean_missed_fraction": float(frac_l.mean()) if len(frac_l) else None,
            "AP_per_class": m.get("AP_per_class")}


class ScoreBank:
    """All candidate uncertainty scores for one model, with the stats model fitted on VAL."""

    def __init__(self, val_records, tau, iou, class_aware=True):
        self.tau = tau
        y, keep = loss_array(val_records, "any", tau, iou, class_aware)
        self.stats = StatsScore(tau).fit([r for r, k in zip(val_records, keep) if k], y)

    def __call__(self, records):
        base = [base_scores(r, self.tau) for r in records]
        out = {k: np.array([b[k] for b in base]) for k in base[0]} if base else {}
        out["stats"] = self.stats(records) if records else np.array([])
        return out


def pick_score(val_scores, val_loss, allowed=None):
    """Score with lowest validation AURC (selection uses VAL only)."""
    table = {k: cal_aurc for k, v in val_scores.items() if (allowed is None or k in allowed)
             for cal_aurc in [aurc(v, val_loss)]}
    best = min(table, key=lambda k: (np.isnan(table[k]), table[k]))
    return best, table


def calibrate_all(u_val, l_val, u_cal, l_cal, alpha, delta, grid_size, binary, u_conf_cal=None):
    """Every calibration method on one (score, loss) pair. Grid order comes from VAL."""
    grid = cal.threshold_grid(u_val, (l_val > 0).astype(int) if not binary else l_val.astype(int),
                              len(u_cal), alpha, delta, grid_size)
    out = [cal.ltt(u_cal, l_cal, grid, alpha, delta, "binomial" if binary else "hb")]
    if binary:
        out.append(cal.ltt(u_cal, l_cal, grid, alpha, delta, "hb"))
    out.append(cal.crc_joint(u_cal, l_cal, alpha))
    out.append(cal.naive(u_cal, l_cal, alpha))
    if u_conf_cal is not None:
        out.append(cal.fixed_confidence(u_conf_cal, l_cal, alpha))
    return out, grid


def resplit_validity(u_pool, l_pool, u_val, l_val, alpha, delta, grid_size, binary, repeats, seed):
    """Random half/half re-splits of the held-out pool (calib U test): how often is the
    realised test selective risk above alpha? LTT should do this at most ~delta of the time
    (up to finite-test-sample noise); naive selection typically does not."""
    rng = np.random.default_rng(seed)
    n = len(u_pool)
    methods = {}
    for _ in range(repeats):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        results, _ = calibrate_all(u_val, l_val, u_pool[a], l_pool[a], alpha, delta, grid_size, binary)
        for res in results:
            ev = cal.evaluate(u_pool[b], l_pool[b], res.threshold)
            m = methods.setdefault(res.method, {"violations": 0, "coverage": [], "risk": [], "uncertified": 0})
            m["coverage"].append(ev["coverage"])
            if ev["risk_accepted"] is not None:
                m["risk"].append(ev["risk_accepted"])
                m["violations"] += int(ev["risk_accepted"] > alpha)
            if not res.certified:
                m["uncertified"] += 1
    return {k: {"violation_rate": v["violations"] / repeats, "uncertified_rate": v["uncertified"] / repeats,
                "coverage_mean": float(np.mean(v["coverage"])), "coverage_std": float(np.std(v["coverage"], ddof=1)),
                "risk_mean": float(np.mean(v["risk"])) if v["risk"] else None,
                "risk_std": float(np.std(v["risk"], ddof=1)) if len(v["risk"]) > 1 else None,
                "repeats": repeats} for k, v in methods.items()}
