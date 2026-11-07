"""Per-image AUTO-ACCEPT / MANUAL-REVIEW decision with a human-readable reason."""
from __future__ import annotations

from .risk_scores import features

ACCEPT, REVIEW = "AUTO-ACCEPT", "MANUAL-REVIEW"


def decide(record, u, threshold, tau):
    f = features(record, tau)
    if threshold is None:
        return REVIEW, "no certified threshold: calibration could not meet the risk target"
    if f["n_accepted"] == 0:
        return REVIEW, f"no detection with confidence >= {tau:.2f}"
    if u <= threshold:
        return ACCEPT, f"uncertainty {u:.3f} <= calibrated threshold {threshold:.3f}"
    reasons = [f"uncertainty {u:.3f} > threshold {threshold:.3f}"]
    if f.get("census_gap", 0) >= 1:
        reasons.append(f"census expects {f['census_count']:.1f} defects, {int(f['n_accepted'])} detected")
    if f.get("unexplained_presence", 0) >= 0.5:
        reasons.append("a class is predicted present with no matching box")
    if f["max_rejected"] >= 0.8 * tau:
        reasons.append(f"near-threshold box at conf {f['max_rejected']:.2f}")
    return REVIEW, "; ".join(reasons)
