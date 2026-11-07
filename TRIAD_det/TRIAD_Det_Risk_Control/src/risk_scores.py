"""Image-level uncertainty scores u(x). Lower u = safer; AUTO-ACCEPT when u <= lambda.

Every score here reads only inference-time fields of a record: predicted boxes, classes,
confidences, image size and (TRIAD-Det only) the census branch. None reads `gt`,
`gt_labels`, `folder_class` or any loss. `features()` enforces this by touching only those keys.

Scores
  conf        1 - highest box confidence among accepted boxes                (confidence only)
  weakest     1 - lowest accepted confidence                                  (confidence only)
  grey        highest rejected confidence / tau: how close a missed box came  (confidence only)
  stats       logistic model of P(any miss) on detection statistics, fitted on the VAL split
  census      TRIAD-Det: expected-count gap and unexplained class presence    (model uncertainty)
  disagree    1 - box agreement between independent detectors                  (multi-model)
Images with no accepted box always get u = 1 and therefore always go to MANUAL-REVIEW.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from triad.evaluate import box_iou

INFERENCE_KEYS = ("preds", "width", "height", "census")


def _view(record):
    return {k: record.get(k) for k in INFERENCE_KEYS}


def features(record, tau):
    r = _view(record)
    conf = np.array([p["conf"] for p in r["preds"]], float)
    acc = conf[conf >= tau]
    rej = conf[conf < tau]
    acc_preds = [p for p in r["preds"] if p["conf"] >= tau]
    area = r["width"] * r["height"]
    covered = sum((p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]) for p in acc_preds) / max(area, 1)
    classes = [p["cls"] for p in acc_preds]
    if classes:
        _, n = np.unique(classes, return_counts=True)
        q = n / n.sum()
        entropy = float(-(q * np.log(q)).sum())
    else:
        entropy = 0.0
    f = {
        "n_accepted": float(len(acc)),
        "max_conf": float(acc.max()) if len(acc) else 0.0,
        "min_accepted": float(acc.min()) if len(acc) else 0.0,
        "mean_accepted": float(acc.mean()) if len(acc) else 0.0,
        "max_rejected": float(rej.max()) if len(rej) else 0.0,
        "grey_mass": float(rej[rej >= tau / 2].sum()),
        "n_grey": float((rej >= tau / 2).sum()),
        "area_covered": float(min(covered, 4.0)),
        "class_entropy": entropy,
        "n_classes": float(len(set(classes))),
    }
    census = r.get("census")
    if census:
        count = float(census["count"])
        f["census_count"] = count
        f["census_gap"] = max(count - len(acc), 0.0)
        f["census_rel_gap"] = min(max(count - len(acc), 0.0) / max(count, 1.0), 1.0)
        present = set(classes)
        f["unexplained_presence"] = max([p for c, p in census["presence"].items() if c not in present], default=0.0)
    return f


def base_scores(record, tau):
    f = features(record, tau)
    if f["n_accepted"] == 0:
        out = {"conf": 1.0, "weakest": 1.0, "grey": 1.0}
        if "census_count" in f:
            out["census"] = 1.0
        return out
    out = {"conf": 1 - f["max_conf"], "weakest": 1 - f["min_accepted"], "grey": min(f["max_rejected"] / tau, 1.0)}
    if "census_count" in f:
        out["census"] = max(f["census_rel_gap"], f["unexplained_presence"], 1 - f["max_conf"])
    return out


class StatsScore:
    """Logistic model P(any miss | detection statistics). Fit on VAL only; never on calib/test."""

    def __init__(self, tau):
        self.tau = tau
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
        self.keys = None

    def fit(self, records, losses):
        rows = [features(r, self.tau) for r in records]
        self.keys = sorted(rows[0])
        y = np.asarray(losses, int)
        if len(set(y)) < 2:
            self.model = None          # degenerate: no misses or all misses in val
            return self
        self.model.fit(np.array([[row[k] for k in self.keys] for row in rows]), y)
        return self

    def __call__(self, records):
        rows = [features(r, self.tau) for r in records]
        if self.model is None:
            return np.array([1 - row["max_conf"] if row["n_accepted"] else 1.0 for row in rows])
        p = self.model.predict_proba(np.array([[row[k] for k in self.keys] for row in rows]))[:, 1]
        return np.where([row["n_accepted"] == 0 for row in rows], 1.0, p)


def disagreement(record, others, tau, iou=0.5):
    """1 - mean class-agnostic F1 agreement between this model's accepted boxes and each other model's."""
    mine = [p["box"] for p in record["preds"] if p["conf"] >= tau[0]]
    if not mine:
        return 1.0
    values = []
    for other, t in zip(others, tau[1:]):
        theirs = [p["box"] for p in other["preds"] if p["conf"] >= t]
        if not theirs:
            values.append(0.0)
            continue
        overlap = box_iou(mine, theirs) >= iou
        a, b = overlap.any(1).sum(), overlap.any(0).sum()
        values.append((a + b) / (len(mine) + len(theirs)))
    return 1.0 - float(np.mean(values))
