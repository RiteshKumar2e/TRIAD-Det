"""Shared set-up for the cross-domain, recalibration and ablation experiments.

A `Prepared` object holds, for one (model, source, target, matching rule):
  tau      box threshold chosen on SOURCE val (F1), never on target data
  bank     candidate scores, stats score fitted on SOURCE val
  splits   target records per split (zero-GT images removed and counted)
  scores   {split: {score_name: array}}
  losses   {split: L_any array}
Source val/calib (in-domain records of the same model) are kept under `src_*`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import calibration as cal
from .evaluation import ScoreBank, choose_tau, pick_score
from .prediction_loader import by_split, load
from .risk_losses import loss_array

SPLITS = ("val", "calib", "test")


def with_gt(records):
    return [r for r in records if r.get("gt")]


@dataclass
class Prepared:
    model: str
    source: str
    target: str
    iou: float
    class_aware: bool
    tau: float
    chosen: str
    val_aurc: dict
    src: dict = field(default_factory=dict)      # source split -> records
    tgt: dict = field(default_factory=dict)      # target split -> records
    src_scores: dict = field(default_factory=dict)
    src_losses: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    losses: dict = field(default_factory=dict)
    dropped_zero_gt: int = 0


def prepare(cfg, model, source, target, variant, iou=None, class_aware=None, loss_kind="any"):
    """Returns Prepared or None if a file / split is missing."""
    iou = cfg["risk"]["match_iou"] if iou is None else iou
    class_aware = (source == target) if class_aware is None else class_aware
    src_all = load(cfg, model, source, source, variant)
    tgt_all = src_all if source == target else load(cfg, model, source, target, variant)
    if not src_all or not tgt_all:
        return None
    src, tgt = by_split(src_all), by_split(tgt_all)
    if not all(k in src for k in ("val", "calib")) or not all(k in tgt for k in SPLITS):
        return None
    dropped = sum(len(tgt[k]) - len(with_gt(tgt[k])) for k in SPLITS)
    src = {k: with_gt(src[k]) for k in ("val", "calib")}
    tgt = {k: with_gt(tgt[k]) for k in SPLITS}
    if not src["val"] or not all(tgt[k] for k in SPLITS):
        return None
    tau = choose_tau(src["val"], iou, cfg["risk"]["tau_grid"], class_aware)
    bank = ScoreBank(src["val"], tau, iou, class_aware)
    p = Prepared(model, source, target, iou, class_aware, tau, "", {}, src, tgt, dropped_zero_gt=dropped)
    for k in ("val", "calib"):
        p.src_scores[k] = bank(src[k])
        p.src_losses[k] = loss_array(src[k], loss_kind, tau, iou, class_aware)[0]
    for k in SPLITS:
        p.scores[k] = bank(tgt[k])
        p.losses[k] = loss_array(tgt[k], loss_kind, tau, iou, class_aware)[0]
    p.chosen, p.val_aurc = pick_score(p.src_scores["val"], p.src_losses["val"])
    return p


def ltt_threshold(u_val, l_val, u_cal, l_cal, alpha, delta, grid_size, grid=None):
    """LTT with exact binomial p-values (binary loss); grid ordered from the given val data."""
    if grid is None:
        grid = cal.threshold_grid(u_val, l_val.astype(int), len(u_cal), alpha, delta, grid_size)
    return cal.ltt(u_cal, l_cal, grid, alpha, delta, "binomial")


def test_row(u_test, l_test, threshold, alpha):
    ev = cal.evaluate(u_test, l_test, threshold)
    ci = ev.pop("risk_accepted_ci95")
    ev["risk_ci95_low"], ev["risk_ci95_high"] = (ci if ci else (None, None))
    ev["violated"] = (ev["risk_accepted"] is not None and ev["risk_accepted"] > alpha)
    return ev


def summarise(values):
    v = np.array([x for x in values if x is not None], float)
    return (float(v.mean()) if len(v) else None, float(v.std(ddof=1)) if len(v) > 1 else None)
