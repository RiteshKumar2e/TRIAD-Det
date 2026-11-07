"""Ground-truth matching. One definition is used everywhere in this study.

A ground-truth box g is DETECTED by a prediction set P if, under greedy highest-confidence-first
one-to-one matching, some prediction p in P with conf(p) >= tau is assigned to g with
IoU(p, g) >= iou and (class-aware mode) cls(p) == cls(g). Otherwise g is MISSED.

Class-agnostic mode drops the class condition. It is used only for cross-dataset transfer,
where the source and target taxonomies differ.
"""
from __future__ import annotations

from triad.evaluate import box_iou, counts, kept, map_metrics, match  # noqa: F401  (re-exported)


def match_record(record, tau, iou, class_aware=True):
    """Matching result for one image record: pairs, false positives, missed GT indices."""
    return match(kept(record, tau), record["gt"], iou, class_aware)


def n_missed(record, tau, iou, class_aware=True):
    return len(match_record(record, tau, iou, class_aware)["missed"])
