"""Image-level missed-defect losses.

L_any(i)      = 1 if at least one ground-truth defect of image i is missed, else 0.
L_fraction(i) = (missed ground-truth defects) / (total ground-truth defects).

Both lie in [0, 1]. L_any is binary, so its conditional risk can be tested with an exact
binomial p-value. L_fraction is bounded but not binary, so it needs a bounded-loss p-value
(Hoeffding-Bentkus). L_fraction <= L_any always, so controlling L_any also controls L_fraction.

Images with zero ground-truth defects have no defined miss loss. They are NOT given loss 0:
they are returned as None and the caller excludes and counts them. (In NEU-DET and GC10-DET
every image has at least one annotated defect, so this case only arises for unannotated
SteelDefectX images, which are excluded from detection losses.)

SteelDefectX image-level loss for the classifier: L_cls(i) = 1 if the top-1 class is wrong.
It is a misclassification loss, not a missed-defect loss, and is always labelled as such.
"""
from __future__ import annotations

import numpy as np

from .detection_matching import n_missed


def loss_any(record, tau, iou, class_aware=True):
    if not record["gt"]:
        return None
    return int(n_missed(record, tau, iou, class_aware) > 0)


def loss_fraction(record, tau, iou, class_aware=True):
    if not record["gt"]:
        return None
    return n_missed(record, tau, iou, class_aware) / len(record["gt"])


def loss_classification(record):
    return int(record["top"][0]["cls"] != record["folder_class"])


LOSSES = {"any": loss_any, "fraction": loss_fraction}


def loss_array(records, kind, tau, iou, class_aware=True):
    """Returns (losses, keep_mask). keep_mask is False for zero-GT images."""
    values = [LOSSES[kind](r, tau, iou, class_aware) for r in records]
    keep = np.array([v is not None for v in values], bool)
    return np.array([v for v in values if v is not None], float), keep
