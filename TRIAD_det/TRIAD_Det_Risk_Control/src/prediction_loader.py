"""Load image-level prediction records for every (model, source, target) combination.

File layout: results/predictions/<variant>/<MODEL>__<SOURCE>__on__<TARGET>.jsonl
  MODEL   TRIAD-Det | TRIAD-YOLO | YOLOv8s | YOLO11s | Classifier
  SOURCE  dataset the model was trained on
  TARGET  dataset the images come from (== SOURCE for in-domain)
Records carry `split`; in-domain files cover train/val/calib/test, transfer files val/calib/test.
"""
from __future__ import annotations

from pathlib import Path

from .common import read_jsonl

DETECTORS = ("TRIAD-Det", "TRIAD-YOLO", "YOLOv8s", "YOLO11s")


def pred_path(cfg, model, source, target, variant="full"):
    return Path(cfg["results"]) / "predictions" / variant / f"{model}__{source}__on__{target}.jsonl"


def available(cfg, variant="full"):
    folder = Path(cfg["results"]) / "predictions" / variant
    out = []
    for path in sorted(folder.glob("*__*__on__*.jsonl")):
        model, rest = path.stem.split("__", 1)
        source, target = rest.split("__on__")
        out.append((model, source, target))
    return out


def load(cfg, model, source, target, variant="full", splits=None):
    path = pred_path(cfg, model, source, target, variant)
    if not path.exists():
        return None
    records = read_jsonl(path)
    if splits:
        records = [r for r in records if r["split"] in splits]
    return records


def by_split(records):
    out = {}
    for r in records:
        out.setdefault(r["split"], []).append(r)
    return out
