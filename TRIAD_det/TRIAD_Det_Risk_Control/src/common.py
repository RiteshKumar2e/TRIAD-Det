"""Paths, configuration and shared I/O for the risk-control study."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]          # TRIAD_Det_Risk_Control/
sys.path.insert(0, str(PROJECT.parent))                # TRIAD_det/, for the shared `triad` package

DATASETS = ("SteelDefectX", "NEU-DET", "GC10-DET", "X-SDD")
BOX_DATASETS = DATASETS          # every dataset now has boxes (SteelDefectX from masks, X-SDD via SteelDefectX)
COLOURS = {"SteelDefectX": "#DC2626", "NEU-DET": "#2563EB", "GC10-DET": "#16A34A", "X-SDD": "#9333EA"}


def load_config(path=None) -> dict:
    path = Path(path) if path else PROJECT / "config" / "experiment_config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("pipeline_run", "data_root", "results"):
        cfg[key] = str((PROJECT / cfg[key]).resolve())
    return cfg


def results_dir(cfg, *parts) -> Path:
    path = Path(cfg["results"]).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        return
    columns = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v, default=_default) if isinstance(v, (list, dict, tuple)) else v)
                             for k, v in row.items()})


def log(cfg, message):
    """Append to results/logs/experiments.log and echo."""
    print(message, flush=True)
    with (results_dir(cfg, "logs") / "experiments.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
