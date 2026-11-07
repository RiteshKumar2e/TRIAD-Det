"""Shared constants, configuration, seeding and file helpers."""
from __future__ import annotations

import json
import platform
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

DATASETS = ("SteelDefectX", "NEU-DET", "GC10-DET", "X-SDD")
DATASET_COLOURS = {"SteelDefectX": "#DC2626", "NEU-DET": "#2563EB", "GC10-DET": "#16A34A", "X-SDD": "#9333EA"}
# Every dataset now has box annotations, so every detector is trained on every dataset.
# SteelDefectX boxes come from the official pixel masks; X-SDD boxes from matched SteelDefectX copies
# (tools/build_detection_datasets.py). Folder names on disk:
DATA_FOLDERS = {"SteelDefectX": "SteelDefectX-DET", "NEU-DET": "NEU-DET", "GC10-DET": "GC10-DET", "X-SDD": "X-SDD-DET"}
DETECTOR_DATASETS = DATASETS
CLASSIFIER_DATASET = "SteelDefectX"   # legacy classifier code only; not part of the default stages
SPLITS = ("train", "val", "calib", "test")

# Where SteelDefectX classes overlap the two box-annotated datasets. Used for
# provenance matching and for scoring the classifier on NEU-DET / GC10-DET.
SDX_TO_NEU = {
    "Crazing": "crazing",
    "Inclusion": "inclusion",
    "Bright scratch": "scratches",
    "Patches": "patches",
    "Pitted surface": "pitted_surface",
    "Rolled in scale": "rolled-in_scale",
}
SDX_TO_GC10 = {
    "Crease": "crease",
    "Crescent gap": "crescent_gap",
    "Inclusion": "inclusion",
    "Oil spot": "oil_spot",
    "Punching": "punching_hole",
    "Rolled pit": "rolled_pit",
    "Silk spot": "silk_spot",
    "Waist folding": "waist_folding",
    "Welding line": "welding_line",
}
SDX_TEXT_ALIASES = {"Dark scratches": "Scratches"}
SDX_TO_XSDD = {
    "Finishing roll printing": "finishing roll printing",
    "Iron sheet ash": "iron sheet ash",
    "Oxide scale of plate system": "oxide scale of plate system",
    "Oxide scale of temperature system": "oxide scale of temperature system",
    "Red iron sheet": "red iron",
    "Slag inclusion": "slag inclusion",
    "Scratches": "surface scratch",
}


class Config(dict):
    """Dict with attribute access for nested configuration."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path, out_root=None, allow_cpu=False) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "base" in raw:
        base = yaml.safe_load((path.parent / raw.pop("base")).read_text(encoding="utf-8"))
        raw = _merge(base, raw)
    code_root = path.parent.parent
    raw["data_root"] = str((code_root / raw["data_root"]).resolve())
    raw["out_root"] = str(Path(out_root).resolve() if out_root else (code_root / raw["out_root"]).resolve())
    raw["config_path"] = str(path)
    raw["device"] = resolve_device(raw["device"], allow_cpu)
    return Config(raw)


def resolve_device(requested: str, allow_cpu: bool) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        if not allow_cpu:
            raise SystemExit(
                "CUDA is not available. Install a CUDA build of PyTorch on the GPU machine, "
                "or pass --allow-cpu for a test run."
            )
        return "cpu"
    return requested


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def out_path(cfg, *parts) -> Path:
    path = Path(cfg.out_root).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def dataset_dir(cfg, dataset: str) -> Path:
    path = Path(cfg.out_root) / dataset
    for sub in ("predictions", "batches", "metrics", "checkpoints"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default) + "\n")


def read_jsonl(path) -> list:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def imread(path, colour: bool = False) -> np.ndarray:
    """Read an image; works with non-ASCII Windows paths, fails loudly on bad files."""
    flag = cv2.IMREAD_COLOR if colour else cv2.IMREAD_GRAYSCALE
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if colour else image


def snapshot_run(cfg) -> None:
    """Save the resolved configuration, seed and software/hardware versions."""
    info = {
        "seed": cfg.seed,
        "smoke": cfg.smoke,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": cfg.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    write_json(out_path(cfg, "manifests", "run_environment.json"), info)
    write_json(out_path(cfg, "manifests", "resolved_config.json"), dict(cfg))
