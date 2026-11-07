"""Train one YOLO-family model on one dataset (used by tools/run_everything.py).

    python tools/train_one.py --dataset NEU-DET --model TRIAD-YOLO
    python tools/train_one.py --dataset GC10-DET --model YOLOv8s
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triad.common import load_config, seed_everything  # noqa: E402
from triad.detector import train_detector  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, help="TRIAD-YOLO, or a name in detector.baselines")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    if args.model == "TRIAD-YOLO":
        train_detector(cfg, args.dataset)
    else:
        train_detector(cfg, args.dataset, weights=cfg.detector["baselines"][args.model], tag=args.model)


if __name__ == "__main__":
    main()
