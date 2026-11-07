"""Run the TRIAD-Det pipeline, or selected stages.

    python run_all.py                                  full GPU run, writes to runs/triad_full
    python run_all.py --stages evaluate,figures        rebuild from saved predictions
    python run_all.py --config configs/smoke.yaml --out runs/smoke --allow-cpu
"""
from __future__ import annotations

import argparse
import time

from triad import data
from triad.common import (DATASETS, DETECTOR_DATASETS, dataset_dir, load_config, seed_everything,
                          out_path, read_jsonl, snapshot_run, write_json, write_jsonl)
from triad.detector import predict_detector, train_detector
from triad.evaluate import compare_detectors, evaluate_all
from triad.export import export_csv
from triad.viz import render_all, render_model_comparison

STAGES = ("index", "dedupe", "provenance", "split", "train", "predict", "evaluate", "figures", "compare", "export")


def train(cfg):
    """TRIAD-YOLO on every dataset. (The separate SteelDefectX classifier is no longer a stage:
    SteelDefectX now has mask-derived boxes and is trained like the other datasets.)"""
    for dataset in DETECTOR_DATASETS:
        train_detector(cfg, dataset)


def predict(cfg):
    """In-domain records on every split, and transfer records on every other dataset's test split.
    Transfer is scored class-agnostically, because the class lists differ between datasets."""
    rows = data.load_final(cfg)
    by_dataset = {d: [r for r in rows if r["dataset"] == d] for d in DATASETS}
    for dataset in DETECTOR_DATASETS:
        write_jsonl(dataset_dir(cfg, dataset) / "predictions" / "all.jsonl", predict_detector(cfg, dataset, by_dataset[dataset]))
        for target in DATASETS:
            if target != dataset:
                subset = [r for r in by_dataset[target] if r["split"] == "test" and r["boxes"]]
                write_jsonl(dataset_dir(cfg, target) / "predictions" / f"transfer_from_{dataset}.jsonl",
                            predict_detector(cfg, dataset, subset) if subset else [])


def compare(cfg):
    """Train the plain YOLOv8 and YOLO11 baselines and score them next to TRIAD-YOLO."""
    rows = data.load_final(cfg)
    results = []
    for dataset in DETECTOR_DATASETS:
        val = [r for r in rows if r["dataset"] == dataset and r["split"] == "val"]
        test = [r for r in rows if r["dataset"] == dataset and r["split"] == "test"]
        saved = read_jsonl(dataset_dir(cfg, dataset) / "predictions" / "all.jsonl")
        results.append(compare_detectors(cfg, dataset, "TRIAD-YOLO", [r for r in saved if r["split"] == "val"],
                                         [r for r in saved if r["split"] == "test"]))
        for name, weights in cfg.detector["baselines"].items():
            if not (dataset_dir(cfg, dataset) / "checkpoints" / f"detector_{name}.pt").exists():
                train_detector(cfg, dataset, weights=weights, tag=name)
            results.append(compare_detectors(cfg, dataset, name, predict_detector(cfg, dataset, val, name),
                                             predict_detector(cfg, dataset, test, name)))
    write_json(out_path(cfg, "COMBINED", "model_comparison.json"), results)
    render_model_comparison(cfg, results)


ACTIONS = {
    "index": data.build_index, "dedupe": data.find_duplicates, "provenance": data.match_gc10_crops,
    "split": data.make_splits, "train": train, "predict": predict, "evaluate": evaluate_all, "figures": render_all,
    "compare": compare, "export": export_csv,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--out", default=None, help="override out_root")
    parser.add_argument("--allow-cpu", action="store_true", help="run without CUDA (tests only)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.out, args.allow_cpu)
    seed_everything(cfg.seed)
    snapshot_run(cfg)
    for stage in args.stages.split(","):
        if stage not in ACTIONS:
            raise SystemExit(f"Unknown stage {stage}; choose from {STAGES}")
        started = time.time()
        print(f"=== {stage} ===", flush=True)
        ACTIONS[stage](cfg)
        print(f"=== {stage} finished in {time.time() - started:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
