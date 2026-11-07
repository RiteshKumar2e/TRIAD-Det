"""Re-run the trained YOLO-family checkpoints over every split of every dataset, in the unified layout.

The base pipeline saved TRIAD-YOLO predictions for all splits but transfer predictions only on the
target TEST split, and baseline predictions only on val/test. Recalibration needs target CALIB too,
so every checkpoint is run here over val/calib/test of every dataset (train too for in-domain).
No retraining happens here. (The SteelDefectX classifier is no longer part of the study.)

    python experiments/predict_yolo.py               # uses ../runs/triad_full
    python experiments/predict_yolo.py --variant smoke --pipeline-run ../runs/smoke
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATASETS, load_config, log, results_dir  # noqa: E402
from triad.common import Config, resolve_device, write_jsonl  # noqa: E402
from triad.data import load_final  # noqa: E402
from triad.detector import predict_detector  # noqa: E402

TAGS = {"TRIAD-YOLO": "best", "YOLOv8s": "YOLOv8s", "YOLO11s": "YOLO11s"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="full")
    parser.add_argument("--pipeline-run", default=None)
    parser.add_argument("--sources", default=None, help="comma list; default all datasets")
    parser.add_argument("--models", default=None, help="comma list of TRIAD-YOLO,YOLOv8s,YOLO11s")
    parser.add_argument("--skip-existing", action="store_true", help="keep prediction files that already exist")
    args = parser.parse_args()
    cfg = load_config()
    run = Path(args.pipeline_run).resolve() if args.pipeline_run else Path(cfg["pipeline_run"])
    # Rebuild the exact pipeline config the run used (resolved_config.json holds absolute paths).
    raw = json.loads((run / "manifests" / "resolved_config.json").read_text(encoding="utf-8"))
    raw["out_root"] = str(run)
    raw["device"] = resolve_device("cuda", False)
    pcfg = Config(raw)

    rows = load_final(pcfg)
    out_dir = results_dir(cfg, "predictions", args.variant)
    sources = args.sources.split(",") if args.sources else DATASETS
    models = args.models.split(",") if args.models else list(TAGS)
    for source in sources:
        for model, tag in ((m, TAGS[m]) for m in models):
            ckpt = run / source / "checkpoints" / f"detector_{tag}.pt"
            if not ckpt.exists():
                log(cfg, f"[predict_yolo] missing checkpoint {ckpt}; skipped")
                continue
            for target in DATASETS:
                if args.skip_existing and (out_dir / f"{model}__{source}__on__{target}.jsonl").exists():
                    continue
                splits = ("train", "val", "calib", "test") if target == source else ("val", "calib", "test")
                subset = [r for r in rows if r["dataset"] == target and r["split"] in splits and r["boxes"]]
                records = predict_detector(pcfg, source, subset, tag) if subset else []
                for r in records:
                    r["model"] = f"{model}:{source}"
                path = out_dir / f"{model}__{source}__on__{target}.jsonl"
                write_jsonl(path, records)
                log(cfg, f"[predict_yolo] {path.name}: {len(records)} records")
    log(cfg, f"[predict_yolo] done -> {out_dir}")


if __name__ == "__main__":
    main()
