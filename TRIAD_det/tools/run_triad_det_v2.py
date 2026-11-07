"""Finish the study with TRIAD-Det v2, unattended (replaces the rest of run_everything.py, stopped 2026-11-05 15:25).

  1. wait for a running predict_yolo process (if any), then finish the YOLO-family predictions, one process
     per (source, model) so GPU memory is released between models; finished files are kept
  2. archive TRIAD-Det v1 (checkpoints, logs, predictions) -> results/v1_archive and predictions/v1
  3. train TRIAD-Det v2 on X-SDD, NEU-DET, GC10-DET, SteelDefectX (wall-clock cap triad_det.max_hours each,
     so the run ends before the 2026-11-06 12:00 deadline)
  4. TRIAD-Det v2 predictions; experiments in_domain, cross_domain, recalibration, ablation, figures (variant full)
  5. in_domain on the archived v1 predictions (variant v1) and a v1 vs v2 table
  6. results.csv, then the base-pipeline stages that failed with OOM (predict, evaluate, figures, compare, export)
Model selection uses validation data only; calib and test are never used for any v2 choice.

    python tools/run_triad_det_v2.py [--wait-pid PID]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_everything as q  # noqa: E402
from rerun_pipeline_after_queue import alive  # noqa: E402

RES = q.RISK / "results"
ORDER = ("X-SDD", "NEU-DET", "GC10-DET", "SteelDefectX")
ALL = ("SteelDefectX", "NEU-DET", "GC10-DET", "X-SDD")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def archive_v1():
    dest = RES / "v1_archive"
    if (dest / "ARCHIVED").exists():
        return
    variant = RES / "predictions" / "v1"
    variant.mkdir(parents=True, exist_ok=True)
    for f in (RES / "predictions" / "full").glob("TRIAD-Det__*.jsonl"):
        shutil.copy2(f, variant / f.name)
    for d in ALL:
        for f in (RES / "checkpoints" / f"triad_det_{d}.pt", RES / "logs" / f"triad_det_{d}_train.csv"):
            if f.exists():
                (dest / f.parent.name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest / f.parent.name / f.name)
    (dest / "ARCHIVED").write_text(time.strftime("%Y-%m-%d %H:%M") + "\nv1 predictions: results/predictions/v1/\n",
                                   encoding="utf-8")
    q.say("TRIAD-Det v1 archived (results/v1_archive, results/predictions/v1)")


def v1_vs_v2():
    def read(path):
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as h:
            return {r["dataset"]: r for r in csv.DictReader(h) if r["model"] == "TRIAD-Det"}
    old, new = read(RES / "tables/v1/in_domain_detection.csv"), read(RES / "tables/full/in_domain_detection.csv")
    keys = ("mAP50", "mAP50_95", "precision", "recall", "f1", "any_miss_image_risk")
    rows = [{"dataset": d, **{f"{v}_{k}": src.get(d, {}).get(k) for k in keys for v, src in (("v1", old), ("v2", new))}}
            for d in ALL if d in old or d in new]
    if rows:
        out = RES / "tables/full/triad_det_v1_vs_v2.csv"
        with out.open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        q.say(f"wrote {out.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, default=None)
    args = parser.parse_args()
    q.say("run_triad_det_v2 started")
    if args.wait_pid:
        while alive(args.wait_pid):
            time.sleep(30)
    for source in ALL:
        for model in q.YOLO_MODELS:
            q.run(["experiments/predict_yolo.py", "--sources", source, "--models", model, "--skip-existing"],
                  cwd=q.RISK, name=f"YOLO-family predictions {model} from {source}")
    archive_v1()
    for d in ORDER:
        marker = RES / "checkpoints" / f"triad_det_v2_{d}.done"
        if not marker.exists():
            if q.run(["-m", "src.triad_det_model.train", "--dataset", d, "--train-only"], cwd=q.RISK,
                     name=f"train TRIAD-Det v2 on {d}"):
                marker.write_text("ok", encoding="utf-8")
    for d in ORDER:
        if (RES / "checkpoints" / f"triad_det_v2_{d}.done").exists():
            q.run(["-m", "src.triad_det_model.train", "--dataset", d, "--predict-only"], cwd=q.RISK,
                  name=f"TRIAD-Det v2 predictions from {d}")
    for script in ("in_domain", "cross_domain", "recalibration", "ablation", "figures"):
        q.run([f"experiments/{script}.py"], cwd=q.RISK, name=f"experiment {script}")
    q.run(["experiments/in_domain.py", "--variant", "v1", "--resplits", "100"], cwd=q.RISK,
          name="experiment in_domain on TRIAD-Det v1 (for v1 vs v2)")
    v1_vs_v2()
    q.refresh()
    q.run(["run_all.py", "--stages", "predict,evaluate,figures,compare,export"],
          name="pipeline predict/evaluate/figures/compare/export (re-run after OOM)")
    q.refresh()
    q.say("run_triad_det_v2 finished")


if __name__ == "__main__":
    main()
