"""Run the whole study unattended, in order, skipping anything already finished.

  A. data stages for the anchor datasets (NEU-DET, GC10-DET)        -> splits fixed
  B. train TRIAD-YOLO, YOLOv8s, YOLO11s, TRIAD-Det on NEU-DET and GC10-DET
     (between jobs: as soon as the SteelDefectX download is complete, build SteelDefectX-DET and
      X-SDD-DET and re-run the data stages for all four datasets; anchor splits are guarded)
  C. train the four models on SteelDefectX and X-SDD
  D. predictions, evaluation, figures, comparison, CSV export, TRIAD-Det predictions,
     risk-control experiments
results.csv is refreshed every 10 minutes and after every job. Each job runs in its own process,
so one failure is logged and the queue continues. Re-running this script resumes where it stopped.

    python tools/run_everything.py            (from TRIAD_det/)
"""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                 # TRIAD_det/
DATA = ROOT.parent                                          # folder with the datasets
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
RUN = ROOT / "runs" / "triad_full"
RISK = ROOT / "TRIAD_Det_Risk_Control"
LOG = ROOT / "runs" / "run_everything.log"
DOWNLOAD = DATA / "_downloads" / "SteelDefectX_hf"
DOWNLOAD_LOG = DATA / "_downloads" / "sdx_download.log"
XSDD_SRC = DATA / "_downloads" / "X-SDD_repo" / "X-SDD" / "datas"

ANCHORS = ("NEU-DET", "GC10-DET")
NEW = ("SteelDefectX", "X-SDD")
YOLO_MODELS = {"TRIAD-YOLO": "best", "YOLOv8s": "YOLOv8s", "YOLO11s": "YOLO11s"}


def say(message):
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(args, cwd=ROOT, name=None):
    name = name or " ".join(args[-4:])
    say(f"START {name}")
    started = time.time()
    with (ROOT / "runs" / "jobs.log").open("a", encoding="utf-8") as out:
        out.write(f"\n===== {dt.datetime.now():%Y-%m-%d %H:%M:%S} {name} =====\n")
        out.flush()
        code = subprocess.call([PY, *args], cwd=str(cwd), stdout=out, stderr=subprocess.STDOUT,
                               env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})
    say(f"{'DONE ' if code == 0 else 'FAILED'} {name} ({(time.time() - started) / 60:.1f} min, exit {code})")
    refresh()
    return code == 0


def refresh():
    subprocess.call([PY, "-m", "triad.results_table"], cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def refresher():
    while True:
        time.sleep(600)
        refresh()


# ------------------------------------------------------------------ state checks
def yolo_done(dataset, model):
    return (RUN / dataset / "checkpoints" / f"detector_{YOLO_MODELS[model]}.pt").exists()


def triad_det_done(dataset):
    return (RISK / "results" / "checkpoints" / f"triad_det_{dataset}.done").exists()


def download_complete():
    """The download command prints a line starting with 'done' when snapshot_download returns."""
    if not DOWNLOAD_LOG.exists():
        return False
    text = DOWNLOAD_LOG.read_text(encoding="utf-8", errors="ignore")
    if not any(line.startswith("done ") for line in text.splitlines()):
        return False
    counts = [len(list((DOWNLOAD / d).glob("*"))) for d in ("train", "train_mask", "val", "val_mask")]
    return all(counts) and counts[0] == counts[1] and counts[2] == counts[3]


def new_data_ready():
    return all((DATA / f / "build_report.json").exists() for f in ("SteelDefectX-DET", "X-SDD-DET"))


def phase_b_done():
    split = RUN / "manifests" / "split_report.json"
    if not split.exists():
        return False
    import json
    per = json.loads(split.read_text(encoding="utf-8"))["per_split"]
    return all(per.get(d) for d in NEW)


def maybe_phase_b():
    """Build the new datasets and redo the data stages once the download is complete."""
    if phase_b_done():
        return True
    if not new_data_ready():
        if not download_complete():
            return False
        if not run(["tools/build_detection_datasets.py", "--sdx", str(DOWNLOAD), "--xsdd", str(XSDD_SRC), "--out", str(DATA)],
                   name="build SteelDefectX-DET and X-SDD-DET"):
            return False
    return run(["run_all.py", "--stages", "index,dedupe,provenance,split"], name="data stages, all four datasets")


# ------------------------------------------------------------------ jobs
def train_models(dataset):
    for model in YOLO_MODELS:
        if not yolo_done(dataset, model):
            run(["tools/train_one.py", "--dataset", dataset, "--model", model], name=f"train {model} on {dataset}")
        maybe_phase_b()
    if not triad_det_done(dataset):
        if run(["-m", "src.triad_det_model.train", "--dataset", dataset, "--train-only"], cwd=RISK,
               name=f"train TRIAD-Det on {dataset}"):
            (RISK / "results" / "checkpoints" / f"triad_det_{dataset}.done").write_text("ok", encoding="utf-8")
    maybe_phase_b()


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    say("run_everything started")
    threading.Thread(target=refresher, daemon=True).start()

    if not (RUN / "manifests" / "anchor_splits.json").exists():
        run(["run_all.py", "--stages", "index,dedupe,provenance,split"], name="data stages, anchor datasets")
    for dataset in ANCHORS:
        train_models(dataset)

    while not maybe_phase_b():
        say("waiting for the SteelDefectX download / dataset build ...")
        time.sleep(300)
    for dataset in NEW:
        train_models(dataset)

    run(["run_all.py", "--stages", "predict,evaluate,figures,compare,export"], name="pipeline predict/evaluate/figures/compare/export")
    for dataset in (*ANCHORS, *NEW):
        run(["-m", "src.triad_det_model.train", "--dataset", dataset, "--predict-only"], cwd=RISK,
            name=f"TRIAD-Det predictions from {dataset}")
    run(["experiments/predict_yolo.py"], cwd=RISK, name="YOLO-family predictions, all splits")
    for script in ("in_domain", "cross_domain", "recalibration", "ablation", "figures"):
        if (RISK / "experiments" / f"{script}.py").exists():
            run([f"experiments/{script}.py"], cwd=RISK, name=f"experiment {script}")
    refresh()
    say("run_everything finished")


if __name__ == "__main__":
    main()
