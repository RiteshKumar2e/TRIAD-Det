"""Build TRIAD_det/OUTPUT: detection batch images for every model on every dataset, plus the study figures.

Every box comes from the saved in-domain prediction files in TRIAD_Det_Risk_Control/results/predictions/full
(TRIAD-Det there is v2). tau comes from tables/full/in_domain_detection.csv, matching uses IoU 0.5 with classes,
as in the test tables. Each model's TP/FP/MISS totals are checked against the precision and recall in that table.

    python tools/build_output_folder.py [--splits test] [--workers 4]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RISK = ROOT / "TRIAD_Det_Risk_Control" / "results"
OUT = ROOT / "OUTPUT"
DATASETS = ("SteelDefectX", "NEU-DET", "GC10-DET", "X-SDD")
MODELS = ("TRIAD-Det", "TRIAD-YOLO", "YOLOv8s", "YOLO11s")
MATCH_IOU = 0.5


def render(job):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from triad.common import DATASET_COLOURS, load_config
    from triad.viz import BATCH_SIZE, TITLE, _legend, _status, draw_tile

    model, dataset, splits, tau, precision, recall, variant, out = job
    out = Path(out)
    cfg = load_config(str(ROOT / "configs" / "default.yaml"), None, False)
    settings = cfg.figures
    path = RISK / "predictions" / variant / f"{model}__{dataset}__on__{dataset}.jsonl"
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    point = {"tau": tau, "iou": MATCH_IOU}
    tp = fp = miss = 0
    written = []
    for split in splits:
        subset = sorted((r for r in records if r["split"] == split), key=lambda r: (r["folder_class"], r["id"]))
        total = math.ceil(len(subset) / settings["per_batch"])
        index = {}
        for batch in range(total):
            chunk = subset[batch * settings["per_batch"]:(batch + 1) * settings["per_batch"]]
            fig, axes = plt.subplots(settings["rows"], settings["cols"], figsize=BATCH_SIZE)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.925, bottom=0.075, hspace=0.25, wspace=0.05)
            for ax in axes.flat[len(chunk):]:
                ax.axis("off")
            for ax, record in zip(axes.flat, chunk):
                draw_tile(ax, cfg, record, point)
                _, result, _ = _status(record, point)
                if result and split == "test":
                    tp += len(result["pairs"])
                    fp += len(result["fp"])
                    miss += len(result["missed"])
            fig.add_artist(Rectangle((0, 0.965), 1, 0.035, transform=fig.transFigure, color=DATASET_COLOURS[dataset]))
            fig.text(0.5, 0.9825, f"{dataset}  |  {model}  |  {split}  |  conf >= {tau}, IoU {MATCH_IOU}  |  "
                                  f"Batch {batch + 1}/{total} ({len(chunk)} images)",
                     ha="center", va="center", fontsize=TITLE, weight="bold", color="white")
            _legend(fig, dataset)
            name = f"{model}_{dataset}_{split}_batch_{batch + 1:03d}.png"
            target = out / "batches" / dataset / model / split / name
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=settings["dpi"])
            plt.close(fig)
            index[name] = [r["id"] for r in chunk]
            written.append(target)
        (out / "batches" / dataset / model / split / "batch_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    got_p = tp / (tp + fp) if tp + fp else 0.0
    got_r = tp / (tp + miss) if tp + miss else 0.0
    return {"dataset": dataset, "model": model, "tau": tau, "batches": len(written), "TP": tp, "FP": fp, "MISS": miss,
            "precision": round(got_p, 4), "table_precision": round(precision, 4),
            "recall": round(got_r, 4), "table_recall": round(recall, 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", default="test")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    with (RISK / "tables" / args.variant / "in_domain_detection.csv").open(encoding="utf-8") as handle:
        table = {(r["dataset"], r["model"]): r for r in csv.DictReader(handle)}
    jobs = [(m, d, args.splits.split(","), float(table[(d, m)]["tau"]), float(table[(d, m)]["precision"]),
             float(table[(d, m)]["recall"]), args.variant, args.out) for d in DATASETS for m in args.models.split(",")]
    out = Path(args.out)
    (out / "batches").mkdir(parents=True, exist_ok=True)
    rows = []
    with mp.Pool(args.workers) as pool:
        for row in pool.imap_unordered(render, jobs):
            rows.append(row)
            print(row, flush=True)
    rows.sort(key=lambda r: (DATASETS.index(r["dataset"]), MODELS.index(r["model"])))
    with (out / "batches" / "batch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    for source in (RISK / "figures" / args.variant).glob("*"):
        if source.is_file():
            shutil.copy2(source, figures / source.name)
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
