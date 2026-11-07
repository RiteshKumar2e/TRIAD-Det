"""All figures. Every mark comes from saved prediction records and annotations."""
from __future__ import annotations

import collections
import math
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from .common import (CLASSIFIER_DATASET, DATASET_COLOURS, DATASETS, DETECTOR_DATASETS, dataset_dir, imread, out_path, read_json,
                     read_jsonl, write_json)
from .evaluate import kept, match

BACKGROUND, PANEL, INK, MUTED, GRID = "#0d1117", "#161b22", "#e6edf3", "#8b949e", "#30363d"
BATCH_SIZE, CHART_SIZE = (20, 19.5), (14, 8)
TITLE, LABEL, SMALL = 16, 11, 7
SMOKE_NOTE = "SMOKE TEST: NOT A RESULT"

plt.rcParams.update({
    "figure.facecolor": BACKGROUND, "savefig.facecolor": BACKGROUND, "axes.facecolor": PANEL, "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": LABEL, "axes.grid": False, "legend.facecolor": PANEL, "legend.edgecolor": GRID,
})


def _stamp(fig, cfg):
    if cfg.smoke:
        fig.text(0.5, 0.5, SMOKE_NOTE, ha="center", va="center", fontsize=34, color=INK, alpha=0.35, rotation=20)


def _short(text, size=26):
    return text if len(text) <= size else text[:size - 1] + "…"


def _heuristic_mask(gray, boxes):
    """Pixels that stand out inside predicted boxes. Display aid only, never evaluated."""
    mask = np.zeros(gray.shape, bool)
    for x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = (max(int(v), 0) for v in (x0, y0, x1, y1))
        patch = gray[y0:y1, x0:x1].astype(np.float32)
        if patch.size:
            median = np.median(patch)
            spread = np.median(np.abs(patch - median)) + 1e-3
            mask[y0:y1, x0:x1] |= np.abs(patch - median) > 2.5 * spread
    return mask


def _status(record, point):
    """Per-box outcome and a one-line tile verdict."""
    preds = kept(record, point["tau"])
    is_classifier = record["model"] == "classifier"
    result = match(preds, record["gt"], point["iou"], class_aware=not is_classifier) if record["gt"] else None
    if is_classifier:
        right = record["top"][0]["cls"] == record["folder_class"]
        verdict = "class correct" if right else "class WRONG"
        if not record["gt"]:
            verdict += ", no GT box"
    else:
        verdict = f'TP {len(result["pairs"])}  FP {len(result["fp"])}  MISS {len(result["missed"])}'
    return preds, result, verdict


def draw_tile(ax, cfg, record, point):
    colour = DATASET_COLOURS[record["dataset"]]
    gray = imread(Path(cfg.data_root) / record["path"])
    height, width = gray.shape
    factor = 512 / max(height, width)
    small = cv2.resize(gray, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA)
    preds, result, verdict = _status(record, point)
    side = max(small.shape)
    pad_x, pad_y = (side - small.shape[1]) / 2, (side - small.shape[0]) / 2
    extent = (pad_x, pad_x + small.shape[1], pad_y + small.shape[0], pad_y)
    ax.imshow(small, cmap="gray", vmin=0, vmax=255, extent=extent, interpolation="bilinear")
    ax.set_xlim(0, side)
    ax.set_ylim(side, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    def scaled(box):
        return [box[0] * factor + pad_x, box[1] * factor + pad_y, box[2] * factor + pad_x, box[3] * factor + pad_y]

    if record["model"] == "classifier":
        cam = cv2.resize(np.asarray(record["cam"], np.float32), small.shape[::-1], interpolation=cv2.INTER_LINEAR)
        mask = cam >= cfg.classifier["cam_threshold"]
    else:
        mask = _heuristic_mask(small, [[v * factor for v in p["box"]] for p in preds])
    overlay = np.zeros((*small.shape, 4), np.float32)
    overlay[..., :3] = to_rgb(colour)
    overlay[..., 3] = mask * 0.32
    ax.imshow(overlay, extent=extent, interpolation="nearest")
    ax.set_xlim(0, side)
    ax.set_ylim(side, 0)

    false_positive = set(result["fp"]) if result else set()
    missed = set(result["missed"]) if result else set()
    for index, item in enumerate(record["gt"]):
        x0, y0, x1, y1 = scaled(item["box"])
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=2.2, ls="-", alpha=0.75))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=colour, lw=1.4, ls=(0, (4, 3))))
        if index in missed:
            ax.text(x0 + 3, y1 - 4, f'MISS {_short(item["cls"], 14)}', fontsize=SMALL - 1, color="white", weight="bold",
                    va="bottom", clip_on=True, bbox={"fc": "black", "ec": colour, "lw": 1, "pad": 1.2})
    ranked = sorted(range(len(preds)), key=lambda i: -preds[i]["conf"])
    for rank, index in enumerate(ranked):
        item = preds[index]
        x0, y0, x1, y1 = scaled(item["box"])
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=colour, lw=1.8))
        if rank < 4 or index in false_positive:
            prefix = "FP " if index in false_positive else ""
            ax.text(x0 + 2, y0 + 3, f'{prefix}{_short(item["cls"], 14)} {item["conf"]:.0%}', fontsize=SMALL - 1,
                    color="white", weight="bold", va="top", clip_on=True, bbox={"fc": colour, "ec": "none", "pad": 1.2})

    truth = ", ".join(sorted({g["cls"] for g in record["gt"]}) or record["gt_labels"])
    if record["model"] == "classifier":
        guess = f'{record["top"][0]["cls"]} {record["top"][0]["p"]:.0%}'
    elif preds:
        best = max(preds, key=lambda p: p["conf"])
        guess = f'{best["cls"]} {best["conf"]:.0%}'
    else:
        guess = "no defect found"
    ax.set_title(f'GT: {_short(truth)}\nPred: {_short(guess)}\n{verdict}', fontsize=SMALL + 1, color=INK, pad=3, linespacing=1.15)
    for spine in ax.spines.values():
        spine.set_edgecolor(colour)
        spine.set_linewidth(1.6)


def _legend(fig, dataset):
    colour = DATASET_COLOURS[dataset]
    overlay = "In-box contrast overlay (display only)"
    handles = [
        Patch(fc=colour, ec="none", label=f"Dataset: {dataset}"),
        Line2D([], [], color=colour, lw=2.4, label="Prediction box: solid + class + confidence"),
        Line2D([], [], color=colour, lw=2.4, ls=(0, (4, 3)), label="Ground truth box: dashed, white edge"),
        Patch(fc=colour, ec="none", alpha=0.35, label=overlay),
        Patch(fc=colour, ec="none", label="FP tag: false positive"),
        Patch(fc="black", ec=colour, label="MISS tag: missed defect"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=LABEL, frameon=True, bbox_to_anchor=(0.5, 0.008))


def render_batches(cfg, dataset):
    folder = dataset_dir(cfg, dataset)
    records = read_jsonl(folder / "predictions" / "all.jsonl")
    point = read_json(folder / "metrics" / "operating_point.json")
    settings = cfg.figures
    per_batch, rows, cols = settings["per_batch"], settings["rows"], settings["cols"]
    index, rendered = {}, 0
    for split in settings["splits"]:
        subset = sorted((r for r in records if r["split"] == split), key=lambda r: (r["folder_class"], r["id"]))
        total = math.ceil(len(subset) / per_batch)
        note = "  (training images, not held out)" if split == "train" else ""
        for batch in range(total):
            chunk = subset[batch * per_batch:(batch + 1) * per_batch]
            fig, axes = plt.subplots(rows, cols, figsize=BATCH_SIZE)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.925, bottom=0.075, hspace=0.25, wspace=0.05)
            for ax in axes.flat[len(chunk):]:
                ax.axis("off")
            for ax, record in zip(axes.flat, chunk):
                draw_tile(ax, cfg, record, point)
            fig.add_artist(Rectangle((0, 0.965), 1, 0.035, transform=fig.transFigure, color=DATASET_COLOURS[dataset]))
            fig.text(0.5, 0.9825, f"{dataset}  |  TRIAD-Det  |  {split}{note}  |  Batch {batch + 1}/{total} ({len(chunk)} images)",
                     ha="center", va="center", fontsize=TITLE, weight="bold", color="white")
            _legend(fig, dataset)
            _stamp(fig, cfg)
            name = f"TRIAD_{dataset}_{split}_batch_{batch + 1:03d}.png"
            target = folder / "batches" / split / name
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=settings["dpi"])
            plt.close(fig)
            index[f"{split}/{name}"] = [r["id"] for r in chunk]
            rendered += len(chunk)
    if rendered != len(records):
        raise SystemExit(f"{dataset}: rendered {rendered} of {len(records)} images")
    write_json(folder / "batches" / "batch_index.json", index)
    print(f"{dataset}: {rendered} images in {len(index)} batches", flush=True)


def _sequence_map(colour):
    return LinearSegmentedColormap.from_list("dataset", [PANEL, colour])


def render_confusion(cfg, dataset):
    folder = dataset_dir(cfg, dataset)
    data = read_json(folder / "metrics" / "confusion_matrix.json")
    matrix = np.array(data["matrix"], float)
    rates = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    labels = data["labels"]
    fig, ax = plt.subplots(figsize=(13, 12))
    image = ax.imshow(rates, cmap=_sequence_map(DATASET_COLOURS[dataset]), vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    for y, x in zip(*np.nonzero(matrix)):
        ax.text(x, y, int(matrix[y, x]), ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(image, ax=ax, fraction=0.04, label="Share of row")
    ax.set_title(f"{dataset}: confusion matrix (test split, counts in cells)", fontsize=TITLE, weight="bold")
    _stamp(fig, cfg)
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix.png", dpi=cfg.figures["dpi"])
    plt.close(fig)


def render_accuracy(cfg, dataset, summary):
    folder = dataset_dir(cfg, dataset)
    report = read_json(folder / "classification_report.json")["per_class"]
    item = summary["datasets"][dataset]
    names = [n for n in report if n not in ("accuracy", "macro avg", "weighted avg", "micro avg")]
    recall = [report[n]["recall"] * 100 for n in names]
    support = [report[n]["support"] for n in names]
    order = np.argsort(recall)
    fig, (left, right) = plt.subplots(1, 2, figsize=CHART_SIZE, gridspec_kw={"width_ratios": [2.3, 1]})
    colour = DATASET_COLOURS[dataset]
    left.barh(range(len(names)), [recall[i] for i in order], color=colour, height=0.72)
    left.set_yticks(range(len(names)), [f"{names[i]} (n={int(support[i])})" for i in order], fontsize=9)
    left.set_xlim(0, 112)
    left.set_xlabel("Recall on the test split (%)")
    left.xaxis.grid(True, color=GRID, lw=0.6)
    left.set_axisbelow(True)
    for row, i in enumerate(order):
        left.text(recall[i] + 1, row, f"{recall[i]:.1f}", va="center", fontsize=8, color=INK)
    left.set_title("Per-class recall", weight="bold")

    test = item["test"]
    lines = ([("Accuracy", item["accuracy"]), ("Macro F1", item["macro_f1"])] if "accuracy" in item else
             [("mAP@.5:.95", item["mAP"]), ("AP50", item["AP50"]), ("Precision", item["precision"]), ("Recall", item["recall"])])
    lines += [(f'Auto-accept (α={item["headline_alpha"]})', test["acceptance_rate"]), ("Manual review", test["manual_review_rate"])]
    risk = test["risk_among_accepted"]
    right.axis("off")
    right.set_title("Test-split summary", weight="bold")
    y = 0.92
    for label, value in lines:
        right.text(0.02, y, label, fontsize=LABEL, color=MUTED, transform=right.transAxes)
        right.text(0.98, y, f"{value:.3f}", fontsize=TITLE, ha="right", weight="bold", transform=right.transAxes)
        y -= 0.115
    right.text(0.02, y, "Risk among accepted", fontsize=LABEL, color=MUTED, transform=right.transAxes)
    right.text(0.98, y, "none accepted" if risk is None else f"{risk:.3f}", fontsize=TITLE, ha="right", weight="bold",
               transform=right.transAxes)
    fig.suptitle(f"{dataset}: accuracy and detection summary", fontsize=TITLE, weight="bold")
    _stamp(fig, cfg)
    fig.tight_layout()
    fig.savefig(folder / "accuracy_summary.png", dpi=cfg.figures["dpi"])
    plt.close(fig)


def render_unified(cfg):
    cols = cfg.figures["cols"]
    fig, axes = plt.subplots(len(DATASETS), cols, figsize=(20, 3.9 * len(DATASETS)))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.895, bottom=0.085, hspace=0.3, wspace=0.05)
    rng = random.Random(cfg.seed)
    for row, dataset in zip(axes, DATASETS):
        folder = dataset_dir(cfg, dataset)
        point = read_json(folder / "metrics" / "operating_point.json")
        records = [r for r in read_jsonl(folder / "predictions" / "all.jsonl") if r["split"] == "test"]
        for ax, record in zip(row, rng.sample(records, min(cols, len(records)))):
            draw_tile(ax, cfg, record, point)
    fig.text(0.5, 0.965, "TRIAD-YOLO: test-split predictions (seeded random sample, not selected by outcome)", ha="center",
             fontsize=TITLE, weight="bold")
    handles = [Patch(fc=DATASET_COLOURS[d], label=d) for d in DATASETS]
    handles += [Line2D([], [], color=MUTED, lw=2.4, label="Prediction: solid"),
                Line2D([], [], color=MUTED, lw=2.4, ls=(0, (4, 3)), label="Ground truth: dashed")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=LABEL, bbox_to_anchor=(0.5, 0.01))
    _stamp(fig, cfg)
    fig.savefig(out_path(cfg, "COMBINED", "unified_predictions.png"), dpi=cfg.figures["dpi"])
    plt.close(fig)


def render_colour_legend(cfg):
    fig, ax = plt.subplots(figsize=(8, 1 + len(DATASETS)))
    ax.axis("off")
    for row, dataset in enumerate(DATASETS):
        y = 0.78 - row * (0.85 / len(DATASETS))
        ax.add_patch(Rectangle((0.03, y), 0.14, 0.15, color=DATASET_COLOURS[dataset], transform=ax.transAxes))
        ax.text(0.21, y + 0.1, dataset, fontsize=TITLE, va="center", weight="bold", transform=ax.transAxes)
        ax.text(0.97, y + 0.1, DATASET_COLOURS[dataset], fontsize=LABEL, va="center", ha="right", color=MUTED,
                transform=ax.transAxes)
    fig.savefig(out_path(cfg, "COMBINED", "dataset_colour_legend.png"), dpi=cfg.figures["dpi"])
    plt.close(fig)


def render_risk_coverage(cfg, summary):
    fig, ax = plt.subplots(figsize=CHART_SIZE)
    for dataset in DATASETS:
        folder = dataset_dir(cfg, dataset)
        curve = read_json(folder / "metrics" / "risk_coverage_curve.json")
        colour = DATASET_COLOURS[dataset]
        ax.plot(curve["coverage"], curve["risk"], color=colour, lw=2, label=dataset)
        if curve["coverage"]:
            ax.annotate(dataset, (curve["coverage"][-1], curve["risk"][-1]), xytext=(6, 0), textcoords="offset points",
                        fontsize=LABEL, color=INK, va="center")
        test = summary["datasets"][dataset]["test"]
        if test["n_accepted"]:
            ax.plot(test["acceptance_rate"], test["risk_among_accepted"], "o", ms=9, color=colour, mec=PANEL, mew=2)
    alpha = cfg.risk["headline_alpha"]
    ax.axhline(alpha, color=MUTED, ls="--", lw=1.2, label=f"Target risk α = {alpha}")
    ax.plot([], [], "o", ms=9, color=MUTED, label="Calibrated operating point")
    ax.set_xlim(0, 1.12)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Coverage: share of test images auto-accepted")
    ax.set_ylabel("Missed-defect risk among accepted images")
    ax.grid(True, color=GRID, lw=0.6)
    ax.legend(loc="upper left")
    ax.set_title("Risk-coverage on each dataset's own test split\n"
                 "Loss on every dataset: at least one ground-truth defect box missed (IoU 0.5, class-aware).",
                 fontsize=TITLE - 2, weight="bold")
    _stamp(fig, cfg)
    fig.tight_layout()
    fig.savefig(out_path(cfg, "COMBINED", "risk_coverage.png"), dpi=cfg.figures["dpi"])
    plt.close(fig)


def render_comparison(cfg, summary):
    measures = (("acceptance_rate", "Auto-accept rate"), ("manual_review_rate", "Manual review rate"),
                ("risk_among_accepted", "Risk among accepted"))
    fig, axes = plt.subplots(1, len(measures), figsize=CHART_SIZE, sharey=True)
    for ax, (key, title) in zip(axes, measures):
        values = [summary["datasets"][d]["test"][key] or 0.0 for d in DATASETS]
        ax.bar(DATASETS, values, color=[DATASET_COLOURS[d] for d in DATASETS], width=0.66)
        for x, value in enumerate(values):
            ax.text(x, value + 0.015, f"{value:.3f}", ha="center", fontsize=LABEL)
        ax.set_title(title, weight="bold")
        ax.set_ylim(0, 1.1)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle(f'Cross-dataset comparison at α = {cfg.risk["headline_alpha"]} (test splits)', fontsize=TITLE, weight="bold")
    _stamp(fig, cfg)
    fig.tight_layout()
    fig.savefig(out_path(cfg, "COMBINED", "cross_dataset_comparison.png"), dpi=cfg.figures["dpi"])
    plt.close(fig)


def render_transfer(cfg, summary):
    rows = [r for r in summary["transfer"] if "AP50" in r]
    sources = sorted({r["source"] for r in rows}, key=DATASETS.index)
    targets = sorted({r["target"] for r in rows}, key=DATASETS.index)
    panels = (("AP50", "AP50, class-agnostic"), ("recall", "Recall at source τ"), ("risk", "Risk among accepted (no guarantee)"))
    fig, axes = plt.subplots(1, 3, figsize=CHART_SIZE)
    cmap = LinearSegmentedColormap.from_list("neutral", [PANEL, "#c9d1d9"])
    for ax, (key, title) in zip(axes, panels):
        grid = np.full((len(sources), len(targets)), np.nan)
        for row in rows:
            value = {"AP50": row["AP50"], "recall": row["at_tau"]["recall"], "risk": row["selective"]["risk_among_accepted"]}[key]
            if value is not None:
                grid[sources.index(row["source"]), targets.index(row["target"])] = value
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(targets)), targets, rotation=25, ha="right", fontsize=9)
        ax.set_yticks(range(len(sources)), sources, fontsize=9)
        ax.set_xlabel("Tested on")
        ax.set_ylabel("Trained on")
        ax.set_title(title, weight="bold", fontsize=LABEL)
        for y, x in np.ndindex(grid.shape):
            text = "n/a" if np.isnan(grid[y, x]) else f"{grid[y, x]:.3f}"
            dark = not np.isnan(grid[y, x]) and grid[y, x] > 0.55
            ax.text(x, y, text, ha="center", va="center", fontsize=LABEL, color=BACKGROUND if dark else INK)
    fig.suptitle("Cross-dataset transfer of detectors (class-agnostic; pairs with SteelDefectX are not independent)", fontsize=TITLE - 1, weight="bold")
    _stamp(fig, cfg)
    fig.tight_layout()
    fig.savefig(out_path(cfg, "COMBINED", "cross_dataset_transfer.png"), dpi=cfg.figures["dpi"])
    plt.close(fig)


def render_model_comparison(cfg, rows):
    fig, axes = plt.subplots(1, len(DETECTOR_DATASETS), figsize=CHART_SIZE, sharey=True)
    for ax, dataset in zip(axes, DETECTOR_DATASETS):
        subset = [r for r in rows if r["dataset"] == dataset]
        positions = np.arange(len(subset))
        ax.bar(positions - 0.2, [r["AP50"] for r in subset], 0.38, color=DATASET_COLOURS[dataset], label="AP50")
        ax.bar(positions + 0.2, [r["mAP"] for r in subset], 0.38, color=DATASET_COLOURS[dataset], alpha=0.5, label="mAP@.5:.95")
        for x, row in zip(positions, subset):
            ax.text(x - 0.2, row["AP50"] + 0.012, f'{row["AP50"]:.3f}', ha="center", fontsize=9)
            ax.text(x + 0.2, row["mAP"] + 0.012, f'{row["mAP"]:.3f}', ha="center", fontsize=9)
        ax.set_xticks(positions, [r["model"] for r in subset])
        ax.set_ylim(0, 1.08)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(dataset, weight="bold")
        ax.legend(loc="upper right")
    fig.suptitle("TRIAD-YOLO versus baselines (test splits, one training seed)", fontsize=TITLE, weight="bold")
    _stamp(fig, cfg)
    fig.tight_layout()
    fig.savefig(out_path(cfg, "COMBINED", "model_comparison.png"), dpi=cfg.figures["dpi"])
    plt.close(fig)


def render_all(cfg):
    summary = read_json(out_path(cfg, "COMBINED", "summary.json"))
    render_colour_legend(cfg)
    render_risk_coverage(cfg, summary)
    render_comparison(cfg, summary)
    render_transfer(cfg, summary)
    render_unified(cfg)
    for dataset in DATASETS:
        render_confusion(cfg, dataset)
        render_accuracy(cfg, dataset, summary)
        render_batches(cfg, dataset)
