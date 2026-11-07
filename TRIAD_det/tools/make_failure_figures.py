"""Failure-case figures for the manuscript, from the saved TRIAD-Det test predictions (full variant). No model is run.

fig13_failures_SteelDefectX   the only dataset where TRIAD-Det is certified at alpha = 0.10; three rows:
                              (i) AUTO-ACCEPT with a missed defect (the residual risk the guarantee bounds),
                              (ii) MANUAL-REVIEW with a missed defect (caught by review),
                              (iii) MANUAL-REVIEW with every defect detected (review that was not needed)
fig14_miss_types              one row per dataset, one column per reason a ground-truth box was missed:
                              no prediction overlaps it / a prediction overlaps with the wrong class /
                              the prediction has the right class but IoU < 0.5

A ground-truth box counts as missed under the study's own rule (class-aware greedy matching, IoU 0.5, box
threshold tau from in_domain_summary.json); the recomputed miss flag is checked against the decision CSV.
Images are drawn at random with a fixed seed from each category; the figure does not look at image content.

    .venv/Scripts/python.exe tools/make_failure_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "TRIAD_Det_Risk_Control"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT))
from src.common import load_config  # noqa: E402
from src.detection_matching import box_iou, match_record  # noqa: E402
from src.prediction_loader import load  # noqa: E402
from triad.common import imread  # noqa: E402

MODEL, VARIANT, SEED = "TRIAD-Det", "full", 0
DATASETS = ("SteelDefectX", "NEU-DET", "GC10-DET", "X-SDD")
OUT = ROOT / "failure_cases"
GREEN, ORANGE, RED = "#16A34A", "#F97316", "#DC2626"
REASONS = ("no prediction, IoU < 0.1", "wrong class, IoU ≥ 0.5", "poor box, 0.1 ≤ IoU < 0.5")
SHORT = dict(zip(REASONS, ("none", "class", "box")))
plt.rcParams.update({"font.family": "serif", "font.size": 8})


def rows(path):
    with open(path, encoding="utf-8") as h:
        return list(csv.DictReader(h))


def reason(rec, kept_preds, j, iou):
    """Why ground-truth box j was missed."""
    if not kept_preds:
        return REASONS[0]
    overlap = box_iou([p["box"] for p in kept_preds], [rec["gt"][j]["box"]])[:, 0]
    if any(o >= iou and p["cls"] != rec["gt"][j]["cls"] for o, p in zip(overlap, kept_preds)):
        return REASONS[1]
    if any(o >= 0.1 for o in overlap):
        return REASONS[2]
    return REASONS[0]


class Data:
    def __init__(self, cfg, dataset, summary):
        self.cfg, self.dataset = cfg, dataset
        self.tau = summary[(MODEL, dataset)]["tau"]
        self.iou = cfg["risk"]["match_iou"]
        self.dec = {r["id"]: r for r in rows(PROJECT / "results/predictions/decisions" / f"{MODEL}__{dataset}__{VARIANT}.csv")}
        self.recs = {r["id"]: r for r in load(cfg, MODEL, dataset, dataset, VARIANT, splits=("test",)) if r["id"] in self.dec}
        self.info = {}
        for i, rec in self.recs.items():
            kept_preds = [p for p in rec["preds"] if p["conf"] >= self.tau]
            missed = match_record(rec, self.tau, self.iou)["missed"]
            assert int(bool(missed)) == int(self.dec[i]["loss_any"]), (dataset, i)  # same rule as the paper
            self.info[i] = {"missed": missed, "reasons": {j: reason(rec, kept_preds, j, self.iou) for j in missed}}

    def draw(self, ax, i, title, colour):
        rec, dec = self.recs[i], self.dec[i]
        img = imread(Path(self.cfg["data_root"]) / rec["path"], colour=True)
        ax.imshow(img)
        missed = set(self.info[i]["missed"])
        for j, g in enumerate(rec["gt"]):
            x0, y0, x1, y1 = g["box"]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       ec=ORANGE if j in missed else GREEN, lw=1.8 if j in missed else 1.2))
        for p in rec["preds"]:
            if p["conf"] >= self.tau:
                x0, y0, x1, y1 = p["box"]
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=RED, lw=1.0, ls="--"))
        why = sorted({SHORT[w] for w in self.info[i]["reasons"].values()})
        tag = f" ({', '.join(why)})" if why else ""
        ax.set_title(f"{title}u = {float(dec['u']):.2f}, missed {len(missed)}/{len(rec['gt'])}{tag}", fontsize=7, color=colour)
        ax.set_xticks([])
        ax.set_yticks([])


def legend(fig):
    handles = [plt.Rectangle((0, 0), 1, 1, fill=False, ec=GREEN, lw=1.2),
               plt.Rectangle((0, 0), 1, 1, fill=False, ec=ORANGE, lw=1.8),
               plt.Rectangle((0, 0), 1, 1, fill=False, ec=RED, lw=1.0, ls="--")]
    fig.legend(handles, ["detected defect (ground truth)", "missed defect (ground truth)", "prediction"],
               loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))


def save(fig, name):
    for ext, dpi in (("png", 300), ("pdf", None)):
        fig.savefig(OUT / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fig13(data, log):
    d = data["SteelDefectX"]
    groups = [("AUTO-ACCEPT,\nmissed defect", lambda r: r["decision"] == "AUTO-ACCEPT" and r["loss_any"] == "1", RED),
              ("MANUAL-REVIEW,\nmissed defect", lambda r: r["decision"] == "MANUAL-REVIEW" and r["loss_any"] == "1", GREEN),
              ("MANUAL-REVIEW,\nall detected", lambda r: r["decision"] == "MANUAL-REVIEW" and r["loss_any"] == "0", "#6B7280")]
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(3, 4, figsize=(8, 6.6))
    for row, (label, keep, colour) in enumerate(groups):
        pool = sorted(i for i, r in d.dec.items() if keep(r) and i in d.recs)
        picks = [pool[k] for k in rng.permutation(len(pool))[:4]]
        log.append(f"fig13 row {row + 1} ({label.replace(chr(10), ' ')}): {len(pool)} test images, shown {picks}")
        for col, i in enumerate(picks):
            d.draw(axes[row, col], i, "", colour)
        axes[row, 0].set_ylabel(label, fontsize=8)
    legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig13_failures_SteelDefectX")


def fig14(data, log):
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(DATASETS), len(REASONS), figsize=(7.2, 8.4))
    for row, name in enumerate(DATASETS):
        d = data[name]
        counts = {k: 0 for k in REASONS}
        for info in d.info.values():
            for why in info["reasons"].values():
                counts[why] += 1
        log.append(f"fig14 {name}: missed boxes by reason {counts}")
        for col, why in enumerate(REASONS):
            ax = axes[row, col]
            # images whose missed boxes all share this one reason, so the panel shows a single failure mode
            pool = sorted(i for i, info in d.info.items() if info["reasons"] and set(info["reasons"].values()) == {why})
            if not pool:
                ax.axis("off")
                ax.text(0.5, 0.5, "no case", ha="center", va="center", transform=ax.transAxes)
                log.append(f"  {why}: 0 images")
                continue
            i = pool[int(rng.integers(len(pool)))]
            log.append(f"  {why}: {len(pool)} images with only this reason, shown {i}")
            d.draw(ax, i, "", "#111827")
            if why == REASONS[1]:
                rec, kept_preds = d.recs[i], [p for p in d.recs[i]["preds"] if p["conf"] >= d.tau]
                j = next(iter(d.info[i]["reasons"]))
                o = box_iou([p["box"] for p in kept_preds], [rec["gt"][j]["box"]])[:, 0]
                p = kept_preds[int(np.argmax(o))]
                ax.set_xlabel(f"true: {rec['gt'][j]['cls']}\npredicted: {p['cls']}".replace("_", " "), fontsize=7)
            if row == 0:
                ax.set_title(f"({'abc'[col]}) {why}\n" + ax.get_title(), fontsize=7)
        axes[row, 0].set_ylabel(name, fontsize=8)
    legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig14_miss_types")


def main():
    OUT.mkdir(exist_ok=True)
    cfg = load_config()
    summary = {(s["model"], s["dataset"]): s for s in
               json.loads((PROJECT / "results/tables" / VARIANT / "in_domain_summary.json").read_text(encoding="utf-8"))}
    data = {d: Data(cfg, d, summary) for d in DATASETS}
    log = []
    fig13(data, log)
    fig14(data, log)
    (OUT / "selection_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
