"""TRIAD_det/results.csv: one row per (model, dataset) with everything measured so far.

Rebuilt from saved files only, so it can be regenerated at any time and never invents a value:
an empty cell means "not measured yet". Called automatically at the end of the pipeline export
stage, after each TRIAD-Det training run and after the in-domain risk experiment. Manual:

    python -m triad.results_table
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                       # TRIAD_det/
RUN = ROOT / "runs" / "triad_full"
RISK = ROOT / "TRIAD_Det_Risk_Control" / "results"
OUT = ROOT / "results.csv"

DATASETS = ("SteelDefectX", "NEU-DET", "GC10-DET", "X-SDD")
MODELS = {"TRIAD-Det": None, "TRIAD-YOLO": "best", "YOLOv8s": "YOLOv8s", "YOLO11s": "YOLO11s"}
HEADLINE_ALPHA = 0.10

COLUMNS = [
    "model", "dataset", "status", "epochs_done", "best_epoch", "val_mAP50", "val_mAP50_95",
    "test_images", "test_mAP50", "test_mAP50_95", "test_precision", "test_recall", "test_f1", "tau",
    "missed_defect_rate", "any_miss_image_risk", "mean_missed_fraction",
    "risk_score", "alpha", "delta", "certified", "threshold",
    "auto_accept_coverage", "manual_review_rate", "risk_among_accepted", "risk_ci95_low", "risk_ci95_high",
    "updated",
]


def _read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [{k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()} for row in csv.DictReader(handle)]


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _training(model, dataset):
    """Training progress and best validation scores from the per-epoch log."""
    if model == "TRIAD-Det":
        v2 = (RISK / "logs" / f"triad_det_v2_{dataset}_train.csv").exists()    # TRIAD-Det v2 supersedes v1
        log = _read_csv(RISK / "logs" / (f"triad_det_v2_{dataset}_train.csv" if v2 else f"triad_det_{dataset}_train.csv"))
        if v2:
            done = (RISK / "checkpoints" / f"triad_det_v2_{dataset}.done").exists()
        else:
            done = ((RISK / "checkpoints" / f"triad_det_{dataset}.done").exists()     # --train-only finished
                    or (RISK / "predictions" / "full" / f"TRIAD-Det__{dataset}__on__{dataset}.jsonl").exists())
        key50, key95 = "val_mAP50", "val_mAP50_95"
    else:
        log = _read_csv(RUN / dataset / "checkpoints" / "yolo" / f"run_{MODELS[model]}" / "results.csv")
        done = (RUN / dataset / "checkpoints" / f"detector_{MODELS[model]}.pt").exists()
        key50, key95 = "metrics/mAP50(B)", "metrics/mAP50-95(B)"
    scored = [r for r in log if _f(r.get(key50)) is not None]
    out = {"epochs_done": len(log) or None}
    if scored:
        best = max(scored, key=lambda r: _f(r[key50]))
        out.update(best_epoch=int(float(best["epoch"])), val_mAP50=_f(best[key50]), val_mAP50_95=_f(best.get(key95)))
    out["status"] = "trained" if done else ("training" if log else "pending")
    return out


def _test_rows():
    det = {(r["model"], r["dataset"]): r for r in _read_csv(RISK / "tables" / "full" / "in_domain_detection.csv")}
    sel = {}
    for r in _read_csv(RISK / "tables" / "full" / "in_domain_selective.csv"):
        if (r.get("loss") == "any" and r.get("score_chosen_on_val") == "True" and r.get("cal_method") == "LTT-binomial"
                and _f(r.get("alpha")) == HEADLINE_ALPHA):
            sel[(r["model"], r["dataset"])] = r
    return det, sel


def build():
    det, sel = _test_rows()
    rows = []
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    for dataset in DATASETS:
        for model in MODELS:
            row = {"model": model, "dataset": dataset, **_training(model, dataset)}
            d = det.get((model, dataset))
            if d:
                row.update(test_images=d.get("n_images"), test_mAP50=d.get("mAP50"), test_mAP50_95=d.get("mAP50_95"),
                           test_precision=d.get("precision"), test_recall=d.get("recall"), test_f1=d.get("f1"),
                           tau=d.get("tau"), missed_defect_rate=d.get("missed_defect_rate"),
                           any_miss_image_risk=d.get("any_miss_image_risk"), mean_missed_fraction=d.get("mean_missed_fraction"),
                           status="evaluated")
            s = sel.get((model, dataset))
            if s:
                ci = json.loads(s["test_risk_accepted_ci95"]) if s.get("test_risk_accepted_ci95") else [None, None]
                row.update(risk_score=s.get("score"), alpha=s.get("alpha"), delta=s.get("delta"),
                           certified=s.get("cal_certified"), threshold=s.get("cal_threshold"),
                           auto_accept_coverage=s.get("test_coverage"), manual_review_rate=s.get("test_manual_review_rate"),
                           risk_among_accepted=s.get("test_risk_accepted"), risk_ci95_low=ci[0], risk_ci95_high=ci[1])
            row["updated"] = now
            rows.append(row)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return OUT


def write_results():
    """Safe hook for other stages: never lets a table problem stop an experiment."""
    try:
        path = build()
        print(f"results table updated: {path}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: results.csv not updated ({exc})", flush=True)


if __name__ == "__main__":
    print(build())
