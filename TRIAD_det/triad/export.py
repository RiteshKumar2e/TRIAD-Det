"""Export every saved result as CSV files under <out_root>/csv, for spreadsheets and tables.

Reads only files the earlier stages wrote, so it can be rerun at any time:
    python run_all.py --stages export
Missing inputs (for example a stage that has not run yet) are skipped with a note.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .common import CLASSIFIER_DATASET, DATASETS, DETECTOR_DATASETS, read_json, read_jsonl


def _flatten(value, prefix=""):
    """Nested dicts become dotted column names; lists become JSON text in one cell."""
    flat = {}
    if isinstance(value, dict):
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        flat[prefix] = json.dumps(value)
    else:
        flat[prefix] = value
    return flat


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.name} ({len(rows)} rows)")


def _read(path: Path):
    if not path.exists():
        print(f"  skipped, not found: {path}")
        return None
    return read_json(path)


def _training_curves(root: Path, out: Path) -> None:
    rows = []
    history = _read(root / CLASSIFIER_DATASET / "metrics" / "classifier_training.json") or []
    rows += [{"dataset": CLASSIFIER_DATASET, "model": "classifier", **row} for row in history]
    _write(out / "training_classifier.csv", rows)

    rows = []
    for dataset in DETECTOR_DATASETS:
        for results in sorted((root / dataset / "checkpoints" / "yolo").glob("run_*/results.csv")):
            model = results.parent.name.removeprefix("run_")
            model = "TRIAD-YOLO" if model == "best" else model
            with results.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    rows.append({"dataset": dataset, "model": model, **{k.strip(): v.strip() for k, v in row.items()}})
    _write(out / "training_detectors.csv", rows)


def _per_dataset(root: Path, out: Path) -> None:
    detection, per_class, ap_per_class, risk, risk_classes, curves, predictions = [], [], [], [], [], [], []
    for dataset in DATASETS:
        folder = root / dataset
        metrics = _read(folder / "detection_metrics.json")
        if metrics:
            ap = metrics.pop("AP_per_class", None) or {}
            ap_per_class += [{"dataset": dataset, "class": k, "AP": v} for k, v in ap.items()]
            detection.append(_flatten(metrics))

        report = _read(folder / "classification_report.json")
        if report:
            for name, values in report["per_class"].items():
                if isinstance(values, dict):
                    per_class.append({"dataset": dataset, "class": name, **_flatten(values)})
            if "accuracy" in report:
                per_class.append({"dataset": dataset, "class": "accuracy", "f1-score": report["accuracy"]})

        matrix = _read(folder / "metrics" / "confusion_matrix.json")
        if matrix:
            _write(out / f"confusion_matrix_{dataset}.csv",
                   [{"ground_truth": label, **dict(zip(matrix["labels"], row))}
                    for label, row in zip(matrix["labels"], matrix["matrix"])])

        control = _read(folder / "metrics" / "risk_control.json")
        if control:
            for point in control["operating_points"]:
                alpha = point["calibration"]["alpha"]
                risk.append({"dataset": dataset, "uncertainty_score": control["uncertainty_score"],
                             "test_aurc": control["test_aurc"],
                             **_flatten({"calibration": point["calibration"], "test": point["test"],
                                         "threshold_uncertainty": point["threshold_uncertainty"]})})
                for name, values in point["class_wise"].items():
                    risk_classes.append({"dataset": dataset, "alpha": alpha, "class": name, **_flatten(values)})

        curve = _read(folder / "metrics" / "risk_coverage_curve.json")
        if curve:
            keys = [k for k, v in curve.items() if isinstance(v, list)]
            curves += [{"dataset": dataset, **dict(zip(keys, values))} for values in zip(*(curve[k] for k in keys))]

        records = folder / "predictions" / "all.jsonl"
        if records.exists():
            for record in read_jsonl(records):
                preds = sorted(record["preds"], key=lambda p: -p["conf"])
                top = record.get("top") or [{"cls": p["cls"], "p": p["conf"]} for p in preds[:1]]
                predictions.append({
                    "dataset": dataset, "split": record["split"], "id": record["id"], "path": record["path"],
                    "true_class": record["folder_class"], "gt_labels": "|".join(record["gt_labels"] or []),
                    "n_gt_boxes": len(record["gt"]),
                    "pred_class": top[0]["cls"] if top else "", "pred_conf": top[0]["p"] if top else "",
                    "correct": (top[0]["cls"] == record["folder_class"]) if top else False,
                    "n_pred_boxes": len(preds),
                    "n_pred_boxes_conf_0.25": sum(p["conf"] >= 0.25 for p in preds),
                })
    _write(out / "detection_metrics.csv", detection)
    _write(out / "per_class_AP.csv", ap_per_class)
    _write(out / "per_class_report.csv", per_class)
    _write(out / "risk_control.csv", risk)
    _write(out / "risk_control_per_class.csv", risk_classes)
    _write(out / "risk_coverage_curve.csv", curves)
    _write(out / "predictions_per_image.csv", predictions)


def _combined(root: Path, out: Path) -> None:
    summary = _read(root / "COMBINED" / "summary.json")
    if summary:
        _write(out / "summary.csv", [{"dataset": d, **_flatten(v)} for d, v in summary["datasets"].items()])
    transfer = _read(root / "COMBINED" / "transfer_metrics.json")
    if transfer:
        _write(out / "transfer.csv", [_flatten(row) for row in transfer])
    comparison = _read(root / "COMBINED" / "model_comparison.json")
    if comparison:
        _write(out / "model_comparison.csv", [_flatten(row) for row in comparison])


def export_csv(cfg) -> None:
    root = Path(cfg.out_root)
    out = root / "csv"
    _training_curves(root, out)
    _per_dataset(root, out)
    _combined(root, out)
    print(f"CSV files in {out}")
    from .results_table import write_results
    write_results()
