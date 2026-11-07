"""Metrics, risk-control reporting and cross-dataset transfer, all from saved records."""
from __future__ import annotations

import collections

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchmetrics.detection import MeanAveragePrecision

from . import conformal
from .common import (CLASSIFIER_DATASET, DATASETS, DETECTOR_DATASETS, SDX_TO_GC10, SDX_TO_NEU, dataset_dir, out_path,
                     read_json, read_jsonl, write_json)

CELLS = (("top-left", "top-center", "top-right"), ("middle-left", "center", "middle-right"),
         ("bottom-left", "bottom-center", "bottom-right"))
DETECTOR_SCORES = ("one_minus_top", "weakest_accepted", "grey_zone")
CLASSIFIER_SCORES = ("one_minus_top", "margin")


def box_iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x0, y0 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x1, y1 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match(preds, gts, iou_threshold, class_aware=True):
    """Greedy highest-confidence-first matching. Returns TP pairs, FP and missed indices."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["conf"])
    overlap = box_iou([p["box"] for p in preds], [g["box"] for g in gts])
    free = set(range(len(gts)))
    pairs, false_positives = [], []
    for i in order:
        options = [j for j in free if overlap[i, j] >= iou_threshold
                   and (not class_aware or preds[i]["cls"] == gts[j]["cls"])]
        if options:
            best = max(options, key=lambda j: overlap[i, j])
            free.remove(best)
            pairs.append((i, best))
        else:
            false_positives.append(i)
    return {"pairs": pairs, "fp": false_positives, "missed": sorted(free)}


def kept(record, tau):
    return [p for p in record["preds"] if p["conf"] >= tau]


def counts(records, tau, iou, class_aware=True):
    tp = fp = fn = 0
    for record in records:
        result = match(kept(record, tau), record["gt"], iou, class_aware)
        tp, fp, fn = tp + len(result["pairs"]), fp + len(result["fp"]), fn + len(result["missed"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def choose_tau(val_records, iou):
    """Box-confidence threshold with the best validation F1."""
    grid = np.round(np.arange(0.05, 0.91, 0.05), 2)
    return float(max(grid, key=lambda t: counts(val_records, t, iou)["f1"]))


def map_metrics(records, class_index=None):
    """COCO mAP. With class_index None every box is treated as one 'defect' class."""
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=class_index is not None)

    def tensors(items, scored):
        boxes = torch.tensor([i["box"] for i in items], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([class_index[i["cls"]] if class_index else 0 for i in items], dtype=torch.long)
        result = {"boxes": boxes, "labels": labels}
        if scored:
            result["scores"] = torch.tensor([i["conf"] for i in items], dtype=torch.float32)
        return result

    for record in records:
        preds = record["preds"] if class_index is None else [p for p in record["preds"] if p["cls"] in class_index]
        metric.update([tensors(preds, True)], [tensors(record["gt"], False)])
    computed = metric.compute()
    result = {"mAP": float(computed["map"]), "AP50": float(computed["map_50"]), "AP75": float(computed["map_75"])}
    if class_index is not None:
        names = {v: k for k, v in class_index.items()}
        present = computed["classes"].reshape(-1).tolist()
        values = computed["map_per_class"].reshape(-1).tolist()
        result["AP_per_class"] = {names[c]: float(v) for c, v in zip(present, values)}
    return result


def detector_scores(record, tau):
    """Candidate uncertainty scores. Images with no accepted box always score 1."""
    confidences = [p["conf"] for p in record["preds"]]
    accepted = [c for c in confidences if c >= tau]
    rejected = [c for c in confidences if c < tau]
    if not accepted:
        return dict.fromkeys(DETECTOR_SCORES, 1.0)
    return {"one_minus_top": 1 - max(accepted), "weakest_accepted": 1 - min(accepted),
            "grey_zone": (max(rejected) / tau) if rejected else 0.0}


def classifier_scores(record):
    top = record["top"]
    second = top[1]["p"] if len(top) > 1 else 0.0
    return {"one_minus_top": 1 - top[0]["p"], "margin": 1 - (top[0]["p"] - second)}


def detector_loss(record, tau, iou, class_aware=True):
    return int(bool(match(kept(record, tau), record["gt"], iou, class_aware)["missed"]))


def classifier_loss(record):
    return int(record["top"][0]["cls"] != record["folder_class"])


def cam_distribution(cam):
    grid = cv2.resize(np.asarray(cam, np.float32), (3, 3), interpolation=cv2.INTER_AREA)
    return grid / max(float(grid.sum()), 1e-6)


def position_agreement(record):
    """Histogram intersection of the CAM with the 3x3 'Position' annotation."""
    position = record.get("position") or {}
    if not position:
        return None
    truth = np.array([[position.get(cell, 0.0) for cell in row] for row in CELLS], float)
    truth /= max(truth.sum(), 1e-9)
    return float(np.minimum(truth, cam_distribution(record["cam"])).sum())


def _rate(errors, total):
    return {"errors": errors, "n": total, "error_rate": errors / total if total else None,
            "ci95": conformal.clopper_pearson(errors, total)}


def _risk_report(cfg, dataset, by_split, score_fn, loss_fn, score_names):
    arrays = {}
    for split in ("val", "calib", "test"):
        records = by_split[split]
        scores = [score_fn(r) for r in records]
        arrays[split] = ({n: np.array([s[n] for s in scores], float) for n in score_names},
                         np.array([loss_fn(r) for r in records], int))
    val_scores, val_loss = arrays["val"]
    aurc = {n: conformal.aurc(val_scores[n], val_loss) for n in score_names}
    chosen = min(aurc, key=lambda n: (np.isnan(aurc[n]), aurc[n]))
    calib_u, calib_loss = arrays["calib"][0][chosen], arrays["calib"][1]
    test_u, test_loss = arrays["test"][0][chosen], arrays["test"][1]
    test_records = by_split["test"]

    report = {"dataset": dataset, "uncertainty_score": chosen, "val_aurc": aurc, "test_aurc": conformal.aurc(test_u, test_loss),
              "overall_error_rate_test": _rate(int(test_loss.sum()), len(test_loss)), "operating_points": [],
              "assumption": "Risk bound holds only for images exchangeable with this dataset's calibration split. "
                            "All images in these datasets contain defects, so defect-free images are not covered."}
    for alpha in cfg.risk["alphas"]:
        delta = cfg.risk["delta"]
        grid = conformal.threshold_grid(val_scores[chosen], val_loss, len(calib_u), alpha, delta, cfg.risk["grid_size"])
        result = conformal.calibrate(calib_u, calib_loss, grid, alpha, delta)
        test = conformal.apply_threshold(test_u, test_loss, result.threshold)
        accepted = np.zeros(len(test_u), bool) if result.threshold is None else test_u <= result.threshold
        per_class = collections.defaultdict(lambda: [0, 0, 0, 0])
        for record, is_accepted, loss in zip(test_records, accepted, test_loss):
            entry = per_class[record["folder_class"]]
            entry[0] += 1
            entry[1] += int(loss)
            entry[2] += int(is_accepted)
            entry[3] += int(loss and is_accepted)
        report["operating_points"].append({
            "calibration": conformal.as_dict(result), "test": test,
            "threshold_uncertainty": conformal.bootstrap_threshold(calib_u, calib_loss, grid, alpha, delta,
                                                                   cfg.risk["bootstrap"], cfg.seed),
            "class_wise": {k: {"overall": _rate(v[1], v[0]), "accepted": _rate(v[3], v[2])} for k, v in sorted(per_class.items())},
        })
    curve = conformal.risk_coverage_curve(test_u, test_loss)
    write_json(dataset_dir(cfg, dataset) / "metrics" / "risk_control.json", report)
    write_json(dataset_dir(cfg, dataset) / "metrics" / "risk_coverage_curve.json", curve)
    return report


def _load(cfg, dataset):
    records = read_jsonl(dataset_dir(cfg, dataset) / "predictions" / "all.jsonl")
    by_split = collections.defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)
    return by_split


def evaluate_detector_dataset(cfg, dataset):
    folder = dataset_dir(cfg, dataset)
    by_split = _load(cfg, dataset)
    classes = read_json(out_path(cfg, "manifests", "class_maps.json"))["classes"][dataset]
    index = {name: i for i, name in enumerate(classes)}
    iou = cfg.evaluation["match_iou"]
    tau = choose_tau(by_split["val"], iou)
    test = by_split["test"]

    metrics = {"dataset": dataset, "split": "test", "n_images": len(test), "n_gt_boxes": sum(len(r["gt"]) for r in test),
               "tau_from_val": tau, "match_iou": iou, **map_metrics(test, index),
               "at_tau": counts(test, tau, iou), "at_tau_iou_sensitivity": {
                   str(cfg.evaluation["sensitivity_iou"]): counts(test, tau, cfg.evaluation["sensitivity_iou"])},
               "class_agnostic": {**map_metrics(test), "at_tau": counts(test, tau, iou, class_aware=False)}}
    write_json(folder / "detection_metrics.json", metrics)

    size = len(classes)
    matrix = np.zeros((size + 1, size + 1), int)   # final row/column = background
    per_class = collections.defaultdict(collections.Counter)
    for record in test:
        preds, gts = kept(record, tau), record["gt"]
        located = match(preds, gts, iou, class_aware=False)
        for p, g in located["pairs"]:
            matrix[index[gts[g]["cls"]], index[preds[p]["cls"]]] += 1
        for p in located["fp"]:
            matrix[size, index[preds[p]["cls"]]] += 1
        for g in located["missed"]:
            matrix[index[gts[g]["cls"]], size] += 1
        strict = match(preds, gts, iou)
        for p, g in strict["pairs"]:
            per_class[gts[g]["cls"]]["tp"] += 1
        for p in strict["fp"]:
            per_class[preds[p]["cls"]]["fp"] += 1
        for g in strict["missed"]:
            per_class[gts[g]["cls"]]["fn"] += 1
    report = {}
    for name in classes:
        c = per_class[name]
        precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        report[name] = {"precision": precision, "recall": recall, "support": c["tp"] + c["fn"],
                        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                        "miss_rate": _rate(c["fn"], c["tp"] + c["fn"])}
    write_json(folder / "classification_report.json", {"level": "box instances at tau, IoU-matched", "per_class": report})
    write_json(folder / "metrics" / "confusion_matrix.json", {
        "labels": classes + ["background"], "rows": "ground truth", "columns": "prediction", "matrix": matrix})

    risk = _risk_report(cfg, dataset, by_split, lambda r: detector_scores(r, tau),
                        lambda r: detector_loss(r, tau, iou), DETECTOR_SCORES)
    write_json(folder / "metrics" / "operating_point.json", {"tau": tau, "iou": iou, "score": risk["uncertainty_score"]})
    return {"detection": metrics, "risk": risk}


def evaluate_classifier_dataset(cfg):
    dataset = CLASSIFIER_DATASET
    folder = dataset_dir(cfg, dataset)
    by_split = _load(cfg, dataset)
    classes = read_json(out_path(cfg, "manifests", "class_maps.json"))["classes"][dataset]
    test = by_split["test"]
    truth = [r["folder_class"] for r in test]
    guess = [r["top"][0]["cls"] for r in test]
    report = classification_report(truth, guess, labels=classes, output_dict=True, zero_division=0)
    write_json(folder / "classification_report.json", {
        "level": "image labels, all 24 classes", "accuracy": accuracy_score(truth, guess), "per_class": report})
    write_json(folder / "metrics" / "confusion_matrix.json", {
        "labels": classes, "rows": "ground truth", "columns": "prediction",
        "matrix": confusion_matrix(truth, guess, labels=classes)})

    boxed = [r for r in test if r["gt"]]
    agreement = [v for v in map(position_agreement, test) if v is not None]
    localisation = {"box_subset_images": len(boxed), "test_images": len(test),
                    "box_sources": dict(collections.Counter(r["box_source"] for r in boxed)),
                    "class_agnostic": map_metrics(boxed) if boxed else None}
    for iou in sorted({cfg.evaluation["cam_iou"], cfg.evaluation["match_iou"]}):
        localisation[f"iou_{iou}"] = counts(boxed, 0.0, iou, class_aware=False)
    metrics = {
        "dataset": dataset, "split": "test", "predictor": "Grad-CAM regions from the 24-class classifier",
        "note": "SteelDefectX has no native boxes. Scores cover only images whose boxes could be inherited from "
                "matched NEU-DET / GC10-DET sources, so they are not a full-dataset mAP.",
        "localisation": localisation,
        "position_agreement_all_images": {"n": len(agreement), "mean": float(np.mean(agreement)) if agreement else None},
    }
    write_json(folder / "detection_metrics.json", metrics)
    risk = _risk_report(cfg, dataset, by_split, classifier_scores, classifier_loss, CLASSIFIER_SCORES)
    write_json(folder / "metrics" / "operating_point.json", {"tau": 0.0, "iou": cfg.evaluation["cam_iou"],
                                                              "score": risk["uncertainty_score"]})
    return {"detection": metrics, "risk": risk, "accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"]}


def _headline(cfg, risk):
    target = cfg.risk["headline_alpha"]
    return next(p for p in risk["operating_points"] if p["calibration"]["alpha"] == target)


def _empirical(cfg, risk, u, loss):
    """Source-calibrated threshold on target data: empirical numbers, no guarantee."""
    threshold = _headline(cfg, risk)["calibration"]["threshold"]
    return {**conformal.apply_threshold(u, loss, threshold), "alpha": cfg.risk["headline_alpha"],
            "source_threshold": threshold, "guarantee_applies": False}


def evaluate_transfer(cfg, results):
    """Every directed (source -> target) pair, class-agnostic, on the target test split."""
    iou = cfg.evaluation["match_iou"]
    rows = []
    for source in DETECTOR_DATASETS:
        point = read_json(dataset_dir(cfg, source) / "metrics" / "operating_point.json")
        for target in DATASETS:
            if source == target:
                records = list(_load(cfg, target)["test"])
            else:
                path = dataset_dir(cfg, target) / "predictions" / f"transfer_from_{source}.jsonl"
                records = read_jsonl(path) if path.exists() else []
            records = [r for r in records if r["gt"]]
            if not records:
                continue
            u = [detector_scores(r, point["tau"])[point["score"]] for r in records]
            loss = [detector_loss(r, point["tau"], iou, class_aware=False) for r in records]
            # SteelDefectX is assembled from NEU, GC10 and X-SDD images, so any pair involving it is not
            # an independent domain-shift test (duplicate groups are still kept within one split).
            independent = "SteelDefectX" not in (source, target) or source == target
            rows.append({"source": source, "target": target, "task": "class-agnostic detection", "n_images": len(records),
                         "independent_of_source": independent, **map_metrics(records),
                         "at_tau": counts(records, point["tau"], iou, class_aware=False),
                         "selective": _empirical(cfg, results[source]["risk"], u, loss)})
    write_json(out_path(cfg, "COMBINED", "transfer_metrics.json"), rows)
    return rows


def compare_detectors(cfg, dataset, name, val_records, test_records):
    """One row of the TRIAD-YOLO versus baseline table; tau comes from validation."""
    classes = read_json(out_path(cfg, "manifests", "class_maps.json"))["classes"][dataset]
    iou = cfg.evaluation["match_iou"]
    tau = choose_tau(val_records, iou)
    scores = map_metrics(test_records, {c: i for i, c in enumerate(classes)})
    return {"dataset": dataset, "model": name, "tau": tau, "mAP": scores["mAP"], "AP50": scores["AP50"],
            "AP75": scores["AP75"], **counts(test_records, tau, iou)}


def evaluate_all(cfg):
    results = {dataset: evaluate_detector_dataset(cfg, dataset) for dataset in DETECTOR_DATASETS}
    transfer = evaluate_transfer(cfg, results)
    summary = {"smoke_test": cfg.smoke, "seed": cfg.seed, "datasets": {}, "transfer": transfer}
    for dataset, result in results.items():
        point = _headline(cfg, result["risk"])
        summary["datasets"][dataset] = {
            "headline_alpha": point["calibration"]["alpha"], "delta": point["calibration"]["delta"],
            "certified": point["calibration"]["certified"], "threshold": point["calibration"]["threshold"],
            "test": point["test"], "uncertainty_score": result["risk"]["uncertainty_score"],
            "overall_error": result["risk"]["overall_error_rate_test"], "test_aurc": result["risk"]["test_aurc"],
            "mAP": result["detection"]["mAP"], "AP50": result["detection"]["AP50"],
            "recall": result["detection"]["at_tau"]["recall"], "precision": result["detection"]["at_tau"]["precision"],
        }
    write_json(out_path(cfg, "COMBINED", "summary.json"), summary)
    return summary
