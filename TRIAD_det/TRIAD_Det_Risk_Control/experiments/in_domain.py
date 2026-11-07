"""Experiment 1: in-domain selective risk control, per dataset and per model.

Protocol (per model, per dataset; nothing below touches TEST until the threshold is frozen):
  1. tau  = box-confidence threshold maximising F1 on VAL (defines "detected").
  2. loss = L_any (binary) and L_fraction (bounded) at IoU 0.5, class-aware.
  3. u    = candidate scores; the stats score is fitted on VAL; the reported score is chosen
            by lowest VAL AURC. All scores are kept for the ablation.
  4. lam  = calibrated on CALIB with LTT (binomial for L_any, Hoeffding-Bentkus for L_fraction),
            grid order from VAL; CRC-joint / naive / fixed-confidence for comparison.
  5. TEST = coverage, manual-review rate, risk among accepted (Clopper-Pearson 95% CI).
  6. Validity: 500 random half/half re-splits of CALIB U TEST.
All four datasets have boxes (SteelDefectX from official masks, X-SDD via matched SteelDefectX copies).
`run_classifier` is kept only for the legacy classifier and is not called.

    python experiments/in_domain.py [--variant smoke]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import calibration as cal  # noqa: E402
from src.common import DATASETS, load_config, log, read_json, results_dir, write_csv, write_json  # noqa: E402
from src.evaluation import ScoreBank, calibrate_all, choose_tau, detection_metrics, pick_score, resplit_validity  # noqa: E402
from src.prediction_loader import DETECTORS, by_split, load  # noqa: E402
from src.risk_losses import loss_array, loss_classification  # noqa: E402
from src.selective_decision import decide  # noqa: E402
from src.statistics import aurc  # noqa: E402


def classes_for(cfg, dataset):
    return read_json(Path(cfg["pipeline_run"]) / "manifests" / "class_maps.json")["classes"][dataset]


def run_detector(cfg, model, dataset, variant, rows_out):
    recs = load(cfg, model, dataset, dataset, variant)
    if not recs:
        return None
    s = by_split(recs)
    if not all(k in s for k in ("val", "calib", "test")):
        log(cfg, f"[in_domain] {model}/{dataset}: missing a split; skipped")
        return None
    r = cfg["risk"]
    iou = r["match_iou"]
    tau = choose_tau(s["val"], iou, r["tau_grid"])
    det = detection_metrics(s["test"], tau, iou, classes_for(cfg, dataset))
    rows_out["detection"].append({"model": model, "dataset": dataset, "split": "test",
                                  **{k: v for k, v in det.items() if k != "AP_per_class"}})
    for c, ap in (det["AP_per_class"] or {}).items():
        rows_out["per_class_ap"].append({"model": model, "dataset": dataset, "class": c, "AP50_95": ap})

    bank = ScoreBank(s["val"], tau, iou)
    scores = {k: bank(s[k]) for k in ("val", "calib", "test")}
    summary = {"model": model, "dataset": dataset, "tau": tau}
    for loss_kind in ("any", "fraction"):
        losses = {k: loss_array(s[k], loss_kind, tau, iou)[0] for k in ("val", "calib", "test")}
        binary = loss_kind == "any"
        chosen, table = pick_score(scores["val"], losses["val"])
        for name, v in table.items():
            rows_out["score_aurc"].append({"model": model, "dataset": dataset, "loss": loss_kind, "score": name,
                                           "val_aurc": v, "test_aurc": aurc(scores["test"][name], losses["test"]),
                                           "chosen": name == chosen})
        summary[f"score_{loss_kind}"] = chosen
        for alpha in r["alphas"]:
            for score_name in scores["val"]:
                results, _ = calibrate_all(scores["val"][score_name], losses["val"], scores["calib"][score_name],
                                           losses["calib"], alpha, r["delta"], r["grid_size"], binary,
                                           scores["calib"]["conf"])
                for res in results:
                    ev = cal.evaluate(scores["test"][score_name], losses["test"], res.threshold)
                    rows_out["selective"].append({
                        "model": model, "dataset": dataset, "loss": loss_kind, "score": score_name,
                        "score_chosen_on_val": score_name == chosen, "alpha": alpha, "delta": r["delta"],
                        **{f"cal_{k}": v for k, v in cal.as_dict(res).items() if k not in ("alpha", "delta")},
                        **{f"test_{k}": v for k, v in ev.items()}})
        # validity of the reported (chosen) score at every alpha
        pool_u = np.r_[scores["calib"][chosen], scores["test"][chosen]]
        pool_l = np.r_[losses["calib"], losses["test"]]
        for alpha in r["alphas"]:
            valid = resplit_validity(pool_u, pool_l, scores["val"][chosen], losses["val"], alpha, r["delta"],
                                     r["grid_size"], binary, r["resplits"], cfg["seed"])
            for method, v in valid.items():
                rows_out["validity"].append({"model": model, "dataset": dataset, "loss": loss_kind, "score": chosen,
                                             "alpha": alpha, "delta": r["delta"], "method": method, **v})
        # per-image decisions on TEST at the headline alpha (for figures / audit)
        if binary:
            res = cal.ltt(scores["calib"][chosen], losses["calib"],
                          cal.threshold_grid(scores["val"][chosen], losses["val"].astype(int), len(losses["calib"]),
                                             r["headline_alpha"], r["delta"], r["grid_size"]),
                          r["headline_alpha"], r["delta"], "binomial")
            out = []
            for rec, u, l in zip(s["test"], scores["test"][chosen], losses["test"]):
                d, why = decide(rec, float(u), res.threshold, tau)
                out.append({"id": rec["id"], "path": rec["path"], "u": float(u), "loss_any": int(l), "decision": d,
                            "reason": why, "n_gt": len(rec["gt"])})
            write_csv(results_dir(cfg, "predictions", "decisions") / f"{model}__{dataset}__{variant}.csv", out)
            summary["headline_threshold"] = res.threshold
    return summary


def run_classifier(cfg, variant, rows_out):
    recs = load(cfg, "Classifier", "SteelDefectX", "SteelDefectX", variant)
    if not recs:
        return None
    s = by_split(recs)
    r = cfg["risk"]

    def sc(records):
        top1 = np.array([x["top"][0]["p"] for x in records])
        top2 = np.array([x["top"][1]["p"] if len(x["top"]) > 1 else 0.0 for x in records])
        return {"conf": 1 - top1, "margin": 1 - (top1 - top2)}

    scores = {k: sc(s[k]) for k in ("val", "calib", "test")}
    losses = {k: np.array([loss_classification(x) for x in s[k]], float) for k in ("val", "calib", "test")}
    acc = 1 - losses["test"].mean()
    rows_out["detection"].append({"model": "Classifier", "dataset": "SteelDefectX", "split": "test",
                                  "n_images": len(losses["test"]), "top1_accuracy": acc,
                                  "note": "image classification; no box metrics (SteelDefectX has no boxes)"})
    chosen, table = pick_score(scores["val"], losses["val"])
    for name, v in table.items():
        rows_out["score_aurc"].append({"model": "Classifier", "dataset": "SteelDefectX", "loss": "misclassification",
                                       "score": name, "val_aurc": v, "test_aurc": aurc(scores["test"][name], losses["test"]),
                                       "chosen": name == chosen})
    for alpha in r["alphas"]:
        for name in scores["val"]:
            results, _ = calibrate_all(scores["val"][name], losses["val"], scores["calib"][name], losses["calib"],
                                       alpha, r["delta"], r["grid_size"], True, scores["calib"]["conf"])
            for res in results:
                ev = cal.evaluate(scores["test"][name], losses["test"], res.threshold)
                rows_out["selective"].append({
                    "model": "Classifier", "dataset": "SteelDefectX", "loss": "misclassification", "score": name,
                    "score_chosen_on_val": name == chosen, "alpha": alpha, "delta": r["delta"],
                    **{f"cal_{k}": v for k, v in cal.as_dict(res).items() if k not in ("alpha", "delta")},
                    **{f"test_{k}": v for k, v in ev.items()}})
        valid = resplit_validity(np.r_[scores["calib"][chosen], scores["test"][chosen]],
                                 np.r_[losses["calib"], losses["test"]], scores["val"][chosen], losses["val"],
                                 alpha, r["delta"], r["grid_size"], True, r["resplits"], cfg["seed"])
        for method, v in valid.items():
            rows_out["validity"].append({"model": "Classifier", "dataset": "SteelDefectX", "loss": "misclassification",
                                         "score": chosen, "alpha": alpha, "delta": r["delta"], "method": method, **v})
    return {"model": "Classifier", "dataset": "SteelDefectX", "score": chosen, "top1_accuracy": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="full")
    parser.add_argument("--pipeline-run", default=None)
    parser.add_argument("--resplits", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config()
    if args.pipeline_run:
        cfg["pipeline_run"] = str(Path(args.pipeline_run).resolve())
    if args.resplits:
        cfg["risk"]["resplits"] = args.resplits
    rows = {k: [] for k in ("detection", "per_class_ap", "score_aurc", "selective", "validity")}
    summaries = []
    for dataset in DATASETS:
        for model in DETECTORS:
            out = run_detector(cfg, model, dataset, args.variant, rows)
            if out:
                summaries.append(out)
                log(cfg, f"[in_domain] {model}/{dataset}: {out}")
    tables = results_dir(cfg, "tables", args.variant)
    for name, data in rows.items():
        write_csv(tables / f"in_domain_{name}.csv", data)
    write_json(tables / "in_domain_summary.json", summaries)
    log(cfg, f"[in_domain] tables -> {tables}")
    if args.variant == "full":
        from triad.results_table import write_results
        write_results()


if __name__ == "__main__":
    main()
