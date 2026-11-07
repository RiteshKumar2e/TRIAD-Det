"""Experiment 2: cross-dataset transfer of a source-calibrated AUTO-ACCEPT threshold.

For every model and every directed pair source -> target (source != target), class-agnostic matching
(source and target label sets differ), IoU 0.5, loss L_any:
  * tau, the stats score and the reported score are chosen on SOURCE val,
  * lambda is calibrated with LTT-binomial on SOURCE calib (grid order from SOURCE val),
  * that frozen lambda is applied to TARGET test: coverage, risk among accepted (Clopper-Pearson CI),
    and whether the risk exceeds alpha (the guarantee does not cover a shifted distribution),
  * reference: lambda re-calibrated on the full TARGET calib split (grid from TARGET val), the best a
    labelled target sample of that size can do,
  * naive empirical thresholding on SOURCE calib, for comparison.
Pairs that involve SteelDefectX share source images with the other datasets (SteelDefectX was built
from NEU, GC10 and X-SDD among others), so they are flagged `shares_source_images` and are not
independent domains. Linked copies always sit in the same split, so no target test image was trained on.

Outputs (results/tables/<variant>/): cross_domain_detection.csv, cross_domain_selective.csv,
cross_domain_feasibility.csv

    python experiments/cross_domain.py [--variant smoke]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import calibration as cal  # noqa: E402
from src.common import DATASETS, load_config, log, results_dir, write_csv  # noqa: E402
from src.detection_matching import counts  # noqa: E402
from src.exp_utils import ltt_threshold, prepare, test_row  # noqa: E402
from src.prediction_loader import DETECTORS  # noqa: E402
from src.statistics import aurc  # noqa: E402


def shares(source, target):
    return "SteelDefectX" in (source, target)


def detection_row(p, records, which):
    c = counts(records, p.tau, p.iou, False)
    n_gt = c["tp"] + c["fn"]
    return {"model": p.model, "source": p.source, "target": p.target, "evaluated_on": which,
            "shares_source_images": shares(p.source, p.target), "tau": p.tau, "n_images": len(records),
            "precision": c["precision"], "recall": c["recall"], "f1": c["f1"],
            "missed_defect_rate": c["fn"] / n_gt if n_gt else None}


def run_pair(cfg, model, source, target, variant, rows):
    r = cfg["risk"]
    p = prepare(cfg, model, source, target, variant, class_aware=False)
    if p is None:
        return False
    base = {"model": model, "source": source, "target": target, "shares_source_images": shares(source, target),
            "tau": p.tau, "score": p.chosen}
    # detection: on target test, and the same detector in-domain on its own source calib (class-agnostic)
    d = detection_row(p, p.tgt["test"], "target_test")
    d["any_miss_image_risk"] = float(p.losses["test"].mean())
    rows["detection"].append(d)
    d = detection_row(p, p.src["calib"], "source_calib")
    d["any_miss_image_risk"] = float(p.src_losses["calib"].mean())
    rows["detection"].append(d)

    u = p.chosen
    su_v, sl_v = p.src_scores["val"][u], p.src_losses["val"]
    su_c, sl_c = p.src_scores["calib"][u], p.src_losses["calib"]
    tu_v, tl_v = p.scores["val"][u], p.losses["val"]
    tu_c, tl_c = p.scores["calib"][u], p.losses["calib"]
    tu_t, tl_t = p.scores["test"][u], p.losses["test"]
    for alpha in r["alphas"]:
        protocols = {
            "source-calibrated LTT": ltt_threshold(su_v, sl_v, su_c, sl_c, alpha, r["delta"], r["grid_size"]),
            "source-calibrated naive": cal.naive(su_c, sl_c, alpha),
            "target-recalibrated LTT (full calib)": ltt_threshold(tu_v, tl_v, tu_c, tl_c, alpha, r["delta"], r["grid_size"]),
        }
        for name, res in protocols.items():
            rows["selective"].append({**base, "alpha": alpha, "delta": r["delta"], "protocol": name,
                                      "cal_certified": res.certified, "cal_threshold": res.threshold,
                                      "cal_n": res.n_calib, "cal_n_accepted": res.n_accepted, "cal_risk": res.calib_risk,
                                      **{f"test_{k}": v for k, v in test_row(tu_t, tl_t, res.threshold, alpha).items()}})
    # feasibility: what risk is reachable on the target at all with this score (target calib, descriptive)
    order = np.argsort(tu_c, kind="stable")
    cum = np.cumsum(tl_c[order]) / np.arange(1, len(order) + 1)
    feas = {**base, "target_calib_n": len(tl_c), "target_any_miss_risk": float(tl_c.mean()),
            "source_any_miss_risk": float(sl_c.mean()), "target_test_aurc": aurc(tu_t, tl_t),
            "source_calib_aurc": aurc(su_c, sl_c)}
    for cov in (0.1, 0.25, 0.5):
        k = max(1, int(round(cov * len(order))))
        feas[f"target_risk_at_cov_{cov}"] = float(cum[k - 1])
    for alpha in r["alphas"]:
        feas[f"min_n_for_zero_miss_cert_alpha_{alpha}"] = int(np.ceil(np.log(r["delta"]) / np.log(1 - alpha)))
    rows["feasibility"].append(feas)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="full")
    args = parser.parse_args()
    cfg = load_config()
    rows = {"detection": [], "selective": [], "feasibility": []}
    for model in DETECTORS:
        for source in DATASETS:
            for target in DATASETS:
                if source != target and run_pair(cfg, model, source, target, args.variant, rows):
                    log(cfg, f"[cross_domain] {model}: {source} -> {target}")
    tables = results_dir(cfg, "tables", args.variant)
    for name, data in rows.items():
        write_csv(tables / f"cross_domain_{name}.csv", data)
    log(cfg, f"[cross_domain] {len(rows['feasibility'])} pairs -> {tables}")


if __name__ == "__main__":
    main()
