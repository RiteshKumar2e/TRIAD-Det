"""Experiment 4: ablations, in-domain, per model and dataset (L_any, LTT-binomial, headline alpha).

  matching    IoU 0.3 / 0.5 / 0.7 (tau re-chosen on val for each), class-aware vs class-agnostic
  score       every uncertainty score forced in turn (conf, weakest, grey, stats, census [TRIAD-Det],
              disagree [1 - box agreement with the other three detectors on the same image]);
              test AURC, coverage at the headline alpha, and a paired bootstrap of AURC against the
              plain confidence score, Holm-corrected over all comparisons
  calib size  n in risk.recal_sizes calibration images drawn from CALIB (repeats), evaluated on TEST
  alpha       every alpha in risk.alphas for the reported score (the full grid is in in_domain_selective.csv)

Outputs (results/tables/<variant>/): ablation_matching.csv, ablation_scores.csv, ablation_calib_size.csv

    python experiments/ablation.py [--variant smoke] [--repeats 20] [--bootstrap 200]
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import calibration as cal  # noqa: E402
from src.common import DATASETS, load_config, log, results_dir, write_csv  # noqa: E402
from src.exp_utils import SPLITS, ltt_threshold, prepare, summarise, test_row  # noqa: E402
from src.prediction_loader import DETECTORS  # noqa: E402
from src.risk_scores import disagreement  # noqa: E402
from src.statistics import aurc, holm, paired_bootstrap_aurc  # noqa: E402


def operating_point(p, score, alpha, r):
    res = ltt_threshold(p.scores["val"][score], p.losses["val"], p.scores["calib"][score], p.losses["calib"],
                        alpha, r["delta"], r["grid_size"])
    return res, test_row(p.scores["test"][score], p.losses["test"], res.threshold, alpha)


def add_disagreement(prepared):
    """Adds a 'disagree' score to each model of one dataset, using the other models' predictions."""
    if len(prepared) < 2:
        return
    index = {m: {k: {rec["id"]: rec for rec in p.tgt[k]} for k in SPLITS} for m, p in prepared.items()}
    for m, p in prepared.items():
        others = [o for o in prepared if o != m]
        for k in SPLITS:
            vals = []
            for rec in p.tgt[k]:
                recs = [index[o][k].get(rec["id"]) for o in others]
                pairs = [(x, prepared[o].tau) for x, o in zip(recs, others) if x is not None]
                vals.append(disagreement(rec, [x for x, _ in pairs], [p.tau] + [t for _, t in pairs])
                            if pairs else 1.0)
            p.scores[k]["disagree"] = np.array(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="full")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config()
    r = cfg["risk"]
    alpha = r["headline_alpha"]
    repeats = args.repeats or r["recal_repeats"]
    n_boot = args.bootstrap or r["bootstrap"]
    matching, scores_rows, size_rows = [], [], []

    for dataset in DATASETS:
        # ---- matching rule
        for model in DETECTORS:
            for iou in r["iou_ablation"]:
                for aware in (True, False):
                    p = prepare(cfg, model, dataset, dataset, args.variant, iou=iou, class_aware=aware)
                    if p is None:
                        continue
                    res, t = operating_point(p, p.chosen, alpha, r)
                    matching.append({"model": model, "dataset": dataset, "match_iou": iou, "class_aware": aware,
                                     "tau": p.tau, "score": p.chosen, "alpha": alpha,
                                     "any_miss_risk_all_test": float(p.losses["test"].mean()),
                                     "cal_certified": res.certified, "cal_threshold": res.threshold,
                                     **{f"test_{k}": v for k, v in t.items()}})
        # ---- scores (incl. disagreement), IoU 0.5 class-aware
        prepared = {m: p for m in DETECTORS
                    if (p := prepare(cfg, m, dataset, dataset, args.variant)) is not None}
        add_disagreement(prepared)
        for model, p in prepared.items():
            for score in p.scores["test"]:
                res, t = operating_point(p, score, alpha, r)
                row = {"model": model, "dataset": dataset, "score": score, "chosen_on_val": score == p.chosen,
                       "val_aurc": aurc(p.scores["val"][score], p.losses["val"]),
                       "test_aurc": aurc(p.scores["test"][score], p.losses["test"]),
                       "alpha": alpha, "cal_certified": res.certified, **{f"test_{k}": v for k, v in t.items()}}
                if score != "conf":
                    b = paired_bootstrap_aurc(p.scores["test"][score], p.scores["test"]["conf"], p.losses["test"],
                                              n_boot, cfg["seed"])
                    row.update(aurc_diff_vs_conf=b["diff"], aurc_diff_ci95_low=b["ci95"][0],
                               aurc_diff_ci95_high=b["ci95"][1], p_value_vs_conf=b["p_value"])
                scores_rows.append(row)
            # ---- calibration-set size
            u = p.chosen
            uv, lv = p.scores["val"][u], p.losses["val"]
            uc, lc = p.scores["calib"][u], p.losses["calib"]
            ut, lt = p.scores["test"][u], p.losses["test"]
            rng = np.random.default_rng(cfg["seed"] + zlib.crc32(f"{model}|{dataset}".encode()))
            for n in sorted({*r["recal_sizes"], len(lc)}):
                if n > len(lc):
                    continue
                grid = cal.threshold_grid(uv, lv.astype(int), n, alpha, r["delta"], r["grid_size"])
                reps = 1 if n == len(lc) else repeats
                cert, viol, covs, risks = [], [], [], []
                for _ in range(reps):
                    idx = np.arange(len(lc)) if n == len(lc) else rng.choice(len(lc), n, replace=False)
                    res = cal.ltt(uc[idx], lc[idx], grid, alpha, r["delta"], "binomial")
                    t = test_row(ut, lt, res.threshold, alpha)
                    cert.append(res.certified)
                    viol.append(t["violated"])
                    covs.append(t["coverage"])
                    risks.append(t["risk_accepted"])
                cm, cs = summarise(covs)
                rm, rs = summarise(risks)
                size_rows.append({"model": model, "dataset": dataset, "score": u, "alpha": alpha, "delta": r["delta"],
                                  "n_calib": n, "repeats": reps, "certified_rate": float(np.mean(cert)),
                                  "violation_rate": float(np.mean(viol)), "coverage_mean": cm, "coverage_std": cs,
                                  "risk_accepted_mean": rm, "risk_accepted_std": rs})
            log(cfg, f"[ablation] {model}/{dataset}")

    tested = [i for i, row in enumerate(scores_rows) if "p_value_vs_conf" in row]
    for i, adj in zip(tested, holm([scores_rows[i]["p_value_vs_conf"] for i in tested]) if tested else []):
        scores_rows[i]["p_value_holm"] = adj
    tables = results_dir(cfg, "tables", args.variant)
    write_csv(tables / "ablation_matching.csv", matching)
    write_csv(tables / "ablation_scores.csv", scores_rows)
    write_csv(tables / "ablation_calib_size.csv", size_rows)
    log(cfg, f"[ablation] tables -> {tables}")


if __name__ == "__main__":
    main()
