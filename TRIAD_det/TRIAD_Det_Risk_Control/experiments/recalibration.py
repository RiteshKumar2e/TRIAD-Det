"""Experiment 3: target recalibration with few labelled target images.

For every model and directed pair source -> target (class-agnostic, IoU 0.5, L_any), the detector,
tau and the score stay as chosen on SOURCE val (see cross_domain.py). Only lambda is re-calibrated,
with LTT-binomial on n randomly drawn labelled images of the TARGET calib split,
n in risk.recal_sizes (sizes larger than the calib split are skipped), risk.recal_repeats draws each.
The grid order comes from SOURCE val, so no extra target data is used beyond the n images.
Evaluated on the full TARGET test split: certification rate, coverage, risk among accepted, and the
violation rate P(test risk > alpha) - which LTT bounds by about delta when target calib and test are
exchangeable. n = 0 is the source-calibrated threshold (no target labels).

Output: results/tables/<variant>/recalibration.csv (one row per model, pair, alpha, n)

    python experiments/recalibration.py [--variant smoke] [--repeats 20]
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
from src.exp_utils import ltt_threshold, prepare, summarise, test_row  # noqa: E402
from src.prediction_loader import DETECTORS  # noqa: E402


def run_pair(cfg, model, source, target, variant, repeats, alphas, out):
    r = cfg["risk"]
    p = prepare(cfg, model, source, target, variant, class_aware=False)
    if p is None:
        return False
    u = p.chosen
    su_v, sl_v = p.src_scores["val"][u], p.src_losses["val"]
    su_c, sl_c = p.src_scores["calib"][u], p.src_losses["calib"]
    tu_c, tl_c = p.scores["calib"][u], p.losses["calib"]
    tu_t, tl_t = p.scores["test"][u], p.losses["test"]
    rng = np.random.default_rng(cfg["seed"] + zlib.crc32(f"{model}|{source}|{target}".encode()))
    base = {"model": model, "source": source, "target": target, "shares_source_images": "SteelDefectX" in (source, target),
            "score": u, "tau": p.tau, "target_calib_available": len(tl_c), "target_test_n": len(tl_t),
            "delta": r["delta"]}
    for alpha in alphas:
        res = ltt_threshold(su_v, sl_v, su_c, sl_c, alpha, r["delta"], r["grid_size"])
        t = test_row(tu_t, tl_t, res.threshold, alpha)
        out.append({**base, "alpha": alpha, "n_target_labels": 0, "repeats": 1,
                    "certified_rate": float(res.certified), "violation_rate": float(t["violated"]),
                    "coverage_mean": t["coverage"], "coverage_std": None,
                    "risk_accepted_mean": t["risk_accepted"], "risk_accepted_std": None})
        for n in r["recal_sizes"]:
            if n > len(tl_c):
                continue
            grid = cal.threshold_grid(su_v, sl_v.astype(int), n, alpha, r["delta"], r["grid_size"])
            cert, viol, covs, risks = [], [], [], []
            for _ in range(repeats):
                idx = rng.choice(len(tl_c), n, replace=False)
                res = ltt_threshold(None, None, tu_c[idx], tl_c[idx], alpha, r["delta"], r["grid_size"], grid=grid)
                t = test_row(tu_t, tl_t, res.threshold, alpha)
                cert.append(res.certified)
                viol.append(t["violated"])
                covs.append(t["coverage"])
                risks.append(t["risk_accepted"])
            cm, cs = summarise(covs)
            rm, rs = summarise(risks)
            out.append({**base, "alpha": alpha, "n_target_labels": n, "repeats": repeats,
                        "certified_rate": float(np.mean(cert)), "violation_rate": float(np.mean(viol)),
                        "coverage_mean": cm, "coverage_std": cs, "risk_accepted_mean": rm, "risk_accepted_std": rs})
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="full")
    parser.add_argument("--repeats", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config()
    repeats = args.repeats or cfg["risk"]["recal_repeats"]
    alphas = sorted({cfg["risk"]["headline_alpha"], 0.05, 0.20} & set(cfg["risk"]["alphas"]))
    out = []
    for model in DETECTORS:
        for source in DATASETS:
            for target in DATASETS:
                if source != target and run_pair(cfg, model, source, target, args.variant, repeats, alphas, out):
                    log(cfg, f"[recalibration] {model}: {source} -> {target}")
    path = results_dir(cfg, "tables", args.variant) / "recalibration.csv"
    write_csv(path, out)
    log(cfg, f"[recalibration] {len(out)} rows -> {path}")


if __name__ == "__main__":
    main()
