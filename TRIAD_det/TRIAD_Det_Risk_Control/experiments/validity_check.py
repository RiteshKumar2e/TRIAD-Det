"""Why did LTT-binomial exceed delta in the half/half re-splits? Separate the guarantee from test-sample noise.

The LTT guarantee is about the TRUE selective risk: P(R(lam_hat) <= alpha) >= 1 - delta for i.i.d. data.
resplit_validity counts something else: the EMPIRICAL risk on a finite test half, drawn without replacement
from the same pool as the calibration half. This script treats calib U test as a known population, so the true
risk of any threshold is known exactly, and draws i.i.d. samples from it (with replacement):

  true_violation       R_pool(lam_hat) > alpha, calib drawn i.i.d.         -> the guaranteed quantity, must be <= delta
  empirical_violation  empirical risk on a fresh i.i.d. test half > alpha  -> includes finite-test noise
  resplit_true         R_pool(lam_hat) > alpha, calib = random half without replacement (original re-split draw)

Grid, tau, score and losses are built exactly as in experiments/in_domain.py, from val only.

    python experiments/validity_check.py [--repeats 1000] [--workers 6]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import calibration as cal  # noqa: E402
from src.common import DATASETS, load_config, results_dir, write_csv  # noqa: E402
from src.evaluation import ScoreBank, choose_tau, pick_score  # noqa: E402
from src.prediction_loader import DETECTORS, by_split, load  # noqa: E402
from src.risk_losses import loss_array  # noqa: E402


def pool_risk(u, loss, threshold):
    accepted = u <= threshold
    return float(loss[accepted].mean()) if accepted.any() else None


def check(job):
    model, dataset, repeats, variant = job
    cfg = load_config()
    r = cfg["risk"]
    recs = load(cfg, model, dataset, dataset, variant)
    if not recs:
        return []
    s = by_split(recs)
    tau = choose_tau(s["val"], r["match_iou"], r["tau_grid"])
    bank = ScoreBank(s["val"], tau, r["match_iou"])
    scores = {k: bank(s[k]) for k in ("val", "calib", "test")}
    rows = []
    for loss_kind, methods in (("any", ("binomial", "hb")), ("fraction", ("hb",))):
        losses = {k: loss_array(s[k], loss_kind, tau, r["match_iou"])[0] for k in ("val", "calib", "test")}
        chosen, _ = pick_score(scores["val"], losses["val"])
        u_val, l_val = scores["val"][chosen], losses["val"]
        u = np.r_[scores["calib"][chosen], scores["test"][chosen]]
        loss = np.r_[losses["calib"], losses["test"]]
        n, half = len(u), len(u) // 2
        ref = l_val.astype(int) if loss_kind == "any" else (l_val > 0).astype(int)
        for alpha in r["alphas"]:
            grid = cal.threshold_grid(u_val, ref, half, alpha, r["delta"], r["grid_size"])
            for pvalue in methods:
                rng = np.random.default_rng(cfg["seed"])
                cert = true_v = emp_v = resplit_cert = resplit_v = 0
                excess = []
                for _ in range(repeats):
                    a = rng.integers(0, n, half)
                    res = cal.ltt(u[a], loss[a], grid, alpha, r["delta"], pvalue)
                    if res.certified:
                        cert += 1
                        risk = pool_risk(u, loss, res.threshold)
                        true_v += int(risk > alpha)
                        excess.append(risk - alpha)
                        b = rng.integers(0, n, n - half)
                        accepted = u[b] <= res.threshold
                        emp_v += int(accepted.any() and loss[b][accepted].mean() > alpha)
                    perm = rng.permutation(n)[:half]
                    res = cal.ltt(u[perm], loss[perm], grid, alpha, r["delta"], pvalue)
                    if res.certified:
                        resplit_cert += 1
                        resplit_v += int(pool_risk(u, loss, res.threshold) > alpha)
                rows.append({"model": model, "dataset": dataset, "loss": loss_kind, "score": chosen, "alpha": alpha,
                             "delta": r["delta"], "pvalue": pvalue, "repeats": repeats, "n_pool": n,
                             "certified_rate": cert / repeats, "true_violation": true_v / repeats,
                             "empirical_violation": emp_v / repeats, "resplit_certified_rate": resplit_cert / repeats,
                             "resplit_true_violation": resplit_v / repeats,
                             "mean_true_risk_minus_alpha": float(np.mean(excess)) if excess else None})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--variant", default="full")
    args = parser.parse_args()
    cfg = load_config()
    jobs = [(m, d, args.repeats, args.variant) for d in DATASETS for m in DETECTORS]
    rows = []
    with mp.Pool(args.workers) as pool:
        for part in pool.imap_unordered(check, jobs):
            rows += part
            print(len(rows), flush=True)
    write_csv(results_dir(cfg, "tables", args.variant) / "validity_decomposition.csv", rows)
    for pvalue, kind in (("binomial", "any"), ("hb", "any"), ("hb", "fraction")):
        part = [x for x in rows if x["pvalue"] == pvalue and x["loss"] == kind and x["certified_rate"] > 0]
        print(pvalue, kind, "settings", len(part),
              "true max %.3f mean %.3f" % (max(x["true_violation"] for x in part), np.mean([x["true_violation"] for x in part])),
              "empirical max %.3f mean %.3f" % (max(x["empirical_violation"] for x in part), np.mean([x["empirical_violation"] for x in part])),
              "resplit_true max %.3f mean %.3f" % (max(x["resplit_true_violation"] for x in part), np.mean([x["resplit_true_violation"] for x in part])))


if __name__ == "__main__":
    main()
