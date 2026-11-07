"""Does the census branch add anything? TRIAD-Det only, saved predictions only, no training.

The reported stats score of TRIAD-Det already reads four census statistics. This compares, per dataset:
  conf            1 - highest accepted confidence
  stats_no_census logistic score on the ten detection statistics
  stats_census    the reported score: ten detection statistics + four census statistics
  census_count    relative count gap
  census_presence unexplained class presence
  census_combined max(count gap, presence, 1 - conf), the census score of the paper
Fitted and ordered on val only; LTT on calib; reported on test. Also: paired bootstrap on test AURC of
stats_census vs stats_no_census, census quality on test, and the "invisible miss" check of Remark 1:
among test images with a miss and high confidence (u_conf <= the median of images without a miss), how many
are ranked as riskier than the median miss-free image by each score.

    python experiments/census_ablation.py [--variant selected] [--bootstrap 2000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import calibration as cal  # noqa: E402
from src.common import DATASETS, load_config, results_dir, write_csv  # noqa: E402
from src.evaluation import choose_tau  # noqa: E402
from src.prediction_loader import by_split, load  # noqa: E402
from src.risk_losses import loss_array  # noqa: E402
from src.risk_scores import features  # noqa: E402
from src.statistics import aurc  # noqa: E402

CENSUS_KEYS = ("census_count", "census_gap", "census_rel_gap", "unexplained_presence")


def logistic(train_rows, y, keys):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    model.fit(np.array([[r[k] for k in keys] for r in train_rows]), y)
    return model


def scores_for(rows, models):
    no_box = np.array([r["n_accepted"] == 0 for r in rows])
    out = {"conf": np.array([1 - r["max_conf"] for r in rows]),
           "census_count": np.array([r["census_rel_gap"] for r in rows]),
           "census_presence": np.array([r["unexplained_presence"] for r in rows]),
           "census_combined": np.array([max(r["census_rel_gap"], r["unexplained_presence"], 1 - r["max_conf"]) for r in rows])}
    for name, (model, keys) in models.items():
        out[name] = model.predict_proba(np.array([[r[k] for k in keys] for r in rows]))[:, 1]
    return {k: np.where(no_box, 1.0, v) for k, v in out.items()}


def census_quality(records):
    count, truth, tp, fp, fn = [], [], 0, 0, 0
    for rec in records:
        c = rec.get("census")
        count.append(float(c["count"]))
        truth.append(len(rec["gt"]))
        labels = {g["cls"] for g in rec["gt"]}
        for cls, p in c["presence"].items():
            hit, want = p >= 0.5, cls in labels
            tp, fp, fn = tp + (hit and want), fp + (hit and not want), fn + (not hit and want)
    count, truth = np.array(count), np.array(truth)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"count_mae": float(np.abs(count - truth).mean()), "count_rmse": float(np.sqrt(((count - truth) ** 2).mean())),
            "count_bias": float((count - truth).mean()), "mean_true_count": float(truth.mean()),
            "presence_precision": precision, "presence_recall": recall,
            "presence_micro_f1": 2 * precision * recall / max(precision + recall, 1e-9)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="selected")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    cfg = load_config()
    r = cfg["risk"]
    rows, comparisons, quality, invisible = [], [], [], []
    rng = np.random.default_rng(cfg["seed"])
    for d in DATASETS:
        s = by_split(load(cfg, "TRIAD-Det", d, d, args.variant))
        tau = choose_tau(s["val"], r["match_iou"], r["tau_grid"])
        feats = {k: [features(x, tau) for x in s[k]] for k in ("val", "calib", "test")}
        loss = {k: loss_array(s[k], "any", tau, r["match_iou"])[0] for k in ("val", "calib", "test")}
        base = sorted(k for k in feats["val"][0] if k not in CENSUS_KEYS)
        both = sorted(feats["val"][0])
        assert len(base) == 10 and len(both) == 14
        y = np.asarray(loss["val"], int)
        models = {"stats_no_census": (logistic(feats["val"], y, base), base), "stats_census": (logistic(feats["val"], y, both), both)}
        u = {k: scores_for(feats[k], models) for k in ("val", "calib", "test")}
        for name in u["val"]:
            row = {"dataset": d, "tau": tau, "score": name, "val_aurc": aurc(u["val"][name], loss["val"]), "test_aurc": aurc(u["test"][name], loss["test"])}
            for alpha in (0.10, 0.20, 0.30):
                grid = cal.threshold_grid(u["val"][name], y, len(u["calib"][name]), alpha, r["delta"], r["grid_size"])
                res = cal.ltt(u["calib"][name], loss["calib"], grid, alpha, r["delta"], "binomial")
                ev = cal.evaluate(u["test"][name], loss["test"], res.threshold)
                tag = f"{alpha:.2f}"
                row.update({f"cert_{tag}": res.certified, f"coverage_{tag}": ev["coverage"], f"risk_{tag}": ev["risk_accepted"]})
            rows.append(row)
        n, diffs = len(loss["test"]), []
        for _ in range(args.bootstrap):
            i = rng.integers(0, n, n)
            diffs.append(aurc(u["test"]["stats_census"][i], loss["test"][i]) - aurc(u["test"]["stats_no_census"][i], loss["test"][i]))
        diffs = np.array(diffs)
        comparisons.append({"dataset": d, "aurc_stats_no_census": aurc(u["test"]["stats_no_census"], loss["test"]),
                            "aurc_stats_census": aurc(u["test"]["stats_census"], loss["test"]), "mean_diff": float(diffs.mean()),
                            "ci95_low": float(np.quantile(diffs, 0.025)), "ci95_high": float(np.quantile(diffs, 0.975)),
                            "p_two_sided": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))})
        quality.append({"dataset": d, "split": "test", **census_quality(s["test"])})
        l, t = loss["test"], u["test"]
        clean = l == 0
        cut = np.median(t["conf"][clean])
        hidden = (l == 1) & (t["conf"] <= cut)
        row = {"dataset": d, "test_images": n, "images_with_miss": int(l.sum()), "confident_images_with_miss": int(hidden.sum())}
        for name in t:
            row[f"caught_{name}"] = float((t[name][hidden] > np.median(t[name][clean])).mean()) if hidden.any() else None
        invisible.append(row)
        print(d, "done", flush=True)
    out = results_dir(cfg, "tables", args.variant)
    write_csv(out / "census_ablation_scores.csv", rows)
    write_csv(out / "census_ablation_stats_comparison.csv", comparisons)
    write_csv(out / "census_quality.csv", quality)
    write_csv(out / "census_invisible_miss.csv", invisible)
    for row in rows:
        print(row["dataset"], row["score"], "aurc %.3f" % row["test_aurc"], "| a=.10", row["cert_0.10"], "%.3f" % row["coverage_0.10"], "| a=.20", row["cert_0.20"], "%.3f" % row["coverage_0.20"])
    for name in (comparisons, quality, invisible):
        for row in name:
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
