"""Figures for the risk-control study, from the saved tables only (300 DPI PNG + PDF, fixed colours).

  fig01_dataset_splits          images per split and dataset
  fig02_detection_map           test mAP50 / mAP50-95 per model and dataset
  fig03_risk_coverage           test risk-coverage curves (L_any, reported score), one panel per dataset
  fig04_coverage_vs_alpha       AUTO-ACCEPT coverage vs alpha (LTT-binomial)
  fig05_risk_vs_alpha           realised test risk among accepted vs alpha, 95% CI, y = alpha line
  fig06_validity                violation rate over random re-splits per method (delta line), headline alpha
  fig07_method_coverage         coverage per calibration method, headline alpha
  fig08_score_aurc              test AURC per uncertainty score (ablation)
  fig09_cross_domain            source-calibrated threshold on each target: risk among accepted (heatmap)
  fig10_recalibration           violation rate and coverage vs number of target labels
  fig11_iou_ablation            coverage and risk vs matching IoU
  fig12_examples_<dataset>      AUTO-ACCEPT and MANUAL-REVIEW test images (GT green, predictions red)
Every figure is drawn independently; a missing table only skips that figure.

    python experiments/figures.py [--variant smoke]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import COLOURS, DATASETS, load_config, log, read_json, results_dir  # noqa: E402
from src.prediction_loader import DETECTORS, load  # noqa: E402

MODEL_COLOURS = {"TRIAD-Det": "#111827", "TRIAD-YOLO": "#F59E0B", "YOLOv8s": "#6B7280", "YOLO11s": "#0EA5E9"}
MODEL_MARKERS = {"TRIAD-Det": "o", "TRIAD-YOLO": "s", "YOLOv8s": "^", "YOLO11s": "D"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False})


def rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as h:
        return list(csv.DictReader(h))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def truthy(x):
    return str(x).lower() == "true"


class Fig:
    def __init__(self, cfg, variant):
        self.cfg, self.variant = cfg, variant
        self.tables = Path(cfg["results"]) / "tables" / variant
        self.out = results_dir(cfg, "figures", variant)
        self.alpha = cfg["risk"]["headline_alpha"]
        self.delta = cfg["risk"]["delta"]
        self.dpi = cfg["figures"]["dpi"]
        self.made = []

    def t(self, name):
        return rows(self.tables / f"{name}.csv")

    def save(self, fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(self.out / f"{name}.{ext}", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        plt.rcParams["font.size"] = 9
        self.made.append(name)

    def panels(self, n):
        if n == 4:                                # 2 above, 2 below, large enough to read in a paper
            plt.rcParams["font.size"] = 12
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            return fig, axes.ravel()
        fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.8), squeeze=False)
        return fig, axes[0]

    @staticmethod
    def left_label(axes, text):
        """The y label on every panel of the left column."""
        for ax in (axes[0], axes[2]) if len(axes) == 4 else (axes[0],):
            ax.set_ylabel(text)

    def present(self, table, key="dataset"):
        return [d for d in DATASETS if any(r.get(key) == d for r in table)]

    # ------------------------------------------------------------------ figures
    def fig01_dataset_splits(self):
        rep = read_json(Path(self.cfg["pipeline_run"]) / "manifests" / "split_report.json")["per_split"]
        splits = ("train", "val", "calib", "test")
        ds = [d for d in DATASETS if rep.get(d)]
        fig, ax = plt.subplots(figsize=(5.5, 2.8))
        w = 0.8 / len(ds)
        for i, d in enumerate(ds):
            vals = [rep[d].get(s, 0) if isinstance(rep[d].get(s, 0), (int, float)) else rep[d][s].get("images", 0)
                    for s in splits]
            ax.bar(np.arange(4) + (i - (len(ds) - 1) / 2) * w, vals, w, color=COLOURS[d], label=d)
        ax.set_xticks(range(4), splits)
        ax.set_ylabel("images")
        ax.legend(ncol=2)
        self.save(fig, "fig01_dataset_splits")

    def fig02_detection_map(self):
        t = [r for r in self.t("in_domain_detection") if r.get("mAP50")]
        ds = self.present(t)
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
        for ax, key in zip(axes, ("mAP50", "mAP50_95")):
            w = 0.8 / len(DETECTORS)
            for i, m in enumerate(DETECTORS):
                vals = [next((f(r[key]) for r in t if r["model"] == m and r["dataset"] == d), 0) or 0 for d in ds]
                ax.bar(np.arange(len(ds)) + (i - 1.5) * w, vals, w, color=MODEL_COLOURS[m], label=m)
            ax.set_xticks(range(len(ds)), ds, rotation=15)
            for tick, d in zip(ax.get_xticklabels(), ds):
                tick.set_color(COLOURS[d])
            ax.set_ylabel(f"test {key.replace('_', '-')}")
            ax.set_ylim(0, 1)
        axes[0].legend(fontsize=7)
        self.save(fig, "fig02_detection_map")

    def fig03_risk_coverage(self):
        folder = Path(self.cfg["results"]) / "predictions" / "decisions"
        ds = [d for d in DATASETS if list(folder.glob(f"*__{d}__{self.variant}.csv"))]
        if not ds:
            return
        fig, axes = self.panels(len(ds))
        for ax, d in zip(axes, ds):
            for m in DETECTORS:
                t = rows(folder / f"{m}__{d}__{self.variant}.csv")
                if not t:
                    continue
                u = np.array([f(r["u"]) for r in t])
                l = np.array([f(r["loss_any"]) for r in t])
                o = np.argsort(u, kind="stable")
                k = np.arange(1, len(o) + 1)
                ax.plot(k / len(o), np.cumsum(l[o]) / k, color=MODEL_COLOURS[m], label=m, lw=1.8)
            ax.axhline(self.alpha, color="k", ls="--", lw=1.0)
            ax.set_title(d, color=COLOURS[d])
            ax.set_xlabel("coverage")
            ax.set_ylim(0, 1)
        self.left_label(axes, "risk among accepted (L_any)")
        axes[0].legend(fontsize=11)
        self.save(fig, "fig03_risk_coverage")

    def _ltt_rows(self):
        return [r for r in self.t("in_domain_selective") if r["loss"] == "any" and truthy(r["score_chosen_on_val"])]

    def fig04_coverage_vs_alpha(self):
        t = [r for r in self._ltt_rows() if r["cal_method"] == "LTT-binomial"]
        ds = self.present(t)
        fig, axes = self.panels(len(ds))
        for ax, d in zip(axes, ds):
            for m in DETECTORS:
                pts = sorted((f(r["alpha"]), f(r["test_coverage"])) for r in t if r["model"] == m and r["dataset"] == d)
                if pts:
                    ax.plot(*zip(*pts), marker=MODEL_MARKERS[m], color=MODEL_COLOURS[m], label=m, ms=6)
            ax.set_title(d, color=COLOURS[d])
            ax.set_xlabel("target risk α")
            ax.set_ylim(0, 1)
        self.left_label(axes, "AUTO-ACCEPT coverage (test)")
        axes[0].legend(fontsize=11)
        self.save(fig, "fig04_coverage_vs_alpha")

    def fig05_risk_vs_alpha(self):
        t = [r for r in self._ltt_rows() if r["cal_method"] == "LTT-binomial"]
        ds = self.present(t)
        fig, axes = self.panels(len(ds))
        for ax, d in zip(axes, ds):
            for j, m in enumerate(DETECTORS):
                sel = sorted((r for r in t if r["model"] == m and r["dataset"] == d and f(r["test_risk_accepted"]) is not None),
                             key=lambda r: f(r["alpha"]))
                if not sel:
                    continue
                a = np.array([f(r["alpha"]) for r in sel]) + (j - 1.5) * 0.004
                y = np.array([f(r["test_risk_accepted"]) for r in sel])
                ci = [json.loads(r["test_risk_accepted_ci95"]) if r["test_risk_accepted_ci95"] else [v, v]
                      for r, v in zip(sel, y)]
                err = np.array([[v - c[0], c[1] - v] for v, c in zip(y, ci)]).T
                ax.errorbar(a, y, yerr=err, fmt=MODEL_MARKERS[m], color=MODEL_COLOURS[m], label=m, ms=6, lw=1.0)
            lim = max(self.cfg["risk"]["alphas"]) * 1.15
            ax.plot([0, lim], [0, lim], "k--", lw=1.0)
            ax.set_title(d, color=COLOURS[d])
            ax.set_xlabel("target risk α")
        self.left_label(axes, "test risk among accepted")
        axes[0].legend(fontsize=11)
        self.save(fig, "fig05_risk_vs_alpha")

    def fig06_validity(self):
        t = [r for r in self.t("in_domain_validity") if r["loss"] == "any" and abs(f(r["alpha"]) - self.alpha) < 1e-9]
        ds = self.present(t)
        methods = sorted({r["method"] for r in t})
        fig, axes = self.panels(len(ds))
        for ax, d in zip(axes, ds):
            w = 0.8 / len(DETECTORS)
            for i, m in enumerate(DETECTORS):
                vals = [next((f(r["violation_rate"]) for r in t if r["model"] == m and r["dataset"] == d
                              and r["method"] == me), 0) for me in methods]
                ax.bar(np.arange(len(methods)) + (i - 1.5) * w, vals, w, color=MODEL_COLOURS[m], label=m)
            ax.axhline(self.delta, color="k", ls="--", lw=1.0)
            ax.set_xticks(range(len(methods)), methods, rotation=20, ha="right", fontsize=11)
            ax.set_title(d, color=COLOURS[d])
            ax.set_ylim(0, 1)
        self.left_label(axes, f"P(test risk > α), α = {self.alpha}")
        axes[0].legend(fontsize=11)
        self.save(fig, "fig06_validity")

    def fig07_method_coverage(self):
        t = [r for r in self._ltt_rows() if abs(f(r["alpha"]) - self.alpha) < 1e-9]
        ds = self.present(t)
        methods = sorted({r["cal_method"] for r in t})
        fig, axes = self.panels(len(ds))
        for ax, d in zip(axes, ds):
            w = 0.8 / len(DETECTORS)
            for i, m in enumerate(DETECTORS):
                vals = [next((f(r["test_coverage"]) for r in t if r["model"] == m and r["dataset"] == d
                              and r["cal_method"] == me), 0) for me in methods]
                ax.bar(np.arange(len(methods)) + (i - 1.5) * w, vals, w, color=MODEL_COLOURS[m], label=m)
            ax.set_xticks(range(len(methods)), methods, rotation=20, ha="right", fontsize=11)
            ax.set_title(d, color=COLOURS[d])
            ax.set_ylim(0, 1)
        self.left_label(axes, f"test coverage, α = {self.alpha}")
        axes[0].legend(fontsize=11)
        self.save(fig, "fig07_method_coverage")

    def fig08_score_aurc(self):
        t = self.t("ablation_scores")
        ds = self.present(t)
        scores = [s for s in ("conf", "weakest", "grey", "stats", "census", "disagree") if any(r["score"] == s for r in t)]
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
        axes = axes.ravel()
        for ax in axes[len(ds):]:
            ax.axis("off")
        top = max(f(r["test_aurc"]) for r in t if r["score"] in scores and r["model"] in DETECTORS)
        for ax, d in zip(axes, ds):
            grid = np.full((len(DETECTORS), len(scores)), np.nan)
            for r in t:
                if r["dataset"] == d and r["score"] in scores and r["model"] in DETECTORS:
                    grid[DETECTORS.index(r["model"]), scores.index(r["score"])] = f(r["test_aurc"])
            im = ax.imshow(grid, cmap="viridis_r", vmin=0, vmax=top, aspect="auto")
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if np.isfinite(grid[i, j]):
                        ink = "w" if grid[i, j] > 0.45 * top else "k"
                        ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=12, color=ink)
                    else:
                        ax.text(j, i, "n/a", ha="center", va="center", fontsize=11, color="#6B7280")
            ax.set_xticks(range(len(scores)), scores, fontsize=12)
            ax.set_yticks(range(len(DETECTORS)), DETECTORS, fontsize=12)
            ax.set_title(d, color=COLOURS[d], fontsize=15)
            ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(False)
        bar = fig.colorbar(im, ax=list(axes), shrink=0.75, aspect=30)
        bar.set_label("test AURC (lower is better)", fontsize=12)
        bar.ax.tick_params(labelsize=11)
        fig.tight_layout = lambda: None           # constrained layout is already set
        self.save(fig, "fig08_score_aurc")

    def fig09_cross_domain(self):
        t = [r for r in self.t("cross_domain_selective")
             if r["protocol"] == "source-calibrated LTT" and abs(f(r["alpha"]) - self.alpha) < 1e-9]
        models = [m for m in DETECTORS if any(r["model"] == m for r in t)]
        if not models:
            return
        fig, axes = self.panels(len(models))
        for ax, m in zip(axes, models):
            grid = np.full((4, 4), np.nan)
            text = [["" for _ in range(4)] for _ in range(4)]
            for r in t:
                if r["model"] != m:
                    continue
                i, j = DATASETS.index(r["source"]), DATASETS.index(r["target"])
                risk, cov = f(r["test_risk_accepted"]), f(r["test_coverage"])
                grid[i, j] = risk if risk is not None else np.nan
                text[i][j] = (f"{risk:.2f}\n{cov:.0%}" if risk is not None else f"none\n{cov:.0%}")
            top = max(0.5, np.nanmax(grid) if np.isfinite(grid).any() else 0.5)
            ax.imshow(grid, cmap="Reds", vmin=0, vmax=top, aspect="auto")
            for i in range(4):
                for j in range(4):
                    dark = np.isfinite(grid[i, j]) and grid[i, j] > 0.6 * top
                    ax.text(j, i, "—" if i == j else text[i][j], ha="center", va="center", fontsize=11,
                            color="w" if dark else "k")
            ax.set_xticks(range(4), DATASETS, rotation=15, ha="right", fontsize=11)
            ax.set_yticks(range(4), DATASETS, fontsize=11)
            for tick, d in zip(ax.get_xticklabels(), DATASETS):
                tick.set_color(COLOURS[d])
            for tick, d in zip(ax.get_yticklabels(), DATASETS):
                tick.set_color(COLOURS[d])
            ax.set_title(m)
            ax.set_xlabel("target")
            ax.grid(False)
        self.left_label(axes, "source (calibration)")
        fig.suptitle(f"risk among accepted / coverage on target test, α = {self.alpha}", fontsize=13)
        self.save(fig, "fig09_cross_domain")

    def fig10_recalibration(self):
        t = [r for r in self.t("recalibration") if abs(f(r["alpha"]) - self.alpha) < 1e-9]
        if not t:
            return
        model = "TRIAD-Det" if any(r["model"] == "TRIAD-Det" for r in t) else t[0]["model"]
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
        for target in DATASETS:
            sel = [r for r in t if r["model"] == model and r["target"] == target]
            for key, ax in zip(("violation_rate", "coverage_mean"), axes):
                by_n = {}
                for r in sel:
                    by_n.setdefault(int(r["n_target_labels"]), []).append(f(r[key]) or 0)
                if by_n:
                    n = sorted(by_n)
                    ax.plot([max(x, 1) for x in n], [np.mean(by_n[x]) for x in n], marker="o", ms=3,
                            color=COLOURS[target], label=f"target {target}")
        axes[0].axhline(self.delta, color="k", ls="--", lw=0.8)
        for ax, lab in zip(axes, ("P(test risk > α)", "AUTO-ACCEPT coverage")):
            ax.set_xscale("log")
            ax.set_xlabel("labelled target images (1 = source calibration only)")
            ax.set_ylabel(lab)
            ax.set_ylim(0, 1)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{model}: target recalibration, mean over sources, α = {self.alpha}", fontsize=13)
        self.save(fig, "fig10_recalibration")

    def fig11_iou_ablation(self):
        t = [r for r in self.t("ablation_matching") if truthy(r["class_aware"])]
        ds = self.present(t)
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
        for d in ds:
            for key, ax in zip(("test_coverage", "any_miss_risk_all_test"), axes):
                by = {}
                for r in t:
                    if r["dataset"] == d:
                        by.setdefault(f(r["match_iou"]), []).append(f(r[key]) or 0)
                x = sorted(by)
                ax.plot(x, [np.mean(by[v]) for v in x], marker="o", ms=3, color=COLOURS[d], label=d)
        axes[0].set_ylabel(f"coverage at α = {self.alpha} (mean over models)")
        axes[1].set_ylabel("any-miss risk, all test images")
        for ax in axes:
            ax.set_xlabel("matching IoU")
            ax.set_ylim(0, 1)
        axes[0].legend(fontsize=7)
        self.save(fig, "fig11_iou_ablation")

    def fig12_examples(self):
        import cv2
        from triad.common import imread
        folder = Path(self.cfg["results"]) / "predictions" / "decisions"
        summary = {(s["model"], s["dataset"]): s for s in
                   (read_json(self.tables / "in_domain_summary.json") if (self.tables / "in_domain_summary.json").exists() else [])}
        for d in DATASETS:
            model = next((m for m in DETECTORS if (folder / f"{m}__{d}__{self.variant}.csv").exists()), None)
            if model is None:
                continue
            dec = rows(folder / f"{model}__{d}__{self.variant}.csv")
            recs = {r["id"]: r for r in (load(self.cfg, model, d, d, self.variant, splits=("test",)) or [])}
            tau = summary.get((model, d), {}).get("tau", 0.25)
            rng = np.random.default_rng(0)
            picks = []
            for decision in ("AUTO-ACCEPT", "MANUAL-REVIEW"):
                pool = [r for r in dec if r["decision"] == decision and r["id"] in recs]
                picks += [(decision, pool[i]) for i in rng.permutation(len(pool))[:4]]
            if not picks:
                continue
            fig, axes = plt.subplots(2, 4, figsize=(8, 4.4))
            for ax in axes.ravel():
                ax.axis("off")
            for row, decision in enumerate(("AUTO-ACCEPT", "MANUAL-REVIEW")):
                for col, (_, r) in enumerate([p for p in picks if p[0] == decision]):
                    rec = recs[r["id"]]
                    img = imread(Path(self.cfg["data_root"]) / rec["path"], colour=True)
                    if img is None:
                        continue
                    ax = axes[row, col]
                    ax.imshow(img)
                    for g in rec["gt"]:
                        x0, y0, x1, y1 = g["box"]
                        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#16A34A", lw=1.2))
                    for p in rec["preds"]:
                        if p["conf"] >= tau:
                            x0, y0, x1, y1 = p["box"]
                            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#DC2626", lw=0.9, ls="--"))
                    ax.set_title(f"u={float(r['u']):.2f}  miss={r['loss_any']}", fontsize=7,
                                 color="#16A34A" if decision == "AUTO-ACCEPT" else "#DC2626")
                axes[row, 0].text(-0.08, 0.5, decision, transform=axes[row, 0].transAxes, rotation=90,
                                  va="center", ha="right", fontsize=8)
            fig.suptitle(f"{d} test images, {model} (green: ground truth, red dashed: predictions)",
                         fontsize=9, color=COLOURS[d])
            self.save(fig, f"fig12_examples_{d}")
        del cv2

    def run(self):
        for name in sorted(n for n in dir(self) if n.startswith("fig")):
            try:
                getattr(self, name)()
            except Exception:                       # one broken figure must not stop the rest
                log(self.cfg, f"[figures] {name} skipped:\n{traceback.format_exc(limit=2)}")
        log(self.cfg, f"[figures] {len(self.made)} figures -> {self.out}: {', '.join(self.made)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="full")
    args = parser.parse_args()
    Fig(load_config(), args.variant).run()


if __name__ == "__main__":
    main()
