"""Write synthetic prediction files (variant 'synthcheck') to exercise the experiment scripts end to end.

Each image has 1-3 GT boxes; a model 'detects' each box with probability that depends on a latent
image difficulty, and its confidences reflect that difficulty, so scores are informative and LTT can
certify. Used only for code testing; never mixed with real results.

    python tests/make_synthetic_predictions.py && python experiments/cross_domain.py --variant synthcheck
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATASETS, load_config  # noqa: E402
from src.prediction_loader import DETECTORS  # noqa: E402

SIZES = {"val": 200, "calib": 400, "test": 400}


def images(dataset, rng):
    out = []
    for split, n in SIZES.items():
        for i in range(n):
            k = int(rng.integers(1, 4))
            gt = []
            for _ in range(k):
                x, y = rng.uniform(0, 150, 2)
                gt.append({"cls": f"c{int(rng.integers(0, 3))}", "box": [x, y, x + 40, y + 40]})
            out.append({"id": f"{dataset}/{split}_{i}", "dataset": dataset, "split": split, "path": "",
                        "width": 200, "height": 200, "gt": gt, "difficulty": float(rng.beta(2, 5))})
    return out


def predict(imgs, shift, rng, census):
    recs = []
    for im in imgs:
        d = min(im["difficulty"] + shift, 1.0)
        preds = []
        for g in im["gt"]:
            conf = float(np.clip(1 - d + rng.normal(0, 0.1), 0.01, 0.99))
            box = [v + rng.normal(0, 2) for v in g["box"]]
            preds.append({"cls": g["cls"], "conf": conf, "box": box})
        r = {k: v for k, v in im.items() if k != "difficulty"}
        r["preds"] = preds
        if census:
            r["census"] = {"count": len(im["gt"]) + d, "presence": {"c0": d, "c1": d / 2, "c2": 0.1}}
        recs.append(r)
    return recs


def main():
    cfg = load_config()
    out = Path(cfg["results"]) / "predictions" / "synthcheck"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    data = {d: images(d, rng) for d in DATASETS}
    for model in DETECTORS:
        for source in DATASETS:
            for target in DATASETS:
                shift = 0.0 if source == target else 0.15
                recs = predict(data[target], shift, rng, model == "TRIAD-Det")
                with (out / f"{model}__{source}__on__{target}.jsonl").open("w", encoding="utf-8") as h:
                    h.writelines(json.dumps(r) + "\n" for r in recs)
    print(f"synthetic predictions -> {out}")


if __name__ == "__main__":
    main()
