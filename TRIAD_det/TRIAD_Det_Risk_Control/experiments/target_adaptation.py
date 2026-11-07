"""Can a small labelled target set restore useful auto-acceptance after a change of dataset?

Design, fixed before any result was seen:
  transfers   the 6 directed pairs among SteelDefectX, NEU-DET, GC10-DET (X-SDD left out: proxy boxes)
  detectors   TRIAD-Det (validation-selected source recipe) and YOLO11s
  conditions  finetune: source weights, new class-specific outputs, trained on n target images
              scratch:  the usual pretraining (ImageNet backbone / COCO), trained on the same n images (control)
  budgets     n = 25, 50, 100, 200, 400 images, class-balanced, drawn from the target TRAIN split;
              a draw is nested (the 25 images are inside the 50), the same for every source and condition
  training    the same fixed number of updates in every run; the best epoch by target val AP50 is kept
  roles       target val: stopping, tau, score, grid order. target calib: lambda only. target test: reporting only.
              n counts adaptation images only; target val and calib labels are needed on top and are reported.
Each run is a subprocess so GPU memory is released; finished runs are skipped, so it can be stopped and resumed.
Weights are deleted after prediction; prediction files stay in results/predictions/adaptation/.

    python experiments/target_adaptation.py [--draws 0,1] [--smoke]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))
from src import calibration as cal  # noqa: E402
from src.evaluation import ScoreBank, choose_tau, detection_metrics, pick_score  # noqa: E402
from src.exp_utils import test_row, with_gt  # noqa: E402
from src.prediction_loader import by_split  # noqa: E402
from src.risk_losses import loss_array  # noqa: E402
from src.triad_det_model.model import TriadDet  # noqa: E402
from src.triad_det_model.train import (DetSet, ModelEMA, class_list, collate, load_cfg, predict_rows, record_base,  # noqa: E402
                                       rows_for, seed_all)
from triad.common import read_jsonl, write_jsonl  # noqa: E402
from triad.evaluate import map_metrics  # noqa: E402

TARGETS = ("SteelDefectX", "NEU-DET", "GC10-DET")
MODELS = ("TRIAD-Det", "YOLO11s")
BUDGETS = (100, 25, 400, 50, 200)
TRIAD_CKPT = {"SteelDefectX": "checkpoints/triad_det_v2_SteelDefectX.pt", "NEU-DET": "checkpoints/triad_det_v2_NEU-DET.pt",
              "GC10-DET": "checkpoints/triad_det_GC10-DET.pt"}   # validation-selected recipe per dataset
YOLO_BATCH = {"SteelDefectX": 16, "NEU-DET": 16, "GC10-DET": 6}
STEPS = 800


def subset(rows, n, draw, target):
    """Class-balanced and nested: round-robin over classes through a fixed shuffled order."""
    rng = np.random.default_rng(1000 * draw + TARGETS.index(target) + 7)
    by_class = {}
    for r in sorted((r for r in rows if r["boxes"]), key=lambda r: r["id"]):
        by_class.setdefault(r["folder_class"], []).append(r)
    order = [list(rng.permutation(len(v))) for v in by_class.values()]
    groups = list(by_class.values())
    out = []
    while len(out) < n and any(order):
        for items, idx in zip(groups, order):
            if idx and len(out) < n:
                out.append(items[idx.pop()])
    return out


def run_name(model, condition, source, target, n, draw):
    return f"{model}__{condition}__{source}__{target}__n{n}__d{draw}"


def adapt_triad(cfg, condition, source, target, rows, steps, device="cuda"):
    s = cfg["triad_det"]
    classes = class_list(cfg, target)
    size, batch = s["imgsz"][target], s["batch"][target]
    model = TriadDet(len(classes), s)
    if condition == "finetune":
        state = torch.load(Path(cfg["results"]) / TRIAD_CKPT[source], map_location="cpu", weights_only=False)["state_dict"]
        mine = model.state_dict()
        shared = {k: v for k, v in state.items() if k in mine and v.shape == mine[k].shape}
        model.load_state_dict(shared, strict=False)
        print(f"loaded {len(shared)} of {len(mine)} tensors from {source}", flush=True)
    model = model.to(device)
    train_set = DetSet(rows, cfg["data_root"], classes, size, True, s.get("aug"))
    backbone = list(model.backbone.parameters())
    ids = {id(p) for p in backbone}
    rest = [p for p in model.parameters() if id(p) not in ids]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": s["lr"] * s["backbone_lr_mult"]}, {"params": rest, "lr": s["lr"]}],
                                  weight_decay=s["weight_decay"])
    base = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=s["amp"])
    ema = ModelEMA(model, tau=200)                      # the runs are short: ramp up the average faster
    val = rows_for(cfg, target, ("val",), False)["val"]
    index = {c: i for i, c in enumerate(classes)}
    warm, every = max(steps // 20, 1), max(steps // 8, 1)
    best, best_state, step = -1.0, None, 0
    for phase_steps, mosaic in ((int(steps * 0.9), None), (steps - int(steps * 0.9), 0.0)):
        if mosaic is not None:
            train_set.mosaic = mosaic
        loader = DataLoader(train_set, batch_size=batch, sampler=RandomSampler(train_set, replacement=True, num_samples=phase_steps * batch),
                            num_workers=s["workers"], collate_fn=collate, drop_last=True, pin_memory=True)
        model.train()
        for images, boxes, labels, _, _ in loader:
            scale = (step + 1) / warm if step < warm else 0.5 * (1 + np.cos(np.pi * (step - warm) / max(steps - warm, 1)))
            for g, lr in zip(optimizer.param_groups, base):
                g["lr"] = lr * scale
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=s["amp"]):
                out = model(images)
            loss, _ = model.loss(out, [b.to(device) for b in boxes], [l.to(device) for l in labels], s)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            step += 1
            if step % every == 0 or step == steps:
                score = map_metrics(predict_rows(ema.ema, val, cfg, classes, size, device, target), index)["AP50"]
                print(f"step {step}/{steps} val AP50 {score:.4f}", flush=True)
                if score > best:
                    best, best_state = score, {k: v.detach().cpu().clone() for k, v in ema.ema.state_dict().items()}
                model.train()
    ema.ema.load_state_dict(best_state)
    rows_all = rows_for(cfg, target, ("val", "calib", "test"), False)
    return [r for k in ("val", "calib", "test") for r in predict_rows(ema.ema, [x for x in rows_all[k] if x["boxes"]], cfg, classes, size, device, target)]


def adapt_yolo(cfg, pcfg, condition, source, target, rows, steps, work):
    from ultralytics import YOLO

    from triad.data import full_path, load_final
    from triad.detector import _link

    settings = pcfg.detector
    classes = class_list(cfg, target)
    index = {c: i for i, c in enumerate(classes)}
    pool = load_final(pcfg)
    root = work / "data"
    val_rows = [r for r in pool if r["dataset"] == target and r["split"] == "val"]
    # Ultralytics validates once per epoch. With few images an epoch is one or two updates, so hundreds of validation
    # passes would dominate the run. Repeating the images (links) makes every run 8 epochs, like TRIAD-Det's 8 evaluations.
    batch = YOLO_BATCH[target]
    per_epoch = max(steps // 8, 1)
    repeats = int(np.ceil(per_epoch * batch / len(rows)))
    for split, chosen, copies in (("train", rows, repeats), ("val", val_rows, 1)):
        for row, copy in ((r, c) for r in chosen for c in range(copies)):
            stem = row["id"].split("/")[-1] + (f"__r{copy}" if copies > 1 else "")
            _link(full_path(pcfg, row), root / "images" / split / f"{stem}.jpg")
            lines = []
            for item in row["boxes"]:
                x0, y0, x1, y1 = item["box"]
                w, h = row["width"], row["height"]
                lines.append(f'{index[item["cls"]]} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}')
            label = root / "labels" / split / f"{stem}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(lines), encoding="utf-8")
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val", "names": dict(enumerate(classes))}), encoding="utf-8")
    run = Path(cfg["pipeline_run"])
    weights = run / source / "checkpoints" / "detector_YOLO11s.pt" if condition == "finetune" else PROJECT.parent / "yolo11s.pt"
    epochs = max(int(np.ceil(steps / max(len(rows) * repeats // batch, 1))), 2)
    os.environ["MLFLOW_TRACKING_URI"] = (work / "mlflow").as_uri()
    model = YOLO(str(weights))
    results = model.train(data=str(data), epochs=epochs, patience=epochs, imgsz=settings["imgsz"][target], batch=batch, workers=settings["workers"],
                          amp=settings["amp"], seed=cfg["seed"], deterministic=True, device=0, flipud=0.5, fliplr=0.5, plots=False, verbose=False,
                          close_mosaic=max(int(epochs * 0.1), 1), val=True, project=str(work / "train"), name="r", exist_ok=True)
    best = YOLO(str(Path(results.save_dir) / "weights" / "best.pt"))
    chosen = [r for r in pool if r["dataset"] == target and r["split"] in ("val", "calib", "test") and r["boxes"]]
    records = []
    for start in range(0, len(chosen), 256):
        part = chosen[start:start + 256]
        output = best.predict([full_path(pcfg, r) for r in part], stream=True, verbose=False, device=0, save=False, half=True,
                              imgsz=settings["imgsz"][target], conf=settings["predict_conf"], iou=settings["predict_iou"], max_det=settings["max_det"],
                              batch=max(1, int(settings["predict_batch"] * min(1.0, (640 / settings["imgsz"][target]) ** 2))))
        for row, result in zip(part, output):
            b = result.boxes
            preds = [{"cls": best.names[int(c)], "conf": float(p), "box": [float(v) for v in xyxy]}
                     for c, p, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]
            records.append({"model": f"YOLO11s:{source}", "preds": preds, **record_base_yolo(row)})
    return records


def record_base_yolo(row):
    from triad.data import record_base as base
    return base(row)


def evaluate(cfg, records, meta, classes):
    r = cfg["risk"]
    s = {k: with_gt(v) for k, v in by_split(records).items()}
    iou = r["match_iou"]
    out = []
    for aware in (True, False):
        tau = choose_tau(s["val"], iou, r["tau_grid"], aware)
        bank = ScoreBank(s["val"], tau, iou, aware)
        u = {k: bank(s[k]) for k in ("val", "calib", "test")}
        loss = {k: loss_array(s[k], "any", tau, iou, aware)[0] for k in ("val", "calib", "test")}
        chosen, _ = pick_score(u["val"], loss["val"])
        det = detection_metrics(s["test"], tau, iou, classes) if aware else {}
        common = {**meta, "match": "class-aware" if aware else "localisation-only", "tau": tau, "score": chosen,
                  "n_val": len(s["val"]), "n_calib": len(s["calib"]), "n_test": len(s["test"]), "mAP50": det.get("mAP50"), "mAP50_95": det.get("mAP50_95"),
                  "test_any_miss": float(np.mean(loss["test"]))}
        for alpha in r["alphas"]:
            grid = cal.threshold_grid(u["val"][chosen], loss["val"].astype(int), len(loss["calib"]), alpha, r["delta"], r["grid_size"])
            res = cal.ltt(u["calib"][chosen], loss["calib"], grid, alpha, r["delta"], "binomial")
            row = test_row(u["test"][chosen], loss["test"], res.threshold, alpha)
            out.append({**common, "alpha": alpha, "delta": r["delta"], "certified": res.certified, "threshold": res.threshold,
                        "coverage": row["coverage"], "risk_accepted": row["risk_accepted"], "risk_ci95_low": row["risk_ci95_low"],
                        "risk_ci95_high": row["risk_ci95_high"], "violated": row["violated"]})
    return out


def pipeline_config():
    from triad.common import Config, resolve_device
    run = PROJECT.parent / "runs" / "triad_full"
    raw = json.loads((run / "manifests" / "resolved_config.json").read_text(encoding="utf-8"))
    raw["out_root"] = str(run)
    raw["device"] = resolve_device("cuda", False)
    return Config(raw)


def one(args):
    cfg = load_cfg(PROJECT / "config" / "experiment_config.yaml")
    seed_all(cfg["seed"] + args.draw)
    started = time.time()
    results = Path(cfg["results"])
    name = run_name(args.model, args.condition, args.source, args.target, args.n, args.draw)
    variant = "adaptation_smoke" if args.steps != STEPS else "adaptation"
    rows = subset(rows_for(cfg, args.target, ("train",), False)["train"], args.n, args.draw, args.target)
    print(name, "images", len(rows), "steps", args.steps, flush=True)
    work = results / "_adaptation_work" / name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    if args.model == "TRIAD-Det":
        records = adapt_triad(cfg, args.condition, args.source, args.target, rows, args.steps)
    else:
        records = adapt_yolo(cfg, pipeline_config(), args.condition, args.source, args.target, rows, args.steps, work)
    pred = results / "predictions" / variant / f"{name}.jsonl"
    pred.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(pred, records)
    shutil.rmtree(work, ignore_errors=True)
    meta = {"model": args.model, "condition": args.condition, "source": args.source, "target": args.target, "n_adapt": args.n, "draw": args.draw, "steps": args.steps,
            "shares_source_images": args.source == "SteelDefectX" or args.target == "SteelDefectX", "seconds": round(time.time() - started)}
    out = evaluate(cfg, read_jsonl(pred), meta, class_list(cfg, args.target))
    table = results / "tables" / variant / "target_adaptation.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    fresh = not table.exists()
    with table.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        if fresh:
            writer.writeheader()
        writer.writerows(out)
    print("finished", name, f"{time.time() - started:.0f} s", flush=True)


def plan(draws, budgets, models):
    for draw in draws:
        for n in budgets:
            for model in models:
                for target in TARGETS:
                    yield model, "scratch", "none", target, n, draw
                    for source in TARGETS:
                        if source != target:
                            yield model, "finetune", source, target, n, draw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--condition")
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--n", type=int)
    parser.add_argument("--draw", type=int, default=0)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--draws", default="0,1")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.one:
        return one(args)
    cfg = load_cfg(PROJECT / "config" / "experiment_config.yaml")
    results = Path(cfg["results"])
    steps = 20 if args.smoke else STEPS
    variant = "adaptation_smoke" if args.smoke else "adaptation"
    jobs = list(plan([int(d) for d in args.draws.split(",")], (25,) if args.smoke else BUDGETS, MODELS))
    if args.smoke:
        jobs = [j for j in jobs if j[3] == "GC10-DET" and j[2] in ("none", "NEU-DET")]
    table = results / "tables" / variant / "target_adaptation.csv"
    for k, (model, condition, source, target, n, draw) in enumerate(jobs, 1):
        done = set()
        if table.exists():
            with table.open(encoding="utf-8") as handle:
                done = {(r["model"], r["condition"], r["source"], r["target"], int(r["n_adapt"]), int(r["draw"])) for r in csv.DictReader(handle)}
        if (model, condition, source, target, n, draw) in done:
            continue
        print(time.strftime("%Y-%m-%d %H:%M:%S"), f"[{k}/{len(jobs)}]", model, condition, source, "->", target, "n", n, "draw", draw, flush=True)
        code = subprocess.run([sys.executable, str(HERE), "--one", "--model", model, "--condition", condition, "--source", source, "--target", target,
                               "--n", str(n), "--draw", str(draw), "--steps", str(steps)], cwd=PROJECT).returncode
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "exit", code, flush=True)
    print("all runs finished", flush=True)


if __name__ == "__main__":
    main()
