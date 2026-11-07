"""Train TRIAD-Det on NEU-DET / GC10-DET and write prediction records.

Uses exactly the splits written by the base pipeline (runs/triad_full/manifests/index_final.jsonl),
so TRIAD-Det, YOLOv8s, YOLO11s and the hybrid TRIAD-YOLO are compared on identical images.
Only train is used for fitting and val for model selection; calib and test are never seen.

    python -m src.triad_det_model.train --dataset NEU-DET
    python -m src.triad_det_model.train --dataset GC10-DET
    python -m src.triad_det_model.train --dataset NEU-DET --smoke     # 1 epoch, 16 images per split
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[2]                       # TRIAD_Det_Risk_Control/
sys.path.insert(0, str(PROJECT.parent))         # TRIAD_det/ for the shared `triad` package
from triad.common import imread, read_jsonl, write_jsonl, DATASETS  # noqa: E402
from triad.evaluate import map_metrics  # noqa: E402

from .model import TriadDet  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_cfg(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["pipeline_run"] = str((PROJECT / cfg["pipeline_run"]).resolve())
    cfg["data_root"] = str((PROJECT / cfg["data_root"]).resolve())
    cfg["results"] = str((PROJECT / cfg["results"]).resolve())
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def letterbox(image, size):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    canvas = np.zeros((size, size, 3), np.float32)
    canvas[:nh, :nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return canvas, scale


def _clip_boxes(boxes, labels, w, h, keep_frac=0.35, min_side=2.0):
    """Clip boxes to the image; drop boxes that lost most of their area or became slivers."""
    if len(boxes) == 0:
        return boxes, labels
    area0 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (bw >= min_side) & (bh >= min_side) & (bw * bh >= keep_frac * np.maximum(area0, 1e-6))
    return boxes[keep], labels[keep]


class DetSet(Dataset):
    """Training augmentation (v2, `aug` block of the config; v1 = flips + brightness only):
    mosaic of four training images (probability aug.mosaic, switched off for the last epochs),
    random scale aug.scale and translation, flips, brightness / contrast / gamma jitter.
    Validation, calibration and test images are only letterboxed."""

    def __init__(self, rows, data_root, classes, size, train, aug=None):
        self.rows, self.root, self.size, self.train = rows, data_root, size, train
        self.index = {c: i for i, c in enumerate(classes)}
        self.aug = aug or {}
        self.mosaic = float(self.aug.get("mosaic", 0.0)) if train else 0.0

    def __len__(self):
        return len(self.rows)

    def _load(self, item):
        row = self.rows[item]
        image = imread(Path(self.root) / row["path"], colour=True).astype(np.float32) / 255
        boxes = np.array([b["box"] for b in row["boxes"]], np.float32).reshape(-1, 4)
        labels = np.array([self.index.get(b["cls"], -1) for b in row["boxes"]], np.int64)   # -1: foreign class (transfer only)
        return image, boxes, labels

    def _flip_jitter(self, image, boxes):
        h, w = image.shape[:2]
        if random.random() < 0.5:
            image = image[:, ::-1]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        if random.random() < 0.5:
            image = image[::-1]
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        return np.ascontiguousarray(image), boxes

    def _photometric(self, image):
        image = np.clip(image * random.uniform(0.75, 1.25) + random.uniform(-0.1, 0.1), 0, 1)
        if self.aug and random.random() < 0.5:
            image = image ** random.uniform(0.7, 1.4)
        return image

    def _scaled(self, item):
        """One image letterboxed to `size`, then randomly scaled and shifted on a size x size canvas."""
        image, boxes, labels = self._load(item)
        image, boxes = self._flip_jitter(image, boxes)
        image, scale = letterbox(image, self.size)
        boxes = boxes * scale
        lo, hi = self.aug.get("scale", (1.0, 1.0))
        s = random.uniform(lo, hi)
        if abs(s - 1.0) > 1e-3 or self.aug.get("translate", 0) > 0:
            size = self.size
            t = self.aug.get("translate", 0.0) * size
            dx, dy = random.uniform(-t, t), random.uniform(-t, t)
            cx = size / 2
            m = np.float32([[s, 0, (1 - s) * cx + dx], [0, s, (1 - s) * cx + dy]])
            image = cv2.warpAffine(image, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            boxes = boxes * s + np.float32([m[0, 2], m[1, 2], m[0, 2], m[1, 2]])
            boxes, labels = _clip_boxes(boxes, labels, size, size)
        return image, boxes, labels

    def _mosaic(self, item):
        size = self.size
        canvas = np.zeros((2 * size, 2 * size, 3), np.float32)
        all_boxes, all_labels = [], []
        items = [item] + [random.randrange(len(self.rows)) for _ in range(3)]
        for k, idx in enumerate(items):
            image, boxes, labels = self._scaled(idx)
            ox, oy = (k % 2) * size, (k // 2) * size
            canvas[oy:oy + size, ox:ox + size] = image
            all_boxes.append(boxes + np.float32([ox, oy, ox, oy]))
            all_labels.append(labels)
        boxes, labels = np.concatenate(all_boxes), np.concatenate(all_labels)
        x0, y0 = random.randint(size // 4, 3 * size // 4), random.randint(size // 4, 3 * size // 4)
        image = canvas[y0:y0 + size, x0:x0 + size]
        boxes = boxes - np.float32([x0, y0, x0, y0])
        boxes, labels = _clip_boxes(boxes, labels, size, size)
        return np.ascontiguousarray(image), boxes, labels, 1.0

    def __getitem__(self, item):
        if self.train and self.mosaic > 0 and random.random() < self.mosaic:
            image, boxes, labels, scale = self._mosaic(item)
            if len(boxes) == 0:                                    # crop missed every defect: plain sample
                image, boxes, labels = self._scaled(item)
        elif self.train and self.aug:
            image, boxes, labels = self._scaled(item)
            scale = 1.0
        else:
            image, boxes, labels = self._load(item)
            if self.train:
                image, boxes = self._flip_jitter(image, boxes)
            image, scale = letterbox(np.ascontiguousarray(image), self.size)
            boxes = boxes * scale
        if self.train:
            image = self._photometric(image)
        image = (image - MEAN) / STD
        return (torch.from_numpy(image.transpose(2, 0, 1).copy()), torch.from_numpy(np.asarray(boxes, np.float32)),
                torch.from_numpy(np.asarray(labels, np.int64)), scale, item)


class ModelEMA:
    """Exponential moving average of the weights (decay ramps up over the first updates)."""

    def __init__(self, model, decay=0.9998, tau=2000):
        import copy
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


def collate(batch):
    images, boxes, labels, scales, items = zip(*batch)
    return torch.stack(images), list(boxes), list(labels), list(scales), list(items)


def rows_for(cfg, dataset, splits, smoke):
    rows = read_jsonl(Path(cfg["pipeline_run"]) / "manifests" / "index_final.jsonl")
    out = {s: [r for r in rows if r["dataset"] == dataset and r["split"] == s] for s in splits}
    if smoke:
        out = {s: v[:16] for s, v in out.items()}
    return out


def class_list(cfg, dataset):
    maps = json.loads((Path(cfg["pipeline_run"]) / "manifests" / "class_maps.json").read_text(encoding="utf-8"))
    return maps["classes"][dataset]


def record_base(row):
    return {"id": row["id"], "dataset": row["dataset"], "split": row["split"], "path": row["path"],
            "width": row["width"], "height": row["height"], "gt": row["boxes"], "gt_labels": row["labels"],
            "folder_class": row["folder_class"], "box_source": row["box_source"], "position": row.get("position")}


@torch.no_grad()
def predict_rows(model, rows, cfg, classes, size, device, source, batch=8, workers=0):
    """Prediction records in the same format as the YOLO pipeline, plus the census branch."""
    s = cfg["triad_det"]
    model.eval()
    loader = DataLoader(DetSet(rows, cfg["data_root"], classes, size, False), batch_size=batch, shuffle=False,
                        num_workers=workers, collate_fn=collate)
    records = []
    for images, _, _, scales, items in loader:
        with torch.autocast("cuda", enabled=device == "cuda" and s["amp"]):
            out = model(images.to(device, non_blocking=True))
        for result, scale, item in zip(model.predict(out, s["predict_conf"], s["predict_iou"], s["max_det"], s["pre_nms_topk"]),
                                       scales, items):
            row = rows[item]
            boxes = (result["boxes"] / scale).cpu()
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, row["width"])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, row["height"])
            preds = [{"cls": classes[int(c)], "conf": float(p), "box": [float(v) for v in b]}
                     for b, p, c in zip(boxes.tolist(), result["scores"].tolist(), result["labels"].tolist())]
            records.append({"model": f"triad_det:{source}", "preds": preds,
                            "census": {"count": result["count"],
                                       "presence": {classes[i]: float(p) for i, p in enumerate(result["presence"].tolist())}},
                            **record_base(row)})
    return records


def train(cfg, dataset, smoke=False, device="cuda"):
    seed_all(cfg["seed"])
    s = cfg["triad_det"]
    classes = class_list(cfg, dataset)
    size = 256 if smoke else s["imgsz"][dataset]
    splits = rows_for(cfg, dataset, ("train", "val"), smoke)
    model = TriadDet(len(classes), {**s, "pretrained_backbone": s["pretrained_backbone"] and not smoke}).to(device)
    epochs = 1 if smoke else s["epochs"][dataset]
    batch = 4 if smoke else s["batch"][dataset]
    aug = s.get("aug")
    train_set = DetSet(splits["train"], cfg["data_root"], classes, size, True, aug)
    close_mosaic = int((aug or {}).get("close_mosaic", 0))
    loader = DataLoader(train_set, batch_size=batch, shuffle=True,
                        num_workers=0 if smoke else s["workers"], collate_fn=collate, drop_last=True,
                        persistent_workers=not smoke and s["workers"] > 0, pin_memory=device == "cuda")
    backbone = list(model.backbone.parameters())
    ids = {id(p) for p in backbone}
    rest = [p for p in model.parameters() if id(p) not in ids]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": s["lr"] * s["backbone_lr_mult"]},
                                   {"params": rest, "lr": s["lr"]}], weight_decay=s["weight_decay"])
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    total_steps = max(epochs * len(loader), 1)
    warm = s["warmup_epochs"] * len(loader)
    use_amp = s["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(cfg["results"]) / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = checkpoint_tag(s, dataset, smoke)
    ema = ModelEMA(model) if s.get("ema", False) else None
    ckpt = out_dir / f"{tag}.pt"
    log_path = Path(cfg["results"]) / "logs" / f"{tag}_train.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history, best, stale, step = [], -1.0, 0, 0
    epoch = 0
    while epoch < epochs:
        if close_mosaic and epoch == epochs - close_mosaic and train_set.mosaic > 0:
            train_set.mosaic = 0.0                     # last epochs on undistorted layouts (as in YOLO)
            loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=s["workers"], collate_fn=collate,
                                drop_last=True, persistent_workers=s["workers"] > 0, pin_memory=device == "cuda")
            print(f"mosaic off from epoch {epoch + 1}", flush=True)
        model.train()
        started, sums = time.time(), {}
        for images, boxes, labels, _, _ in loader:
            lr_scale = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total_steps - warm, 1)))
            for g, lr in zip(optimizer.param_groups, base_lrs):
                g["lr"] = lr * lr_scale
            images = images.to(device, non_blocking=True)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            with torch.autocast("cuda", enabled=use_amp):
                out = model(images)
            loss, parts = model.loss(out, boxes, labels, s)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            step += 1
            for k, v in {"loss": float(loss), **parts}.items():
                sums[k] = sums.get(k, 0.0) + v
        budget = (s.get("max_hours") or {}).get(dataset)
        if budget and not smoke and epoch == 1:
            # fit the run into a wall-clock budget: fix the epoch count after the second epoch (the first one
            # carries worker start-up and cuDNN tuning, 67 s vs 14 s on X-SDD), so the
            # cosine schedule and the mosaic switch-off still complete (val passes counted in)
            per_epoch = (time.time() - started) * (1 + 0.6 / s["eval_every"])
            fitted = max(10, min(epochs, int(budget * 3600 / per_epoch)))
            if fitted < epochs:
                print(f"time budget {budget} h: {epochs} -> {fitted} epochs", flush=True)
                epochs = fitted
                total_steps = max(epochs * len(loader), 1)
                close_mosaic = min(close_mosaic, max(epochs // 7, 1)) if close_mosaic else 0
        entry = {"epoch": epoch + 1, "seconds": round(time.time() - started, 1),
                 **{f"train_{k}": v / max(len(loader), 1) for k, v in sums.items()}}
        if (epoch + 1) % s["eval_every"] == 0 or epoch + 1 == epochs:
            eval_model = ema.ema if ema is not None else model
            records = predict_rows(eval_model, splits["val"], cfg, classes, size, device, dataset)
            scores = map_metrics(records, {c: i for i, c in enumerate(classes)})
            entry.update(val_mAP50=scores["AP50"], val_mAP50_95=scores["mAP"])
            if scores["AP50"] > best:
                best, stale = scores["AP50"], 0
                torch.save({"state_dict": eval_model.state_dict(), "classes": classes, "epoch": epoch + 1, "val_AP50": best,
                            "cfg": s, "imgsz": size, "dataset": dataset, "seed": cfg["seed"]}, ckpt)
            else:
                stale += s["eval_every"]
        history.append(entry)
        print(json.dumps(entry), flush=True)
        _write_csv(log_path, history)
        if not smoke:
            from triad.results_table import write_results
            write_results()
        if stale >= s["patience"]:
            print(f"early stop at epoch {epoch + 1}; best val AP50 {best:.4f}", flush=True)
            break
        epoch += 1
    return ckpt


def checkpoint_tag(s, dataset, smoke=False):
    version = s.get("version", "v1")
    return (f"triad_det_{dataset}" if version == "v1" else f"triad_det_{version}_{dataset}") + ("_smoke" if smoke else "")


def _write_csv(path, rows):
    import csv
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with Path(path).open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def load_model(ckpt, device):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = TriadDet(len(state["classes"]), {**state["cfg"], "pretrained_backbone": False})
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval(), state


def predict_all(cfg, dataset, ckpt, smoke=False, device="cuda"):
    """In-domain records for val/calib/test (and train), and transfer records on the other datasets."""
    model, state = load_model(ckpt, device)
    classes, size = state["classes"], state["imgsz"]
    pred_dir = Path(cfg["results"]) / "predictions" / ("smoke" if smoke else "full")
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = rows_for(cfg, dataset, ("train", "val", "calib", "test"), smoke)
    records = [r for split in ("train", "val", "calib", "test") for r in predict_rows(model, rows[split], cfg, classes, size, device, dataset)]
    write_jsonl(pred_dir / f"TRIAD-Det__{dataset}__on__{dataset}.jsonl", records)
    for target in DATASETS:
        if target == dataset:
            continue
        target_rows = rows_for(cfg, target, ("val", "calib", "test"), smoke)
        subset = [r for split in ("val", "calib", "test") for r in target_rows[split] if r["boxes"]]
        write_jsonl(pred_dir / f"TRIAD-Det__{dataset}__on__{target}.jsonl",
                    predict_rows(model, subset, cfg, classes, size, device, dataset))
    print("predictions written to", pred_dir, flush=True)
    if not smoke:
        from triad.results_table import write_results
        write_results()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT / "config" / "experiment_config.yaml"))
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--train-only", action="store_true", help="skip prediction (run it once all splits exist)")
    parser.add_argument("--pipeline-run", default=None, help="override pipeline_run (e.g. ../runs/smoke)")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    if args.pipeline_run:
        cfg["pipeline_run"] = str((PROJECT / args.pipeline_run).resolve())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    ckpt = Path(cfg["results"]) / "checkpoints" / f"{checkpoint_tag(cfg['triad_det'], args.dataset, args.smoke)}.pt"
    if not args.predict_only:
        ckpt = train(cfg, args.dataset, args.smoke)
    if not args.train_only:
        predict_all(cfg, args.dataset, ckpt, args.smoke)


if __name__ == "__main__":
    main()
