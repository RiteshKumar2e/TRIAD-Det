"""SteelDefectX 24-class classifier with Grad-CAM localisation.

SteelDefectX ships no box or mask annotations, so a detector cannot be trained on
all 24 classes. Image-level labels train this classifier; Grad-CAM supplies the
predicted regions, which are scored separately against whatever annotation exists.
"""
from __future__ import annotations

import collections

import cv2
import numpy as np
import timm
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .common import dataset_dir, imread, seed_everything, write_json
from .data import full_path, load_final, record_base

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class ImageSet(Dataset):
    def __init__(self, cfg, rows, class_index, train=False):
        self.cfg, self.rows, self.index, self.train = cfg, rows, class_index, train
        self.size = cfg.classifier["image_size"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = self.rows[item]
        image = imread(full_path(self.cfg, row), colour=True)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        if self.train:
            if np.random.rand() < 0.5:
                image = image[:, ::-1]
            if np.random.rand() < 0.5:
                image = image[::-1]
            image = np.clip(image * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.05, 0.05), 0, 1)
        image = (image - MEAN) / STD
        label = self.index.get(row["folder_class"], -1)
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))), label, item


def _classes(rows):
    return sorted({r["folder_class"] for r in rows})


def _build(cfg, num_classes, pretrained):
    return timm.create_model(cfg.classifier["model"], pretrained=pretrained, num_classes=num_classes)


def _checkpoint(cfg):
    return dataset_dir(cfg, "SteelDefectX") / "checkpoints" / "classifier_best.pt"


def train_classifier(cfg):
    seed_everything(cfg.seed)
    settings = cfg.classifier
    rows = [r for r in load_final(cfg) if r["dataset"] == "SteelDefectX"]
    classes = _classes(rows)
    index = {name: i for i, name in enumerate(classes)}
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    counts = collections.Counter(r["folder_class"] for r in train_rows)
    missing = [c for c in classes if counts[c] == 0]
    if missing:
        print("WARNING: no training images for", missing)
    weights = torch.tensor([(len(train_rows) / (len(classes) * counts[c])) ** 0.5 if counts[c] else 0.0 for c in classes])

    device = cfg.device
    model = _build(cfg, len(classes), settings["pretrained"]).to(device)
    loader_args = {"batch_size": settings["batch_size"], "num_workers": settings["workers"],
                   "pin_memory": device == "cuda", "persistent_workers": settings["workers"] > 0}
    train_loader = DataLoader(ImageSet(cfg, train_rows, index, train=True), shuffle=True, drop_last=False, **loader_args)
    val_loader = DataLoader(ImageSet(cfg, val_rows, index), shuffle=False, **loader_args)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=settings["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings["epochs"])
    use_amp = settings["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, stale, history = -1.0, 0, []
    for epoch in range(settings["epochs"]):
        model.train()
        total = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(labels)
        scheduler.step()

        model.eval()
        truth, guess = [], []
        with torch.inference_mode():
            for images, labels, _ in val_loader:
                guess += model(images.to(device)).argmax(1).cpu().tolist()
                truth += labels.tolist()
        macro_f1 = f1_score(truth, guess, average="macro", zero_division=0)
        history.append({"epoch": epoch + 1, "train_loss": total / max(len(train_rows), 1), "val_macro_f1": macro_f1})
        print(history[-1], flush=True)
        if macro_f1 > best:
            best, stale = macro_f1, 0
            torch.save({"state_dict": model.state_dict(), "classes": classes, "epoch": epoch + 1,
                        "val_macro_f1": macro_f1, "model": settings["model"], "seed": cfg.seed}, _checkpoint(cfg))
        else:
            stale += 1
            if stale >= settings["patience"]:
                break
    write_json(dataset_dir(cfg, "SteelDefectX") / "metrics" / "classifier_training.json", history)


def cam_boxes(cam, width, height, threshold, min_area):
    """Connected regions of a Grad-CAM map, as boxes in image coordinates."""
    resized = cv2.resize(np.asarray(cam, np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    mask = (resized >= threshold * max(float(resized.max()), 1e-6)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area >= min_area * width * height:
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
    return boxes


def predict_classifier(cfg, rows):
    """Softmax scores plus Grad-CAM for every given row, using the best checkpoint."""
    state = torch.load(_checkpoint(cfg), map_location="cpu", weights_only=False)
    classes = state["classes"]
    device = cfg.device
    model = _build(cfg, len(classes), False)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    settings = cfg.classifier
    loader = DataLoader(ImageSet(cfg, rows, {}), batch_size=settings["batch_size"], num_workers=settings["workers"],
                        shuffle=False, pin_memory=device == "cuda")
    records = []
    for images, _, indices in loader:
        images = images.to(device)
        features = model.forward_features(images)
        logits = model.forward_head(features)
        probabilities = logits.detach().softmax(1)
        top = logits.argmax(1)
        gradient = torch.autograd.grad(logits.gather(1, top[:, None]).sum(), features)[0]
        cams = (gradient.mean((2, 3), keepdim=True) * features).sum(1).relu().detach()
        cams = (cams / cams.flatten(1).amax(1).clamp_min(1e-6).view(-1, 1, 1)).cpu().numpy()
        ranked = probabilities.topk(min(5, len(classes)), dim=1)
        for b, item in enumerate(indices.tolist()):
            row = rows[item]
            top_classes = [{"cls": classes[c], "p": float(p)}
                           for c, p in zip(ranked.indices[b].tolist(), ranked.values[b].tolist())]
            boxes = cam_boxes(cams[b], row["width"], row["height"], settings["cam_threshold"], settings["cam_min_area"])
            records.append({
                "model": "classifier", "top": top_classes, "cam": np.round(cams[b], 3).tolist(),
                "preds": [{"cls": top_classes[0]["cls"], "conf": top_classes[0]["p"], "box": box} for box in boxes],
                **record_base(row),
            })
    return records
