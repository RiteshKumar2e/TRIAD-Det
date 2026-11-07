"""YOLO training and inference for the box-annotated datasets."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

from .common import dataset_dir, out_path, read_json
from .data import full_path, load_final, record_base

YOLOV8_LAYERS = (2, 4, 13, 16)   # C2f positions in configs/triad_yolo.yaml


def _link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare_yolo(cfg, dataset):
    """Write the YOLO layout. Only train and val are exposed to training."""
    classes = read_json(out_path(cfg, "manifests", "class_maps.json"))["classes"][dataset]
    index = {name: i for i, name in enumerate(classes)}
    root = Path(cfg.out_root) / "_work" / "yolo" / dataset
    shutil.rmtree(root, ignore_errors=True)
    rows = [r for r in load_final(cfg) if r["dataset"] == dataset and r["split"] in ("train", "val")]
    for row in rows:
        stem = row["id"].split("/")[-1]
        _link(full_path(cfg, row), root / "images" / row["split"] / f"{stem}.jpg")
        lines = []
        for item in row["boxes"]:
            x0, y0, x1, y1 = item["box"]
            w, h = row["width"], row["height"]
            lines.append(f'{index[item["cls"]]} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} '
                         f'{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}')
        label = root / "labels" / row["split"] / f"{stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines), encoding="utf-8")
    config = root / "data.yaml"
    config.write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val",
                                      "names": dict(enumerate(classes))}), encoding="utf-8")
    return config


def _checkpoint(cfg, dataset, tag="best"):
    return dataset_dir(cfg, dataset) / "checkpoints" / f"detector_{tag}.pt"


def init_hybrid(model, pretrained):
    """Start TRIAD-YOLO from both parents: YOLO11 overall, YOLOv8 in its C2f layers."""
    from ultralytics import YOLO

    model.load(pretrained["yolo11"])
    donor = YOLO(pretrained["yolov8"]).model.float().state_dict()
    target = model.model.state_dict()
    shared = {k: v for k, v in donor.items()
              if int(k.split(".")[1]) in YOLOV8_LAYERS and k in target and v.shape == target[k].shape}
    model.model.load_state_dict(shared, strict=False)
    return len(shared)


def build_model(cfg):
    from ultralytics import YOLO

    settings = cfg.detector
    model = YOLO(str(Path(cfg.config_path).parent.parent / settings["model"]))
    if settings["pretrained"]:
        print("YOLOv8 tensors loaded into C2f layers:", init_hybrid(model, settings["pretrained"]))
    return model


def train_detector(cfg, dataset, weights=None, tag="best"):
    """Train TRIAD-YOLO, or a baseline when weights are given."""
    from ultralytics import YOLO

    settings = cfg.detector
    os.environ["MLFLOW_TRACKING_URI"] = (Path(cfg.out_root) / "_work" / "mlflow").as_uri()   # keep logs out of the code folder
    data = prepare_yolo(cfg, dataset)
    model = YOLO(weights) if weights else build_model(cfg)
    results = model.train(
        data=str(data), epochs=settings["epochs"], patience=settings["patience"], imgsz=settings["imgsz"][dataset],
        batch=settings["batch"], workers=settings["workers"], amp=settings["amp"], seed=cfg.seed, deterministic=True,
        device=0 if cfg.device == "cuda" else "cpu", flipud=0.5, fliplr=0.5, plots=False, verbose=False,
        project=str(dataset_dir(cfg, dataset) / "checkpoints" / "yolo"), name=f"run_{tag}", exist_ok=True)
    shutil.copy2(Path(results.save_dir) / "weights" / "best.pt", _checkpoint(cfg, dataset, tag))
    from .results_table import write_results
    write_results()


def predict_detector(cfg, source_dataset, rows, tag="best"):
    """Run the source dataset's detector over rows. Low-score boxes are kept on purpose."""
    from ultralytics import YOLO

    settings = cfg.detector
    model = YOLO(str(_checkpoint(cfg, source_dataset, tag)))
    device = 0 if cfg.device == "cuda" else "cpu"
    records = []
    step = 256
    for start in range(0, len(rows), step):
        chunk = rows[start:start + step]
        results = model.predict(
            [full_path(cfg, r) for r in chunk], stream=True, verbose=False, device=device, save=False,
            imgsz=settings["imgsz"][source_dataset], conf=settings["predict_conf"], iou=settings["predict_iou"],
            max_det=settings["max_det"], half=device == 0,
            # GC10-DET at 1024 px with batch 8 ran out of memory at warm-up (2026-11-05): scale the batch by area
            batch=max(1, int(settings["predict_batch"] * min(1.0, (640 / settings["imgsz"][source_dataset]) ** 2))))
        for row, result in zip(chunk, results):
            boxes = result.boxes
            preds = [{"cls": model.names[int(c)], "conf": float(p), "box": [float(v) for v in xyxy]}
                     for c, p, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist())]
            records.append({"model": f"detector:{source_dataset}", "preds": preds, **record_base(row)})
    del model                                   # free GPU memory before the next detector is loaded
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records
