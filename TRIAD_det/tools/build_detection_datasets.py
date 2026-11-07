"""Build detection-format copies of SteelDefectX and X-SDD, laid out like NEU-DET / GC10-DET.

    <out>/SteelDefectX-DET/IMAGES/<class>/<stem>.jpg   + ANNOTATIONS/<stem>.xml   (Pascal VOC)
    <out>/X-SDD-DET/IMAGES/<class>/<stem>.jpg          + ANNOTATIONS/<stem>.xml

SteelDefectX boxes come from the official pixel masks (Hugging Face Zhaosxian/SteelDefectX, CC-BY-4.0):
  1. binarise the mask (> 127),
  2. progressive grouping: morphological closing with an elliptical kernel of g x (short image side),
     g = 0.02, 0.05, 0.08, 0.12, 0.20; the smallest g that leaves <= 10 regions is used. The same rule
     is applied to every image and class (no per-class tuning),
  3. regions with fewer defect pixels than max(16 px, 0.05% of the image) are dropped,
  4. one box per region, spanning the defect pixels in it, labelled with the image's class.
X-SDD (Apache-2.0 repository, image labels only) has no boxes. Each X-SDD image is matched to its copy
inside SteelDefectX (32x32 correlation >= 0.95 and 128x128 mean absolute grey error < 10, same mapped
class); the SteelDefectX boxes are rescaled to the X-SDD image. Unmatched X-SDD images get no boxes and
are listed in the build report; they are not given invented boxes.

The source folders are only read. Nothing in SteelDefectX/, NEU-DET/ or GC10-DET/ is changed.

    python tools/build_detection_datasets.py --sdx ../_downloads/SteelDefectX_hf --xsdd ../_downloads/X-SDD_repo/X-SDD/datas --out ..
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triad.common import imread  # noqa: E402

SDX_ALIASES = {"Dark scratches": "Scratches"}
XSDD_TO_SDX = {
    "finishing roll printing": "Finishing roll printing",
    "iron sheet ash": "Iron sheet ash",
    "oxide scale of plate system": "Oxide scale of plate system",
    "oxide scale of temperature system": "Oxide scale of temperature system",
    "red iron": "Red iron sheet",
    "slag inclusion": "Slag inclusion",
    "surface scratch": "Scratches",   # verified empirically by the match report (see build_report.json)
}


GROUP_STEPS = (0.02, 0.05, 0.08, 0.12, 0.20)   # grouping distance, fraction of the short image side
MAX_BOXES = 10


def _grouped_boxes(binary, g, min_area):
    h, w = binary.shape
    k = max(3, int(round(g * min(h, w))) | 1)
    grouped = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, _, _ = cv2.connectedComponentsWithStats(grouped)
    boxes = []
    for i in range(1, n):
        ys, xs = np.nonzero((labels == i) & (binary > 0))     # box spans real defect pixels only
        if len(xs) >= min_area:
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
    return boxes


def mask_boxes(mask, min_frac=0.0005, min_px=16):
    """Progressive grouping, identical for every image and class: the smallest grouping distance
    in GROUP_STEPS that yields 1..MAX_BOXES regions. Fine textures (crazing, pitted surface, scale)
    whose masks are many specks become a few region boxes, as in NEU-DET; compact defects are
    unchanged. Returns (boxes, grouping distance used)."""
    h, w = mask.shape
    binary = (mask > 127).astype(np.uint8)
    min_area = max(min_px, min_frac * h * w)
    for g in GROUP_STEPS:
        boxes = _grouped_boxes(binary, g, min_area)
        if 1 <= len(boxes) <= MAX_BOXES:          # 0 boxes = only sub-threshold specks: keep grouping
            return boxes, g
    return boxes, GROUP_STEPS[-1]


def write_voc(path, folder, filename, width, height, cls, boxes):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = folder
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    for key, value in (("width", width), ("height", height), ("depth", 3)):
        ET.SubElement(size, key).text = str(value)
    for box in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = cls
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), box):
            ET.SubElement(bnd, key).text = str(value)
    ET.indent(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=False)


def _find_mask(mask_dir, stem):
    for ext in (".png", ".jpg", ".bmp"):
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def build_sdx(src, out, report):
    dest = out / "SteelDefectX-DET"
    rows = []
    for split in ("train", "val"):
        text = json.loads((src / f"{split}-text.json").read_text(encoding="utf-8"))
        classes = {r["image_name"]: SDX_ALIASES.get(r["class_name"], r["class_name"]) for r in text}
        for image in sorted((src / split).glob("*.jpg")):
            cls = classes.get(image.name)
            if cls is None:
                report["sdx_no_text_record"].append(image.name)
                continue
            mask_path = _find_mask(src / f"{split}_mask", image.stem)
            if mask_path is None:
                report["sdx_no_mask"].append(image.name)
                continue
            mask = imread(mask_path)
            gray = imread(image)
            if mask.shape != gray.shape:
                mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            boxes, g_used = mask_boxes(mask)
            report["sdx_grouping_used"].append(g_used)
            if not boxes:
                report["sdx_empty_mask"].append(image.name)
            stem = image.stem
            target = dest / "IMAGES" / cls / f"{stem}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target)
            write_voc(dest / "ANNOTATIONS" / f"{stem}.xml", cls, f"{stem}.jpg", gray.shape[1], gray.shape[0], cls, boxes)
            rows.append({"stem": stem, "class": cls, "official_split": split, "boxes": boxes,
                         "path": str(target), "width": gray.shape[1], "height": gray.shape[0]})
    report["sdx_images"] = len(rows)
    used = report.pop("sdx_grouping_used", [])
    report["sdx_grouping_distance_counts"] = {str(k): v for k, v in sorted(collections.Counter(used).items())}
    report["sdx_per_class"] = dict(sorted(collections.Counter(r["class"] for r in rows).items()))
    report["sdx_boxes_per_image"] = dict(sorted(collections.Counter(min(len(r["boxes"]), 20) for r in rows).items()))
    return rows


def _vec(gray):
    v = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def build_xsdd(src, out, sdx_rows, report, corr=0.95, mae_max=10.0):
    dest = out / "X-SDD-DET"
    by_class = collections.defaultdict(list)
    for r in sdx_rows:
        by_class[r["class"]].append(r)
    cache = {}

    def sdx_feat(r):
        if r["stem"] not in cache:
            g = imread(r["path"])
            cache[r["stem"]] = (_vec(g), cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA))
        return cache[r["stem"]]

    links = []
    for folder in sorted(p for p in src.iterdir() if p.is_dir()):
        x_cls = folder.name
        sdx_cls = XSDD_TO_SDX.get(x_cls)
        candidates = by_class.get(sdx_cls, [])
        feats = np.stack([sdx_feat(r)[0] for r in candidates]) if candidates else None
        for image in sorted(folder.iterdir()):
            if image.suffix.lower() not in (".jpg", ".png", ".bmp"):
                continue
            gray = imread(image)
            h, w = gray.shape
            boxes, match = [], None
            if feats is not None:
                sims = feats @ _vec(gray)
                j = int(sims.argmax())
                thumb = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
                mae = float(np.abs(thumb.astype(np.float32) - sdx_feat(candidates[j])[1].astype(np.float32)).mean())
                if sims[j] >= corr and mae < mae_max:
                    s = candidates[j]
                    fx, fy = w / s["width"], h / s["height"]
                    boxes = [[int(round(b[0] * fx)), int(round(b[1] * fy)), int(round(b[2] * fx)), int(round(b[3] * fy))]
                             for b in s["boxes"]]
                    match = {"xsdd": f"{x_cls}/{image.name}", "sdx": s["stem"], "corr": float(sims[j]), "mae": mae}
                    links.append(match)
            if match is None:
                report["xsdd_unmatched"].append(f"{x_cls}/{image.name}")
            stem = f"{x_cls.replace(' ', '_')}_{image.stem}"
            target = dest / "IMAGES" / x_cls / f"{stem}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            colour = imread(image, colour=True)
            cv2.imencode(".jpg", cv2.cvtColor(colour, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tofile(str(target))
            write_voc(dest / "ANNOTATIONS" / f"{stem}.xml", x_cls, f"{stem}.jpg", w, h, x_cls, boxes)
    report["xsdd_matched"] = len(links)
    report["xsdd_unmatched_count"] = len(report["xsdd_unmatched"])
    (dest / "xsdd_to_sdx_links.json").write_text(json.dumps(links, indent=1), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdx", required=True)
    parser.add_argument("--xsdd", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    report = collections.defaultdict(list)
    report["mask_to_box"] = {"threshold": 127, "group_steps": GROUP_STEPS, "max_boxes": MAX_BOXES,
                             "min_frac": 0.0005, "min_px": 16, "box_extent": "defect pixels inside each group"}
    rows = build_sdx(Path(args.sdx), out, report)
    build_xsdd(Path(args.xsdd), out, rows, report)
    for name in ("SteelDefectX-DET", "X-SDD-DET"):
        (out / name / "build_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
