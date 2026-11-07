"""Dataset indexing, duplicate detection, provenance matching and splitting."""
from __future__ import annotations

import collections
import hashlib
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import (DATA_FOLDERS, DATASETS, SDX_TEXT_ALIASES, SDX_TO_GC10, SDX_TO_NEU, SDX_TO_XSDD, SPLITS, imread, out_path,
                     read_json, read_jsonl, write_json, write_jsonl)


def _parse_xml(path):
    root = ET.parse(path).getroot()
    boxes = []
    for obj in root.findall("object"):
        box = [float(obj.find(f"bndbox/{key}").text) for key in ("xmin", "ymin", "xmax", "ymax")]
        boxes.append({"cls": obj.find("name").text, "box": box})
    return boxes


def _limit(paths, limit):
    return sorted(paths)[:limit] if limit else sorted(paths)


def _index_box_dataset(cfg, dataset):
    root = Path(cfg.data_root) / DATA_FOLDERS[dataset]
    rows = []
    for folder in sorted((root / "IMAGES").iterdir()):
        for image in _limit(folder.glob("*.jpg"), cfg.data.limit_per_class):
            boxes = _parse_xml(root / "ANNOTATIONS" / f"{image.stem}.xml")
            rows.append({
                "id": f"{dataset}/{image.stem}", "dataset": dataset, "folder_class": folder.name,
                "path": str(image.relative_to(cfg.data_root)), "boxes": boxes, "box_source": "xml",
                "labels": sorted({box["cls"] for box in boxes}),
            })
    return rows


def _index_steeldefectx(cfg):
    root = Path(cfg.data_root) / "SteelDefectX"
    text = {r["image_name"]: r for r in read_json(root / "train-text.json")}
    rows = []
    for folder in sorted((root / "train_by_class").iterdir()):
        for image in _limit(folder.glob("*.jpg"), cfg.data.limit_per_class):
            meta = text.get(image.name, {})
            attributes = meta.get("structured_attributes", {})
            rows.append({
                "id": f"SteelDefectX/{folder.name}/{image.stem}", "dataset": "SteelDefectX",
                "folder_class": folder.name, "path": str(image.relative_to(cfg.data_root)),
                "boxes": [], "box_source": None, "labels": [folder.name],
                "position": dict(attributes.get("Position", [])),
                "text_class": meta.get("class_name"),
            })
    return rows


def build_index(cfg):
    """Index all four box-annotated datasets. Images whose annotation has no box (an empty
    SteelDefectX mask, or an X-SDD image with no SteelDefectX match) are not detection data:
    they are left out and listed in the audit, never given invented boxes."""
    rows, dropped, missing = [], collections.defaultdict(list), []
    for dataset in DATASETS:
        if not (Path(cfg.data_root) / DATA_FOLDERS[dataset]).exists():
            print(f"WARNING: {DATA_FOLDERS[dataset]} not found; {dataset} skipped in this index", flush=True)
            missing.append(dataset)
            continue
        for row in _index_box_dataset(cfg, dataset):
            if row["boxes"]:
                rows.append(row)
            else:
                dropped[dataset].append(row["id"])
    for row in rows:
        full = Path(cfg.data_root) / row["path"]
        row["sha1"] = hashlib.sha1(full.read_bytes()).hexdigest()
        image = imread(full)
        row["height"], row["width"] = image.shape

    audit = {}
    for dataset in DATASETS:
        subset = [r for r in rows if r["dataset"] == dataset]
        classes = collections.Counter(r["folder_class"] for r in subset)
        boxes = [len(r["boxes"]) for r in subset]
        audit[dataset] = {"images": len(subset), "classes": len(classes), "per_class": dict(sorted(classes.items())),
                          "dropped_no_box": len(dropped[dataset]), "boxes": int(sum(boxes)),
                          "boxes_per_image_mean": float(np.mean(boxes)) if boxes else 0.0,
                          "sizes": dict(collections.Counter(f'{r["width"]}x{r["height"]}' for r in subset).most_common(5))}
        expected = cfg.data.expected.get(dataset)
        if expected and dataset not in missing and not cfg.data.limit_per_class and (len(subset), len(classes)) != (expected["images"], expected["classes"]):
            raise SystemExit(f"{dataset}: found {len(subset)} images / {len(classes)} classes, expected {expected}")
    audit["dropped_ids"] = {k: v for k, v in dropped.items()}
    audit["datasets_missing"] = missing
    audit["exact_byte_duplicates"] = sum(
        count - 1 for count in collections.Counter(r["sha1"] for r in rows).values() if count > 1)

    classes = {}
    for dataset in DATASETS:
        names = {box["cls"] for r in rows if r["dataset"] == dataset for box in r["boxes"]}
        classes[dataset] = sorted(names | {r["folder_class"] for r in rows if r["dataset"] == dataset})
    write_jsonl(out_path(cfg, "manifests", "index.jsonl"), rows)
    write_json(out_path(cfg, "manifests", "class_maps.json"), {
        "classes": classes, "steeldefectx_to_neu": SDX_TO_NEU, "steeldefectx_to_gc10": SDX_TO_GC10,
        "steeldefectx_to_xsdd": SDX_TO_XSDD, "steeldefectx_text_aliases": SDX_TEXT_ALIASES})
    write_json(out_path(cfg, "manifests", "data_audit.json"), audit)
    print({d: audit[d]["images"] for d in DATASETS})
    return rows


def _features(cfg, rows):
    vectors, thumbs = [], []
    for row in rows:
        image = imread(Path(cfg.data_root) / row["path"])
        vector = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        vector -= vector.mean()
        vectors.append(vector / (np.linalg.norm(vector) + 1e-6))
        thumbs.append(cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA))
    return np.stack(vectors), np.stack(thumbs)


def _mae(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def _cross_copies(cfg, sdx, other, f_sdx, f_other, mapping, kind):
    """SteelDefectX images that are copies of an image in another dataset (same mapped class)."""
    if not sdx or not other:
        return []
    edges = []
    similarity = f_sdx[0] @ f_other[0].T
    for i, j in enumerate(similarity.argmax(1)):
        mae = _mae(f_sdx[1][i], f_other[1][j])
        if (similarity[i, j] >= cfg.dedupe.corr_neu and mae < cfg.dedupe.mae_neu
                and mapping.get(sdx[i]["folder_class"]) == other[j]["folder_class"]):
            edges.append({"a": sdx[i]["id"], "b": other[j]["id"], "kind": kind, "mae": mae, "same_class": True})
    return edges


def find_duplicates(cfg):
    """Confirm candidate pairs by pixel error; correlation alone over-matches smooth images."""
    rows = read_jsonl(out_path(cfg, "manifests", "index.jsonl"))
    by_dataset = {d: [r for r in rows if r["dataset"] == d] for d in DATASETS}
    present = [d for d in DATASETS if by_dataset[d]]
    feats = {d: _features(cfg, by_dataset[d]) for d in present}
    edges = []
    for dataset in present:
        vectors, thumbs = feats[dataset]
        subset = by_dataset[dataset]
        for start in range(0, len(subset), 512):
            similarity = vectors[start:start + 512] @ vectors.T
            for i, j in zip(*np.where(similarity >= cfg.dedupe.corr_within)):
                a, b = start + int(i), int(j)
                if a < b and (mae := _mae(thumbs[a], thumbs[b])) < cfg.dedupe.mae_within:
                    edges.append({"a": subset[a]["id"], "b": subset[b]["id"], "kind": "within", "mae": mae,
                                  "same_class": subset[a]["folder_class"] == subset[b]["folder_class"]})
    for target, mapping, kind in (("NEU-DET", SDX_TO_NEU, "sdx_copy_of_neu"), ("X-SDD", SDX_TO_XSDD, "sdx_copy_of_xsdd")):
        if "SteelDefectX" in feats and target in feats:
            edges += _cross_copies(cfg, by_dataset["SteelDefectX"], by_dataset[target], feats["SteelDefectX"], feats[target],
                                   mapping, kind)
    write_jsonl(out_path(cfg, "manifests", "duplicate_edges.jsonl"), edges)
    print("duplicate edges:", dict(collections.Counter(e["kind"] for e in edges)))
    return edges


def _ncc(frames, templates):
    """Normalised cross-correlation of every template with every frame, as one conv."""
    h, w = templates.shape[-2:]
    centred = templates - templates.mean((2, 3), keepdim=True)
    centred = centred / (centred.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-6)
    numerator = F.conv2d(frames, centred)
    mean = F.avg_pool2d(frames, (h, w), stride=1)
    variance = (F.avg_pool2d(frames * frames, (h, w), stride=1) - mean * mean).clamp_min(0)
    score = numerator / (variance * h * w).sqrt().clamp_min(1e-4)
    return score.masked_fill(variance < 1e-5, -1).clamp(max=1.0)


def match_gc10_crops(cfg):
    """Locate SteelDefectX images inside GC10-DET frames of the matching class."""
    rows = read_jsonl(out_path(cfg, "manifests", "index.jsonl"))
    gc10 = [r for r in rows if r["dataset"] == "GC10-DET"]
    device = cfg.device
    scale = cfg.provenance.downscale
    links = []
    with torch.inference_mode():
        for sdx_class, gc10_class in SDX_TO_GC10.items():
            crops = [r for r in rows if r["dataset"] == "SteelDefectX" and r["folder_class"] == sdx_class]
            frames_meta = [r for r in gc10 if gc10_class in r["labels"]]
            if not crops or not frames_meta:
                continue
            frames = torch.stack([torch.from_numpy(cv2.resize(
                imread(Path(cfg.data_root) / r["path"]), (r["width"] // scale, r["height"] // scale),
                interpolation=cv2.INTER_AREA)) for r in frames_meta]).unsqueeze(1).float().div_(255)
            images = [imread(Path(cfg.data_root) / r["path"]) for r in crops]
            best = np.full(len(crops), -1.0)
            where = [None] * len(crops)
            for crop_size in cfg.provenance.crop_sizes:
                side = crop_size // scale
                templates = torch.stack([torch.from_numpy(cv2.resize(i, (side, side), interpolation=cv2.INTER_AREA))
                                         for i in images]).unsqueeze(1).float().div_(255)
                for t0 in range(0, len(crops), cfg.provenance.template_chunk):
                    chunk = templates[t0:t0 + cfg.provenance.template_chunk].to(device)
                    for f0 in range(0, len(frames_meta), cfg.provenance.frame_chunk):
                        block = frames[f0:f0 + cfg.provenance.frame_chunk].to(device)
                        score = _ncc(block, chunk)
                        width = score.shape[-1]
                        flat = score.permute(1, 0, 2, 3).flatten(1)
                        values, index = flat.max(1)
                        for offset, (value, position) in enumerate(zip(values.tolist(), index.tolist())):
                            if value > best[t0 + offset]:
                                frame, rest = divmod(position, score.shape[-2] * width)
                                y, x = divmod(rest, width)
                                best[t0 + offset] = value
                                where[t0 + offset] = (f0 + frame, x * scale, y * scale, crop_size)
            for i, crop in enumerate(crops):
                if where[i] is None:
                    continue
                frame, x, y, size = where[i]
                links.append({"sdx": crop["id"], "gc10": frames_meta[frame]["id"], "ncc": float(best[i]),
                              "x": x, "y": y, "crop_size": size})
            hits = int((best >= cfg.provenance.link_ncc).sum())
            print(f"{sdx_class}: {hits}/{len(crops)} matched at NCC >= {cfg.provenance.link_ncc}", flush=True)
    write_jsonl(out_path(cfg, "manifests", "gc10_crop_links.jsonl"), links)
    return links


def inherit_neu_boxes(sdx, neu):
    factor_x, factor_y = sdx["width"] / neu["width"], sdx["height"] / neu["height"]
    return [{"cls": sdx["folder_class"], "box": [b["box"][0] * factor_x, b["box"][1] * factor_y,
                                                 b["box"][2] * factor_x, b["box"][3] * factor_y]}
            for b in neu["boxes"]]


def inherit_gc10_boxes(sdx, frame, link, min_visible):
    """Project frame boxes into crop coordinates, keeping those substantially inside it."""
    x0, y0, size = link["x"], link["y"], link["crop_size"]
    factor = sdx["width"] / size
    boxes = []
    for item in frame["boxes"]:
        a, b, c, d = item["box"]
        ia, ib, ic, id_ = max(a, x0), max(b, y0), min(c, x0 + size), min(d, y0 + size)
        if ic <= ia or id_ <= ib:
            continue
        if (ic - ia) * (id_ - ib) / ((c - a) * (d - b)) >= min_visible:
            boxes.append({"cls": sdx["folder_class"], "box": [(ia - x0) * factor, (ib - y0) * factor,
                                                              (ic - x0) * factor, (id_ - y0) * factor]})
    return boxes


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def assign_splits(groups, fractions, seed, preassigned=None):
    """Stratified assignment of whole groups, rarest stratum first.

    groups: {group_id: [stratum, ...]} with one stratum per image.
    Every image of a group lands in the same split, so duplicates cannot leak.
    """
    rng = random.Random(seed)
    preassigned = preassigned or []                       # [(stratum, split)] already fixed elsewhere
    totals = collections.Counter(s for members in groups.values() for s in members)
    totals.update(st for st, _ in preassigned)
    current = {s: dict.fromkeys(fractions, 0) for s in totals}
    assigned_total = dict.fromkeys(fractions, 0)
    for stratum, split in preassigned:
        current[stratum][split] += 1
        assigned_total[split] += 1
    grand_total = sum(totals.values())
    order = list(groups)
    rng.shuffle(order)
    order.sort(key=lambda g: (min(totals[s] for s in groups[g]), -len(groups[g])))   # preassigned strata included
    result = {}
    for group in order:
        counts = collections.Counter(groups[group])
        strata = sorted(counts, key=lambda s: totals[s])[:1]

        def need(split):
            stratum = strata[0]
            target = totals[stratum] * fractions[split]
            return ((target - current[stratum][split]) / max(target, 1e-9),
                    fractions[split] * grand_total - assigned_total[split])

        split = max(fractions, key=need)
        result[group] = split
        for stratum, count in counts.items():
            current[stratum][split] += count
        assigned_total[split] += sum(counts.values())
    return result


ANCHORS = ("NEU-DET", "GC10-DET")


def make_splits(cfg):
    rows = read_jsonl(out_path(cfg, "manifests", "index.jsonl"))
    by_id = {r["id"]: r for r in rows}
    edges = read_jsonl(out_path(cfg, "manifests", "duplicate_edges.jsonl"))
    links_path = out_path(cfg, "manifests", "gc10_crop_links.jsonl")
    links = read_jsonl(links_path) if links_path.exists() else []

    union = _UnionFind()
    for row in rows:
        union.find(row["id"])
    # Duplicate and provenance links only group images (so a group never spans splits).
    # Boxes are not inherited any more: every dataset carries its own boxes.
    for edge in edges:
        union.union(edge["a"], edge["b"])
    for link in links:
        if link["ncc"] >= cfg.provenance.link_ncc:
            union.union(link["sdx"], link["gc10"])

    groups = collections.defaultdict(list)
    for row in rows:
        row["group"] = union.find(row["id"])
        groups[row["group"]].append(f'{row["dataset"]}/{row["folder_class"]}')
    fractions = {s: cfg.split[s] for s in SPLITS}
    # Two-phase assignment. ANCHOR datasets (NEU-DET, GC10-DET) are split first, from their own images
    # and within-dataset duplicate groups only, so their splits never change when SteelDefectX / X-SDD
    # are added later. Every other image then joins the split of any anchor image it is linked to
    # (copy or crop); unlinked groups are split stratified, counting the inherited images.
    anchor_union = _UnionFind()
    for row in rows:
        if row["dataset"] in ANCHORS:
            anchor_union.find(row["id"])
    for edge in edges:
        if edge["a"].split("/")[0] in ANCHORS and edge["b"].split("/")[0] in ANCHORS:
            anchor_union.union(edge["a"], edge["b"])
    anchor_groups = collections.defaultdict(list)
    for row in rows:
        if row["dataset"] in ANCHORS:
            anchor_groups[anchor_union.find(row["id"])].append(f'{row["dataset"]}/{row["folder_class"]}')
    anchor_split = assign_splits(anchor_groups, fractions, cfg.seed)
    fixed = {r["id"]: anchor_split[anchor_union.find(r["id"])] for r in rows if r["dataset"] in ANCHORS}

    group_anchor_splits = collections.defaultdict(set)
    for row in rows:
        if row["id"] in fixed:
            group_anchor_splits[row["group"]].add(fixed[row["id"]])
    conflicts = {g for g, v in group_anchor_splits.items() if len(v) > 1}
    if conflicts:
        # a non-anchor image linked to anchors in different splits: drop the non-anchor members
        rows = [r for r in rows if r["dataset"] in ANCHORS or r["group"] not in conflicts]
        for row in rows:      # the bridge is gone: anchors fall back to their anchor-only groups
            if row["group"] in conflicts:
                row["group"] = "anchor:" + anchor_union.find(row["id"])
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row["group"]].append(f'{row["dataset"]}/{row["folder_class"]}')
    # Guard: anchor splits must be identical in every later run (training may already have used them).
    frozen = out_path(cfg, "manifests", "anchor_splits.json")
    if frozen.exists():
        previous = read_json(frozen)
        changed = [k for k, v in fixed.items() if previous.get(k, v) != v]
        if changed or set(previous) != set(fixed):
            raise SystemExit(f"Anchor splits changed for {len(changed)} images; refusing to continue (leakage risk).")
    else:
        write_json(frozen, fixed)
    preassigned, free_groups = [], {}
    for row in rows:
        if row["dataset"] in ANCHORS:
            row["split"] = fixed[row["id"]]
        elif group_anchor_splits.get(row["group"]):
            row["split"] = next(iter(group_anchor_splits[row["group"]]))
            preassigned.append((f'{row["dataset"]}/{row["folder_class"]}', row["split"]))
    for group, members in groups.items():
        if not group_anchor_splits.get(group):
            free_groups[group] = members
    free_split = assign_splits(free_groups, fractions, cfg.seed, preassigned)
    for row in rows:
        if "split" not in row:
            row["split"] = free_split[row["group"]]

    per_class = collections.defaultdict(collections.Counter)
    for row in rows:
        per_class[f'{row["dataset"]}/{row["folder_class"]}'][row["split"]] += 1
    sizes = collections.Counter(collections.Counter(r["group"] for r in rows).values())
    report = {
        "seed": cfg.seed, "fractions": fractions, "n_groups": len(groups), "group_size_histogram": dict(sorted(sizes.items())),
        "per_split": {d: dict(collections.Counter(r["split"] for r in rows if r["dataset"] == d)) for d in DATASETS},
        "per_class": {k: dict(v) for k, v in sorted(per_class.items())},
        "classes_missing_a_split": [k for k, v in sorted(per_class.items()) if any(v[s] == 0 for s in SPLITS)],
        "cross_dataset_groups": sum(len({m.split("/")[0] for m in members}) > 1 for members in groups.values()),
        "groups_crossing_splits": 0,
        "anchor_datasets": list(ANCHORS),
        "inherited_from_anchor": len(preassigned),
        "dropped_conflicting_links": len(conflicts),
    }
    seen = collections.defaultdict(set)
    for row in rows:
        seen[row["group"]].add(row["split"])
    report["groups_crossing_splits"] = sum(len(v) > 1 for v in seen.values())
    if report["groups_crossing_splits"]:
        raise SystemExit("Leakage: a duplicate group spans several splits")
    write_jsonl(out_path(cfg, "manifests", "index_final.jsonl"), rows)
    for split in SPLITS:
        write_jsonl(out_path(cfg, "manifests", f"split_{split}.jsonl"),
                    [{"id": r["id"], "dataset": r["dataset"], "path": r["path"], "group": r["group"],
                      "class": r["folder_class"]} for r in rows if r["split"] == split])
    write_jsonl(out_path(cfg, "manifests", "ground_truth.jsonl"),
                [{k: r.get(k) for k in ("id", "dataset", "split", "labels", "boxes", "box_source", "position")} for r in rows])
    write_json(out_path(cfg, "manifests", "split_report.json"), report)
    print({k: report[k] for k in ("n_groups", "per_split", "classes_missing_a_split", "cross_dataset_groups")})
    return rows


def load_final(cfg):
    return read_jsonl(out_path(cfg, "manifests", "index_final.jsonl"))


def full_path(cfg, row):
    return os.path.join(cfg.data_root, row["path"])


def record_base(row):
    """Fields every prediction record carries, so metrics need no second lookup."""
    return {"id": row["id"], "dataset": row["dataset"], "split": row["split"], "path": row["path"],
            "width": row["width"], "height": row["height"], "gt": row["boxes"], "gt_labels": row["labels"],
            "folder_class": row["folder_class"], "box_source": row["box_source"], "position": row.get("position")}
