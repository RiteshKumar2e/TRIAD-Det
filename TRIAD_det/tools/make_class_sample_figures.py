"""Class sample figures for SteelDefectX and X-SDD, in the layout of the GC10-DET figure of the manuscript.

Each class is a column of six square crops, each centred on an annotated defect of that class, from six different
training images (different duplicate groups). Selection is random with a fixed seed and does not look at image content.
A crop is shifted only where it would leave the image, so defects near an edge sit off-centre.

    python tools/make_class_sample_figures.py
Writes class_samples/ (PNG and PDF pages) and class_samples/single/<dataset>_<class>.png (one unlabelled column per class).
"""
from __future__ import annotations

import json
import string
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "runs" / "triad_full" / "manifests"
DATA_ROOT = Path(json.loads((MANIFESTS / "resolved_config.json").read_text(encoding="utf-8"))["data_root"])
OUT = ROOT / "class_samples"
SEED = 42
TILES, PER_ROW, ROWS_PER_PAGE = 6, 5, 2
TILE_IN, GAP_IN, COL_IN, CAPTION_IN = 0.62, 0.025, 1.3, 0.46
TILE_PX = 256
CONTEXT, MIN_SHARE = 1.5, 0.5  # crop side = max(1.5 x long box side, 0.5 x short image side), at most the short image side


def jsonl(name):
    with open(MANIFESTS / name, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def crop_tile(path, box):
    with Image.open(path) as source:
        image = source.convert("L")
    w, h = image.size
    x1, y1, x2, y2 = max(box[0], 0), max(box[1], 0), min(box[2], w), min(box[3], h)
    short = min(w, h)
    side = int(round(min(max(CONTEXT * max(x2 - x1, y2 - y1), MIN_SHARE * short), short)))
    left = int(np.clip(round((x1 + x2) / 2 - side / 2), 0, w - side))
    top = int(np.clip(round((y1 + y2) / 2 - side / 2), 0, h - side))
    return image.crop((left, top, left + side, top + side)).resize((TILE_PX, TILE_PX), Image.LANCZOS)


def samples(dataset, classes):
    paths = {r["id"]: r for r in jsonl("split_train.jsonl") if r["dataset"] == dataset}
    truth = [r for r in jsonl("ground_truth.jsonl") if r["dataset"] == dataset and r["split"] == "train" and r["id"] in paths]
    rng = np.random.default_rng(SEED)
    out = {}
    for cls in classes:
        rows = sorted((r for r in truth if any(b["cls"] == cls for b in r["boxes"])), key=lambda r: r["id"])
        tiles, groups = [], set()
        for i in rng.permutation(len(rows)):
            row = rows[int(i)]
            info = paths[row["id"]]
            if info["group"] in groups:
                continue
            box = next(b["box"] for b in row["boxes"] if b["cls"] == cls)
            tiles.append(crop_tile(DATA_ROOT / info["path"], box))
            groups.add(info["group"])
            if len(tiles) == TILES:
                break
        assert len(tiles) == TILES, (dataset, cls, len(tiles))
        out[cls] = tiles
    return out


def page(items, stem, offset):
    rows = [items[i:i + PER_ROW] for i in range(0, len(items), PER_ROW)]
    col_h = TILES * TILE_IN + (TILES - 1) * GAP_IN
    row_h = col_h + CAPTION_IN
    width, height = PER_ROW * COL_IN, len(rows) * row_h
    fig = plt.figure(figsize=(width, height))
    for r, row in enumerate(rows):
        start = (width - len(row) * COL_IN) / 2
        top = height - r * row_h
        for c, (cls, tiles) in enumerate(row):
            x = start + c * COL_IN + (COL_IN - TILE_IN) / 2
            for t, tile in enumerate(tiles):
                y = top - (t + 1) * TILE_IN - t * GAP_IN
                ax = fig.add_axes((x / width, y / height, TILE_IN / width, TILE_IN / height))
                ax.imshow(np.asarray(tile), cmap="gray", vmin=0, vmax=255, interpolation="none")
                ax.axis("off")
            letter = string.ascii_lowercase[offset + r * PER_ROW + c]
            label = "\n".join(textwrap.wrap(f"({letter}) {cls[0].upper()}{cls[1:]}", 22))
            fig.text((x + TILE_IN / 2) / width, (top - col_h - 0.07) / height, label, ha="center", va="top",
                     family="serif", fontweight="bold", fontsize=7.5, linespacing=1.05)
    for ext, dpi in (("png", 400), ("pdf", None)):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=dpi)
    plt.close(fig)
    print(stem, len(items), "classes")


def single(dataset, cls, tiles):
    gap = int(round(TILE_PX * GAP_IN / TILE_IN))
    column = Image.new("L", (TILE_PX, TILES * TILE_PX + (TILES - 1) * gap), 255)
    for t, tile in enumerate(tiles):
        column.paste(tile, (0, t * (TILE_PX + gap)))
    folder = OUT / "single"
    folder.mkdir(exist_ok=True)
    column.save(folder / f"{dataset}_{cls.lower().replace(' ', '_')}.png")


def main():
    OUT.mkdir(exist_ok=True)
    classes = json.loads((MANIFESTS / "class_maps.json").read_text(encoding="utf-8"))["classes"]
    per_page = PER_ROW * ROWS_PER_PAGE
    sdx = list(samples("SteelDefectX", classes["SteelDefectX"]).items())
    for cls, tiles in sdx:
        single("SteelDefectX", cls, tiles)
    for n, start in enumerate(range(0, len(sdx), per_page), 1):
        page(sdx[start:start + per_page], f"fig_classes_SteelDefectX_part{n}", start)
    xsdd = list(samples("X-SDD", classes["X-SDD"]).items())
    for cls, tiles in xsdd:
        single("X-SDD", cls, tiles)
    page(xsdd, "fig_classes_X-SDD", 0)


if __name__ == "__main__":
    main()
