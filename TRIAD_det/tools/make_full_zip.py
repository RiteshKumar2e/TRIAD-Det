"""Full backup zip: TRIAD_det/TRIAD-Det_Full.zip. Safe to re-run.

Holds the four datasets and, from TRIAD_det, OUTPUT, runs, tests, tools, triad, TRIAD_Det_Risk_Control, README.md,
requirements.txt, results.csv, run_all.py, yolo11s.pt and yolov8s.pt. The folder layout is kept
(TRIAD-Det/<dataset>/ next to TRIAD-Det/TRIAD_det/), so the relative data_root paths in the configs still work.
Left out: trained checkpoints and other model weights (.pt .pth .ckpt .onnx .safetensors .bin; everything under
checkpoints folders except CSV training logs), results/_adaptation_work (copies of dataset images from the stopped
adaptation run), .venv, __pycache__, .pytest_cache, .ipynb_checkpoints and zip files.

    .venv/Scripts/python.exe tools/make_full_zip.py
"""
import os
import zipfile
from pathlib import Path

TOP = Path(__file__).resolve().parents[2]            # the TRIAD-Det folder
REPO = TOP / "TRIAD_det"
TARGET = REPO / "TRIAD-Det_Full.zip"
TEMP = TARGET.with_suffix(".zip.part")
DATASETS = ("SteelDefectX-DET", "NEU-DET", "GC10-DET", "X-SDD-DET")
FOLDERS = ("OUTPUT", "runs", "tests", "tools", "triad", "TRIAD_Det_Risk_Control")
FILES = ("README.md", "requirements.txt", "requirements-lock.txt", "results.csv", "run_all.py", "yolo11s.pt", "yolov8s.pt")
SKIP_DIRS = {".venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git", "_adaptation_work"}
WEIGHTS = {".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".bin"}
STORED = {".png", ".jpg", ".jpeg", ".bmp", ".pdf", ".gz", ".pt", ".pth"}


def walk(root):
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            path = Path(folder) / name
            suffix = path.suffix.lower()
            if suffix in (".zip", ".part") or suffix in WEIGHTS:
                continue
            if "checkpoints" in path.parts and suffix != ".csv":
                continue
            yield path


def main():
    paths = [p for d in DATASETS for p in walk(TOP / d)]
    paths += [p for d in FOLDERS for p in walk(REPO / d)]
    paths += [REPO / f for f in FILES]                # yolo11s.pt / yolov8s.pt: pretrained starting weights, kept
    missing = [p for p in paths if not p.exists()]
    assert not missing, missing
    total = 0
    with zipfile.ZipFile(TEMP, "w", allowZip64=True) as z:
        for path in paths:
            kind = zipfile.ZIP_STORED if path.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
            z.write(path, Path("TRIAD-Det") / path.relative_to(TOP), kind)
            total += path.stat().st_size
    with zipfile.ZipFile(TEMP) as z:
        bad = z.testzip()
        names = z.namelist()
    assert bad is None, bad
    assert len(names) == len(paths)
    os.replace(TEMP, TARGET)
    print(TARGET, f"{TARGET.stat().st_size / 2**30:.2f} GB (from {total / 2**30:.2f} GB)", len(names), "files; corrupt: None")
    print("weights inside:", [n for n in names if Path(n).suffix.lower() in WEIGHTS])
    print("non-CSV under checkpoints:", sum("/checkpoints/" in n and not n.endswith(".csv") for n in names))
    for part in DATASETS + tuple(f"TRIAD_det/{d}" for d in FOLDERS):
        print(f"  {part}: {sum(n.startswith(f'TRIAD-Det/{part}/') for n in names)} files")
    print("  top files:", [f for f in FILES if f"TRIAD-Det/TRIAD_det/{f}" in names])


if __name__ == "__main__":
    main()
