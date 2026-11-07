"""Small zip of the whole TRIAD-Det folder: TRIAD_det/TRIAD-Det_NoCheckpoints.zip. Safe to re-run.

Left out: model weights and everything else under checkpoints folders except CSV logs, .venv, caches, the
pipeline's temporary work folder, smoke-test runs, zips, and runs/triad_full/<dataset>/batches (those images
are titled TRIAD-Det but show TRIAD-YOLO; correct ones are in OUTPUT/batches).

    python tools/make_project_zip.py
"""
import os
import zipfile
from pathlib import Path

TOP = Path(__file__).resolve().parents[2]            # the TRIAD-Det folder
TARGET = TOP / "TRIAD_det" / "TRIAD-Det_NoCheckpoints.zip"
SKIP_DIRS = {".venv", "__pycache__", ".git", "_work", "smoke", ".pytest_cache", ".ipynb_checkpoints"}
WEIGHTS = {".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".bin"}
STORED = {".png", ".jpg", ".jpeg", ".bmp", ".zip", ".pdf", ".gz"}
FAT32_LIMIT = 4 * 2**30 - 1


def keep(path, parts):
    if path.suffix.lower() in WEIGHTS or path.suffix.lower() == ".zip":
        return False
    if "checkpoints" in parts and path.suffix.lower() != ".csv":
        return False
    return not (len(parts) > 3 and parts[:2] == ("TRIAD_det", "runs") and "batches" in parts)


def main():
    total = count = 0
    with zipfile.ZipFile(TARGET, "w", allowZip64=True) as z:
        for folder, dirs, files in os.walk(TOP):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in files:
                path = Path(folder) / name
                rel = path.relative_to(TOP)
                if keep(path, rel.parts):
                    kind = zipfile.ZIP_STORED if path.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
                    z.write(path, Path("TRIAD-Det") / rel, kind)
                    total += path.stat().st_size
                    count += 1
    size = TARGET.stat().st_size
    with zipfile.ZipFile(TARGET) as z:
        bad = z.testzip()
        names = z.namelist()
    print(TARGET, f"{size / 2**30:.2f} GB", f"(from {total / 2**30:.2f} GB)", count, "files; corrupt:", bad)
    print("fits FAT32 4 GB limit:", size <= FAT32_LIMIT)
    print("weights inside:", sum(Path(n).suffix.lower() in WEIGHTS for n in names),
          "| non-CSV under checkpoints:", sum("/checkpoints/" in n and not n.endswith(".csv") for n in names))
    for key in ("TRIAD_det/paper.md", "TRIAD_det/results.csv", "TRIAD_det/run_all.py", "TRIAD_det/OUTPUT/batches/batch_summary.csv",
                "TRIAD_det/OUTPUT/figures/architecture/tikz/triad_det.tex"):
        print(key, f"TRIAD-Det/{key}" in names)
    print("top folders:", sorted({n.split("/")[1] for n in names}))


if __name__ == "__main__":
    main()
