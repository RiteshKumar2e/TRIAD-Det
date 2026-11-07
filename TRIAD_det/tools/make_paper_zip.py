"""Everything needed to write the paper, in one zip: TRIAD_det/TRIAD_Paper_Package.zip. Safe to re-run.

    python tools/make_paper_zip.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "TRIAD_Paper_Package"
TABLES = ROOT / "TRIAD_Det_Risk_Control" / "results" / "tables" / "full"
README = """TRIAD-Det paper package

paper.md                          Working draft with all results. Start here.
results.csv                       One row per model and dataset: test metrics and risk control at alpha = 0.10.
Approach.md, MODELS.md            Why the study exists, the approach, and model details.
tables/                           Every result table behind paper.md.
OUTPUT/figures/                   fig01-fig12 result figures (PNG and PDF).
OUTPUT/figures/architecture/      TRIAD-Det, TRIAD-YOLO, selective-inspection pipeline, data protocol (PNG and PDF).
OUTPUT/figures/architecture/tikz/  TikZ source (.tex) and PDF for all 4 detectors; architecture_figures.tex has them all with captions.
OUTPUT/batches/<dataset>/<model>/test/   Detection images on the test split, 30 images per page.
OUTPUT/batches/batch_summary.csv  TP, FP and MISS per model and dataset.

TRIAD-Det means v2 unless a table says v1.
"""


def main():
    items = [(ROOT / n, n) for n in ("paper.md", "results.csv", "Approach.md", "MODELS.md")]
    items += [(p, f"tables/{p.name}") for p in sorted(TABLES.iterdir()) if p.is_file()]
    items += [(p, p.relative_to(ROOT).as_posix()) for p in sorted((ROOT / "OUTPUT").rglob("*")) if p.is_file()]
    target = ROOT / f"{NAME}.zip"
    with zipfile.ZipFile(target, "w") as z:
        z.writestr(f"{NAME}/README.txt", README, zipfile.ZIP_DEFLATED)
        for path, name in items:
            kind = zipfile.ZIP_STORED if path.suffix == ".png" else zipfile.ZIP_DEFLATED
            z.write(path, f"{NAME}/{name}", kind)
    with zipfile.ZipFile(target) as z:
        bad = z.testzip()
        names = z.namelist()
    print(target, f"{target.stat().st_size / 2**20:.0f} MB", len(names), "files; corrupt:", bad)
    for key in ("/paper.md", "/results.csv", "figures/architecture/fig_arch_triad_det.pdf", "figures/fig12_examples_X-SDD.png", "batch_summary.csv", "tikz/triad_det.tex", "tikz/yolov8s.tex", "tikz/yolo11s.tex", "tikz/triad_yolo.tex", "tikz/architecture_figures.tex", "tables/validity_decomposition.csv"):
        print(key, any(n.endswith(key) for n in names))
    print("batch pngs", sum("/batches/" in n and n.endswith(".png") for n in names), "tables", sum("/tables/" in n for n in names))


if __name__ == "__main__":
    main()
