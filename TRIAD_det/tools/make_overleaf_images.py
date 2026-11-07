"""Copy every image the manuscript uses into TRIAD_det/overleaf_images (flat, as Overleaf needs). Safe to re-run.

Result figures and TRIAD-Det batches use the primary analysis: results/figures/full and OUTPUT/batches (extended recipe everywhere).

    python tools/make_overleaf_images.py
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "overleaf_images"
FIG = ROOT / "TRIAD_Det_Risk_Control" / "results" / "figures"
PAPER = ROOT / "manuscript"


def main():
    tex = (PAPER / "sn-article.tex").read_text(encoding="utf-8")
    names = sorted(set(re.findall(r"\\figfile\{([^}#]+\.png)\}", tex)))
    OUT.mkdir(exist_ok=True)
    final, provisional, missing = [], [], []
    for name in names:
        full = FIG / "full" / name
        arch = ROOT / "OUTPUT" / "figures" / "architecture" / name
        tikz = OUT / name
        batch = re.match(r"TRIAD-Det_(.+)_test_batch_\d+\.png", name)
        if batch:
            old = ROOT / "OUTPUT" / "batches" / batch[1] / "TRIAD-Det" / "test" / name
            source = old if old.exists() else None
            group = final
        elif full.exists():
            source, group = full, final
        elif arch.exists():
            source, group = arch, final
        elif name.startswith("fig_tikz") and tikz.exists():
            source, group = tikz, final
        elif full.exists():
            source, group = full, provisional
        else:
            source, group = None, missing
        if source is None:
            missing.append(name)
            continue
        if source != OUT / name:
            shutil.copy2(source, OUT / name)
        group.append(name)
    print(OUT, "| used by tex:", len(names), "| final:", len(final), "| provisional:", len(provisional), "| missing:", len(missing))
    print("provisional:", provisional)
    print("missing:", missing)


if __name__ == "__main__":
    main()
