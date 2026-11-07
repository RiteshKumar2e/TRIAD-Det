"""Architecture and method diagrams for the paper, drawn from the code.

  fig_arch_triad_det     TRIAD_Det_Risk_Control/src/triad_det_model/model.py + config (neck_channels 192, head_convs 3)
  fig_arch_triad_yolo    configs/triad_yolo.yaml
  fig_pipeline_selective src/calibration.py, src/evaluation.py, src/selective_decision.py
  fig_data_protocol      paper.md 7.0-7.1, MODELS.md

Writes PDF and PNG to OUTPUT/figures/architecture.   python tools/make_architecture_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "OUTPUT" / "figures" / "architecture"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "pdf.fonttype": 42, "mathtext.fontset": "stix"})
BLUE, GREEN, ORANGE, PURPLE, RED, GREY, YELLOW = "#DBEAFE", "#DCFCE7", "#FFEDD5", "#EDE9FE", "#FEE2E2", "#F3F4F6", "#FEF9C3"
EDGE = "#374151"


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def box(ax, x, y, w, h, text, fc=GREY, size=9.5, weight="normal", ec=EDGE):
    """x, y = lower-left corner."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=ec, lw=0.9, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, zorder=3, linespacing=1.15)


def group(ax, x, y, w, h, title, ec="#9CA3AF", size=10.5):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=ec, lw=0.9, ls=(0, (4, 3)), zorder=1))
    ax.text(x + w / 2, y + h - 0.08, title, ha="center", va="top", fontsize=size, weight="bold", color="#111827")


def arrow(ax, a, b, rad=0.0, color=EDGE, ls="-", lw=1.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=lw, color=color, ls=ls, zorder=4,
                                 shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def elbow(ax, points, color=EDGE, ls="-"):
    """Orthogonal connector through the given points, arrow head on the last segment."""
    for p, q in zip(points[:-2], points[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=1.0, ls=ls, zorder=4, solid_capstyle="round")
    arrow(ax, points[-2], points[-1], color=color, ls=ls)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def triad_det():
    fig, ax = canvas(13.6, 6.6)
    rows = {"3": 1.7, "4": 2.95, "5": 4.2}
    bw, bh = 1.25, 0.8
    box(ax, 0.1, 2.75, 0.95, 1.0, "Input\nimage\n3 × H × W", GREY)

    group(ax, 1.3, 1.45, 1.55, 4.15, "Backbone\nConvNeXt-Tiny")
    for k, (name, c, s) in {"3": ("C3", 192, 8), "4": ("C4", 384, 16), "5": ("C5", 768, 32)}.items():
        box(ax, 1.45, rows[k], bw, bh, f"{name}\n{c} ch, stride {s}", BLUE)
    elbow(ax, [(1.05, 3.25), (1.2, 3.25), (1.2, rows["3"] + bh / 2), (1.45, rows["3"] + bh / 2)])
    arrow(ax, (1.05, 3.25), (1.45, rows["4"] + bh / 2))
    elbow(ax, [(1.2, 3.25), (1.2, rows["5"] + bh / 2), (1.45, rows["5"] + bh / 2)])

    group(ax, 3.05, 0.35, 4.55, 5.25, "BiFusion neck (192 channels, Conv + GroupNorm + SiLU)")
    for k in rows:
        y = rows[k]
        box(ax, 3.2, y, 0.95, bh, "1×1 conv\nlateral", GREEN)
        box(ax, 4.55, y, 1.05, bh, f"Top-down\nTD{k}", GREEN)
        box(ax, 6.2, y, 1.25, bh, f"P{k}\nstride {2 ** int(k)}" if k == "3" else f"Bottom-up\nP{k}, stride {2 ** int(k)}", GREEN)
        arrow(ax, (2.7, y + bh / 2), (3.2, y + bh / 2))
        arrow(ax, (4.15, y + bh / 2), (4.55, y + bh / 2))
        arrow(ax, (5.6, y + bh / 2), (6.2, y + bh / 2))
    for a, b in (("5", "4"), ("4", "3")):
        arrow(ax, (5.075, rows[a]), (5.075, rows[b] + bh))
    ax.text(5.14, 4.07, "×2 up, add", fontsize=8, va="center")
    ax.text(5.14, 2.82, "×2 up, add", fontsize=8, va="center")
    for a, b in (("3", "4"), ("4", "5")):
        arrow(ax, (6.82, rows[a] + bh), (6.82, rows[b]))
    ax.text(6.88, 2.82, "stride 2, concat", fontsize=8, va="center")
    ax.text(6.88, 4.07, "stride 2, concat", fontsize=8, va="center")
    box(ax, 6.2, 0.5, 1.25, 0.72, "Extra conv\nP6, stride 64", GREEN)
    arrow(ax, (6.82, rows["3"]), (6.82, 1.22))

    ax.plot([7.6, 7.85], [3.35, 3.35], color=EDGE, lw=1.4, zorder=4)
    ax.plot([7.85, 7.85], [1.45, 5.0], color=EDGE, lw=1.4, zorder=4)
    ax.text(7.83, 5.12, "P3–P6", fontsize=8.5, ha="center", weight="bold")

    group(ax, 8.05, 2.45, 2.85, 3.15, "Detection head (shared across levels)")
    box(ax, 8.2, 4.2, 1.3, 0.8, "Class tower\n3 conv blocks", ORANGE)
    box(ax, 8.2, 2.9, 1.3, 0.8, "Box tower\n3 conv blocks", ORANGE)
    box(ax, 9.8, 4.2, 0.98, 0.8, "Class\nK", ORANGE)
    box(ax, 9.8, 3.42, 0.98, 0.62, "Box 4\n× level scale", ORANGE, size=8.5)
    box(ax, 9.8, 2.6, 0.98, 0.62, "Quality\n1", ORANGE)
    arrow(ax, (7.85, 4.6), (8.2, 4.6))
    arrow(ax, (7.85, 3.3), (8.2, 3.3))
    arrow(ax, (9.5, 4.6), (9.8, 4.6))
    arrow(ax, (9.5, 3.3), (9.8, 3.7))
    arrow(ax, (9.5, 3.3), (9.8, 2.91))

    group(ax, 8.05, 0.35, 2.85, 1.95, "Census head (image level)")
    box(ax, 8.2, 0.55, 1.15, 1.15, "Global avg pool\neach level,\nconcat\n192 × 4", PURPLE, size=8.5)
    box(ax, 9.55, 0.55, 1.23, 1.15, "MLP\n768 → 256\n→ 1 + K\n(dropout 0.2)", PURPLE, size=8.5)
    elbow(ax, [(7.85, 1.45), (7.85, 1.12), (8.2, 1.12)])
    arrow(ax, (9.35, 1.12), (9.55, 1.12))

    box(ax, 11.25, 3.75, 1.85, 1.1, "Boxes and classes\nscore = √(class × quality)", YELLOW, size=9)
    box(ax, 11.25, 1.3, 1.85, 1.1, "Defect count (Poisson)\nClass presence (K)", YELLOW, size=9)
    box(ax, 11.25, 0.05, 1.85, 0.9, "Census uncertainty score\n(count gap, unexplained\nclass presence, 1 − max conf)", RED, size=8)
    arrow(ax, (10.9, 4.3), (11.25, 4.3))
    arrow(ax, (10.9, 1.5), (11.25, 1.85))
    arrow(ax, (12.17, 1.3), (12.17, 0.95))
    elbow(ax, [(13.1, 4.3), (13.42, 4.3), (13.42, 0.5), (13.1, 0.5)], ls=(0, (3, 2)))
    ax.text(0.1, 6.28, "TRIAD-Det: non-YOLO anchor-free detector with an image-level census branch", fontsize=13, weight="bold")
    ax.text(0.1, 5.98, "K = number of classes in the dataset.  Training: ATSS assignment, Quality Focal Loss, GIoU box loss, EMA weights.", fontsize=10)
    save(fig, "fig_arch_triad_det")


def triad_yolo():
    fig, ax = canvas(12.6, 5.4)
    col = {"v8": BLUE, "11": GREEN, "base": GREY, "out": YELLOW}
    bw, bh = 1.3, 0.52
    y0 = 0.3
    backbone = [("Conv\nP1/2", "base"), ("Conv\nP2/4", "base"), ("C2f × 3\nP2", "v8"), ("Conv\nP3/8", "base"), ("C2f × 6\nP3", "v8"),
                ("Conv\nP4/16", "base"), ("C3k2 × 2\nP4", "11"), ("Conv\nP5/32", "base")]
    backbone = [(t, c) for t, c in backbone]
    y = {}
    group(ax, 0.15, 0.1, 1.7, 4.62, "Backbone")
    items = backbone + [("C3k2 × 2\nP5", "11")]
    ys = [y0 + 0.0 + i * 0.47 for i in range(len(items))]
    for i, ((t, c), yy) in enumerate(zip(items, ys)):
        box(ax, 0.35, yy, bw, 0.4, t, col[c], size=7.8)
        y[i] = yy
        if i:
            arrow(ax, (1.0, ys[i - 1] + 0.4), (1.0, yy))
    group(ax, 2.05, 3.1, 1.6, 1.62, "P5 context")
    box(ax, 2.2, 3.25, 1.3, 0.45, "SPPF", col["base"], size=9)
    box(ax, 2.2, 3.88, 1.3, 0.45, "C2PSA × 2\nattention", col["11"], size=8)
    elbow(ax, [(1.65, ys[8] + 0.2), (1.85, ys[8] + 0.2), (1.85, 3.47), (2.2, 3.47)])
    arrow(ax, (2.85, 3.7), (2.85, 3.88))

    group(ax, 4.15, 0.1, 2.05, 4.62, "Top-down neck")
    box(ax, 4.35, 3.88, 1.65, 0.45, "Upsample + concat P4", col["base"], size=8.2)
    box(ax, 4.35, 3.05, 1.65, 0.5, "C2f (512)\nYOLOv8", col["v8"], size=8.5)
    box(ax, 4.35, 2.2, 1.65, 0.45, "Upsample + concat P3", col["base"], size=8.2)
    box(ax, 4.35, 1.3, 1.65, 0.55, "C2f (256)\nYOLOv8, P3 output", col["v8"], size=8.5)
    arrow(ax, (3.5, 4.1), (4.35, 4.1))
    arrow(ax, (5.17, 3.88), (5.17, 3.55))
    arrow(ax, (5.17, 3.05), (5.17, 2.65))
    arrow(ax, (5.17, 2.2), (5.17, 1.85))

    group(ax, 6.7, 0.1, 2.05, 4.62, "Bottom-up neck")
    box(ax, 6.9, 1.3, 1.65, 0.55, "Conv stride 2\n+ concat", col["base"], size=8.5)
    box(ax, 6.9, 2.1, 1.65, 0.55, "C3k2 (512)\nYOLO11, P4 output", col["11"], size=8.5)
    box(ax, 6.9, 2.95, 1.65, 0.45, "Conv stride 2 + concat", col["base"], size=8.2)
    box(ax, 6.9, 3.7, 1.65, 0.6, "C3k2 (1024)\nYOLO11, P5 output", col["11"], size=8.5)
    arrow(ax, (6.0, 1.57), (6.9, 1.57))
    arrow(ax, (7.72, 1.85), (7.72, 2.1))
    arrow(ax, (7.72, 2.65), (7.72, 2.95))
    arrow(ax, (7.72, 3.4), (7.72, 3.7))

    box(ax, 9.3, 1.9, 1.5, 1.3, "Detect\nanchor-free\nDFL head\nP3, P4, P5", col["out"], size=9.5)
    box(ax, 11.2, 2.15, 1.25, 0.8, "Boxes,\nclass, conf", col["out"], size=9.5)
    arrow(ax, (10.8, 2.55), (11.2, 2.55))
    elbow(ax, [(6.0, 1.45), (6.45, 1.45), (6.45, 0.45), (9.05, 0.45), (9.05, 2.1), (9.3, 2.1)])
    arrow(ax, (8.55, 2.37), (9.3, 2.55))
    elbow(ax, [(8.55, 4.0), (9.05, 4.0), (9.05, 3.0), (9.3, 3.0)])

    # skip connections, drawn dashed
    elbow(ax, [(1.65, ys[6] + 0.2), (1.95, ys[6] + 0.2), (1.95, 2.95), (3.85, 2.95), (3.85, 3.98), (4.35, 3.98)], ls=(0, (3, 2)))
    elbow(ax, [(6.0, 3.3), (6.52, 3.3), (6.52, 1.72), (6.9, 1.72)], ls=(0, (3, 2)))
    elbow(ax, [(3.5, 4.25), (3.72, 4.25), (3.72, 4.39), (6.62, 4.39), (6.62, 3.17), (6.9, 3.17)], ls=(0, (3, 2)))
    elbow(ax, [(1.65, ys[4] + 0.2), (3.95, ys[4] + 0.2), (3.95, 2.42), (4.35, 2.42)], ls=(0, (3, 2)))
    ax.text(0.15, 5.13, "TRIAD-YOLO: a YOLOv8 + YOLO11 hybrid in one network", fontsize=13, weight="bold")
    ax.text(0.15, 4.86, "Depth and width factors 0.50 (scale s).  “× n” and channel numbers are YAML values, before those factors.  Dashed = skip connection.", fontsize=9.5)
    box(ax, 9.3, 4.1, 1.05, 0.34, "YOLOv8 C2f", col["v8"], size=8.5)
    box(ax, 10.45, 4.1, 1.05, 0.34, "YOLO11", col["11"], size=8.5)
    box(ax, 11.6, 4.1, 0.85, 0.34, "Shared", col["base"], size=8.5)
    save(fig, "fig_arch_triad_yolo")


def pipeline():
    fig, ax = canvas(12.6, 5.6)
    ax.text(0.1, 5.3, "Risk-controlled selective inspection", fontsize=13, weight="bold")
    group(ax, 0.1, 2.85, 12.4, 2.25, "Deployment: one decision per image, no labels needed")
    box(ax, 0.3, 3.2, 1.2, 1.1, "Steel\nsurface\nimage", GREY)
    box(ax, 1.9, 3.2, 1.6, 1.1, "Detector\nTRIAD-Det, TRIAD-YOLO,\nYOLOv8s or YOLO11s", BLUE, size=8.5)
    box(ax, 3.9, 3.2, 1.5, 1.1, "Boxes kept at\nconf ≥ τ", BLUE)
    box(ax, 5.8, 3.2, 1.75, 1.1, "Uncertainty score u(x)\nread from predictions\n(stats, grey, census…)", ORANGE, size=8.8)
    box(ax, 7.95, 3.2, 1.3, 1.1, r"$u(x) \leq \hat{\lambda}$ ?", YELLOW, size=11, weight="bold")
    box(ax, 9.85, 3.85, 2.5, 0.72, "AUTO-ACCEPT\nboxes are reported without review", GREEN, size=9)
    box(ax, 9.85, 3.0, 2.5, 0.72, "MANUAL-REVIEW\ninspector checks the image", RED, size=9)
    for a, b in ((1.5, 1.9), (3.5, 3.9), (5.4, 5.8), (7.55, 7.95)):
        arrow(ax, (a, 3.75), (b, 3.75))
    arrow(ax, (9.25, 3.85), (9.85, 4.2))
    arrow(ax, (9.25, 3.6), (9.85, 3.36))
    ax.text(9.5, 4.2, "yes", fontsize=9, ha="right")
    ax.text(9.5, 3.28, "no", fontsize=9, ha="right")

    group(ax, 0.1, 0.1, 4.1, 2.55, "Validation split (labels)")
    box(ax, 0.3, 1.5, 3.7, 0.62, "Choose τ: best F1", BLUE)
    box(ax, 0.3, 0.85, 3.7, 0.56, "Fit stats score; pick score with lowest AURC", ORANGE, size=9)
    box(ax, 0.3, 0.25, 3.7, 0.52, "Fix the order of the λ grid", YELLOW)
    group(ax, 4.4, 0.1, 4.45, 2.55, "Calibration split (labels): Learn-then-Test")
    box(ax, 4.6, 1.5, 4.05, 0.62, r"$H_\lambda$: $R(\lambda) > \alpha$,   $R(\lambda) = E[L \mid u \leq \lambda]$", YELLOW)
    box(ax, 4.6, 0.85, 4.05, 0.56, r"p-value: binomial ($L_{any}$), Hoeffding-Bentkus ($L_{fraction}$)", YELLOW, size=8.8)
    box(ax, 4.6, 0.25, 4.05, 0.52, r"Stop at first $p > \delta$; $\hat{\lambda}$ = last rejected; none → review all", YELLOW, size=8.8)
    group(ax, 9.05, 0.1, 3.45, 2.55, "Test split (labels, held out)")
    box(ax, 9.25, 1.5, 3.05, 0.62, "Coverage, risk among accepted", GREEN, size=9.5)
    box(ax, 9.25, 0.85, 3.05, 0.56, "Re-split validity", GREEN, size=9.5)
    box(ax, 9.25, 0.25, 3.05, 0.52, r"$P(R(\hat{\lambda}) \leq \alpha) \geq 1-\delta$, i.i.d. calib and test", GREEN, size=9)
    arrow(ax, (4.2, 0.51), (4.6, 0.51))
    arrow(ax, (8.65, 0.51), (9.25, 0.51))
    elbow(ax, [(8.6, 2.65), (8.6, 2.85), (8.6, 3.2)], ls=(0, (3, 2)))
    save(fig, "fig_pipeline_selective")


def protocol():
    fig, ax = canvas(12.6, 5.0)
    ax.text(0.1, 4.72, "Leakage-safe data protocol", fontsize=13, weight="bold")
    group(ax, 0.1, 0.1, 4.6, 4.35, "Four datasets, shared source images")
    box(ax, 0.35, 3.15, 1.5, 0.62, "NEU-DET", "#DBEAFE", size=10)
    box(ax, 0.35, 2.05, 1.5, 0.62, "GC10-DET", "#DCFCE7", size=10)
    box(ax, 0.35, 0.95, 1.5, 0.62, "X-SDD", "#EDE9FE", size=10)
    box(ax, 3.0, 1.9, 1.5, 0.92, "SteelDefectX", "#FEE2E2", size=10, weight="bold")
    arrow(ax, (1.85, 3.46), (3.0, 2.65))
    arrow(ax, (1.85, 2.36), (3.0, 2.36))
    arrow(ax, (1.85, 1.26), (3.0, 2.07))
    ax.text(2.42, 3.2, "copies", fontsize=9)
    ax.text(2.42, 2.43, "crops", fontsize=9)
    ax.text(2.42, 1.33, "copies", fontsize=9)
    group(ax, 4.9, 0.1, 3.0, 4.35, "Split building")
    box(ax, 5.1, 3.0, 2.6, 0.8, "Detect image links\nacross datasets", YELLOW)
    box(ax, 5.1, 1.8, 2.6, 0.8, "Keep each linked group\nin a single split", YELLOW)
    box(ax, 5.1, 0.45, 2.6, 0.95, "Exclude groups that conflict\n580 SteelDefectX images\n(579 Inclusion, 1 Silk spot)", RED, size=9)
    arrow(ax, (6.4, 3.0), (6.4, 2.6))
    arrow(ax, (6.4, 1.8), (6.4, 1.4))
    arrow(ax, (4.7, 2.3), (5.1, 2.3))
    group(ax, 8.1, 0.1, 4.4, 4.35, "Final splits (images)")
    head = ("", "train", "val", "calib", "test")
    table = [("SteelDefectX", 4371, 702, 992, 1128), ("NEU-DET", 1080, 180, 270, 270), ("GC10-DET", 1368, 232, 346, 346), ("X-SDD", 811, 140, 206, 203)]
    xs = (8.25, 9.65, 10.35, 11.05, 11.75)
    for x, h in zip(xs, head):
        ax.text(x, 3.65, h, fontsize=10, weight="bold")
    for i, row in enumerate(table):
        for x, v in zip(xs, row):
            ax.text(x, 3.15 - i * 0.48, str(v), fontsize=10)
    ax.plot([8.25, 12.35], [3.5, 3.5], color=EDGE, lw=0.7)
    ax.text(8.25, 0.68, r"train, val: fit and choose; calib: $\hat{\lambda}$ only;", fontsize=9.5)
    ax.text(8.25, 0.38, "test: reported results only", fontsize=9.5)
    arrow(ax, (7.9, 2.3), (8.1, 2.3))
    save(fig, "fig_data_protocol")


if __name__ == "__main__":
    triad_det()
    triad_yolo()
    pipeline()
    protocol()
    print("wrote", OUT)
