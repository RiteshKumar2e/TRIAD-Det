r"""TikZ architecture diagrams for all four detectors: OUTPUT/figures/architecture/tikz/*.tex.

Each .tex compiles alone (pdflatex) and can be input into a paper, see architecture_figures.tex.
Layer types, repeats and output channels of the YOLO-family models were read from the built networks
(ultralytics 8.4.202, yolov8s.yaml, yolo11s.yaml, configs/triad_yolo.yaml, input 640). TRIAD-Det follows
TRIAD_Det_Risk_Control/src/triad_det_model/model.py and config/experiment_config.yaml.

    python tools/make_tikz_architectures.py
"""
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parents[1] / "OUTPUT" / "figures" / "architecture" / "tikz"

HEAD = r"""%% %(title)s. Compile with pdflatex, or \input it after \usepackage{standalone,tikz} and \usetikzlibrary{positioning,arrows.meta,calc}.
\documentclass[tikz,border=4pt]{standalone}
\usepackage{newtxtext,newtxmath}
\usetikzlibrary{positioning,arrows.meta,calc}
\begin{document}
"""
STYLES = r"""\definecolor{cConv}{HTML}{F3F4F6}
\definecolor{cCtwo}{HTML}{DBEAFE}
\definecolor{cCthree}{HTML}{DCFCE7}
\definecolor{cSppf}{HTML}{FFEDD5}
\definecolor{cPsa}{HTML}{EDE9FE}
\definecolor{cOut}{HTML}{FEF9C3}
\definecolor{cRisk}{HTML}{FEE2E2}
\definecolor{cEdge}{HTML}{374151}
"""
PICTURE = r"""\begin{tikzpicture}[
  font=\small, >={Latex[length=2mm]}, line width=0.5pt,
  block/.style={draw=cEdge, rounded corners=2pt, align=center, minimum width=#1, minimum height=8mm, inner sep=2pt},
  block/.default=30mm,
  flow/.style={->, draw=cEdge},
  skip/.style={->, draw=cEdge, dashed},
  group/.style={draw=gray, dashed, rounded corners=1pt},
  title/.style={font=\small\bfseries, anchor=north}]
"""
COLOR = {"Conv": "cConv", "C2f": "cCtwo", "C3k2": "cCthree", "SPPF": "cSppf", "C2PSA": "cPsa"}


def block(name, x, y, kind, label, width="30mm"):
    return rf"\node[block={width}, fill={COLOR[kind]}] ({name}) at ({x},{y}) {{{label}}};" + "\n"


def yolo(title, backbone, neck, legend):
    """backbone: [(type, repeats, channels, stride label)], neck: top-down and bottom-up blocks, see MODELS below."""
    gap = 1.05
    t = HEAD % {"title": title} + STYLES + PICTURE
    names = []
    for i, (kind, n, ch, tag) in enumerate(backbone):
        rep = rf"$\times${n}, " if n else ""
        t += block(f"b{i}", 0, i * gap, kind, rf"[{i}] {kind}{' ' + tag if tag else ''}\\{rep}{ch} ch")
        names.append(f"b{i}")
        if i:
            t += rf"\draw[flow] (b{i - 1}) -- (b{i});" + "\n"
    top = (len(backbone) - 1) * gap
    td1, td2, bu1, bu2 = neck["td1"], neck["td2"], neck["bu1"], neck["bu2"]
    first = len(backbone)                       # index of the first head layer
    ys = [top, top - 1.9, top - 3.8, top - 5.7]
    t += block("u1", 4.6, ys[0], "Conv", rf"[{first},{first + 1}] Upsample $\times$2\\Concat with P4")
    t += block("c1", 4.6, ys[1], td1[0], rf"[{first + 2}] {td1[0]}\\$\times${td1[1]}, {td1[2]} ch")
    t += block("u2", 4.6, ys[2], "Conv", rf"[{first + 3},{first + 4}] Upsample $\times$2\\Concat with P3")
    t += block("c2", 4.6, ys[3], td2[0], rf"[{first + 5}] {td2[0]}, P3 output\\$\times${td2[1]}, {td2[2]} ch")
    t += block("d1", 9.4, ys[3], "Conv", rf"[{first + 6},{first + 7}] Conv stride 2, {neck['down'][0]} ch\\Concat with [{first + 2}]")
    t += block("c3", 9.4, ys[2], bu1[0], rf"[{first + 8}] {bu1[0]}, P4 output\\$\times${bu1[1]}, {bu1[2]} ch")
    t += block("d2", 9.4, ys[1], "Conv", rf"[{first + 9},{first + 10}] Conv stride 2, {neck['down'][1]} ch\\Concat with [{first - 1}]")
    t += block("c4", 9.4, ys[0], bu2[0], rf"[{first + 11}] {bu2[0]}, P5 output\\$\times${bu2[1]}, {bu2[2]} ch")
    t += rf"\node[block=26mm, fill=cOut, minimum height=32mm] (det) at (13.7,{top - 2.85}) {{[{first + 12}] Detect\\anchor-free\\DFL head\\P3, P4, P5}};" + "\n"
    t += rf"\node[block=20mm, fill=cOut] (res) at (16.8,{top - 2.85}) {{Boxes,\\class, conf}};" + "\n"
    last = f"b{len(backbone) - 1}"
    t += rf"""\draw[flow] ({last}.east) -- (u1.west);
\draw[flow] (u1) -- (c1);
\draw[flow] (c1) -- (u2);
\draw[flow] (u2) -- (c2);
\draw[flow] (c2) -- (d1);
\draw[flow] (d1) -- (c3);
\draw[flow] (c3) -- (d2);
\draw[flow] (d2) -- (c4);
\draw[skip] (b6.east) -- ++(0.3,0) |- ($(u1.west)+(0,-0.25)$);
\draw[skip] (b4.east) -- ++(0.75,0) |- (u2.west);
\draw[skip] (c1.east) -- ++(0.6,0) |- ($(d1.west)+(0,0.25)$);
\draw[skip] ({last}.north) |- ($(c4.north west)+(-0.65,0.5)$) |- ($(d2.west)+(0,0.25)$);
\draw[flow] (c4.east) -| (det.north);
\draw[flow] (c3.east) -- (c3.east -| det.west);
\draw[flow] (c2.south) |- ($(d1.south)+(0,-0.55)$) -| (det.south);
\draw[flow] (det) -- (res);
"""
    t += rf"\node[anchor=west, font=\footnotesize] at (11.4,{ys[0] + 0.25}) {{P5}};" + "\n"
    t += rf"\node[anchor=west, font=\footnotesize] at (11.4,{ys[2] + 0.2}) {{P4}};" + "\n"
    t += rf"\node[anchor=west, font=\footnotesize] at (11.4,{ys[3] - 0.35}) {{P3}};" + "\n"
    y0 = top - 7.6
    x = 3.0
    for kind, text in legend:
        t += rf"\node[block=16mm, fill={COLOR[kind]}, minimum height=6mm, font=\footnotesize] at ({x},{y0}) {{{text}}};" + "\n"
        x += 2.0
    t += rf"\node[anchor=west, font=\footnotesize] at ({x - 0.8},{y0}) {{solid: main path; dashed: skip connection; [n]: layer index}};" + "\n"
    t += r"\node[title] at (0,%s) {Backbone};" % (top + 1.6) + "\n"
    t += r"\node[title] at (4.6,%s) {Top-down neck};" % (top + 1.6) + "\n"
    t += r"\node[title] at (9.4,%s) {Bottom-up neck};" % (top + 1.6) + "\n"
    t += "\\end{tikzpicture}\n\\end{document}\n"
    return t


MODELS = {
    "yolov8s": ("YOLOv8s", dict(
        backbone=[("Conv", 0, 32, "P1/2"), ("Conv", 0, 64, "P2/4"), ("C2f", 1, 64, ""), ("Conv", 0, 128, "P3/8"), ("C2f", 2, 128, ""),
                  ("Conv", 0, 256, "P4/16"), ("C2f", 2, 256, ""), ("Conv", 0, 512, "P5/32"), ("C2f", 1, 512, ""), ("SPPF", 0, 512, "")],
        neck=dict(td1=("C2f", 1, 256), td2=("C2f", 1, 128), bu1=("C2f", 1, 256), bu2=("C2f", 1, 512), down=(128, 256)),
        legend=[("Conv", "Conv / neck ops"), ("C2f", "C2f"), ("SPPF", "SPPF")])),
    "yolo11s": ("YOLO11s", dict(
        backbone=[("Conv", 0, 32, "P1/2"), ("Conv", 0, 64, "P2/4"), ("C3k2", 1, 128, ""), ("Conv", 0, 128, "P3/8"), ("C3k2", 1, 256, ""),
                  ("Conv", 0, 256, "P4/16"), ("C3k2", 1, 256, ""), ("Conv", 0, 512, "P5/32"), ("C3k2", 1, 512, ""), ("SPPF", 0, 512, ""),
                  ("C2PSA", 1, 512, "")],
        neck=dict(td1=("C3k2", 1, 256), td2=("C3k2", 1, 128), bu1=("C3k2", 1, 256), bu2=("C3k2", 1, 512), down=(128, 256)),
        legend=[("Conv", "Conv / neck ops"), ("C3k2", "C3k2"), ("SPPF", "SPPF"), ("C2PSA", "C2PSA")])),
    "triad_yolo": ("TRIAD-YOLO: YOLOv8 C2f in the shallow backbone and top-down neck, YOLO11 C3k2 and C2PSA elsewhere", dict(
        backbone=[("Conv", 0, 32, "P1/2"), ("Conv", 0, 64, "P2/4"), ("C2f", 2, 64, ""), ("Conv", 0, 128, "P3/8"), ("C2f", 3, 128, ""),
                  ("Conv", 0, 256, "P4/16"), ("C3k2", 1, 256, ""), ("Conv", 0, 512, "P5/32"), ("C3k2", 1, 512, ""), ("SPPF", 0, 512, ""),
                  ("C2PSA", 1, 512, "")],
        neck=dict(td1=("C2f", 2, 256), td2=("C2f", 2, 128), bu1=("C3k2", 1, 256), bu2=("C3k2", 1, 512), down=(128, 256)),
        legend=[("Conv", "Conv / neck ops"), ("C2f", "C2f (v8)"), ("C3k2", "C3k2 (v11)"), ("SPPF", "SPPF"), ("C2PSA", "C2PSA (v11)")])),
}


def triad_det():
    t = HEAD % {"title": "TRIAD-Det"} + STYLES + PICTURE
    ys = {3: 0.0, 4: 2.0, 5: 4.0}
    stage = {3: 192, 4: 384, 5: 768}
    t += r"\node[block=18mm, fill=cConv] (in) at (-3.4,2) {Input image\\$3\times H\times W$};" + "\n"
    for k, y in ys.items():
        stride = 2 ** k
        t += block(f"c{k}", 0, y, "C2f", rf"C{k}\\{stage[k]} ch, stride {stride}", "28mm")
        t += block(f"l{k}", 3.4, y, "C3k2", r"$1\times1$ conv\\lateral, 192 ch", "26mm")
        t += block(f"t{k}", 6.8, y, "C3k2", rf"Top-down TD{k}\\Conv+GN+SiLU", "26mm")
        t += block(f"p{k}", 10.8, y, "C3k2", (rf"P{k}, stride {stride}" if k == 3 else rf"Bottom-up P{k}\\stride {stride}"), "26mm")
        t += rf"\draw[flow] (in.east) -- (c{k}.west);" + "\n"
        t += rf"\draw[flow] (c{k}) -- (l{k}); \draw[flow] (l{k}) -- (t{k}); \draw[flow] (t{k}) -- (p{k});" + "\n"
    t += block("p6", 10.8, -2.0, "C3k2", r"Extra conv\\P6, stride 64", "26mm")
    t += r"""\draw[flow] (t5) -- node[right, font=\footnotesize] {$\times$2 up, add} (t4);
\draw[flow] (t4) -- node[right, font=\footnotesize] {$\times$2 up, add} (t3);
\draw[flow] (p3) -- node[right, font=\footnotesize] {stride 2, concat} (p4);
\draw[flow] (p4) -- node[right, font=\footnotesize] {stride 2, concat} (p5);
\draw[flow] (p3) -- node[right, font=\footnotesize] {stride 2} (p6);
\coordinate (bus) at (13.8,2);
\draw[line width=1.2pt, draw=cEdge] (13.8,-2.75) -- (13.8,5.0);
\draw[line width=1.2pt, draw=cEdge] (p5.east) -- (p5.east -| bus);
\draw[line width=1.2pt, draw=cEdge] (p4.east) -- (p4.east -| bus);
\draw[line width=1.2pt, draw=cEdge] (p3.east) -- (p3.east -| bus);
\draw[line width=1.2pt, draw=cEdge] (p6.east) -- (p6.east -| bus);
\node[font=\footnotesize\bfseries] at (13.8,5.25) {P3--P6};
"""
    t += block("ct", 16.3, 4.0, "SPPF", r"Class tower\\3 conv blocks", "26mm")
    t += block("bt", 16.3, 1.5, "SPPF", r"Box tower\\3 conv blocks", "26mm")
    t += block("co", 20.0, 4.0, "SPPF", r"Class\\$K$", "20mm")
    t += block("bo", 20.0, 2.2, "SPPF", r"Box, 4\\$\times$ level scale", "20mm")
    t += block("qo", 20.0, 0.7, "SPPF", r"Quality\\1", "20mm")
    t += block("gp", 16.3, -2.0, "C2PSA", r"Global avg pool\\each level, concat", "30mm")
    t += block("ml", 20.0, -2.0, "C2PSA", r"MLP $768\to256$\\$\to 1+K$, dropout 0.2", "30mm")
    t += r"\node[block=38mm, fill=cOut] (d) at (25,2.4) {Boxes and classes\\score $=\sqrt{\mathrm{class}\times\mathrm{quality}}$};" + "\n"
    t += r"\node[block=38mm, fill=cOut] (cn) at (25,-2) {Defect count (Poisson)\\Class presence ($K$)};" + "\n"
    t += r"\node[block=38mm, fill=cRisk] (u) at (25,-4.4) {Census uncertainty score\\count gap, unexplained presence,\\$1-\max$ conf};" + "\n"
    t += r"""\draw[flow] (13.8,4) -- (ct.west);
\draw[flow] (13.8,1.5) -- (bt.west);
\draw[flow] (13.8,-2) -- (gp.west);
\draw[flow] (ct) -- (co);
\draw[flow] (bt.east) -- (bo.west);
\draw[flow] (bt.east) -- (qo.west);
\draw[flow] (gp) -- (ml);
\draw[flow] (co.east) -- ($(d.west)+(0,0.25)$);
\draw[flow] (bo.east) -- (d.west);
\draw[flow] (qo.east) -- ($(d.west)+(0,-0.25)$);
\draw[flow] (ml) -- (cn);
\draw[flow] (cn) -- (u);
\draw[skip] (d.east) -- ++(0.35,0) |- (u.east);
\draw[group] (-1.75,-0.8) rectangle (1.75,5.75);
\node[title] at (0,5.7) {Backbone: ConvNeXt-Tiny};
\draw[group] (1.95,-2.95) rectangle (13.45,5.75);
\node[title] at (7.7,5.7) {BiFusion neck, 192 channels};
\draw[group] (14.3,-0.05) rectangle (21.55,5.75);
\node[title] at (17.9,5.7) {Detection head, shared across levels};
\draw[group] (14.3,-3.3) rectangle (21.95,-0.45);
\node[title] at (18.1,-0.5) {Census head, image level};
"""
    t += "\\end{tikzpicture}\n\\end{document}\n"
    return t


ALL = r"""% All four architectures in one document. Copy the \begin{figure} blocks into the paper.
\documentclass[10pt]{article}
\usepackage[margin=2cm,landscape]{geometry}
\usepackage{newtxtext,newtxmath}
\usepackage{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,calc}
\begin{document}
\begin{figure}[p]\centering
\resizebox{\linewidth}{!}{\input{triad_det}}
\caption{TRIAD-Det. A ConvNeXt-Tiny backbone and a two-way fusion neck feed an anchor-free detection head and an image-level census head. $K$ is the number of classes.}
\end{figure}
\begin{figure}[p]\centering
\resizebox{\linewidth}{!}{\input{triad_yolo}}
\caption{TRIAD-YOLO. YOLOv8 C2f blocks in the shallow backbone and top-down neck; YOLO11 C3k2 and C2PSA in the deep backbone and bottom-up neck.}
\end{figure}
\begin{figure}[p]\centering
\resizebox{\linewidth}{!}{\input{yolov8s}}
\caption{YOLOv8s baseline, as built from the Ultralytics configuration.}
\end{figure}
\begin{figure}[p]\centering
\resizebox{\linewidth}{!}{\input{yolo11s}}
\caption{YOLO11s baseline, as built from the Ultralytics configuration.}
\end{figure}
\end{document}
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"triad_det": triad_det(), "architecture_figures": ALL}
    for name, (title, spec) in MODELS.items():
        files[name] = yolo(title, **spec)
    for name, text in files.items():
        (OUT / f"{name}.tex").write_text(text, encoding="utf-8")
    for name in files:
        for _ in range(2 if name == "architecture_figures" else 1):
            run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"], cwd=OUT,
                                 capture_output=True, text=True, errors="replace")
        errors = [l for l in run.stdout.splitlines() if l.startswith("!")]
        print(name, "exit", run.returncode, (OUT / f"{name}.pdf").exists(), errors[:3])
    for path in OUT.iterdir():
        if path.suffix in (".aux", ".log", ".out"):
            path.unlink()


if __name__ == "__main__":
    main()
