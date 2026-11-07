"""Write manuscript/sn-article.tex from the draft as received (manuscript/sn-article_draft_rev2.tex) and the "full" result tables.

"full": TRIAD-Det uses the extended recipe (v2) on every dataset; this is the primary analysis. The base recipe (v1) on
GC10-DET is read from results/tables/selected and reported only in Table tab:gc10recipes. Safe to re-run.

    python tools/update_manuscript.py
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "manuscript" / "sn-article_draft_rev2.tex"   # the draft as received
OUT = ROOT / "manuscript" / "sn-article.tex"
T = ROOT / "TRIAD_Det_Risk_Control" / "results" / "tables"
VARIANT = "full"
BASE_RECIPE_GC10 = T / "selected"
DS = ("SteelDefectX", "NEU-DET", "GC10-DET", "X-SDD")
MD = ("TRIAD-Det", "TRIAD-YOLO", "YOLOv8s", "YOLO11s")


def rows(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(x):
    return float(x) if x not in ("", None) else None


s = BASE.read_text(encoding="utf-8")


def rep(old, new, count=1):
    global s
    assert s.count(old) >= 1, old[:70]
    s = s.replace(old, new) if count == 0 else s.replace(old, new, count)


def block(start_label, new_body):
    """Replace everything between the \\midrule after a table label and that table's \\botrule."""
    global s
    i = s.index(start_label)
    a = s.index("\\midrule\n", i) + len("\\midrule\n")
    b = s.index("\\botrule", a)
    s = s[:a] + new_body + s[b:]


# ---------------------------------------------------------------- detection and selective, GC10-DET TRIAD-Det, primary recipe
det = {(r["dataset"], r["model"]): r for r in rows(T / VARIANT / "in_domain_detection.csv")}
g = det[("GC10-DET", "TRIAD-Det")]
g_base = next(r for r in rows(BASE_RECIPE_GC10 / "in_domain_detection.csv") if r["dataset"] == "GC10-DET" and r["model"] == "TRIAD-Det")
assert max(f(r["mAP50"]) for r in det.values() if r["dataset"] == "GC10-DET") == f(det[("GC10-DET", "YOLO11s")]["mAP50"])
rep(r"""& TRIAD-Det  & \textcolor{red}{--} & \textbf{0.624} & 0.266 & \textcolor{red}{--} & \textcolor{red}{--} & 0.572 & \textcolor{red}{--} & 0.454\\""",
    f"& TRIAD-Det  & {f(g['tau']):.2f} & {f(g['mAP50']):.3f} & {f(g['mAP50_95']):.3f} & {f(g['precision']):.3f} & "
    f"{f(g['recall']):.3f} & {f(g['f1']):.3f} & {f(g['missed_defect_rate']):.3f} & {f(g['any_miss_image_risk']):.3f}\\\\")
rep(r"& YOLO11s    & 0.25 & 0.615 & \textbf{0.293}", r"& YOLO11s    & 0.25 & \textbf{0.615} & \textbf{0.293}")


def lowest(path):
    q = [r for r in rows(path) if r["model"] == "TRIAD-Det" and r["dataset"] == "GC10-DET" and r["loss"] == "any"
         and r["score_chosen_on_val"] == "True" and r["cal_method"] == "LTT-binomial" and r["cal_certified"] == "True"]
    return min(q, key=lambda r: f(r["alpha"]))


low, low_base = lowest(T / VARIANT / "in_domain_selective.csv"), lowest(BASE_RECIPE_GC10 / "in_domain_selective.csv")
rep(r"GC10-DET     & \textcolor{red}{--} & 0.10 (41\,\%)",
    f"GC10-DET     & {f(low['alpha']):.2f} ({f(low['test_coverage']) * 100:.0f}\\,\\%) & 0.10 (41\\,\\%)")

# ---------------------------------------------------------------- one primary recipe; both recipes reported on GC10-DET
assert f(g["mAP50"]) < f(g_base["mAP50"]) and round(f(g_base["mAP50"]), 3) == 0.624 and round(f(g["mAP50"]), 3) == 0.598
rep(r"TRIAD-Det was trained with two recipes, and for each dataset the one with the higher validation mAP$_{50}$ is used throughout the paper.",
    r"TRIAD-Det was trained with two recipes. The extended recipe is the primary recipe and is used on every dataset throughout the paper; the base recipe is kept for comparison.")
rep(r"Validation selected the extended recipe on NEU-DET, SteelDefectX and X-SDD and the base recipe on GC10-DET (Appendix Table~\ref{tab:valmap}).",
    r"Validation mAP$_{50}$ favoured the extended recipe on NEU-DET, SteelDefectX and X-SDD and the base recipe on GC10-DET (Appendix Table~\ref{tab:valmap}); Table~\ref{tab:gc10recipes} reports both recipes on GC10-DET.")
rep(r"TRIAD-Det = validation-selected recipe per dataset (GC10-DET: base recipe = v1; others: extended = v2).",
    r"TRIAD-Det = extended recipe (v2) on every dataset; base recipe (v1) on GC10-DET only in tab:gc10recipes.")
rep(r"For each dataset, TRIAD-Det uses the training recipe selected on validation (Section~\ref{sec:triad}).",
    r"TRIAD-Det uses the extended recipe on every dataset (Section~\ref{sec:triad}).")
rep(r"TRIAD-Det has the highest mAP$_{50}$ on three of four datasets: NEU-DET (0.742), GC10-DET (0.624) and SteelDefectX (0.629, 0.3 points above TRIAD-YOLO); on X-SDD it is last (0.596 against 0.629 for YOLO11s).",
    r"TRIAD-Det has the highest mAP$_{50}$ on two of four datasets: NEU-DET (0.742) and SteelDefectX (0.629, 0.3 points above TRIAD-YOLO); on GC10-DET it is second (0.598 against 0.615 for YOLO11s) and on X-SDD last (0.596 against 0.629 for YOLO11s).")
rep(r"TRIAD-Det uses the validation-selected training recipe per dataset.", r"TRIAD-Det uses the extended recipe on all datasets.")
rep(r"the selected run per dataset is in bold.", r"the primary recipe is in bold.")
rep(r"TRIAD-Det (base)     & 32 & \textbf{111} & 144 & 10\\", r"TRIAD-Det (base)     & 32 & 111 & 144 & 10\\")
rep(r"TRIAD-Det (extended) & \textbf{77} & 187 & \textbf{195} & \textbf{32}\\", r"TRIAD-Det (extended) & \textbf{77} & \textbf{187} & \textbf{195} & \textbf{32}\\")
rep(r"bold marks the TRIAD-Det recipe selected for each dataset.", r"bold marks the primary TRIAD-Det recipe. The base recipe has the higher value on GC10-DET.")
rep(r"GC10-DET     & \textbf{0.584} / 0.278 & 0.552 / 0.278", r"GC10-DET     & 0.584 / 0.278 & \textbf{0.552} / 0.278")
recipe_row = lambda name, val, d, l: (f"{name} & {val} & {f(d['mAP50']):.3f} & {f(d['mAP50_95']):.3f} & {f(d['precision']):.3f} & {f(d['recall']):.3f} & {f(d['f1']):.3f} & "
                                     f"{f(d['any_miss_image_risk']):.3f} & {f(l['alpha']):.2f} ({f(l['test_coverage']) * 100:.0f}\\,\\%)\\\\\n")
recipe_table = (
    "\\begin{table}[t]\n\\caption{TRIAD-Det on GC10-DET under both training recipes. Val.: validation mAP$_{50}$; other columns on test. "
    "Lowest $\\alpha$: lowest certified level with its test coverage ($\\Lany$, LTT-binomial). The extended recipe is the primary recipe in this paper.}\\label{tab:gc10recipes}\n"
    "\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular*}{\\columnwidth}{@{\\extracolsep\\fill}lrrrrrrrr}\n\\toprule\n"
    "Recipe & Val. & mAP$_{50}$ & mAP$_{50:95}$ & Prec. & Rec. & F1 & Any-miss & Lowest $\\alpha$\\\\\n\\midrule\n"
    + recipe_row("Extended", "0.552", g, low) + recipe_row("Base", "0.584", g_base, low_base)
    + "\\botrule\n\\end{tabular*}\n\\end{table}\n\n")
rep(r"Each model was trained once, so differences of one or two points should be read as ties.",
    r"Each model was trained once, so differences of one or two points should be read as ties, and these results do not rank the detectors."
    + "\n\nOn GC10-DET the base recipe had the higher validation mAP$_{50}$ and a test mAP$_{50}$ of 0.624, above every detector in Table~\\ref{tab:det}, while the extended recipe has the higher mAP$_{50:95}$ and F1 (Table~\\ref{tab:gc10recipes}). "
    "We do not claim that lead. Choosing a recipe per dataset was considered only after the test results of both recipes had been computed, and the decision to report one recipe everywhere was also taken after seeing them, so this comparison is descriptive.\n\n" + recipe_table.rstrip("\n"))

# ---------------------------------------------------------------- table widths
rep("\\label{tab:det}\n\\scriptsize\n", "\\label{tab:det}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n")
rep("\\label{tab:valmap}\n\\scriptsize\n", "\\label{tab:valmap}\n\\scriptsize\n\\setlength{\\tabcolsep}{1pt}\n")
i = s.index("\\label{tab:valmap}")
j = s.index("\\botrule", i)
s = s[:i] + s[i:j].replace(" / ", "/") + s[j:]

# ---------------------------------------------------------------- methods
all_sel = rows(T / VARIANT / "in_domain_selective.csv")
chosen = [r for r in all_sel if r["score_chosen_on_val"] == "True"]
lines = []
for method, label in (("LTT-binomial", "LTT-binomial"), ("LTT-hb", "LTT-HB"), ("CRC-joint", "CRC-joint"), ("naive-empirical", "Naive"), ("fixed-conf-0.5", "Fixed-0.5")):
    v = [r for r in chosen if r["loss"] == "any" and f(r["alpha"]) == 0.10 and r["cal_method"] == method and r["cal_certified"] == "True"]
    risk = [f(r["test_risk_accepted"]) for r in v if r["test_risk_accepted"]]
    cov = [f(r["test_coverage"]) * 100 for r in v]
    lines.append(f"{label:<12} & {len(v)} / 16 & {min(risk):.3f}--{max(risk):.3f} & {sum(x > 0.10 for x in risk)} & {min(cov):.1f}--{max(cov):.1f}\\\\\n")
block("\\label{tab:methods}", "".join(lines))

# ---------------------------------------------------------------- cross-domain and recalibration
cross = rows(T / VARIANT / "cross_domain_selective.csv")
lines = []
for a in ("0.05", "0.1", "0.15", "0.2", "0.3"):
    cells = []
    for protocol in ("source-calibrated LTT", "source-calibrated naive", "target-recalibrated LTT (full calib)"):
        q = [r for r in cross if r["protocol"] == protocol and r["alpha"] == a]
        cells += [sum(r["cal_certified"] == "True" for r in q), sum(r["test_violated"] == "True" for r in q)]
    lines.append(f"{float(a):.2f} & " + " & ".join(str(c) for c in cells) + "\\\\\n")
block("\\label{tab:cross}", "".join(lines))
recal = rows(T / VARIANT / "recalibration.csv")
lines = []
for a in ("0.1", "0.2"):
    if a == "0.2":
        lines.append("\\midrule\n")
    for n in (0, 25, 50, 100, 200, 400):
        q = [r for r in recal if r["alpha"] == a and int(r["n_target_labels"]) == n]
        vals = [sum(f(r[k]) for r in q) / len(q) for k in ("certified_rate", "violation_rate", "coverage_mean")]
        lines.append(f"{float(a):.2f} & {n} & " + " & ".join(f"{v:.3f}" for v in vals) + "\\\\\n")
block("\\label{tab:recal}", "".join(lines))

# ---------------------------------------------------------------- score ablation
abl = rows(T / VARIANT / "ablation_scores.csv")
lines = []
SCORES = ("conf", "stats", "grey", "weakest", "disagree", "census")
best_count, stats_sig = {}, 0
for k, d in enumerate(DS):
    if k:
        lines.append("\\midrule\n")
    for n, m in enumerate(MD):
        q = {r["score"]: r for r in abl if r["dataset"] == d and r["model"] == m}
        best = min(q, key=lambda x: f(q[x]["test_aurc"]))
        best_count[best] = best_count.get(best, 0) + 1
        cells = []
        for score in SCORES:
            if score not in q:
                cells.append("--")
                continue
            text = f"{f(q[score]['test_aurc']):.3f}"
            text = f"\\textbf{{{text}}}" if score == best else text
            p = f(q[score]["p_value_holm"])
            if p is not None and p < 0.05:
                text += "$^\\ast$"
                stats_sig += score == "stats"
            cells.append(text)
        lead = f"\\multirow{{4}}{{*}}{{{d}}} & " if n == 0 else " & "
        lines.append(f"{lead}{m} & " + " & ".join(cells) + "\\\\\n")
block("\\label{tab:scoreabl}", "".join(lines))
print("best-score counts", best_count, "stats significant", stats_sig)
assert best_count["stats"] == 13 and stats_sig == 12
conf_cert = [r for r in abl if r["score"] == "conf" and r["cal_certified"] == "True"]
assert [(r["model"], r["dataset"]) for r in conf_cert] == [("TRIAD-Det", "GC10-DET")] and round(f(conf_cert[0]["test_coverage"]) * 100, 1) == 9.8
census_p = {r["dataset"]: f(r["p_value_holm"]) for r in abl if r["model"] == "TRIAD-Det" and r["score"] == "census"}
assert census_p["SteelDefectX"] < 0.001 and round(census_p["NEU-DET"], 3) == 0.033 and census_p["GC10-DET"] > 0.05 and census_p["X-SDD"] > 0.05

rep(r"Confidence alone is weak: forced as the score, it was certified at $\alpha=0.10$ in at most one of the 16 settings. %% CHECK: was GC10 TRIAD-Det (other recipe), coverage 9.8%",
    r"Confidence alone is weak: forced as the score, it was certified at $\alpha=0.10$ in only one of the 16 settings (GC10-DET TRIAD-Det, coverage 9.8\,\%).")
rep(r"SteelDefectX ($p<0.001$) and NEU-DET ($p=0.033$), with no significant difference on X-SDD \tbd{and GC10-DET}, and it is worse than \emph{stats} on all \tbd{four} datasets.",
    r"SteelDefectX ($p<0.001$) and NEU-DET ($p=0.033$), with no significant difference on X-SDD and GC10-DET, and it is worse than \emph{stats} on all four datasets.")

# ---------------------------------------------------------------- statements that did not match the code
rep(r"An image with no accepted box is given $u(x)=1$; since every candidate satisfies $\lambda<1$, such images are always reviewed.",
    r"An image with no accepted box is given $u(x)=1$ and is reviewed unless the certified threshold reaches $\lambda=1$, which occurred only for YOLO11s with $\Lfrac$ at $\alpha=0.30$ on NEU-DET and GC10-DET.")
rep(r"stats    & logistic regression on ten detection statistics, fitted on validation (Appendix~\ref{app:stats})\\",
    r"stats    & logistic regression on ten detection statistics, plus four census statistics for TRIAD-Det, fitted on validation (Appendix~\ref{app:stats})\\")
rep(r"conf, weakest, grey, stats and disagree are all of this form (disagree reads several detectors' boxes, and fails when all of them miss the same defect).",
    r"conf, weakest, grey and disagree are all of this form (disagree reads several detectors' boxes, and fails when all of them miss the same defect), as is stats for the YOLO-family detectors. For TRIAD-Det, stats also reads the census outputs (Appendix~\ref{app:stats}).")
rep("%% CHECK: confirm Eq. (census) matches the implementation in\n%% TRIAD_Det_Risk_Control/src (normalisation of the count gap, presence\n%% threshold).\n", "")
rep(" %% CHECK in cross_domain.py", "")
rep(r"All design choices were made on validation data. The total loss is \tbd{$\mathcal{L}=\mathcal{L}_{\mathrm{cls}}+w_{\mathrm{box}}\mathcal{L}_{\mathrm{GIoU}}+w_{\mathrm{cnt}}\mathcal{L}_{\mathrm{Pois}}+w_{\mathrm{pres}}\mathcal{L}_{\mathrm{pres}}$ with weights from the config}.",
    r"The total loss is $\mathcal{L}=\mathcal{L}_{\mathrm{cls}}+2\,\mathcal{L}_{\mathrm{GIoU}}+\mathcal{L}_{\mathrm{qual}}+0.2\,\mathcal{L}_{\mathrm{Pois}}+0.2\,\mathcal{L}_{\mathrm{pres}}$, where $\mathcal{L}_{\mathrm{cls}}$ is the recipe's class loss and $\mathcal{L}_{\mathrm{qual}}$ the binary cross-entropy between the predicted quality and the IoU of the predicted box.")
rep(r"A learned statistics score that already combines many box-level signals is stronger still, and including the census outputs among its inputs is a direct next step.",
    r"A learned statistics score that combines many signals is stronger still; for TRIAD-Det its inputs already include the census outputs, so the two signals are not independent. Measuring how much of its coverage comes from those inputs is a direct next step.")
rep(r"Feeding the census count gap and class-presence outputs into the detection-statistics score, and training the census branch jointly with a missed-defect objective, could combine the two strongest signals found here and raise coverage on datasets such as NEU-DET and X-SDD.",
    r"The detection-statistics score of TRIAD-Det already reads the census count gap and class-presence outputs. Training the census branch jointly with a missed-defect objective, and giving such a branch to the YOLO-family detectors, could raise coverage on datasets such as NEU-DET and X-SDD.")
rep(r"\tbd{List the ten statistics fed to the logistic regression exactly as implemented.}",
    r"""The \emph{stats} score is a logistic regression (standardised inputs, balanced class weights, $C=1$) fitted on the validation split to predict $\Lany$, and it uses these ten statistics at the box threshold $\tau$:
\begin{enumerate}
\item number of accepted boxes;
\item highest accepted confidence;
\item lowest accepted confidence;
\item mean accepted confidence;
\item highest rejected confidence;
\item sum of rejected confidences in $[\tau/2,\tau)$;
\item number of rejected boxes in $[\tau/2,\tau)$;
\item fraction of the image area covered by accepted boxes, capped at 4;
\item entropy of the accepted class distribution;
\item number of distinct accepted classes.
\end{enumerate}
For TRIAD-Det it also uses four census statistics: the predicted count $\hat\mu(x)$, the absolute gap $\max\{0,\hat\mu(x)-N(x)\}$, the relative gap $g_{\mathrm{cnt}}$ capped at 1, and the unexplained presence $g_{\mathrm{pres}}$ of Eq.~\eqref{eq:census}. Images with no accepted box are assigned $u(x)=1$.""")
rep("%% manuscript.tex  (revision 2, 2026-11-05)", "%% manuscript.tex  (revision 4, 2026-11-07)")

# ---------------------------------------------------------------- review fixes: guarantee wording, proxy boxes, localisation-only transfer, census
rep(r"and, among such thresholds, the largest coverage.",
    r"and take the last threshold certified by the fixed testing sequence of Section~\ref{sec:ltt}. This rule does not guarantee the largest possible coverage, and coverage is observed on test data rather than guaranteed. The bound concerns the population selective risk under the stated assumptions; it is not a statement about any individual accepted image.")
rep(r"All 1,360 images were matched; an unmatched image would have been excluded rather than given an invented box.",
    r"All 1,360 images were matched; an unmatched image would have been excluded rather than given an invented box. These are proxy annotations, not independently drawn ground truth, so X-SDD detection and risk results should be read as auxiliary evidence.")
rep(r"Matching is class-agnostic for both source calibration and target evaluation.",
    r"Matching is class-agnostic for both source calibration and target evaluation, because the label sets differ, so E2 measures whether defect regions are localised across datasets, not class-aware defect detection. In 24 of the 48 model$\times$pair settings, source and target contain visually linked images; every linked group stays in one split.")
rep(r"summarise the 48 model$\times$pair settings at each $\alpha$ (class-agnostic matching). A source-calibrated threshold almost never survives a change of dataset: at $\alpha=0.10$, 14 of the 15 certified source thresholds violated $\alpha$ on the target, and at $\alpha=0.30$ all 42 did.",
    r"summarise the 48 model$\times$pair settings at each $\alpha$ (class-agnostic matching, so ``miss'' here means an unlocalised region). A source-calibrated threshold almost never survives a change of dataset: at $\alpha=0.10$, 14 of the 15 certified source thresholds violated $\alpha$ on the target, and at $\alpha=0.30$ all 42 did. The settings without linked images behave alike: 2 of 2 certified thresholds violated $\alpha$ at $\alpha=0.10$, and 20 of 20 at $\alpha=0.30$.")
cross_ltt = [r for r in cross if r["protocol"] == "source-calibrated LTT" and r["shares_source_images"] == "False"]
for a, n in (("0.1", 2), ("0.3", 20)):
    q = [r for r in cross_ltt if r["alpha"] == a]
    assert sum(r["cal_certified"] == "True" for r in q) == n and sum(r["test_violated"] == "True" for r in q) == n
rep(r"whose image-level census branch predicts how many defects an image holds, exposing misses that leave no box.",
    r"whose image-level census branch predicts how many defects an image holds, as a signal for misses that leave no box.")
rep(r"the census score improved significantly on confidence on two of four datasets.}",
    r"the census score improved significantly on confidence on two of four datasets, and removing the census inputs from the statistics score changed its AURC by at most 0.005. Each detector was trained once, and the study uses public datasets of defective images only.}")
rep(r"TRIAD-Det reached the highest test mAP$_{50}$ on three of four datasets and the highest certified coverage (49.8\,\%~on SteelDefectX)." if False else r"TRIAD-Det reached the highest test mAP$_{50}$ on three of four datasets and the highest certified coverage (49.8\,\% on SteelDefectX).",
    r"With one training recipe on all datasets, TRIAD-Det reached the highest test mAP$_{50}$ on two of four datasets and the highest certified coverage (49.8\,\% on SteelDefectX).")
rep(r"and declined to certify, sending every image to review, on NEU-DET and X-SDD.",
    r"and declined to certify, sending every image to review, on NEU-DET and X-SDD (whose proxy boxes make it auxiliary evidence).")
rep(r"We address this blind spot with TRIAD-Det, a detector whose",
    r"We test whether this blind spot can be reduced with TRIAD-Det, a detector whose")
rep(r"A census that exceeds the boxes found is direct evidence of a miss (Fig.~\ref{fig:census}).",
    r"A census that exceeds the boxes found can indicate a miss (Fig.~\ref{fig:census}), and we measure how much this adds to box-derived signals.")
rep(r"It improves significantly on confidence on two of four datasets.",
    r"It improves significantly on confidence on two of four datasets, but adds little once detection statistics are available.")
rep(r"and introduced a census branch that exposes defects which leave no box.",
    r"and evaluated a census branch intended to expose defects which leave no box.")
rep(r"the census score added information beyond confidence on two datasets.",
    r"the census score added information beyond confidence on two datasets but little beyond detection statistics.")

rep(r"Let $\Lambda=\{\lambda_1,\dots,\lambda_G\}\subset[0,1)$ be a grid of candidate thresholds and $H_\lambda: R(\lambda)>\alpha$ the null hypothesis for each.",
    r"Let $\Lambda=\{\lambda_1,\dots,\lambda_G\}\subset[0,1]$ be a grid of candidate thresholds and $H_\lambda: R(\lambda)>\alpha$ the null hypothesis for each. The grid holds the empirical quantiles of the validation scores at $G=200$ evenly spaced levels $0,1/199,\dots,1$, with duplicate values removed, so at most 200 candidates remain. Testing starts at the candidate with the lowest projected binomial p-value, computed from the validation scores and losses and the calibration-set size only, and moves upward from there. No calibration score or loss enters the grid or its order, so both are fixed before calibration.")
rep(r"Moving a detector to a new line requires adapting the detector, not only its threshold.",
    r"Moving a detector to a new line requires adapting the detector, not only its threshold."
    + "\n\n\\paragraph{Scope and limitations.} Each detector was trained once, with one seed, so detector differences carry no estimate of training variance and were not tested for significance; the guarantee concerns the threshold of a given trained detector and does not depend on this. "
    "Every image used holds at least one annotated defect, so the false-alarm rate on defect-free strip was not measured. All four datasets are public benchmarks: nothing here has been validated on images from a production line, and the guarantee would hold there only if calibration and production images were exchangeable. X-SDD boxes are proxy annotations, so its results are auxiliary evidence.")

abl2 = rows(T / VARIANT / "census_ablation_scores.csv")
cmp2 = {r["dataset"]: r for r in rows(T / VARIANT / "census_ablation_stats_comparison.csv")}
qual = {r["dataset"]: r for r in rows(T / VARIANT / "census_quality.csv")}
hid = {r["dataset"]: r for r in rows(T / VARIANT / "census_invisible_miss.csv")}
census_gap = max(abs(f(c["aurc_stats_census"]) - f(c["aurc_stats_no_census"])) for c in cmp2.values())
assert census_gap < 0.0055
assert [d for d, c in cmp2.items() if f(c["p_two_sided"]) < 0.05] == ["SteelDefectX"]
lines = []
for d in DS:
    q = {r["score"]: r for r in abl2 if r["dataset"] == d}
    cov = lambda s: (f"{f(q[s]['coverage_0.20']) * 100:.1f}" if q[s]["cert_0.20"] == "True" else "--")
    p = f(cmp2[d]["p_two_sided"])
    lines.append(f"{d} & " + " & ".join(f"{f(q[s]['test_aurc']):.3f}" for s in ("conf", "census_count", "census_presence", "census_combined", "stats_no_census", "stats_census"))
                 + (" & $<0.001$" if p < 0.001 else f" & {p:.2f}") + f" & {cov('stats_no_census')} & {cov('stats_census')}\\\\\n")
qline = "; ".join(f"{d} {f(qual[d]['count_mae']):.2f}" for d in DS)
sdx = {r["score"]: r for r in abl2 if r["dataset"] == "SteelDefectX"}
cov_a, cov_b = f(sdx["stats_no_census"]["coverage_0.10"]) * 100, f(sdx["stats_census"]["coverage_0.10"]) * 100
hide_stats = [f(hid[d]["caught_stats_no_census"]) * 100 for d in DS]
hide_cen = [f(hid[d]["caught_census_combined"]) * 100 for d in DS]
gc = {r["score"]: r for r in abl2 if r["dataset"] == "GC10-DET"}
assert gc["stats_no_census"]["cert_0.20"] == "True" and gc["stats_census"]["cert_0.20"] == "False"
gc_no = f(gc["stats_no_census"]["coverage_0.20"]) * 100
census_block = (
    "\\paragraph{What the census adds.} For TRIAD-Det the reported \\emph{stats} score reads ten detection statistics and four census statistics. "
    "Table~\\ref{tab:censusabl} removes the census inputs. The AURC changes by at most 0.005, significantly only on SteelDefectX (0.136 to 0.134); "
    f"certified coverage on SteelDefectX at $\\alpha=0.10$ moves from {cov_a:.1f} to {cov_b:.1f}\\,\\%, so most of TRIAD-Det's coverage there does not come from the census. "
    f"The one large coverage change is on GC10-DET at $\\alpha=0.20$, where the score without census inputs is certified at {gc_no:.1f}\\,\\% and the reported score is not certified, because fixed-sequence testing stops at the first failure; this single setting should not be generalised. "
    "To test Remark~\\ref{rem:blind} directly we took test images with a miss whose confidence score is at or below the median of miss-free images; "
    f"the census score ranks {min(hide_cen):.0f}--{max(hide_cen):.0f}\\,\\% of them above the median miss-free image, and detection statistics without census inputs {min(hide_stats):.0f}--{max(hide_stats):.0f}\\,\\%. "
    f"The census count has a test mean absolute error of {qline} defects; for SteelDefectX and X-SDD the counted regions depend on the box-derivation rule of Section~\\ref{{sec:datasets}}.\n\n"
    "\\begin{table*}[t]\n\\caption{TRIAD-Det with and without census information ($\\Lany$). AURC on test, lower is better. s+c: stats with census inputs. $p$: paired bootstrap of s+c against stats alone (2000 resamples, two-sided, uncorrected). "
    "Cov.: test coverage at $\\alpha=0.20$, LTT-binomial; --: not certified.}\\label{tab:censusabl}\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
    "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lrrrrrrrrr}\n\\toprule\n"
    "Dataset & conf & count & presence & census & stats & s+c & $p$ & Cov.\\ stats & Cov.\\ s+c\\\\\n\\midrule\n"
    + "".join(lines) + "\\botrule\n\\end{tabular*}\n\\end{table*}\n\n")
rep("\\paragraph{Matching rule.}", census_block + "\\paragraph{Matching rule.}")
rep(r"A learned statistics score that combines many signals is stronger still; for TRIAD-Det its inputs already include the census outputs, so the two signals are not independent. Measuring how much of its coverage comes from those inputs is a direct next step.",
    r"A learned statistics score is stronger still, and adding the census outputs to it changes its AURC little (Table~\ref{tab:censusabl}). On this evidence the census is a supporting signal rather than the source of TRIAD-Det's coverage.")
rep(r"The census score improves on confidence where TRIAD-Det misses defects that leave no box, which Remark~\ref{rem:blind} predicts, and it does so significantly on two of four datasets.",
    r"The census score improves significantly on confidence on two of four datasets, in line with Remark~\ref{rem:blind}.")

# ---------------------------------------------------------------- validity tables, once the selected in_domain run exists
val_path, dec_path = T / VARIANT / "in_domain_validity.csv", T / VARIANT / "validity_decomposition.csv"
if val_path.exists() and dec_path.exists():
    val = rows(val_path)
    names = (("LTT-hb", "any", "LTT-HB       & $\\Lany$ "), ("LTT-hb", "fraction", "LTT-HB       & $\\Lfrac$"), ("LTT-binomial", "any", "LTT-binomial & $\\Lany$ "),
             ("naive-empirical", "any", "Naive        & $\\Lany$ "), ("CRC-joint", "any", "CRC-joint$^\\dagger$ & $\\Lany$"))
    lines, stat = [], {}
    for method, loss, label in names:
        means, worst = [], 0.0
        for a in ("0.05", "0.1", "0.15", "0.2", "0.3"):
            q = [f(r["violation_rate"]) for r in val if r["method"] == method and r["loss"] == loss and r["alpha"] == a and f(r["uncertified_rate"]) < 1]
            means.append(sum(q) / len(q))
            worst = max(worst, max(q))
        stat[(method, loss)] = (min(means), max(means), worst)
        lines.append(f"{label} & " + " & ".join(f"{m:.3f}" for m in means) + f" & {worst:.3f}\\\\\n")
    block("\\label{tab:validity}", "".join(lines))
    hb, nv, bn = stat[("LTT-hb", "any")], stat[("naive-empirical", "any")], stat[("LTT-binomial", "any")]
    rep(r"LTT-HB stayed within $\delta$ in every setting (worst 0.096). The naive threshold violated $\alpha$ in 43--57\,\% of re-splits on average. LTT-binomial averaged 0.061--0.121 with a worst setting of 0.178.",
        f"LTT-HB stayed within $\\delta$ in every setting (worst {hb[2]:.3f}). The naive threshold violated $\\alpha$ in {nv[0] * 100:.0f}--{nv[1] * 100:.0f}\\,\\% of re-splits on average. "
        f"LTT-binomial averaged {bn[0]:.3f}--{bn[1]:.3f} with a worst setting of {bn[2]:.3f}.")
    assert hb[2] <= 0.10
    dec = rows(dec_path)
    lines, stat = [], {}
    for pv, loss, label in (("binomial", "any", "binomial, $\\Lany$"), ("hb", "any", "HB, $\\Lany$      "), ("hb", "fraction", "HB, $\\Lfrac$     ")):
        q = [r for r in dec if r["pvalue"] == pv and r["loss"] == loss and f(r["certified_rate"]) > 0]
        tr, em = [f(r["true_violation"]) for r in q], [f(r["empirical_violation"]) for r in q]
        stat[(pv, loss)] = (sum(tr) / len(tr), max(tr), max(em))
        lines.append(f"{label} & {len(q)} & {sum(tr) / len(tr):.3f} / {max(tr):.3f} & {sum(em) / len(em):.3f} / {max(em):.3f}\\\\\n")
    block("\\label{tab:truerisk}", "".join(lines))
    over = sum(f(r["true_violation"]) > 0.10 for r in dec)
    b = stat[("binomial", "any")]
    print("true-risk violations above 0.10:", over, "worst", b[1], "of", len(dec))
    print("CHECK text of sec:res-validity by hand against:", b)
    assert len(dec) == 240 and b[1] < 0.12
    tail = f"one of 240 settings at {b[1]:.3f}" if over == 1 else (f"{over} of 240 settings above 0.10, worst {b[1]:.3f}" if over else f"worst of 240 settings {b[1]:.3f}")
    rep(r"(mean 0.042; one of 240 settings at 0.102, within the Monte Carlo standard error of about 0.01), while the empirical violation of the same thresholds on a fresh test half reaches 0.157,",
        f"(mean {b[0]:.3f}; {tail}, within the Monte Carlo standard error of about 0.01), while the empirical violation of the same thresholds on a fresh test half reaches {b[2]:.3f},")
    print("validity tables:", VARIANT)
else:
    raise SystemExit("validity tables missing")

for stale in ("validation-selected training recipe", "the recipe selected", "three of four datasets", "0.014"):
    assert stale not in s, stale
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(s, encoding="utf-8")
print("wrote", OUT, "| remaining tbd:", len(re.findall(r"\\tbd\{", s)) - 1, "| red dashes:", s.count(r"\textcolor{red}{--}"))
