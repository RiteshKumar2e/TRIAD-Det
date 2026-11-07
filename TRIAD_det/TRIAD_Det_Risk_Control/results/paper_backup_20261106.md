# Paper draft: Risk-Controlled Selective Prediction for Reliable Steel Surface Defect Detection Across Multiple Datasets

*Working draft, last updated 2026-11-05 18:25. Everything below is true as of now. Places marked **[TBD]** are filled in once the final run ends (expected 2026-11-06 ~06:00). Do not put numbers into the manuscript that are not in the result tables.*

**Status of the numbers in this file**

- Training is complete for TRIAD-YOLO, YOLOv8s and YOLO11s.
- TRIAD-Det **v2** has finished on X-SDD and NEU-DET and is training on GC10-DET. SteelDefectX follows. Live progress is in §8.4.
- The only performance numbers that exist so far are **validation** mAP (Â§7.1). Test-set and risk-control numbers come from `TRIAD_Det_Risk_Control/results/tables/full/` after the run.

---

## 0. One-paragraph pitch

Steel defect detectors are reported by mAP, an average over a test set. A plant needs something else: a rule that says, for each new image, whether the detector's result can be used without a person looking at it. It also needs a guarantee on how often such an auto-accepted image hides a missed defect.

We turn four detectors into **selective inspectors**. Each image gets an uncertainty score. Images below a calibrated threshold are AUTO-ACCEPTED and the rest go to MANUAL-REVIEW. The threshold is calibrated with Learn-then-Test so that, with probability at least 1 âˆ’ Î´, the miss rate among accepted images is at most Î±.

We test this:

- on four steel datasets (SteelDefectX, NEU-DET, GC10-DET, X-SDD), with leakage-safe splits across datasets that share images;
- with four detectors, including TRIAD-Det, a non-YOLO detector with an image-level census branch;
- under the dataset shift a plant meets when a model trained on one line meets another, measuring how many labelled target images restore the guarantee.

---

## 1. Title options

1. Risk-Controlled Selective Prediction for Reliable Steel Surface Defect Detection Across Multiple Datasets *(working title)*
2. When Can a Steel Defect Detector Be Trusted? Calibrated Auto-Accept / Manual-Review Decisions Across Four Datasets
3. TRIAD-Det: Certified Missed-Defect Risk for Selective Steel Surface Inspection

## 2. Abstract (skeleton; fill [TBD] after the run)

> Deep detectors for steel surface defects are evaluated by mean average precision, which does not tell an inspection line how often an image accepted without review still hides a missed defect. We formulate inspection as selective prediction: every image is either auto-accepted or sent to manual review, and the acceptance threshold on an image-level uncertainty score is calibrated with Learn-then-Test so that the probability that the miss rate among accepted images exceeds a target Î± is at most Î´. We study two image-level losses (any missed defect; fraction of missed defects), six label-free uncertainty scores, and four detectors â€” TRIAD-Det, a non-YOLO anchor-free detector with an image-level census branch, a hybrid TRIAD-YOLO, YOLOv8s and YOLO11s â€” on SteelDefectX, NEU-DET, GC10-DET and X-SDD with leakage-safe splits across datasets that share source images. In-domain, the calibrated rule [TBD: kept the realised risk below Î± in X of Y settings; violation rate over 500 re-splits â‰¤ Î´], auto-accepting [TBD: aâ€“b %] of test images at Î± = 0.10, whereas naive thresholding [TBD]. Across the 12 directed dataset pairs, a source-calibrated threshold [TBD: violated Î± in N pairs]; recalibrating with [TBD: n] labelled target images restored [TBD]. [TBD: one sentence on TRIAD-Det v2 vs baselines and on the census / disagreement scores.]

## 3. Contributions (only claims we can defend)

1. **Problem formulation.** Image-level AUTO-ACCEPT / MANUAL-REVIEW for steel defect *detection*, with a finite-sample guarantee on the *selective* missed-defect risk.
   - We use Learn-then-Test (LTT) as published. The calibration procedure itself is **not** new.
2. **A multi-dataset study.**
   - Four datasets, four detectors, two losses, six scores and five Î± values.
   - Validity checked over 500 random re-splits.
   - Paired bootstrap with Holm correction for comparing scores.
3. **Leakage-safe cross-dataset protocol.**
   - SteelDefectX contains copies of NEU-DET and X-SDD images and crops of GC10-DET frames. We detect these links and keep every linked group in one split.
   - Boxes are derived for SteelDefectX from its official masks, and for X-SDD from its matched SteelDefectX copies.
4. **Shift and recalibration.**
   - All 12 directed sourceâ†’target transfers: whether a source-calibrated threshold still holds.
   - How many labelled target images (25â€“400) are needed to restore it.
5. **TRIAD-Det.** A detector written without YOLO components:
   - ConvNeXt-Tiny backbone and a two-way fusion neck;
   - anchor-free head with detection, IoU-quality and an image-level **census** branch (defect count and class presence);
   - the census output is used as a label-free uncertainty signal.

**Claims we must NOT make**

- "First risk control for steel defects": Shen & Liu (2025) already apply CRC to steel defect segmentation.
- "First selective industrial certification with exact binomial bounds": SafeRestore (2026) does this for restoration.
- "TRIAD-Det is the most accurate detector", unless the final test tables show it on every dataset. Current validation numbers do not (Â§7.1).
- Any guarantee across datasets. The guarantee needs exchangeable calibration and test data.

## 4. Related work (verified entries only; see `TRIAD_Det_Risk_Control/reports/literature_review.csv`)

**Steel defect detectors.** Mostly YOLO variants tuned for mAP. None gives an image-level acceptance rule or a missed-defect guarantee.

- GRACE: Chen et al., *Sci. Rep.* 2026. YOLO11s; NEU-DET, GC10-DET, X-SDD; NEU-DET mAP50 75.88 %.
- ATFL-Swin-YOLO: Yu et al., *Sci. Rep.* 2026. YOLOv8n; NEU-DET mAP50 77.6 %, as reported.
- CFL-YOLO: Ren et al., *Measurement* 266, 2026.
- IMTS-YOLO: Fu et al., *Coatings* 16(1) 51, 2026.
- AGE-YOLO (*EAAI* 2026): authors not verified. Cite only after checking.
- Not citable:
  - SDM-YOLO and SteelYOLO11: repositories only.
  - SFEDet: title only.
  - TAD-YOLO11n: not found.

**Risk control and selective prediction.**

- Selective classification with guaranteed risk: Geifman & El-Yaniv, NeurIPS 2017.
- RCPS: Bates et al., *JACM* 2021.
- Conformal Risk Control (CRC): Angelopoulos et al., ICLR 2024. Needs a monotone loss; selective risk is not monotone.
- Learn-then-Test: Angelopoulos et al., *Ann. Appl. Stat.* 2025. Handles non-monotone risks through multiple testing. This is what we use.
- Selective CRC: Xu, Guo & Wei, arXiv 2512.12844, 2025. Classification sets.

**Conformal detection.**

- AndÃ©ol et al., COPA 2023: railway signals, box-level.
- SeqCRC: AndÃ©ol et al., arXiv 2505.24038, 2025. Sequential CRC for detection outputs; it does not decide per image whether to accept.

**Closest prior art.**

- Shen & Liu, arXiv 2504.17721 / *Mathematics* 2025: CRC on Mask R-CNN pixel maps for steel defects, bounding expected FDR/FNR.
  - We differ in four ways: image-level accept/review, selective risk with a (1 âˆ’ Î´) guarantee, detection, and multi-dataset shift.
- SafeRestore: Yang & Wang, arXiv 2609.03475, 2026. Auto-return vs human review with exact binomial certificates, for restoration.

**Datasets.**

- NEU-DET: He et al., *IEEE TIM* 2020.
- GC10-DET: Lv et al., *Sensors* 2020.
- SteelDefectX: arXiv 2603.21824, 2026. Copy the author list from arXiv.
- X-SDD: [TBD: verify the reference; believed to be Feng et al., *Symmetry* 2021]. Code: github.com/Fighter20092392/X-SDD-A-New-benchmark.

**Detector components** (standard references; check the BibTeX before submission):

- FCOS: Tian et al., ICCV 2019.
- ATSS: Zhang et al., CVPR 2020.
- Generalized / Quality Focal Loss: Li et al., NeurIPS 2020.
- Focal loss: Lin et al., ICCV 2017.
- ConvNeXt: Liu et al., CVPR 2022.
- Clopperâ€“Pearson 1934; Holm 1979.

## 5. Method

### 5.1 Selective inspection

Each image x gets a detector output and an uncertainty score u(x) âˆˆ [0, 1]. The decision is:

- **AUTO-ACCEPT** if u(x) â‰¤ Î»;
- **MANUAL-REVIEW** otherwise.

Images with no confident box always go to review.

The **selective risk** is

  R(Î») = E[ L(x) | u(x) â‰¤ Î» ].

Target: P_calib( R(Î»Ì‚) â‰¤ Î± ) â‰¥ 1 âˆ’ Î´. Headline setting: Î± = 0.10, Î´ = 0.10. Also reported: Î± âˆˆ {0.05, 0.10, 0.15, 0.20, 0.30}.

### 5.2 Losses (image level)

- **L_any(x)** = 1 if at least one ground-truth defect is missed.
- **L_frac(x)** = missed defects / all defects.
- A defect is *detected* if a prediction with confidence â‰¥ Ï„ matches it at IoU â‰¥ 0.5 with the same class.
  - Matching is greedy.
  - Cross-dataset matching is class-agnostic, because label sets differ.
- Ï„ maximises F1 on **val** and is fixed before calibration.
- Images with no ground truth have no defined loss and are excluded. Every image used has at least one box.

### 5.3 Uncertainty scores (label-free; read predictions only)

| Score    | Definition                                                                             |
| -------- | -------------------------------------------------------------------------------------- |
| conf     | 1 âˆ’ max accepted box confidence                                                   |
| weakest  | 1 âˆ’ min accepted box confidence                                                   |
| grey     | max rejected confidence / Ï„                                                         |
| stats    | logistic regression on 10 detection statistics, fitted on val                          |
| census   | TRIAD-Det only: max(relative count gap, unexplained class presence, 1 âˆ’ max conf) |
| disagree | 1 âˆ’ mean box agreement (F1 at IoU 0.5) with the other three detectors             |

The reported score per (model, dataset, loss) is the one with the lowest **val** AURC.

### 5.4 Calibration

- **LTT with fixed-sequence testing** over a Î» grid.
  - The grid order comes from val only: start at the candidate with the lowest projected p-value, then go upward.
  - Null hypothesis H_Î»: R(Î») > Î±.
- **p-values:**
  - L_any: exact binomial, BinomCDF(k; n_Î», Î±), where n_Î» is the number of accepted calibration images.
  - L_frac: Hoeffdingâ€“Bentkus.
- **Stopping rule:** stop at the first p > Î´. Î»Ì‚ is the last rejected Î». If none is rejected, every image goes to MANUAL-REVIEW.
- **Comparison methods:**
  - CRC on the *joint* risk E[LÂ·1{accept}] (a different quantity, guaranteed in expectation);
  - naive empirical threshold (no guarantee);
  - fixed confidence 0.5.
- A synthetic check is in `tests/test_calibration.py`: n_cal 300, Î± = Î´ = 0.10, 400 trials. Violation rates: LTT-binomial 0.06, LTT-HB 0.02, naive 0.51.

### 5.5 Statistics

- Clopperâ€“Pearson 95 % intervals on test risk.
- 500 random half/half re-splits of calib âˆª test for the violation rate.
- Paired bootstrap of the AURC difference, 2000 resamples, with Holm correction.

## 6. Models

|                       | TRIAD-Det (ours)                                            | TRIAD-YOLO                       | YOLOv8s     | YOLO11s      |
| --------------------- | ----------------------------------------------------------- | -------------------------------- | ----------- | ------------ |
| Framework             | plain PyTorch, no YOLO blocks                               | Ultralytics YAML hybrid          | Ultralytics | Ultralytics  |
| Backbone              | ConvNeXt-Tiny (ImageNet-22kâ†’1k init)                   | YOLOv8 C2f + YOLO11 C3k2 / C2PSA | CSP-C2f     | C3k2 + C2PSA |
| Neck                  | BiFusion: top-down then bottom-up, GroupNorm, extra P6      | PAN-FPN                          | PAN-FPN     | PAN-FPN      |
| Head                  | anchor-free, strides 8/16/32/64; detection, quality, census | Detect (DFL)                     | Detect      | Detect       |
| Score                 | âˆš(class Ã— quality)                                  | class                            | class       | class        |
| Params                | 33.6 M                                                      | 9.84 M                           | ~11 M       | ~9 M         |
| Detection pretraining | none (ImageNet backbone only)                               | COCO (YOLO11s / YOLOv8s weights) | COCO        | COCO         |

**TRIAD-Det v1** (trained 2026-11-04/05):

- FCOS assignment with fixed size ranges, focal loss, GIoU loss, and IoU-quality BCE;
- census losses: Poisson count and presence BCE;
- flips and brightness augmentation only, no EMA;
- early stop with patience 16 on val AP50.

**TRIAD-Det v2** (training now). The changes give it the training the YOLO baselines already get. Every choice is made on validation data only.

- ATSS assignment, with no hand-set size ranges.
- Quality Focal Loss: the class target is the IoU of the predicted box.
- EMA weights.
- Augmentation:
  - mosaic with probability 0.5, switched off for the last 10 epochs;
  - scale 0.6â€“1.4 and translate 0.1;
  - flips, brightness/contrast and gamma.
- NEU-DET at 640 px, the same as the YOLO runs (v1 used 512).
- 80â€“150 epochs, patience 30, with a wall-clock cap per dataset.
- v1 is kept (`results/v1_archive`, `results/predictions/v1`) for a v1 vs v2 ablation.

**Fairness note for the paper.** v2 got more design iterations than the baselines, which use Ultralytics defaults. Say so. The YOLO baselines also start from COCO detection weights and TRIAD-Det does not.

## 7. Data and protocol

### 7.0 Datasets

| Dataset      | Images used                              | Classes | Boxes                                   | Size         | Licence    |
| ------------ | ---------------------------------------- | ------- | --------------------------------------- | ------------ | ---------- |
| SteelDefectX | 7,193 (7,773 with boxes; 7,778 released) | 25      | official soft masks â†’ boxes        | varies       | CC-BY-4.0  |
| NEU-DET      | 1,800                                    | 6       | original XML                            | 200Ã—200   | â€”     |
| GC10-DET     | 2,292                                    | 10      | original XML                            | 2048Ã—1000 | â€”     |
| X-SDD        | 1,360                                    | 7       | rescaled from matched SteelDefectX copy | 128Ã—128   | Apache-2.0 |

**Mask â†’ box rule** (one rule for every image and class):

1. Binarise the mask at > 127.
2. Group fragments with morphological closing at g Ã— (short side), g âˆˆ {0.02, 0.05, 0.08, 0.12, 0.20]. Use the smallest g that gives 1â€“10 regions.
3. Drop regions smaller than max(16 px, 0.05 % of the image).
4. One box per region, spanning its defect pixels.

Results of the rule:

- 6,033 images needed only g = 0.02; 162 needed g = 0.20.
- 5 masks were empty; those images were dropped.
- Visual check: `results/figures/dataset_checks/steeldefectx_mask_boxes.jpg`.

**X-SDD â†’ SteelDefectX matching:**

- Criteria: 32Ã—32 correlation â‰¥ 0.95, 128Ã—128 mean absolute grey error < 10, and the same mapped class.
- All 1,360 images were matched.

### 7.1 Splits

- 60 / 10 / 15 / 15 % train / val / calib / test, seed 42, stratified by class.
- Duplicates are grouped by unionâ€“find over exact, near-duplicate and cross-dataset links. In total there are 6,734 groups, 3,077 of which span datasets.
- **Two-phase anchored split:**
  - NEU-DET and GC10-DET are split first and frozen behind a guard file.
  - Linked SteelDefectX and X-SDD images inherit the anchor split (3,905 images).
  - The remaining images are split stratified.
- **Excluded:** 580 SteelDefectX images (579 Inclusion, 1 Silk spot). They are linked to anchor images in different splits (19 groups), so no split could hold them without leakage. SteelDefectX Inclusion keeps 211 images.
- No group crosses splits (checked).

| Dataset      | train | val | calib | test  |
| ------------ | ----- | --- | ----- | ----- |
| SteelDefectX | 4,371 | 702 | 992   | 1,128 |
| NEU-DET      | 1,080 | 180 | 270   | 270   |
| GC10-DET     | 1,368 | 232 | 346   | 346   |
| X-SDD        | 811   | 140 | 206   | 203   |

**Split roles:**

| Split | Used for                                                             |
| ----- | -------------------------------------------------------------------- |
| train | fitting                                                              |
| val   | early stopping, Ï„, the stats score, score choice, Î»-grid order |
| calib | Î» only                                                            |
| test  | reporting only                                                       |

### 7.2 Experiments

| #  | Experiment                                                                                                                   | Output tables                                                            |
| -- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| E1 | In-domain: detection metrics; selective risk for all scores, losses, Î± and methods; validity over 500 re-splits           | `in_domain_{detection,per_class_ap,score_aurc,selective,validity}.csv` |
| E2 | Cross-domain, 12 directed pairs, class-agnostic: source-calibrated Î» on target test, target-recalibrated reference, naive | `cross_domain_{detection,selective,feasibility}.csv`                   |
| E3 | Target recalibration: n âˆˆ {25, 50, 100, 200, 400} target calib images Ã— 200 draws; grid order from source val        | `recalibration.csv`                                                    |
| E4 | Ablation: IoU 0.3 / 0.5 / 0.7 Ã— class-aware vs class-agnostic; each score forced; calibration-set size                    | `ablation_{matching,scores,calib_size}.csv`                            |
| E5 | TRIAD-Det v1 vs v2                                                                                                           | `triad_det_v1_vs_v2.csv`                                               |

Figures (300 DPI, PNG + PDF, in `results/figures/full/`):

- fig01: splits
- fig02: mAP
- fig03: riskâ€“coverage
- fig04: coverage vs Î±
- fig05: risk vs Î±
- fig06: validity
- fig07: methods
- fig08: score AURC
- fig09: cross-domain heatmap
- fig10: recalibration
- fig11: IoU ablation
- fig12: AUTO-ACCEPT / MANUAL-REVIEW examples, one per dataset

Fixed dataset colours:

| Dataset      | Colour  |
| ------------ | ------- |
| SteelDefectX | #DC2626 |
| NEU-DET      | #2563EB |
| GC10-DET     | #16A34A |
| X-SDD        | #9333EA |

### 7.3 Implementation

- Hardware: RTX 3060 12 GB, i9-10900, Windows 11.
- Software: Python 3.10, torch 2.11 + CUDA 12.8, Ultralytics 8.4.202, timm 1.0.31.
- YOLO family:
  - 150 epochs, patience 30, auto-batch at 60 % of VRAM;
  - imgsz 640 (GC10-DET 1024, X-SDD 320).
- TRIAD-Det: AdamW at lr 2e-4 (backbone Ã— 0.5), weight decay 0.05, cosine schedule with 2 warm-up epochs, AMP.
- Prediction keeps boxes with confidence â‰¥ 0.01, max 100 per image.
- Training times so far (wall clock):

| Model        | NEU-DET             | GC10-DET            | SteelDefectX        | X-SDD                           |
| ------------ | ------------------- | ------------------- | ------------------- | ------------------------------- |
| TRIAD-YOLO   | 38 min              | 80 min              | 134 min             | 13 min                          |
| YOLOv8s      | 34 min              | 66 min              | 113 min             | 11 min                          |
| YOLO11s      | 34 min              | 94 min              | 117 min             | 11 min                          |
| TRIAD-Det v1 | 32 min              | 111 min             | 144 min             | 10 min                          |
| TRIAD-Det v2 | 77 min (82 epochs, early stop) | running (cap 3.75 h) | queued (cap 3.75 h) | 32 min (118 epochs, early stop) |

The v2 wall-clock cap fixes the epoch count from the time of the **second** epoch. The first epoch carries worker start-up and cuDNN tuning (67 s vs 14 s on X-SDD); measuring on it had cut the X-SDD run to 30 epochs. That run was restarted.

## 8. Results

### 8.1 What exists now: validation mAP (not for the abstract; test numbers replace these)

| Dataset      | TRIAD-Det v1  | TRIAD-YOLO                        | YOLOv8s                           | YOLO11s                 |
| ------------ | ------------- | --------------------------------- | --------------------------------- | ----------------------- |
| NEU-DET      | 0.749 / 0.378 | 0.768 / 0.431                     | **0.769** / **0.446** | 0.764 / 0.442           |
| GC10-DET     | 0.584 / 0.278 | **0.617** / **0.314** | 0.599 / 0.293                     | 0.594 / 0.304           |
| SteelDefectX | 0.584 / 0.335 | **0.667** / **0.425** | 0.633 / 0.405                     | 0.658 / 0.417           |
| X-SDD        | 0.483 / 0.213 | 0.616 / 0.337                     | 0.618 / 0.332                     | **0.632** / 0.337 |

Cells are val mAP50 / mAP50-95, at each model's best epoch on val. Source: [results.csv](results.csv).

**Reading:**

- TRIAD-YOLO is best or tied on three of the four datasets.
- On NEU-DET the four models are within 2 points.
- TRIAD-Det v1 is 2â€“15 points behind. The likely reasons are weak augmentation, no EMA, early stopping and no COCO pretraining, which is what v2 addresses.
- For context, published NEU-DET mAP50 is ~76â€“78 % (GRACE, ATFL-Swin-YOLO). Those use different splits, so the numbers are not directly comparable.

### 8.2 To fill after the run

- **Table: test detection**
  - Columns: mAP50, mAP50-95, precision, recall, F1, missed-defect rate, any-miss image risk.
  - Rows: 4 models Ã— 4 datasets.
  - Source: `in_domain_detection.csv`.
- **Table: in-domain selective risk at Î± = 0.10, L_any**
  - Columns: chosen score, certified (y/n), test coverage, test risk among accepted [95 % CI], violation rate over 500 re-splits.
  - Methods: LTT-binomial, CRC-joint, naive, fixed-0.5.
  - Source: `in_domain_selective.csv`, `in_domain_validity.csv`.
- **Table: L_frac.** Same layout, LTT-HB only.
- **Table: cross-domain.** 12 pairs Ã— models: target risk and coverage with the source Î»; violated (y/n); target-recalibrated coverage.
- **Figure: recalibration.** Violation rate and coverage vs n.
- **Table: score ablation.** Test AURC per score; Holm-adjusted p vs conf; whether census and disagree help.
- **Table: TRIAD-Det v1 vs v2.**

### 8.4 TRIAD-Det v2 progress (validation only)

| Dataset      | v1 best val mAP50 / mAP50-95 | v2 best val mAP50 / mAP50-95              | Best YOLO-family val mAP50 | Status                              |
| ------------ | ---------------------------- | ----------------------------------------- | -------------------------- | ----------------------------------- |
| X-SDD        | 0.483 / 0.213                | **0.572 / 0.291** (epoch 88 of 118) | 0.632 (YOLO11s)            | done, 16:21                         |
| NEU-DET      | 0.749 / 0.378                | **0.761 / 0.407** (epoch 52 of 82)        | 0.769 (YOLOv8s)            | done, 17:38                         |
| GC10-DET     | 0.584 / 0.278                | 0.519 / 0.249 at epoch 16 (running)       | 0.617 (TRIAD-YOLO)         | running; expected end ~21:25        |
| SteelDefectX | 0.584 / 0.335                | —                                        | 0.667 (TRIAD-YOLO)         | queued; expected end ~01:15 (11-06) |

**Reading so far:**

- On X-SDD, v2 gains +8.9 mAP50 and +7.8 mAP50-95 over v1.
- The gap to the best YOLO model on X-SDD shrinks from 14.9 to 6.0 points.
- On NEU-DET, v2 gains +1.2 mAP50 and +2.9 mAP50-95 over v1. It is 0.8 mAP50 behind YOLOv8s, and its mAP50-95 (0.407) is still below all three YOLO-family models (0.431–0.446).
- NEU-DET v2 stopped early: best at epoch 52, no gain in the next 30 epochs.
- Whether v2 is best on any dataset is open until the test tables. Do not write that it is.

### 8.3 Early observation (v1 predictions, partial in-domain run before it was stopped; check against the final tables)

- On SteelDefectX, YOLOv8s and YOLO11s used Ï„ = 0.25.
- The val-chosen score was **stats** for both losses.
- LTT certified a threshold at Î± = 0.10 (Î»Ì‚ â‰ˆ 0.10 and 0.13).

## 9. Discussion points to develop

- **Coverage is the price of the guarantee.** A weaker detector is still safe under LTT; it just auto-accepts fewer images. So mAP and coverage at a fixed Î± should be reported together.
- **Selective risk vs joint risk.** CRC controls E[LÂ·1{accept}], which can look small simply because few images are accepted. Show this with E1.
- **Shift.** Report how often a source Î» fails on a target. Recalibration with a small labelled target sample is the practical fix.
- **Dataset overlap.** Pairs involving SteelDefectX are not independent domains (flag `shares_source_images`). Present NEU-DET â†” GC10-DET â†” X-SDD separately.
- **Census branch.** Report whether it adds information beyond confidence, using the E4 paired bootstrap. If it does not, say so.

## 11. Reproducibility

- Code and runs: `TRIAD_det/`. Read `README.md`, `Approach.md` and `MODELS.md`.
- Every step is logged in `TRIAD_Det_Risk_Control/EXECUTION_LOG.md`.
- Splits: `runs/triad_full/manifests/split_*.jsonl` and `split_report.json`.
- Data builder: `tools/build_detection_datasets.py`.
- Full study: `tools/run_everything.py` followed by `tools/run_triad_det_v2.py`.

## 12. To-do after results (2026-11-06)

1. Fill Â§2 and Â§8 from `results/tables/full/*.csv`. Copy the figures from `results/figures/full/`.
2. Decide which TRIAD-Det version is the main model. Report the other in the ablation.
3. Verify these references and fetch their BibTeX from DOIs: X-SDD, AGE-YOLO authors, SteelDefectX authors, ATSS, GFL, FCOS, ConvNeXt.
4. Write `paper/manuscript.tex` (10â€“15 pages) from this file.
