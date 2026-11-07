# TRIAD-Det

**Risk-controlled selective prediction for steel surface defect detection.**

Automated inspection needs to know when it can be trusted. TRIAD-Det routes every image to one of two outcomes:

- **AUTO-ACCEPT:** the detector's output is used without review.
- **MANUAL-REVIEW:** a person inspects the image.

The acceptance threshold is calibrated so that the miss rate among accepted images is at most α, with probability at least 1 − δ. If no threshold can be certified, every image goes to manual review.

The study covers four detectors on four datasets. It includes TRIAD-Det, a new anchor-free detector with a census branch that estimates how many defects an image should contain.

| | |
|---|---|
| **Detectors** | TRIAD-Det, TRIAD-YOLO, YOLOv8s, YOLO11s |
| **Datasets** | SteelDefectX, NEU-DET, GC10-DET, X-SDD |
| **Calibration** | Learn-then-Test with fixed-sequence testing |
| **Headline setting** | α = 0.10, δ = 0.10, IoU 0.5 |

## Contents

- [Key results](#key-results)
- [Method](#method)
- [Repository layout](#repository-layout)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Results and outputs](#results-and-outputs)
- [Limitations](#limitations)
- [Reproducibility status](#reproducibility-status)
- [License](#license)

## Key results

All numbers are on held-out test splits and come from [`results.csv`](TRIAD_det/results.csv).

### Risk control at α = 0.10, δ = 0.10

"Coverage" is the share of test images auto-accepted. "Risk" is the share of accepted images with at least one missed defect, with its 95% Clopper–Pearson interval. Only 5 of the 16 model × dataset combinations were certified. The other 11 send every image to manual review, which is the intended safe outcome.

| Dataset | Model | Certified | Coverage | Risk among accepted (95% CI) |
|---|---|:---:|---:|---|
| SteelDefectX | **TRIAD-Det** | Yes | **49.8%** | 6.4% (4.5–8.8) |
| SteelDefectX | TRIAD-YOLO | Yes | 30.9% | 6.9% (4.5–10.1) |
| SteelDefectX | YOLO11s | Yes | 23.8% | 7.8% (4.9–11.7) |
| SteelDefectX | YOLOv8s | Yes | 8.2% | 4.3% (1.2–10.6) |
| GC10-DET | TRIAD-YOLO | Yes | 41.0% | 10.6% (6.0–16.8) |
| GC10-DET | TRIAD-Det, YOLOv8s, YOLO11s | No | 0% | n/a |
| NEU-DET | all four | No | 0% | n/a |
| X-SDD | all four | No | 0% | n/a |

The guarantee is a probability over calibration draws, so one test split can land slightly above α. The GC10-DET row at 10.6% is an example, and its interval contains 10%.

### Detection (test mAP50 / mAP50-95)

| Model | SteelDefectX | NEU-DET | GC10-DET | X-SDD |
|---|---|---|---|---|
| TRIAD-Det | **0.629** / 0.401 | **0.742** / 0.379 | 0.598 / 0.278 | 0.596 / 0.314 |
| TRIAD-YOLO | 0.626 / 0.400 | 0.715 / 0.379 | 0.570 / 0.250 | 0.615 / 0.330 |
| YOLOv8s | 0.615 / 0.390 | 0.697 / **0.382** | 0.589 / 0.268 | 0.612 / 0.333 |
| YOLO11s | 0.624 / **0.403** | 0.681 / 0.360 | **0.615** / **0.293** | **0.629** / **0.342** |

No single detector wins everywhere. TRIAD-Det leads mAP50 on SteelDefectX and NEU-DET, and YOLO11s leads on GC10-DET and X-SDD.

<p align="center">
  <img src="TRIAD_det/TRIAD_Det_Risk_Control/results/figures/full/fig03_risk_coverage.png" alt="Risk-coverage curves for the four detectors on each dataset" width="85%">
</p>

## Method

### TRIAD-Det detector

TRIAD-Det (Triple-branch Risk-Aware Inspection Detector) is a single-stage, anchor-free detector in plain PyTorch. It shares no building blocks with the YOLO family.

| Part | Design |
|---|---|
| Backbone | ConvNeXt-Tiny from timm, ImageNet-pretrained |
| Neck | Bidirectional fusion: top-down FPN pass, then bottom-up pass, with GroupNorm |
| Detection branch | Per-location class logits and box distances |
| Quality branch | Per-location IoU-aware localisation quality |
| Census branch | Image-level defect count (Poisson rate) and class presence |
| Training (v2) | ATSS assignment, quality focal loss, EMA weights, mosaic augmentation |

The census branch gives a signal that needs no ground truth: a gap between the expected number of defects and the number of confident boxes suggests a defect may have been missed.

### Compared models

| | TRIAD-Det | TRIAD-YOLO | YOLOv8s | YOLO11s |
|---|---|---|---|---|
| Implementation | Plain PyTorch | Ultralytics YAML hybrid | Ultralytics | Ultralytics |
| Backbone | ConvNeXt-Tiny | YOLOv8 C2f + YOLO11 C3k2/C2PSA | CSPDarknet-C2f | C3k2 + C2PSA |
| Head | Anchor-free, three branches | YOLO Detect | YOLO Detect | YOLO Detect |
| Parameters | 33.6 M | 9.84 M | ~11 M | ~9 M |

### Risk control

1. **Loss.** `L_any` is 1 if any ground-truth box is missed at IoU 0.5 with the correct class. `L_fraction` is the share of boxes missed. Cross-dataset matching is class-agnostic.
2. **Uncertainty score.** Candidates are `conf`, `weakest`, `grey`, `stats` (a logistic model fitted on val), `census` (TRIAD-Det only) and `disagree` (between detectors). The score with the lowest AURC on **val** is reported.
3. **Calibration.** Learn-then-Test fixed-sequence testing runs on **calib**, with the threshold order taken from val. P-values are exact binomial for `L_any` and Hoeffding–Bentkus for `L_fraction`.
4. **Decision.** An image is auto-accepted when its uncertainty is at or below the calibrated threshold. The guarantee is:

   ```text
   P_calib( R(λ̂) ≤ α ) ≥ 1 − δ,   where R(λ) = E[ L | u ≤ λ ]
   ```

5. **Evaluation.** The **test** split reports coverage, manual-review rate and risk among accepted images. Validity is checked over 500 random calib/test re-splits. Score comparisons use a paired bootstrap with Holm correction.

Baselines are Conformal Risk Control on the joint risk, a naive empirical threshold, and a fixed confidence of 0.5.

## Repository layout

```text
TRIAD-Det/
├── GC10-DET/  NEU-DET/              Datasets: IMAGES/<class>/, ANNOTATIONS/*.xml
├── SteelDefectX-DET/  X-SDD-DET/    Datasets built for detection, same layout
└── TRIAD_det/
    ├── run_all.py                   Base pipeline entry point
    ├── results.csv                  One row per model × dataset
    ├── requirements.txt             Minimum package versions
    ├── requirements-lock.txt        Exact versions used for the results
    ├── triad/                       Data, splits, YOLO-family detector, evaluation, export
    ├── tools/                       Study runner, dataset builder, figure and packaging scripts
    ├── tests/                       Core unit tests
    ├── OUTPUT/batches/              Annotated test-image batches per model and dataset
    └── TRIAD_Det_Risk_Control/
        ├── config/experiment_config.yaml
        ├── src/                     Calibration, losses, scores, statistics
        │   └── triad_det_model/     TRIAD-Det model and training
        ├── experiments/             In-domain, cross-domain, ablation and figure scripts
        ├── tests/                   Calibration validity tests
        ├── reports/                 Literature review
        ├── results/                 Predictions, tables, figures, logs
        └── EXECUTION_LOG.md         Dated log of the study's main steps
```

## Datasets

| Dataset | Images used | Classes | Test images | Box source |
|---|---:|---:|---:|---|
| SteelDefectX | 7,193 | 25 | 1,128 | Official pixel masks converted to boxes |
| NEU-DET | 1,800 | 6 | 270 | Original XML |
| GC10-DET | 2,292 | 10 | 346 | Original XML |
| X-SDD | 1,360 | 7 | 203 | Matched SteelDefectX copies |

**Sources for the built datasets**

- SteelDefectX: Hugging Face `Zhaosxian/SteelDefectX` (CC-BY-4.0)
- X-SDD: `github.com/Fighter20092392/X-SDD-A-New-benchmark` (Apache-2.0)

**Splits**

- Train / val / calib / test is 60 / 10 / 15 / 15%, with seed 42, stratified by class.
- Duplicates and cross-dataset copies always stay in one split. SteelDefectX contains copies from NEU-DET and X-SDD, and crops from GC10-DET.
- NEU-DET and GC10-DET are split first and then frozen.
- 580 of the 7,773 SteelDefectX images are excluded because they link anchor images in different splits, which would cause leakage.

## Installation

Tested on Windows 11 with an NVIDIA RTX 3060 12 GB and Python 3.10.11. Key packages are torch 2.11.0+cu128, ultralytics 8.4.202 and timm 1.0.31. Training requires CUDA. Linux and macOS are untested.

```powershell
git clone https://github.com/RiteshKumar2e/TRIAD-Det.git
cd TRIAD-Det\TRIAD_det

py -3.10 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Check the install:

```powershell
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv\Scripts\python -m pytest tests TRIAD_Det_Risk_Control/tests -q
```

The first command must print `True`. If it prints `False`, the CPU-only torch was installed: uninstall `torch` and `torchvision`, then repeat the install commands. Pretrained weights download on first use.

## Usage

Run these from `TRIAD_det/`. Read [Reproducibility status](#reproducibility-status) first, because some commands need files that are not in this repository.

**Whole study** (unattended, skips finished jobs):

```powershell
.venv\Scripts\python tools/run_everything.py
```

**Individual steps:**

```powershell
# Train a YOLO-family model: TRIAD-YOLO | YOLOv8s | YOLO11s
.venv\Scripts\python tools/train_one.py --dataset GC10-DET --model YOLOv8s

cd TRIAD_Det_Risk_Control

# Train TRIAD-Det, then write its predictions
..\.venv\Scripts\python -m src.triad_det_model.train --dataset NEU-DET --train-only
..\.venv\Scripts\python -m src.triad_det_model.train --dataset NEU-DET --predict-only

# Risk-control experiments
..\.venv\Scripts\python experiments/predict_yolo.py
..\.venv\Scripts\python experiments/in_domain.py
..\.venv\Scripts\python experiments/cross_domain.py
..\.venv\Scripts\python experiments/recalibration.py
..\.venv\Scripts\python experiments/ablation.py
..\.venv\Scripts\python experiments/figures.py
```

**Further analyses** in `TRIAD_Det_Risk_Control/experiments/`:

| Script | Question |
|---|---|
| `census_ablation.py` | Does the census branch improve the uncertainty score? |
| `validity_check.py` | Is a re-split violation a real guarantee failure or finite-test noise? |
| `target_adaptation.py` | Can a small labelled target set restore auto-acceptance on a new dataset? |

**Rebuild the derived datasets** from fresh downloads:

```powershell
.venv\Scripts\python tools/build_detection_datasets.py --sdx <SteelDefectX download> --xsdd <X-SDD repo>/X-SDD/datas --out ..
```

## Results and outputs

Paths are under `TRIAD_det/`.

| Location | Contents |
|---|---|
| `results.csv` | Training, detection and risk-control summary |
| `TRIAD_Det_Risk_Control/results/tables/full/` | In-domain, cross-domain, recalibration, ablation, census and validity tables |
| `TRIAD_Det_Risk_Control/results/tables/adaptation/` | Target adaptation results |
| `TRIAD_Det_Risk_Control/results/figures/full/` | Figures 1–12 as 300-DPI PNG and PDF |
| `TRIAD_Det_Risk_Control/results/predictions/` | AUTO-ACCEPT / MANUAL-REVIEW decisions and adaptation predictions |
| `OUTPUT/batches/` | Annotated test batches and `batch_summary.csv` |

**Result variants.** `full` is the primary analysis, with the v2 TRIAD-Det recipe on every dataset. `selected` uses the v1 recipe for TRIAD-Det on GC10-DET. `v1` and `v1_archive` keep the earlier recipe, `smoke` is a plumbing test, and none of these three should be read as final results.

## Limitations

- **Exchangeability.** The guarantee holds only when new images are exchangeable with the calib split. Cross-dataset results are empirical, not guaranteed.
- **Certification is often infeasible.** Most combinations at α = 0.10 certify nothing, including every model on NEU-DET and X-SDD.
- **Defect-only data.** Every image contains a defect, so false alarms on defect-free steel cannot be measured.
- **Shared sources.** Pairs involving SteelDefectX share source images and are flagged `shares_source_images`.
- **Subjective boxes.** Boxes derived from masks of diffuse texture defects are partly subjective.
- **Proxy boxes.** X-SDD boxes come from matched SteelDefectX copies, not from original annotations.

## Reproducibility status

| Item | Status |
|---|---|
| Datasets, source code, experiment config | Included |
| Result tables, figures, AUTO-ACCEPT / MANUAL-REVIEW decisions | Included |
| Per-image predictions in `results/predictions/{full,selected,smoke,v1}/` | **Not included.** Excluded by `.gitignore` because of their size. Re-generate them with the prediction steps |
| `TRIAD_det/configs/` (`default.yaml`, `smoke.yaml`, `triad_yolo.yaml`) | **Missing.** `run_all.py`, `tools/train_one.py` and `tools/run_everything.py` need these files |
| `TRIAD_det/runs/` (split manifests, YOLO-family checkpoints, training logs) | **Not included.** Excluded by `.gitignore`. `in_domain.py` and `figures.py` read manifests from it |
| Model weights (`*.pt`) | **Not included.** Excluded by `.gitignore`, so trained checkpoints must be retrained |

The published tables and figures can be inspected as they are. Re-running the pipeline requires restoring `configs/` first.

## License

No license file is included yet, so default copyright applies to the code. The datasets keep the terms of their original sources.
