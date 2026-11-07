"""Unit tests. Synthetic values here test the maths only and never reach any output."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triad import conformal
from triad.common import DATASET_COLOURS
from triad.data import _ncc, assign_splits, inherit_gc10_boxes
from triad.evaluate import box_iou, match


def test_dataset_colours_are_fixed():
    assert DATASET_COLOURS == {"SteelDefectX": "#DC2626", "NEU-DET": "#2563EB", "GC10-DET": "#16A34A"}


def test_min_accepted():
    assert conformal.min_accepted(0.05, 0.1) == 45


def _sample(rng, n):
    u = rng.uniform(size=n)
    return u, (rng.uniform(size=n) < 0.02 + 0.5 * u ** 2).astype(int)


def test_risk_control_holds_at_nominal_rate():
    """Selected thresholds should exceed the true conditional risk at most delta of the time."""
    rng = np.random.default_rng(0)
    alpha, delta = 0.10, 0.10
    population_u, population_loss = _sample(rng, 2_000_000)
    order = np.argsort(population_u)
    risk_at = np.cumsum(population_loss[order]) / np.arange(1, len(order) + 1)
    sorted_u = population_u[order]
    violations = certified = 0
    for _ in range(600):
        grid = conformal.threshold_grid(*_sample(rng, 300), 400, alpha, delta, 100)
        u, loss = _sample(rng, 400)
        result = conformal.calibrate(u, loss, grid, alpha, delta)
        if result.threshold is not None:
            certified += 1
            violations += risk_at[np.searchsorted(sorted_u, result.threshold) - 1] > alpha
    assert certified > 500
    assert violations / 600 <= delta


def test_no_threshold_when_every_image_is_wrong():
    result = conformal.calibrate(np.linspace(0, 1, 200), np.ones(200), np.linspace(0.3, 1, 10), 0.1, 0.1)
    assert result.threshold is None and not result.certified
    report = conformal.apply_threshold(np.linspace(0, 1, 50), np.zeros(50), result.threshold)
    assert report["n_accepted"] == 0 and report["manual_review_rate"] == 1


def test_risk_curve_merges_ties():
    curve = conformal.risk_coverage_curve([0.1, 0.1, 0.5], [1, 0, 0])
    assert curve["coverage"].tolist() == pytest.approx([2 / 3, 1])
    assert curve["risk"].tolist() == pytest.approx([0.5, 1 / 3])


def test_clopper_pearson_edges():
    assert conformal.clopper_pearson(0, 10)[0] == 0
    assert conformal.clopper_pearson(10, 10)[1] == 1


def test_match_outcomes():
    gts = [{"cls": "a", "box": [0, 0, 10, 10]}, {"cls": "b", "box": [20, 20, 30, 30]}]
    preds = [{"cls": "a", "conf": 0.9, "box": [0, 0, 10, 10]}, {"cls": "a", "conf": 0.8, "box": [20, 20, 30, 30]},
             {"cls": "a", "conf": 0.7, "box": [50, 50, 60, 60]}]
    strict = match(preds, gts, 0.5)
    assert strict["pairs"] == [(0, 0)] and strict["fp"] == [1, 2] and strict["missed"] == [1]
    assert len(match(preds, gts, 0.5, class_aware=False)["pairs"]) == 2


def test_match_handles_empty_inputs():
    assert match([], [{"cls": "a", "box": [0, 0, 1, 1]}], 0.5)["missed"] == [0]
    assert match([{"cls": "a", "conf": 1, "box": [0, 0, 1, 1]}], [], 0.5)["fp"] == [0]
    assert box_iou([], []).shape == (0, 0)


def test_groups_stay_whole_and_splits_are_stratified():
    groups = {f"g{i}": ["x"] for i in range(100)} | {f"h{i}": ["y", "y"] for i in range(50)} | {"r0": ["z"], "r1": ["z"]}
    fractions = {"train": 0.6, "val": 0.1, "calib": 0.15, "test": 0.15}
    result = assign_splits(groups, fractions, 1)
    assert set(result) == set(groups)
    x_counts = [sum(result[f"g{i}"] == s for i in range(100)) for s in fractions]
    assert x_counts == pytest.approx([60, 10, 15, 15], abs=2)


def test_ncc_finds_a_planted_crop():
    generator = torch.Generator().manual_seed(0)
    frame = torch.rand(1, 1, 60, 90, generator=generator)
    template = frame[:, :, 11:31, 37:57].clone()
    score = _ncc(frame, template)
    assert score.max().item() == pytest.approx(1.0, abs=1e-3)
    assert divmod(score.flatten().argmax().item(), score.shape[-1]) == (11, 37)


def test_box_projection_into_a_crop():
    sdx = {"width": 256, "folder_class": "Oil spot"}
    frame = {"boxes": [{"cls": "oil_spot", "box": [100, 100, 200, 200]}, {"cls": "oil_spot", "box": [900, 900, 950, 950]}]}
    link = {"x": 50, "y": 50, "crop_size": 512}
    boxes = inherit_gc10_boxes(sdx, frame, link, 0.3)
    assert len(boxes) == 1
    assert boxes[0]["box"] == pytest.approx([25, 25, 75, 75])
