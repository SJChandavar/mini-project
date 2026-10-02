"""
Unit tests for quantitative evaluation metrics.
"""

import numpy as np
import pytest
import torch
from metrics import (
    compute_dice,
    compute_iou,
    compute_accuracy,
    compute_fscore,
    compute_psnr,
    compute_map,
)


def test_perfect_match_metrics():
    """Test metric outputs when prediction perfectly matches target mask."""
    target = np.ones((100, 100), dtype=np.float32)
    pred = np.ones((100, 100), dtype=np.float32)

    assert compute_dice(pred, target) == pytest.approx(1.0)
    assert compute_iou(pred, target) == pytest.approx(1.0)
    assert compute_accuracy(pred, target) == pytest.approx(1.0)

    f_res = compute_fscore(pred, target)
    assert f_res["precision"] == pytest.approx(1.0)
    assert f_res["recall"] == pytest.approx(1.0)
    assert f_res["fscore"] == pytest.approx(1.0)


def test_zero_match_metrics():
    """Test metric outputs when prediction has zero overlap with target mask."""
    target = np.zeros((100, 100), dtype=np.float32)
    target[20:40, 20:40] = 1.0

    pred = np.zeros((100, 100), dtype=np.float32)
    pred[60:80, 60:80] = 1.0

    assert compute_dice(pred, target) < 0.01
    assert compute_iou(pred, target) < 0.01


def test_psnr_calculation():
    """Test PSNR computation."""
    target = np.ones((50, 50), dtype=np.float32)
    pred = np.ones((50, 50), dtype=np.float32)
    assert compute_psnr(pred, target) == 100.0  # Perfect match convention


def test_map_calculation():
    """Test instance mAP calculation."""
    target = np.zeros((100, 100), dtype=np.float32)
    target[10:30, 10:30] = 1.0
    target[50:70, 50:70] = 1.0

    pred = np.zeros((100, 100), dtype=np.float32)
    pred[10:30, 10:30] = 1.0

    map_val = compute_map(pred, target, method="instance", iou_threshold=0.5)
    assert 0.0 <= map_val <= 1.0
