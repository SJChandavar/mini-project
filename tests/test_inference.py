"""
Unit tests for Inference Pipeline.
"""

import numpy as np
import pytest
from inference import InferencePipeline


def test_inference_pipeline_predict():
    """Test inference pipeline execution on synthetic image."""
    dummy_xray = np.random.randint(50, 200, (512, 512), dtype=np.uint8)
    pipeline = InferencePipeline(checkpoint_path=None, model_name="unetplusplus_eca", device="cpu")

    res = pipeline.predict(dummy_xray, threshold=0.5, target_size=(256, 256))

    assert "original_image" in res
    assert "probability_map" in res
    assert "binary_mask" in res
    assert "overlay" in res
    assert "inference_time" in res

    assert res["binary_mask"].shape == (512, 512)
    assert res["overlay"].shape == (512, 512, 3)
    assert res["inference_time"] > 0.0
