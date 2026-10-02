"""
Unit tests for U-Net++, ECA-U-Net++, and Baseline Architectures.
"""

import pytest
import torch
from models import create_model, UNet, UNetPlusPlus, ECAUNetPlusPlus, FCN8s, ENet, UNet3Plus, SwiftNet


@pytest.mark.parametrize("model_name", [
    "unetplusplus_eca",
    "unetplusplus",
    "unet",
    "fcn",
    "enet",
    "unet3plus",
    "swiftnet",
])
def test_model_forward_shape(model_name):
    """Verify forward pass output shape [B, 1, 256, 256] for all architectures."""
    b, c, h, w = 2, 1, 256, 256
    x = torch.randn(b, c, h, w)
    config = {"image": {"channels": c}, "model": {"name": model_name}}
    model = create_model(model_name, config)
    model.eval()

    with torch.no_grad():
        out = model(x)
        if isinstance(out, list):
            out = out[-1]
        assert out.shape == (b, 1, h, w), f"Model '{model_name}' expected output shape {(b, 1, h, w)}, got {out.shape}"


def test_eca_unet_plus_plus_attention_extraction():
    """Verify ECA-U-Net++ bottleneck attention weight extraction."""
    x = torch.randn(2, 1, 256, 256)
    model = ECAUNetPlusPlus(in_channels=1, out_channels=1, use_eca=True)
    logits, att_weights = model(x, return_attention=True)
    assert logits.shape == (2, 1, 256, 256)
    assert att_weights is not None
    assert att_weights.ndim == 2  # [B, C]


def test_backward_pass():
    """Test backward pass optimization step on ECA-U-Net++."""
    x = torch.randn(2, 1, 128, 128)
    target = torch.randint(0, 2, (2, 1, 128, 128)).float()
    model = ECAUNetPlusPlus(in_channels=1, out_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss), "Loss should be a valid number"
