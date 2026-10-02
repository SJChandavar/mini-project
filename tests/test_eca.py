"""
Unit tests for Efficient Channel Attention (ECA-Net) Module.
"""

import pytest
import torch
from models.eca import ECABlock


def test_eca_dimensions():
    """Test ECA forward pass tensor shape retention."""
    b, c, h, w = 2, 512, 32, 32
    x = torch.randn(b, c, h, w)
    eca = ECABlock(channels=c)
    out = eca(x)
    assert out.shape == (b, c, h, w), f"Expected shape {(b, c, h, w)}, got {out.shape}"


def test_eca_attention_weights():
    """Test ECA attention weights extraction in range [0, 1]."""
    b, c, h, w = 2, 128, 16, 16
    x = torch.randn(b, c, h, w)
    eca = ECABlock(channels=c)
    weights = eca.get_attention_weights(x)
    assert weights.shape == (b, c), f"Expected weights shape {(b, c)}, got {weights.shape}"
    assert (weights >= 0.0).all() and (weights <= 1.0).all(), "Attention weights must be within [0, 1]"


def test_eca_gradient_flow():
    """Test backward pass and gradient computation through ECA block."""
    x = torch.randn(2, 64, 16, 16, requires_grad=True)
    eca = ECABlock(channels=64)
    out = eca(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should flow back to input x"
    assert not torch.isnan(x.grad).any(), "Gradients should not contain NaNs"
