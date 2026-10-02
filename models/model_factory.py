"""
Model Factory for instantiating Tooth Segmentation models.
"""

from typing import Dict, Any
import torch.nn as nn

from .unet import UNet
from .unet_plus_plus import UNetPlusPlus
from .eca_unet_plus_plus import ECAUNetPlusPlus
from .fcn import FCN8s
from .enet import ENet
from .unet3plus import UNet3Plus
from .swiftnet import SwiftNet


def create_model(model_name: str, config: Dict[str, Any] = None) -> nn.Module:
    """
    Factory function to instantiate models by name.

    Args:
        model_name (str): Identifier for model architecture.
        config (dict, optional): Configuration dictionary.

    Returns:
        nn.Module: Instantiated PyTorch model.
    """
    name = model_name.lower().replace("-", "").replace("_", "")

    if config is None:
        config = {}

    in_channels = config.get("image", {}).get("channels", 1)
    out_channels = 1
    dropout = config.get("model", {}).get("dropout", 0.1)
    encoder_channels = config.get("model", {}).get("encoder_channels", [64, 128, 256, 512, 1024])

    if name in ["unetpluspluseca", "ecaunetplusplus", "ecaunet++", "unetplusplusecanet"]:
        eca_cfg = config.get("model", {}).get("eca", {})
        return ECAUNetPlusPlus(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_channels=encoder_channels,
            dropout=dropout,
            use_eca=True,
            eca_gamma=eca_cfg.get("gamma", 2.0),
            eca_b=eca_cfg.get("b", 1.0),
        )
    elif name in ["unetplusplus", "unet++"]:
        return UNetPlusPlus(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_channels=encoder_channels,
            dropout=dropout,
        )
    elif name in ["unet"]:
        return UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_channels=encoder_channels,
            dropout=dropout,
        )
    elif name in ["fcn", "fcn8s"]:
        return FCN8s(in_channels=in_channels, out_channels=out_channels)
    elif name in ["enet"]:
        return ENet(in_channels=in_channels, out_channels=out_channels)
    elif name in ["unet3plus", "unet3+"]:
        return UNet3Plus(in_channels=in_channels, out_channels=out_channels)
    elif name in ["swiftnet"]:
        return SwiftNet(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(
            f"Unknown model name: '{model_name}'. Available choices: "
            "['unetplusplus_eca', 'unetplusplus', 'unet', 'fcn', 'enet', 'unet3plus', 'swiftnet']"
        )
