from .efficient_b0_unet import EfficientB0UNetBoundary
from .unet import UNet
from .unetpp import UNetPlusPlus
from .transunet import TransUNet
from .umamba import UMamba
from .vmunet import VMUNet


def _common_model_kwargs(m):
    return {
        "in_channels": m.get("in_channels", 1),
        "num_classes": m.get("num_classes", 1),
    }


def build_model(cfg):
    """Factory for all models used in ablation/baseline experiments.

    Supported model.name values:
      - efficientnet_b0_unet_boundary
      - unet
      - unetpp
      - transunet
      - umamba
      - vmunet
    """
    m = cfg["model"]
    name = str(m.get("name", "efficientnet_b0_unet_boundary")).lower()
    common = _common_model_kwargs(m)

    if name in ["efficientnet_b0_unet_boundary", "effb0_unet", "be_unet"]:
        return EfficientB0UNetBoundary(
            encoder_name=m.get("encoder_name", "efficientnet_b0"),
            pretrained=m.get("pretrained", True),
            in_channels=m.get("in_channels", 1),
            num_classes=m.get("num_classes", 1),
            decoder_channels=m.get("decoder_channels", [256, 160, 96, 64, 32]),
            ms_channels=m.get("ms_channels", 24),
            use_multiscale=m.get("use_multiscale", True),
            use_se=m.get("use_se", True),
            use_boundary_head=m.get("use_boundary_head", True),
            use_deep_supervision=m.get("use_deep_supervision", True),
        )

    if name == "unet":
        return UNet(
            **common,
            base_channels=m.get("base_channels", 32),
            channels=m.get("channels", None),
        )

    if name in ["unetpp", "unet++"]:
        return UNetPlusPlus(
            **common,
            base_channels=m.get("base_channels", 32),
            channels=m.get("channels", None),
            deep_supervision=m.get("use_deep_supervision", False),
        )

    if name == "transunet":
        return TransUNet(
            **common,
            base_channels=m.get("base_channels", 32),
            channels=m.get("channels", None),
            transformer_depth=m.get("transformer_depth", 2),
            transformer_heads=m.get("transformer_heads", 4),
            deep_supervision=m.get("use_deep_supervision", False),
        )

    if name in ["umamba", "u-mamba"]:
        return UMamba(
            **common,
            base_channels=m.get("base_channels", 32),
            channels=m.get("channels", None),
        )

    if name in ["vmunet", "vm-unet", "vm_unet"]:
        return VMUNet(
            **common,
            base_channels=m.get("base_channels", 32),
            channels=m.get("channels", None),
        )

    raise ValueError(f"Unknown model.name='{name}'. See models/__init__.py for supported names.")
