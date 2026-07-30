from .efficient_b0_unet import EfficientB0UNetBoundary
from .unet import UNet
from .unetpp import UNetPlusPlus
from .transunet import TransUNet
from .umamba import UMamba
from .official_umamba import UMambaBot2D, UMambaEnc2D
from .vmunet import VMUNet as VMUNetProxy
from .original_baselines import (
    OriginalUNet,
    OriginalUNetPlusPlus,
    OriginalAttentionUNet,
    DeepLabV3PlusResNet50,
)
from .official_vmunet import OfficialVMUNet


def _common_model_kwargs(m):
    return {
        "in_channels": m.get("in_channels", 1),
        "num_classes": m.get("num_classes", 1),
    }


def build_model(cfg):
    """Factory for all models used in ablation/baseline experiments.

    Paper-faithful, non-proxy model.name values:
      - unet_original
      - unetpp_original
      - attention_unet_original
      - deeplabv3plus_resnet50
      - vmunet_official (or vmunet)
      - umamba_bot_official
      - umamba_enc_official

    Legacy/custom model.name values:
      - efficientnet_b0_unet_boundary
      - unet
      - unetpp
      - transunet
      - umamba
      - vmunet_proxy
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

    # Paper-faithful baselines -------------------------------------------------
    if name in ["unet_original", "paper_unet", "canonical_unet"]:
        return OriginalUNet(
            **common,
            base_channels=m.get("base_channels", 32),
            batch_norm=m.get("batch_norm", False),
        )

    if name in ["unetpp_original", "paper_unetpp", "canonical_unetpp"]:
        return OriginalUNetPlusPlus(
            **common,
            base_channels=m.get("base_channels", 32),
            deep_supervision=m.get("use_deep_supervision", True),
            batch_norm=m.get("batch_norm", False),
        )

    if name in ["attention_unet_original", "attention_unet", "attunet"]:
        return OriginalAttentionUNet(
            **common,
            base_channels=m.get("base_channels", 32),
            batch_norm=m.get("batch_norm", True),
        )

    if name in ["deeplabv3plus_resnet50", "deeplabv3plus", "deeplabv3+"]:
        return DeepLabV3PlusResNet50(
            **common,
            pretrained=m.get("pretrained", False),
            aspp_channels=m.get("aspp_channels", 256),
            low_level_channels=m.get("low_level_channels", 48),
            atrous_rates=m.get("atrous_rates", [6, 12, 18]),
        )

    if name in ["umamba_bot_official", "u-mamba-bot", "umamba-bot", "umamba_bot"]:
        return UMambaBot2D(
            **common,
            input_size=m.get("input_size", cfg.get("training", {}).get("image_size", 256)),
            channels=m.get("channels", [32, 64, 128, 256, 320, 320]),
            strides=m.get("strides", [1, 2, 2, 2, 2, 2]),
            encoder_blocks=m.get("encoder_blocks", 2),
            decoder_blocks=m.get("decoder_blocks", 2),
            d_state=m.get("d_state", 16),
            d_conv=m.get("d_conv", 4),
            expand=m.get("expand", 2),
            deep_supervision=m.get("use_deep_supervision", False),
            conv_bias=m.get("conv_bias", True),
        )

    if name in ["umamba_enc_official", "u-mamba-enc", "umamba-enc", "umamba_enc"]:
        return UMambaEnc2D(
            **common,
            input_size=m.get("input_size", cfg.get("training", {}).get("image_size", 256)),
            channels=m.get("channels", [32, 64, 128, 256, 320, 320]),
            strides=m.get("strides", [1, 2, 2, 2, 2, 2]),
            encoder_blocks=m.get("encoder_blocks", 2),
            decoder_blocks=m.get("decoder_blocks", 2),
            d_state=m.get("d_state", 16),
            d_conv=m.get("d_conv", 4),
            expand=m.get("expand", 2),
            deep_supervision=m.get("use_deep_supervision", False),
            conv_bias=m.get("conv_bias", True),
        )

    if name in ["vmunet_official", "vm-unet-official", "vmunet", "vm-unet", "vm_unet"]:
        return OfficialVMUNet(
            **common,
            repo_path=m.get("repo_path", "external/VM-UNet"),
            depths=m.get("depths", [2, 2, 9, 2]),
            depths_decoder=m.get("depths_decoder", [2, 9, 2, 2]),
            drop_path_rate=m.get("drop_path_rate", 0.2),
            repeat_grayscale_to_rgb=m.get("repeat_grayscale_to_rgb", True),
            load_ckpt_path=m.get("load_ckpt_path", None),
        )

    # Legacy/custom baselines -------------------------------------------------
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

    if name in ["vmunet_proxy", "vm-unet-proxy", "vm_unet_proxy"]:
        return VMUNetProxy(
            **common,
            base_channels=m.get("base_channels", 32),
            channels=m.get("channels", None),
        )

    raise ValueError(f"Unknown model.name='{name}'. See models/__init__.py for supported names.")
