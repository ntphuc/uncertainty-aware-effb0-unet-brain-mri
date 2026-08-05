from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models.hcsaf_br import (
    BoundaryRefinementBlock,
    ChannelReliabilityEstimator,
    HCSAF,
    HCSAFFeatureProjection,
    LearnedResidualUpsample,
    SpatialReliabilityEstimator,
)


def test_hcsaf_feature_projection_shape_and_gradient():
    module = HCSAFFeatureProjection(24, fusion_channels=64)
    x = torch.randn(2, 24, 32, 32, requires_grad=True)
    y = module(x)
    assert y.shape == (2, 64, 32, 32)
    y.mean().backward()
    assert x.grad is not None


def test_channel_reliability_shape_and_gradient():
    module = ChannelReliabilityEstimator(channels=64, hidden_channels=32)
    encoder = torch.randn(2, 64, 16, 16, requires_grad=True)
    query = torch.randn(2, 64, 32, 32, requires_grad=True)
    logits = module(encoder, query)
    assert logits.shape == (2, 64)
    logits.mean().backward()
    assert encoder.grad is not None
    assert query.grad is not None


def test_spatial_reliability_shape_and_gradient():
    module = SpatialReliabilityEstimator(hidden_channels=8)
    encoder = torch.randn(2, 64, 32, 32, requires_grad=True)
    query = torch.randn(2, 64, 32, 32, requires_grad=True)
    logits = module(encoder, query)
    assert logits.shape == (2, 1, 32, 32)
    logits.mean().backward()
    assert encoder.grad is not None
    assert query.grad is not None


def _make_encoder_features(batch_size: int = 2):
    channels = [24, 40, 80, 112, 320]
    sizes = [128, 64, 32, 16, 8]
    return [
        torch.randn(batch_size, ch, size, size, requires_grad=True)
        for ch, size in zip(channels, sizes)
    ]


def test_hcsaf_channel_only_shapes_normalization_and_gradient():
    features = _make_encoder_features(batch_size=2)
    decoder = torch.randn(2, 256, 16, 16, requires_grad=True)
    module = HCSAF(
        encoder_channels=[24, 40, 80, 112, 320],
        decoder_channels=256,
        fusion_channels=64,
        channel_hidden_channels=32,
        use_spatial=False,
    )
    fused, diagnostics = module(features, decoder, target_size=(16, 16))

    assert fused.shape == (2, 64, 16, 16)
    channel_weights = diagnostics["channel_weights"]
    assert channel_weights.shape == (2, 5, 64)
    assert diagnostics["spatial_weights"] is None
    assert diagnostics["scale_summary"].shape == (2, 5)
    assert torch.all(channel_weights >= 0)
    assert torch.allclose(
        channel_weights.sum(dim=1),
        torch.ones(2, 64),
        atol=1e-6,
    )
    assert torch.allclose(
        diagnostics["scale_summary"].sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )

    fused.mean().backward()
    assert decoder.grad is not None
    assert all(feature.grad is not None for feature in features)


def test_hcsaf_channel_spatial_shapes_normalization_and_gradient():
    features = _make_encoder_features(batch_size=1)
    decoder = torch.randn(1, 96, 64, 64, requires_grad=True)
    module = HCSAF(
        encoder_channels=[24, 40, 80, 112, 320],
        decoder_channels=96,
        fusion_channels=64,
        channel_hidden_channels=32,
        use_spatial=True,
        spatial_hidden_channels=8,
    )
    fused, diagnostics = module(features, decoder, target_size=(64, 64))

    assert fused.shape == (1, 64, 64, 64)
    channel_weights = diagnostics["channel_weights"]
    spatial_weights = diagnostics["spatial_weights"]
    assert channel_weights.shape == (1, 5, 64)
    assert spatial_weights.shape == (1, 5, 64, 64)
    assert torch.allclose(
        channel_weights.sum(dim=1),
        torch.ones(1, 64),
        atol=1e-6,
    )
    assert torch.allclose(
        spatial_weights.sum(dim=1),
        torch.ones(1, 64, 64),
        atol=1e-6,
    )
    assert torch.allclose(
        diagnostics["scale_summary"].sum(dim=1),
        torch.ones(1),
        atol=1e-6,
    )

    fused.mean().backward()
    assert decoder.grad is not None
    assert all(feature.grad is not None for feature in features)


def test_learned_residual_upsample_shape_and_gradient():
    module = LearnedResidualUpsample(channels=64)
    x = torch.randn(2, 64, 32, 32, requires_grad=True)
    y = module(x, target_size=(64, 64))
    assert y.shape == (2, 64, 64, 64)
    y.mean().backward()
    assert x.grad is not None


def test_boundary_refinement_shape_and_gradient():
    module = BoundaryRefinementBlock(channels=32)
    x = torch.randn(2, 32, 64, 64, requires_grad=True)
    refined, guide = module(x)
    assert refined.shape == x.shape
    assert guide.shape == (2, 1, 64, 64)
    (refined.mean() + guide.mean()).backward()
    assert x.grad is not None


def test_complete_hcsaf_br_unet_forward_with_fake_encoder(monkeypatch):
    import types
    import torch.nn as nn
    import torch.nn.functional as F
    import models.efficient_b0_unet as efficient_module

    channels = [24, 40, 80, 112, 320]
    sizes = [128, 64, 32, 16, 8]

    class FakeFeatureInfo:
        def channels(self):
            return channels

    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_info = FakeFeatureInfo()

        def forward(self, x):
            outputs = []
            base = x.mean(dim=1, keepdim=True)
            for ch, size in zip(channels, sizes):
                resized = F.interpolate(
                    base,
                    size=(size, size),
                    mode="bilinear",
                    align_corners=False,
                )
                outputs.append(resized.repeat(1, ch, 1, 1))
            return outputs

    fake_timm = types.SimpleNamespace(create_model=lambda *args, **kwargs: FakeEncoder())
    monkeypatch.setattr(efficient_module, "timm", fake_timm)

    model = efficient_module.EfficientB0UNetBoundary(
        pretrained=False,
        in_channels=1,
        num_classes=1,
        decoder_channels=[256, 160, 96, 64, 32],
        use_multiscale=True,
        fusion_type="hcsaf_br",
        hcsaf_channels=64,
        hcsaf_channel_hidden_channels=32,
        hcsaf_spatial_hidden_channels=8,
        hcsaf_spatial_stage_indices=[2, 3],
        hcsaf_learned_upsample_stage_indices=[3],
        hcsaf_use_learned_final_upsample=True,
        hcsaf_use_boundary_refinement=True,
        expected_encoder_channels=channels,
        use_se=True,
        use_boundary_head=True,
        use_deep_supervision=True,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 1, 256, 256))

    assert outputs["seg"].shape == (1, 1, 256, 256)
    assert outputs["boundary"].shape == (1, 1, 256, 256)
    assert outputs["boundary_guide"].shape == (1, 1, 256, 256)
    assert len(outputs["deep_outputs"]) == 4
    assert outputs["rgamf_weights"] is None
    assert len(outputs["hcsaf_weights"]) == 4

    for stage_index, diagnostics in enumerate(outputs["hcsaf_weights"]):
        channel_weights = diagnostics["channel_weights"]
        assert channel_weights.shape == (1, 5, 64)
        assert torch.allclose(
            channel_weights.sum(dim=1),
            torch.ones(1, 64),
            atol=1e-6,
        )
        if stage_index in {2, 3}:
            assert diagnostics["spatial_weights"] is not None
        else:
            assert diagnostics["spatial_weights"] is None


def test_boundary_guide_receives_auxiliary_supervision():
    from losses.combined import CombinedSegBoundaryLoss

    criterion = CombinedSegBoundaryLoss(
        lambda_boundary=0.15,
        lambda_boundary_guide=0.05,
        beta_deep_supervision=0.0,
    )
    outputs = {
        "seg": torch.randn(2, 1, 32, 32, requires_grad=True),
        "boundary": torch.randn(2, 1, 32, 32, requires_grad=True),
        "boundary_guide": torch.randn(2, 1, 32, 32, requires_grad=True),
        "deep_outputs": [],
    }
    masks = (torch.rand(2, 1, 32, 32) > 0.8).float()
    losses = criterion(outputs, masks)
    assert "boundary_guide_loss" in losses
    assert losses["boundary_guide_loss"].item() >= 0.0
    losses["loss"].backward()
    assert outputs["boundary_guide"].grad is not None
