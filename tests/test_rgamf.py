from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models.rgamf import FeatureProjection, ReliabilityEstimator, RGAMF


def test_feature_projection_shape():
    module = FeatureProjection(24, 64)
    x = torch.randn(2, 24, 128, 128)
    y = module(x)
    assert y.shape == (2, 64, 128, 128)


def test_reliability_estimator_shape_and_gradient():
    module = ReliabilityEstimator(64, 256, hidden_channels=32)
    encoder = torch.randn(2, 64, 8, 8, requires_grad=True)
    decoder = torch.randn(2, 256, 16, 16, requires_grad=True)
    score = module(encoder, decoder)
    assert score.shape == (2,)
    score.sum().backward()
    assert encoder.grad is not None
    assert decoder.grad is not None


def test_rgamf_shapes_weights_and_gradient():
    batch_size = 2
    encoder_channels = [24, 40, 80, 112, 320]
    features = [
        torch.randn(batch_size, 24, 128, 128, requires_grad=True),
        torch.randn(batch_size, 40, 64, 64, requires_grad=True),
        torch.randn(batch_size, 80, 32, 32, requires_grad=True),
        torch.randn(batch_size, 112, 16, 16, requires_grad=True),
        torch.randn(batch_size, 320, 8, 8, requires_grad=True),
    ]
    decoder = torch.randn(batch_size, 256, 16, 16, requires_grad=True)

    module = RGAMF(
        encoder_channels=encoder_channels,
        decoder_channels=256,
        fusion_channels=64,
        reliability_hidden_channels=32,
    )
    fused, alpha = module(features, decoder, target_size=(16, 16))

    assert fused.shape == (batch_size, 64, 16, 16)
    assert alpha.shape == (batch_size, 5)
    assert torch.all(alpha >= 0)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(batch_size), atol=1e-6)

    (fused.mean() + alpha.mean()).backward()
    assert decoder.grad is not None
    assert all(feature.grad is not None for feature in features)


def test_complete_rgamf_unet_forward_with_fake_encoder(monkeypatch):
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
                resized = F.interpolate(base, size=(size, size), mode="bilinear", align_corners=False)
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
        fusion_type="rgamf",
        rgamf_channels=64,
        rgamf_hidden_channels=32,
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
    assert len(outputs["deep_outputs"]) == 4
    assert len(outputs["rgamf_weights"]) == 4
    for alpha in outputs["rgamf_weights"]:
        assert alpha.shape == (1, 5)
        assert torch.allclose(alpha.sum(dim=1), torch.ones(1), atol=1e-6)
