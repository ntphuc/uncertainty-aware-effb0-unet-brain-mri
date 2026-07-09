import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models import build_model
from utils.misc import load_config
from utils.postprocess import postprocess_single_prediction


def to_tensor(sample, in_channels: int, image_size: int):
    arr = np.asarray(sample)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        if arr.shape[-1] <= 4 and arr.shape[0] > 4:
            arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Input sample must be 2D or 3D, got {arr.shape}")
    arr = arr.astype(np.float32)
    if arr.shape[0] == 1 and in_channels == 3:
        arr = np.repeat(arr, 3, axis=0)
    elif arr.shape[0] != in_channels:
        arr = arr[:in_channels]
    x = torch.from_numpy(np.ascontiguousarray(arr)).float()
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True)
    x = (x - mean) / (std + 1e-6)
    x = F.interpolate(x.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)
    return x


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/efficient_b0_boundary.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pth")
    parser.add_argument("--input_npy", required=True, help="Path to a .npy file containing one slice [H,W] or a batch [N,H,W].")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="outputs/inference_result.png")
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    mcfg = cfg["model"]
    tcfg = cfg["training"]

    data = np.load(args.input_npy)
    sample = data[args.index] if data.ndim >= 3 else data
    x = to_tensor(sample, in_channels=mcfg.get("in_channels", 1), image_size=tcfg.get("image_size", 256)).to(device)

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    outputs = model(x)
    prob = torch.sigmoid(outputs["seg"])[0, 0].detach().cpu().numpy()
    threshold = float(tcfg.get("threshold", 0.5))
    pcfg = cfg.get("postprocess", {})
    if bool(pcfg.get("enabled", False)) and not args.no_postprocess:
        pred, _ = postprocess_single_prediction(
            prob=prob,
            image=x[0, 0].detach().cpu().numpy(),
            threshold=threshold,
            enabled=True,
            remove_small_components=pcfg.get("remove_small_components", True),
            min_component_area=pcfg.get("min_component_area", 20),
            intensity_filter=pcfg.get("intensity_filter", False),
            intensity_percentile=pcfg.get("intensity_percentile", 70.0),
            foreground_percentile=pcfg.get("foreground_percentile", 5.0),
            min_component_mean_z=pcfg.get("min_component_mean_z", None),
            min_component_p75_z=pcfg.get("min_component_p75_z", None),
            connectivity=pcfg.get("connectivity", 8),
        )
    else:
        pred = prob > threshold

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x[0, 0].detach().cpu().numpy(), cmap="gray")
    axes[0].set_title("MRI")
    axes[1].imshow(prob, cmap="gray")
    axes[1].set_title("Probability")
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
