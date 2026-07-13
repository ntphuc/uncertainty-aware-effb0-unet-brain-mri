from pathlib import Path

import matplotlib.pyplot as plt
import torch

from utils.boundary import mask_to_boundary


def save_prediction_grid(images, masks, outputs, save_path, max_items: int = 4, threshold: float = 0.5):
    """Save MRI / GT / Prediction / Boundary visualization."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        seg_probs = torch.sigmoid(outputs["seg"]).detach().cpu()
        pred_masks = (seg_probs > threshold).float()
        gt_boundary = mask_to_boundary(masks.detach().cpu())
        if outputs.get("boundary") is not None:
            pred_boundary = torch.sigmoid(outputs["boundary"]).detach().cpu()
        else:
            pred_boundary = torch.zeros_like(gt_boundary)

    images = images.detach().cpu()
    masks = masks.detach().cpu()
    n = min(max_items, images.shape[0])

    fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        img = images[i, 0].numpy()
        gt = masks[i, 0].numpy()
        pred = pred_masks[i, 0].numpy()
        gb = gt_boundary[i, 0].numpy()
        pb = pred_boundary[i, 0].numpy()

        titles = ["MRI", "GT mask", "Pred mask", "GT boundary", "Pred boundary"]
        data = [img, gt, pred, gb, pb]
        for j in range(5):
            axes[i, j].imshow(data[j], cmap="gray")
            axes[i, j].set_title(titles[j])
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)
