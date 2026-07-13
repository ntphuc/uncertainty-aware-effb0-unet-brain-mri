import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import NPYSliceDataset
from models import build_model
from utils.metrics import (
    dice_score_from_logits, iou_score_from_logits, batch_surface_metrics,
    dice_score_from_binary, iou_score_from_binary, batch_surface_metrics_from_binary,
    precision_recall_fbeta_from_logits, precision_recall_fbeta_from_binary, AverageMeter,
)
from utils.misc import load_config, save_json, estimate_flops, count_parameters
from utils.visualization import save_prediction_grid
from utils.tta import multiscale_logits
from utils.postprocess import postprocess_batch_predictions, binary_mask_to_logits


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/efficient_b0_boundary.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--threshold", type=float, default=None, help="Override cfg training.threshold")
    parser.add_argument("--tta-scales", nargs="+", type=float, default=None, help="Optional multi-scale inference scales, e.g. 1.0 1.25 1.5")
    parser.add_argument("--no-postprocess", action="store_true", help="Disable postprocess even if config postprocess.enabled=true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    paths = cfg["paths"]
    tcfg = cfg["training"]
    mcfg = cfg["model"]

    if args.split == "test":
        x_path, y_path = paths["x_test"], paths["y_test"]
    else:
        x_path, y_path = paths["x_train"], paths["y_train"]

    dataset = NPYSliceDataset(
        x_path,
        y_path,
        image_size=tcfg.get("image_size", 256),
        in_channels=mcfg.get("in_channels", 1),
        normalize="zscore",
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=tcfg.get("batch_size", 8),
        shuffle=False,
        num_workers=tcfg.get("num_workers", 2),
        pin_memory=True,
    )

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    threshold = float(args.threshold if args.threshold is not None else tcfg.get("threshold", 0.5))
    tta_scales = args.tta_scales or [1.0]
    meters = {k: AverageMeter() for k in ["dice", "iou", "precision", "recall", "f2", "hd95", "assd", "boundary_f1"]}
    first_batch = None
    post_stats = {"components_before": 0, "components_after": 0, "removed_small": 0, "removed_intensity": 0}

    for images, masks in tqdm(loader, desc=f"eval-{args.split}"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        seg_logits = multiscale_logits(model, images, scales=tta_scales)
        probs = torch.sigmoid(seg_logits)
        bs = images.size(0)

        pcfg = cfg.get("postprocess", {})
        use_pp = bool(pcfg.get("enabled", False)) and not args.no_postprocess
        if use_pp:
            pred_mask, st = postprocess_batch_predictions(images, probs, pcfg, threshold=threshold)
            for k in post_stats:
                post_stats[k] += st.get(k, 0)
            meters["dice"].update(dice_score_from_binary(pred_mask, masks), bs)
            meters["iou"].update(iou_score_from_binary(pred_mask, masks), bs)
            pr = precision_recall_fbeta_from_binary(pred_mask, masks, beta=2.0)
            meters["precision"].update(pr["precision"], bs)
            meters["recall"].update(pr["recall"], bs)
            meters["f2"].update(pr["f_beta"], bs)
            surf = batch_surface_metrics_from_binary(pred_mask, masks)
            vis_seg = binary_mask_to_logits(pred_mask)
        else:
            meters["dice"].update(dice_score_from_logits(seg_logits, masks, threshold), bs)
            meters["iou"].update(iou_score_from_logits(seg_logits, masks, threshold), bs)
            pr = precision_recall_fbeta_from_logits(seg_logits, masks, threshold=threshold, beta=2.0)
            meters["precision"].update(pr["precision"], bs)
            meters["recall"].update(pr["recall"], bs)
            meters["f2"].update(pr["f_beta"], bs)
            surf = batch_surface_metrics(seg_logits, masks, threshold=threshold)
            vis_seg = seg_logits

        for k in ["hd95", "assd", "boundary_f1"]:
            meters[k].update(surf[k], bs)
        outputs = {"seg": vis_seg, "boundary": None, "deep_outputs": []}
        if first_batch is None:
            first_batch = (images.detach().cpu(), masks.detach().cpu(), {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in outputs.items()})

    params = count_parameters(model)
    flops, _ = estimate_flops(
        model,
        input_shape=(1, mcfg.get("in_channels", 1), tcfg.get("image_size", 256), tcfg.get("image_size", 256)),
        device=device,
    )

    results = {k: v.avg for k, v in meters.items()}
    results["params"] = params
    results["gflops"] = None if flops is None else flops / 1e9
    results["threshold"] = threshold
    results["tta_scales"] = tta_scales
    results["postprocess_enabled"] = bool(cfg.get("postprocess", {}).get("enabled", False)) and not args.no_postprocess
    results.update({f"post_{k}": v for k, v in post_stats.items()})
    results["checkpoint_epoch"] = ckpt.get("epoch")
    results["checkpoint_best_dice"] = ckpt.get("best_dice")

    out_dir = Path(paths.get("output_dir", "outputs")) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, out_dir / f"{args.split}_results.json")

    if first_batch is not None:
        images, masks, cpu_outputs = first_batch
        save_prediction_grid(images, masks, cpu_outputs, out_dir / f"{args.split}_predictions.png", threshold=threshold)

    print(results)
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
