import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets import NPYSliceDataset
from losses import CombinedSegBoundaryLoss
from models import build_model
from utils.metrics import AverageMeter, dice_score_from_logits, iou_score_from_logits, batch_surface_metrics, precision_recall_fbeta_from_logits
from utils.misc import ensure_dir, load_config, set_seed, count_parameters
from utils.visualization import save_prediction_grid


def make_loaders(cfg):
    tcfg = cfg["training"]
    acfg = cfg.get("augmentation", {})
    mcfg = cfg["model"]
    paths = cfg["paths"]

    full_dataset = NPYSliceDataset(
        paths["x_train"],
        paths["y_train"],
        image_size=tcfg.get("image_size", 256),
        in_channels=mcfg.get("in_channels", 1),
        normalize="zscore",
        augment=False,
    )

    val_ratio = float(tcfg.get("val_ratio", 0.15))
    val_len = max(1, int(len(full_dataset) * val_ratio))
    train_len = len(full_dataset) - val_len
    train_subset, val_subset = random_split(
        full_dataset,
        lengths=[train_len, val_len],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
    )

    # Enable augmentation on the underlying dataset only during training would also affect val because random_split shares dataset.
    # To avoid leakage, create two independent datasets and reuse indices.
    train_dataset = NPYSliceDataset(
        paths["x_train"],
        paths["y_train"],
        image_size=tcfg.get("image_size", 256),
        in_channels=mcfg.get("in_channels", 1),
        normalize="zscore",
        augment=True,
        horizontal_flip=acfg.get("horizontal_flip", True),
        vertical_flip=acfg.get("vertical_flip", False),
        random_rotate90=acfg.get("random_rotate90", True),
        lesion_crop_prob=acfg.get("lesion_crop_prob", 0.0),
        lesion_crop_scale_min=acfg.get("lesion_crop_scale_min", 1.3),
        lesion_crop_scale_max=acfg.get("lesion_crop_scale_max", 2.5),
        lesion_crop_min_size=acfg.get("lesion_crop_min_size", 96),
        lesion_crop_mode=acfg.get("lesion_crop_mode", "whole"),
        component_crop_prob=acfg.get("component_crop_prob", 0.75),
        component_min_area=acfg.get("component_min_area", 5),
        component_connectivity=acfg.get("component_connectivity", 8),
        small_component_bias=acfg.get("small_component_bias", 0.75),
        component_jitter_ratio=acfg.get("component_jitter_ratio", 0.15),
        intensity_aug=acfg.get("intensity_aug", False),
        noise_std=acfg.get("noise_std", 0.03),
        contrast_range=tuple(acfg.get("contrast_range", [0.9, 1.1])),
        brightness_range=tuple(acfg.get("brightness_range", [-0.05, 0.05])),
    )
    val_dataset = NPYSliceDataset(
        paths["x_train"],
        paths["y_train"],
        image_size=tcfg.get("image_size", 256),
        in_channels=mcfg.get("in_channels", 1),
        normalize="zscore",
        augment=False,
    )
    train_subset.dataset = train_dataset
    val_subset.dataset = val_dataset

    train_loader = DataLoader(
        train_subset,
        batch_size=tcfg.get("batch_size", 8),
        shuffle=True,
        num_workers=tcfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=tcfg.get("batch_size", 8),
        shuffle=False,
        num_workers=tcfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp=True, threshold=0.5):
    model.train()
    meters = {k: AverageMeter() for k in ["loss", "seg_loss", "tversky_loss", "component_weight_loss", "hard_negative_loss", "boundary_loss", "deep_loss", "dice", "iou", "precision", "recall", "f2"]}

    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(images)
            loss_dict = criterion(outputs, masks)
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        meters["loss"].update(loss.item(), bs)
        meters["seg_loss"].update(loss_dict["seg_loss"].item(), bs)
        meters["tversky_loss"].update(loss_dict.get("tversky_loss", torch.tensor(0.0, device=masks.device)).item(), bs)
        meters["component_weight_loss"].update(loss_dict.get("component_weight_loss", torch.tensor(0.0, device=masks.device)).item(), bs)
        meters["hard_negative_loss"].update(loss_dict.get("hard_negative_loss", torch.tensor(0.0, device=masks.device)).item(), bs)
        meters["boundary_loss"].update(loss_dict["boundary_loss"].item(), bs)
        meters["deep_loss"].update(loss_dict["deep_loss"].item(), bs)
        meters["dice"].update(dice_score_from_logits(outputs["seg"], masks, threshold), bs)
        meters["iou"].update(iou_score_from_logits(outputs["seg"], masks, threshold), bs)
        pr = precision_recall_fbeta_from_logits(outputs["seg"], masks, threshold=threshold, beta=2.0)
        meters["precision"].update(pr["precision"], bs)
        meters["recall"].update(pr["recall"], bs)
        meters["f2"].update(pr["f_beta"], bs)

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, amp=True, threshold=0.5):
    model.eval()
    meters = {k: AverageMeter() for k in ["loss", "seg_loss", "tversky_loss", "component_weight_loss", "hard_negative_loss", "boundary_loss", "deep_loss", "dice", "iou", "precision", "recall", "f2", "hd95", "assd", "boundary_f1"]}
    first_batch = None

    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(images)
            loss_dict = criterion(outputs, masks)

        bs = images.size(0)
        meters["loss"].update(loss_dict["loss"].item(), bs)
        meters["seg_loss"].update(loss_dict["seg_loss"].item(), bs)
        meters["tversky_loss"].update(loss_dict.get("tversky_loss", torch.tensor(0.0, device=masks.device)).item(), bs)
        meters["component_weight_loss"].update(loss_dict.get("component_weight_loss", torch.tensor(0.0, device=masks.device)).item(), bs)
        meters["hard_negative_loss"].update(loss_dict.get("hard_negative_loss", torch.tensor(0.0, device=masks.device)).item(), bs)
        meters["boundary_loss"].update(loss_dict["boundary_loss"].item(), bs)
        meters["deep_loss"].update(loss_dict["deep_loss"].item(), bs)
        meters["dice"].update(dice_score_from_logits(outputs["seg"], masks, threshold), bs)
        meters["iou"].update(iou_score_from_logits(outputs["seg"], masks, threshold), bs)
        pr = precision_recall_fbeta_from_logits(outputs["seg"], masks, threshold=threshold, beta=2.0)
        meters["precision"].update(pr["precision"], bs)
        meters["recall"].update(pr["recall"], bs)
        meters["f2"].update(pr["f_beta"], bs)

        surf = batch_surface_metrics(outputs["seg"], masks, threshold=threshold)
        meters["hd95"].update(surf["hd95"], bs)
        meters["assd"].update(surf["assd"], bs)
        meters["boundary_f1"].update(surf["boundary_f1"], bs)

        if first_batch is None:
            first_batch = (images.detach().cpu(), masks.detach().cpu(), {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in outputs.items()})

    return {k: v.avg for k, v in meters.items()}, first_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/efficient_b0_boundary.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    out_dir = Path(cfg["paths"].get("output_dir", "outputs"))
    ckpt_dir = out_dir / "checkpoints"
    vis_dir = out_dir / "visualizations"
    ensure_dir(ckpt_dir)
    ensure_dir(vis_dir)

    device = torch.device(args.device)
    train_loader, val_loader = make_loaders(cfg)

    model = build_model(cfg).to(device)
    print(f"Trainable params: {count_parameters(model):,}")

    lcfg = cfg["loss"]
    criterion = CombinedSegBoundaryLoss(
        lambda_boundary=lcfg.get("lambda_boundary", 0.15),
        beta_deep_supervision=lcfg.get("beta_deep_supervision", 0.25),
        boundary_kernel_size=lcfg.get("boundary_kernel_size", 3),
        gamma_tversky=lcfg.get("gamma_tversky", 0.0),
        alpha_tversky=lcfg.get("alpha_tversky", 0.3),
        beta_tversky=lcfg.get("beta_tversky", 0.7),
        focal_tversky_gamma=lcfg.get("focal_tversky_gamma", 0.75),
        use_focal_tversky=lcfg.get("use_focal_tversky", True),
        use_soft_boundary=lcfg.get("use_soft_boundary", False),
        soft_boundary_sigma=lcfg.get("soft_boundary_sigma", 2.0),
        soft_boundary_radius=lcfg.get("soft_boundary_radius", 5),
        use_component_weight=lcfg.get("use_component_weight", False),
        component_weight_lambda=lcfg.get("component_weight_lambda", 0.0),
        component_weight_max=lcfg.get("component_weight_max", 4.0),
        component_weight_power=lcfg.get("component_weight_power", 0.5),
        component_weight_min_area=lcfg.get("component_weight_min_area", 5),
        use_hard_negative_loss=lcfg.get("use_hard_negative_loss", False),
        lambda_hard_negative=lcfg.get("lambda_hard_negative", 0.0),
        hard_negative_threshold=lcfg.get("hard_negative_threshold", 0.60),
        hard_negative_topk_percent=lcfg.get("hard_negative_topk_percent", 0.01),
        hard_negative_min_pixels=lcfg.get("hard_negative_min_pixels", 16),
    )

    tcfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.get("learning_rate", 3e-4),
        weight_decay=tcfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg.get("epochs", 50))
    amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "lr",
            "train_loss", "train_seg_loss", "train_tversky_loss", "train_component_weight_loss", "train_hard_negative_loss", "train_boundary_loss", "train_deep_loss",
            "train_dice", "train_iou", "train_precision", "train_recall", "train_f2",
            "val_loss", "val_seg_loss", "val_tversky_loss", "val_component_weight_loss", "val_hard_negative_loss", "val_boundary_loss", "val_deep_loss",
            "val_dice", "val_iou", "val_precision", "val_recall", "val_f2", "val_hd95", "val_assd", "val_boundary_f1",
        ])

    best_dice = -1.0
    epochs = int(tcfg.get("epochs", 50))
    threshold = float(tcfg.get("threshold", 0.5))

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, amp=amp, threshold=threshold)
        val_metrics, first_batch = validate(model, val_loader, criterion, device, amp=amp, threshold=threshold)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f}, train_dice={train_metrics['dice']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}, val_dice={val_metrics['dice']:.4f}, "
            f"val_iou={val_metrics['iou']:.4f}, recall={val_metrics['recall']:.4f}, "
            f"hd95={val_metrics['hd95']:.3f}, assd={val_metrics['assd']:.3f}"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, lr,
                train_metrics["loss"], train_metrics["seg_loss"], train_metrics["tversky_loss"], train_metrics["component_weight_loss"], train_metrics["hard_negative_loss"], train_metrics["boundary_loss"], train_metrics["deep_loss"],
                train_metrics["dice"], train_metrics["iou"], train_metrics["precision"], train_metrics["recall"], train_metrics["f2"],
                val_metrics["loss"], val_metrics["seg_loss"], val_metrics["tversky_loss"], val_metrics["component_weight_loss"], val_metrics["hard_negative_loss"], val_metrics["boundary_loss"], val_metrics["deep_loss"],
                val_metrics["dice"], val_metrics["iou"], val_metrics["precision"], val_metrics["recall"], val_metrics["f2"],
                val_metrics["hd95"], val_metrics["assd"], val_metrics["boundary_f1"],
            ])

        if first_batch is not None and (epoch == 1 or epoch % int(tcfg.get("save_every", 10)) == 0):
            images, masks, cpu_outputs = first_batch
            save_prediction_grid(images, masks, cpu_outputs, vis_dir / f"epoch_{epoch:03d}.png", threshold=threshold)

        is_best = val_metrics["dice"] > best_dice
        if is_best:
            best_dice = val_metrics["dice"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_dice": best_dice,
                "cfg": cfg,
            }, ckpt_dir / "best.pth")
            print(f"Saved best checkpoint: val_dice={best_dice:.4f}")

        if epoch % int(tcfg.get("save_every", 10)) == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_dice": best_dice,
                "cfg": cfg,
            }, ckpt_dir / f"epoch_{epoch:03d}.pth")

    print(f"Done. Best val Dice: {best_dice:.4f}. Log: {log_path}")


if __name__ == "__main__":
    main()
