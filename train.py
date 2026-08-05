import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets import NPYSliceDataset
from losses import CombinedSegBoundaryLoss
from models import build_model
from utils.metrics import AverageMeter, dice_score_from_logits, iou_score_from_logits, precision_recall_fbeta_from_logits, segmentation_metrics_per_sample_from_logits
from utils.misc import ensure_dir, load_config, set_seed, count_parameters, save_json
from utils.visualization import save_prediction_grid
from utils.selection import compare_metric_dicts, normalize_selection_policy


def make_loaders(cfg):
    """Build a fixed, auditable train/validation split.

    The training seed controls model initialization and augmentation randomness,
    while ``training.split_seed`` controls the split.  Keeping split_seed fixed
    across every factorial cell is essential for paired architecture selection.
    """
    tcfg = cfg["training"]
    acfg = cfg.get("augmentation", {})
    mcfg = cfg["model"]
    paths = cfg["paths"]

    base_dataset = NPYSliceDataset(
        paths["x_train"],
        paths["y_train"],
        image_size=tcfg.get("image_size", 256),
        in_channels=mcfg.get("in_channels", 1),
        normalize="zscore",
        augment=False,
    )

    val_ratio = float(tcfg.get("val_ratio", 0.15))
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"training.val_ratio must be in (0, 1), got {val_ratio}")
    split_seed = int(tcfg.get("split_seed", 42))
    val_len = max(1, int(len(base_dataset) * val_ratio))
    train_len = len(base_dataset) - val_len

    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(len(base_dataset), generator=generator).tolist()
    val_indices = sorted(permutation[:val_len])
    train_indices = sorted(permutation[val_len:])
    split_payload = {
        "dataset_length": len(base_dataset),
        "train_length": train_len,
        "validation_length": val_len,
        "validation_ratio": val_ratio,
        "split_seed": split_seed,
        "train_indices": train_indices,
        "validation_indices": val_indices,
    }
    split_bytes = json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    split_payload["split_sha256"] = hashlib.sha256(split_bytes).hexdigest()

    # Independent dataset objects prevent train augmentations from affecting validation.
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

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
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
    return train_loader, val_loader, split_payload


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp=True, threshold=0.5, accumulation_steps=1):
    model.train()
    meters = {k: AverageMeter() for k in ["loss", "seg_loss", "tversky_loss", "component_weight_loss", "hard_negative_loss", "boundary_loss", "boundary_guide_loss", "deep_loss", "dice", "iou", "precision", "recall", "f2"]}

    accumulation_steps = max(1, int(accumulation_steps))
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(loader)
    for step, (images, masks) in enumerate(tqdm(loader, desc="train", leave=False)):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(images)
            loss_dict = criterion(outputs, masks)
            loss = loss_dict["loss"]
            scaled_loss = loss / accumulation_steps

        scaler.scale(scaled_loss).backward()
        should_step = ((step + 1) % accumulation_steps == 0) or ((step + 1) == num_batches)
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = images.size(0)
        meters["loss"].update(loss.item(), bs)
        meters["seg_loss"].update(loss_dict["seg_loss"].item(), bs)
        meters["tversky_loss"].update(loss_dict.get("tversky_loss", torch.tensor(0.0, device=masks.device)).item(), bs)
        meters["component_weight_loss"].update(loss_dict.get("component_weight_loss", torch.tensor(0.0, device=masks.device)).item(), bs)
        meters["hard_negative_loss"].update(loss_dict.get("hard_negative_loss", torch.tensor(0.0, device=masks.device)).item(), bs)
        meters["boundary_loss"].update(loss_dict["boundary_loss"].item(), bs)
        meters["boundary_guide_loss"].update(loss_dict.get("boundary_guide_loss", torch.tensor(0.0, device=masks.device)).item(), bs)
        meters["deep_loss"].update(loss_dict["deep_loss"].item(), bs)
        meters["dice"].update(dice_score_from_logits(outputs["seg"], masks, threshold), bs)
        meters["iou"].update(iou_score_from_logits(outputs["seg"], masks, threshold), bs)
        pr = precision_recall_fbeta_from_logits(outputs["seg"], masks, threshold=threshold, beta=2.0)
        meters["precision"].update(pr["precision"], bs)
        meters["recall"].update(pr["recall"], bs)
        meters["f2"].update(pr["f_beta"], bs)

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    amp=True,
    threshold=0.5,
    empty_surface_penalty=None,
):
    """Validate with per-case metrics suitable for checkpoint selection.

    HD95/ASSD are undefined when exactly one mask is empty.  Rather than
    silently dropping those failures, validation selection replaces non-finite
    distances with a pre-specified finite penalty (the resized image diagonal
    by default).  The number and rate of missing predictions are also logged.
    """
    model.eval()
    loss_names = [
        "loss", "seg_loss", "tversky_loss", "component_weight_loss",
        "hard_negative_loss", "boundary_loss", "boundary_guide_loss", "deep_loss",
    ]
    loss_meters = {name: AverageMeter() for name in loss_names}
    metric_names = [
        "dice", "iou", "precision", "recall", "f2",
        "hd95", "assd", "boundary_f1",
    ]
    metric_values = {name: [] for name in metric_names}
    missing_predictions = 0
    n_cases = 0
    first_batch = None

    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(images)
            loss_dict = criterion(outputs, masks)

        bs = images.size(0)
        loss_meters["loss"].update(loss_dict["loss"].item(), bs)
        loss_meters["seg_loss"].update(loss_dict["seg_loss"].item(), bs)
        loss_meters["tversky_loss"].update(loss_dict.get("tversky_loss", torch.tensor(0.0, device=masks.device)).item(), bs)
        loss_meters["component_weight_loss"].update(loss_dict.get("component_weight_loss", torch.tensor(0.0, device=masks.device)).item(), bs)
        loss_meters["hard_negative_loss"].update(loss_dict.get("hard_negative_loss", torch.tensor(0.0, device=masks.device)).item(), bs)
        loss_meters["boundary_loss"].update(loss_dict["boundary_loss"].item(), bs)
        loss_meters["boundary_guide_loss"].update(loss_dict.get("boundary_guide_loss", torch.tensor(0.0, device=masks.device)).item(), bs)
        loss_meters["deep_loss"].update(loss_dict["deep_loss"].item(), bs)

        per_case = segmentation_metrics_per_sample_from_logits(
            outputs["seg"], masks, threshold=threshold, beta=2.0
        )
        probs = torch.sigmoid(outputs["seg"])
        pred_empty = ((probs > threshold).flatten(1).sum(dim=1) == 0)
        target_nonempty = ((masks > 0.5).flatten(1).sum(dim=1) > 0)
        missing_predictions += int((pred_empty & target_nonempty).sum().item())
        n_cases += bs

        for name in metric_names:
            values = [float(value) for value in per_case[name]]
            if name in {"hd95", "assd"}:
                if empty_surface_penalty is None:
                    if any(not math.isfinite(value) for value in values):
                        raise ValueError(
                            "Non-finite validation surface distance encountered. "
                            "Set training.validation_empty_surface_penalty."
                        )
                else:
                    values = [
                        value if math.isfinite(value) else float(empty_surface_penalty)
                        for value in values
                    ]
            metric_values[name].extend(values)

        if first_batch is None:
            first_batch = (
                images.detach().cpu(),
                masks.detach().cpu(),
                {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in outputs.items()},
            )

    metrics = {name: meter.avg for name, meter in loss_meters.items()}
    for name, values in metric_values.items():
        metrics[name] = float(sum(values) / max(len(values), 1))
    metrics["missing_prediction_count"] = int(missing_predictions)
    metrics["missing_prediction_rate"] = float(missing_predictions / max(n_cases, 1))
    metrics["n_validation_cases"] = int(n_cases)
    metrics["empty_surface_penalty"] = (
        None if empty_surface_penalty is None else float(empty_surface_penalty)
    )
    return metrics, first_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/efficient_b0_boundary.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(
        cfg.get("seed", 42),
        deterministic=bool(cfg.get("training", {}).get("deterministic", False)),
    )

    out_dir = Path(cfg["paths"].get("output_dir", "outputs"))
    ckpt_dir = out_dir / "checkpoints"
    vis_dir = out_dir / "visualizations"
    ensure_dir(ckpt_dir)
    ensure_dir(vis_dir)

    device = torch.device(args.device)
    train_loader, val_loader, split_info = make_loaders(cfg)
    save_json(split_info, out_dir / "data_split.json")

    model = build_model(cfg).to(device)
    print(f"Trainable params: {count_parameters(model):,}")

    lcfg = cfg["loss"]
    criterion = CombinedSegBoundaryLoss(
        lambda_boundary=lcfg.get("lambda_boundary", 0.15),
        lambda_boundary_guide=lcfg.get("lambda_boundary_guide", 0.0),
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
            "val_dice", "val_iou", "val_precision", "val_recall", "val_f2",
            "val_hd95", "val_assd", "val_boundary_f1",
            "val_missing_prediction_count", "val_missing_prediction_rate",
        ])

    selection_policy = normalize_selection_policy(tcfg.get("checkpoint_selection"))
    best_metrics = None
    best_epoch = None
    epochs = int(tcfg.get("epochs", 50))
    threshold = float(tcfg.get("threshold", 0.5))
    penalty_cfg = tcfg.get("validation_empty_surface_penalty", "image_diagonal")
    if isinstance(penalty_cfg, str):
        if penalty_cfg != "image_diagonal":
            raise ValueError(
                "training.validation_empty_surface_penalty must be a number, null, "
                "or 'image_diagonal'."
            )
        image_size = tcfg.get("image_size", 256)
        if isinstance(image_size, int):
            height = width = int(image_size)
        else:
            height, width = map(int, image_size)
        empty_surface_penalty = math.hypot(height - 1, width - 1)
    elif penalty_cfg is None:
        empty_surface_penalty = None
    else:
        empty_surface_penalty = float(penalty_cfg)
    selection_log_path = out_dir / "checkpoint_selection.csv"
    selection_fields = [
        "epoch", "selected", "reason",
        "dice", "iou", "hd95", "assd", "boundary_f1",
        "missing_prediction_count", "missing_prediction_rate", "loss",
    ]
    with selection_log_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=selection_fields).writeheader()

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            amp=amp,
            threshold=threshold,
            accumulation_steps=tcfg.get("accumulation_steps", 1),
        )
        val_metrics, first_batch = validate(
            model,
            val_loader,
            criterion,
            device,
            amp=amp,
            threshold=threshold,
            empty_surface_penalty=empty_surface_penalty,
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f}, train_dice={train_metrics['dice']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}, val_dice={val_metrics['dice']:.4f}, "
            f"val_iou={val_metrics['iou']:.4f}, recall={val_metrics['recall']:.4f}, "
            f"hd95={val_metrics['hd95']:.3f}, assd={val_metrics['assd']:.3f}, "
            f"missing={val_metrics['missing_prediction_rate']:.2%}"
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
                val_metrics["missing_prediction_count"], val_metrics["missing_prediction_rate"],
            ])

        if first_batch is not None and (epoch == 1 or epoch % int(tcfg.get("save_every", 10)) == 0):
            images, masks, cpu_outputs = first_batch
            save_prediction_grid(images, masks, cpu_outputs, vis_dir / f"epoch_{epoch:03d}.png", threshold=threshold)

        eligible = epoch >= selection_policy.get("min_epoch", 1)
        selected, reason = (False, "epoch below selection min_epoch")
        if eligible:
            selected, reason = compare_metric_dicts(val_metrics, best_metrics, selection_policy)

        selection_row = {
            "epoch": epoch,
            "selected": int(selected),
            "reason": reason,
            "dice": val_metrics["dice"],
            "iou": val_metrics["iou"],
            "hd95": val_metrics["hd95"],
            "assd": val_metrics["assd"],
            "boundary_f1": val_metrics["boundary_f1"],
            "missing_prediction_count": val_metrics["missing_prediction_count"],
            "missing_prediction_rate": val_metrics["missing_prediction_rate"],
            "loss": val_metrics["loss"],
        }
        with selection_log_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=selection_fields).writerow(selection_row)

        if selected:
            best_metrics = dict(val_metrics)
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_dice": best_metrics["dice"],  # backward compatibility
                "best_validation_metrics": best_metrics,
                "checkpoint_selection_policy": selection_policy,
                "selection_reason": reason,
                "validation_metric_policy": {
                    "threshold": threshold,
                    "empty_surface_penalty": empty_surface_penalty,
                    "surface_space": "resized pixel coordinates",
                },
                "split_sha256": split_info["split_sha256"],
                "cfg": cfg,
            }
            torch.save(checkpoint, ckpt_dir / "best_validation.pth")
            # Keep the legacy path so existing evaluation commands continue to work.
            torch.save(checkpoint, ckpt_dir / "best.pth")
            save_json({
                "best_epoch": best_epoch,
                "best_validation_metrics": best_metrics,
                "checkpoint_selection_policy": selection_policy,
                "selection_reason": reason,
                "validation_metric_policy": {
                    "threshold": threshold,
                    "empty_surface_penalty": empty_surface_penalty,
                    "surface_space": "resized pixel coordinates",
                },
                "split_sha256": split_info["split_sha256"],
                "training_seed": int(cfg.get("seed", 42)),
                "split_seed": int(tcfg.get("split_seed", 42)),
                "experiment": cfg.get("experiment", {}),
            }, out_dir / "validation_best.json")
            print(
                "Saved validation-selected checkpoint: "
                f"epoch={best_epoch}, dice={best_metrics['dice']:.4f}, "
                f"hd95={best_metrics['hd95']:.3f}, assd={best_metrics['assd']:.3f}"
            )

        if epoch % int(tcfg.get("save_every", 10)) == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_dice": None if best_metrics is None else best_metrics["dice"],
                "best_validation_metrics": best_metrics,
                "checkpoint_selection_policy": selection_policy,
                "split_sha256": split_info["split_sha256"],
                "cfg": cfg,
            }, ckpt_dir / f"epoch_{epoch:03d}.pth")

    if best_metrics is None:
        raise RuntimeError("No checkpoint satisfied the validation selection policy.")
    print(
        f"Done. Validation-selected epoch: {best_epoch}. "
        f"Dice={best_metrics['dice']:.4f}, HD95={best_metrics['hd95']:.3f}, "
        f"ASSD={best_metrics['assd']:.3f}. Log: {log_path}"
    )


if __name__ == "__main__":
    main()
