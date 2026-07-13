import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets import NPYSliceDataset
from kd import MedicalSambaTeacher, SegKDLoss
from losses import CombinedSegBoundaryLoss
from models import build_model
from utils.metrics import AverageMeter, dice_score_from_logits, iou_score_from_logits, batch_surface_metrics
from utils.misc import count_parameters, ensure_dir, load_config, set_seed
from utils.visualization import save_prediction_grid


def unpack_batch(batch):
    if len(batch) == 2:
        images, masks = batch
        indices = None
    elif len(batch) == 3:
        images, masks, indices = batch
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")
    return images, masks, indices


def make_loaders(cfg):
    tcfg = cfg["training"]
    acfg = cfg.get("augmentation", {})
    mcfg = cfg["model"]
    paths = cfg["paths"]

    common_ds = dict(
        image_size=tcfg.get("image_size", 256),
        in_channels=mcfg.get("in_channels", 1),
        normalize=tcfg.get("normalize", "zscore"),
        return_index=False,
    )
    full_dataset = NPYSliceDataset(paths["x_train"], paths["y_train"], augment=False, **common_ds)
    val_ratio = float(tcfg.get("val_ratio", 0.15))
    val_len = max(1, int(len(full_dataset) * val_ratio))
    train_len = len(full_dataset) - val_len
    train_subset, val_subset = random_split(
        full_dataset,
        lengths=[train_len, val_len],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
    )

    train_dataset = NPYSliceDataset(
        paths["x_train"],
        paths["y_train"],
        augment=True,
        horizontal_flip=acfg.get("horizontal_flip", True),
        vertical_flip=acfg.get("vertical_flip", False),
        random_rotate90=acfg.get("random_rotate90", True),
        lesion_crop_prob=acfg.get("lesion_crop_prob", 0.0),
        lesion_crop_scale_min=acfg.get("lesion_crop_scale_min", 1.3),
        lesion_crop_scale_max=acfg.get("lesion_crop_scale_max", 2.5),
        lesion_crop_min_size=acfg.get("lesion_crop_min_size", 96),
        lesion_crop_mode=acfg.get("lesion_crop_mode", "none"),
        component_crop_prob=acfg.get("component_crop_prob", 0.75),
        component_min_area=acfg.get("component_min_area", 5),
        component_connectivity=acfg.get("component_connectivity", 8),
        small_component_bias=acfg.get("small_component_bias", 0.75),
        component_jitter_ratio=acfg.get("component_jitter_ratio", 0.15),
        intensity_aug=acfg.get("intensity_aug", False),
        noise_std=acfg.get("noise_std", 0.03),
        contrast_range=tuple(acfg.get("contrast_range", [0.9, 1.1])),
        brightness_range=tuple(acfg.get("brightness_range", [-0.05, 0.05])),
        **common_ds,
    )
    val_dataset = NPYSliceDataset(paths["x_train"], paths["y_train"], augment=False, **common_ds)
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


def build_losses(cfg):
    lcfg = cfg.get("loss", {})
    base_criterion = CombinedSegBoundaryLoss(
        lambda_boundary=lcfg.get("lambda_boundary", 0.10),
        beta_deep_supervision=lcfg.get("beta_deep_supervision", 0.25),
        boundary_kernel_size=lcfg.get("boundary_kernel_size", 3),
        gamma_tversky=lcfg.get("gamma_tversky", 0.0),
        alpha_tversky=lcfg.get("alpha_tversky", 0.3),
        beta_tversky=lcfg.get("beta_tversky", 0.7),
        focal_tversky_gamma=lcfg.get("focal_tversky_gamma", 0.75),
        use_soft_boundary=lcfg.get("use_soft_boundary", False),
        soft_boundary_sigma=lcfg.get("soft_boundary_sigma", 2.0),
        soft_boundary_radius=lcfg.get("soft_boundary_radius", 5),
        use_component_weight=lcfg.get("use_component_weight", False),
        component_weight_lambda=lcfg.get("component_weight_lambda", 0.0),
        use_hard_negative_loss=lcfg.get("use_hard_negative_loss", False),
        lambda_hard_negative=lcfg.get("lambda_hard_negative", 0.0),
    )
    kcfg = cfg.get("kd", {})
    kd_enabled = bool(kcfg.get("enabled", False))
    kd_criterion = None
    if kd_enabled:
        kd_criterion = SegKDLoss(
            alpha_logit=kcfg.get("alpha_logit", 0.5),
            alpha_dice=kcfg.get("alpha_dice", 0.5),
            alpha_boundary=kcfg.get("alpha_boundary", 0.0),
            temperature=kcfg.get("temperature", 1.0),
            teacher_output_type=kcfg.get("teacher_output_type", cfg.get("teacher", {}).get("output_type", "logit")),
            selective=kcfg.get("selective", True),
            confidence_threshold=kcfg.get("confidence_threshold", 0.65),
            uncertainty_weighted=kcfg.get("uncertainty_weighted", True),
            uncertainty_power=kcfg.get("uncertainty_power", 1.0),
            boundary_kernel_size=kcfg.get("boundary_kernel_size", lcfg.get("boundary_kernel_size", 3)),
            soft_boundary_sigma=kcfg.get("soft_boundary_sigma", 2.0),
            soft_boundary_radius=kcfg.get("soft_boundary_radius", 5),
        )
    return base_criterion, kd_criterion


def teacher_to_pred(teacher_out: Dict[str, torch.Tensor]):
    if "logits" in teacher_out:
        return teacher_out["logits"], "logit"
    if "prob" in teacher_out:
        return teacher_out["prob"], "prob"
    raise KeyError(f"Teacher output missing logits/prob keys: {list(teacher_out.keys())}")


def train_one_epoch(model, teacher, loader, base_criterion, kd_criterion, optimizer, scaler, device, cfg, amp=True):
    model.train()
    meters = {k: AverageMeter() for k in ["loss", "base_loss", "kd_loss", "dice", "iou", "kd_weight_mean"]}
    kd_weight = float(cfg.get("kd", {}).get("lambda_kd", 1.0))
    kd_enabled = bool(cfg.get("kd", {}).get("enabled", False))

    for batch in tqdm(loader, desc="train", leave=False):
        images, masks, _ = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(images)
            base_parts = base_criterion(outputs, masks)
            loss = base_parts["loss"]
            kd_loss_val = images.new_tensor(0.0)
            kd_weight_mean = images.new_tensor(0.0)
            if kd_enabled and kd_criterion is not None:
                if teacher is None:
                    raise RuntimeError("kd.enabled=true but teacher is None")
                teacher_out = teacher(images, masks=masks)
                teacher_pred, _ = teacher_to_pred(teacher_out)
                kd_parts = kd_criterion(outputs["seg"], teacher_pred, teacher_uncertainty=teacher_out.get("uncertainty", None))
                kd_loss_val = kd_parts["loss"]
                kd_weight_mean = kd_parts["kd_weight_mean"]
                loss = loss + kd_weight * kd_loss_val

        scaler.scale(loss).backward()
        if cfg["training"].get("grad_clip", 0.0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("grad_clip", 1.0))
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        meters["loss"].update(loss.item(), bs)
        meters["base_loss"].update(base_parts["loss"].item(), bs)
        meters["kd_loss"].update(float(kd_loss_val.detach().item()), bs)
        meters["kd_weight_mean"].update(float(kd_weight_mean.detach().item()), bs)
        meters["dice"].update(dice_score_from_logits(outputs["seg"].detach(), masks), bs)
        meters["iou"].update(iou_score_from_logits(outputs["seg"].detach(), masks), bs)
    return {k: m.avg for k, m in meters.items()}


@torch.no_grad()
def validate(model, loader, base_criterion, device, amp=True, threshold=0.5):
    model.eval()
    meters = {k: AverageMeter() for k in ["loss", "dice", "iou", "hd95", "assd"]}
    first_batch = None
    for batch in tqdm(loader, desc="val", leave=False):
        images, masks, _ = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(images)
            loss_parts = base_criterion(outputs, masks)
        bs = images.size(0)
        meters["loss"].update(loss_parts["loss"].item(), bs)
        meters["dice"].update(dice_score_from_logits(outputs["seg"], masks, threshold=threshold), bs)
        meters["iou"].update(iou_score_from_logits(outputs["seg"], masks, threshold=threshold), bs)
        surf = batch_surface_metrics(outputs["seg"], masks, threshold=threshold)
        meters["hd95"].update(surf.get("hd95", 0.0), bs)
        meters["assd"].update(surf.get("assd", 0.0), bs)
        if first_batch is None:
            first_batch = (images.detach().cpu(), masks.detach().cpu(), {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in outputs.items()})
    return {k: m.avg for k, m in meters.items()}, first_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg["paths"].get("output_dir", "outputs/student_kd"))
    ckpt_dir = out_dir / "checkpoints"
    vis_dir = out_dir / "vis"
    ensure_dir(ckpt_dir)
    ensure_dir(vis_dir)

    train_loader, val_loader = make_loaders(cfg)
    model = build_model(cfg).to(device)
    print(f"[Student] params={count_parameters(model):,}")

    base_criterion, kd_criterion = build_losses(cfg)
    base_criterion = base_criterion.to(device)
    if kd_criterion is not None:
        kd_criterion = kd_criterion.to(device)

    teacher = None
    if bool(cfg.get("kd", {}).get("enabled", False)):
        teacher = MedicalSambaTeacher(cfg.get("teacher", {}), image_size=cfg["training"].get("image_size", 256)).to(device)
        teacher.eval()
        print(f"[Teacher] enabled mode={cfg.get('teacher', {}).get('mode')} prompt={cfg.get('teacher', {}).get('prompt', {})}")

    tcfg = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.get("learning_rate", 3e-4), weight_decay=tcfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg.get("epochs", 50))
    amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    threshold = float(tcfg.get("threshold", 0.5))
    best_dice = -1.0

    log_path = out_dir / "train_kd_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_base_loss", "train_kd_loss", "train_dice", "train_iou", "train_kd_weight_mean", "val_loss", "val_dice", "val_iou", "val_hd95", "val_assd"])

    for epoch in range(1, int(tcfg.get("epochs", 50)) + 1):
        train_m = train_one_epoch(model, teacher, train_loader, base_criterion, kd_criterion, optimizer, scaler, device, cfg, amp=amp)
        val_m, first_batch = validate(model, val_loader, base_criterion, device, amp=amp, threshold=threshold)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | train_loss={train_m['loss']:.4f} base={train_m['base_loss']:.4f} kd={train_m['kd_loss']:.4f} "
            f"train_dice={train_m['dice']:.4f} | val_dice={val_m['dice']:.4f} val_iou={val_m['iou']:.4f} hd95={val_m['hd95']:.3f}"
        )
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, lr, train_m["loss"], train_m["base_loss"], train_m["kd_loss"], train_m["dice"], train_m["iou"], train_m["kd_weight_mean"], val_m["loss"], val_m["dice"], val_m["iou"], val_m["hd95"], val_m["assd"]])
        if first_batch is not None and (epoch == 1 or epoch % int(tcfg.get("save_every", 10)) == 0):
            images, masks, outputs = first_batch
            save_prediction_grid(images, masks, outputs, vis_dir / f"epoch_{epoch:03d}.png", threshold=threshold)
        if val_m["dice"] > best_dice:
            best_dice = val_m["dice"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best_dice": best_dice, "cfg": cfg}, ckpt_dir / "best.pth")
            print(f"Saved best: {best_dice:.4f}")
        if epoch % int(tcfg.get("save_every", 10)) == 0:
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best_dice": best_dice, "cfg": cfg}, ckpt_dir / f"epoch_{epoch:03d}.pth")
    print(f"Done. best_val_dice={best_dice:.4f}. log={log_path}")


if __name__ == "__main__":
    main()
