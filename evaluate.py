import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import NPYSliceDataset
from models import build_model
from utils.metrics import (
    segmentation_metrics_per_sample_from_binary,
    summarize_metric_values,
)
from utils.misc import load_config, save_json, estimate_flops, count_parameters
from utils.visualization import save_prediction_grid
from utils.tta import multiscale_logits
from utils.postprocess import postprocess_batch_predictions, binary_mask_to_logits


METRIC_NAMES = [
    "dice",
    "iou",
    "precision",
    "recall",
    "f2",
    "hd95",
    "assd",
    "boundary_f1",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_selection_lock(config_path: Path, checkpoint_path: Path, lock_path: Path) -> dict:
    """Verify that a factorial test evaluation occurs only after validation lock."""
    if not lock_path.exists():
        raise FileNotFoundError(
            f"Required validation selection lock not found: {lock_path}. "
            "Do not evaluate the test set before architecture selection is frozen."
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    config_resolved = config_path.resolve()
    checkpoint_resolved = checkpoint_path.resolve()
    matching = []
    for record in lock.get("all_locked_files", []):
        if (
            Path(record["config"]).resolve() == config_resolved
            and Path(record["checkpoint"]).resolve() == checkpoint_resolved
        ):
            matching.append(record)
    if len(matching) != 1:
        raise RuntimeError(
            "Config/checkpoint pair is not present exactly once in the selection lock: "
            f"{config_path}, {checkpoint_path}"
        )
    record = matching[0]
    if _sha256_file(config_path) != record["config_sha256"]:
        raise RuntimeError(f"Config changed after architecture lock: {config_path}")
    if _sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint changed after architecture lock: {checkpoint_path}")
    return lock


def save_per_case_csv(rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_index", *METRIC_NAMES]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/efficient_b0_boundary.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--threshold", type=float, default=None, help="Override cfg training.threshold")
    parser.add_argument(
        "--tta-scales",
        nargs="+",
        type=float,
        default=None,
        help="Optional multi-scale inference scales, e.g. 1.0 1.25 1.5",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Disable postprocess even if config postprocess.enabled=true",
    )
    parser.add_argument(
        "--std-ddof",
        type=int,
        default=1,
        choices=[0, 1],
        help="0=population std, 1=sample std for per-case ± reporting",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    paths = cfg["paths"]
    lock_info = None
    eval_cfg = cfg.get("evaluation", {})
    if args.split == "test" and bool(eval_cfg.get("require_selection_lock_for_test", False)):
        lock_path = Path(eval_cfg.get("selection_lock", "outputs/factorial/selection/SELECTION_LOCK.json"))
        lock_info = verify_selection_lock(Path(args.config), Path(args.checkpoint), lock_path)
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
    metric_values = {name: [] for name in METRIC_NAMES}
    per_case_rows = []
    first_batch = None
    case_index = 0
    post_stats = {
        "components_before": 0,
        "components_after": 0,
        "removed_small": 0,
        "removed_intensity": 0,
    }

    for images, masks in tqdm(loader, desc=f"eval-{args.split}"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        seg_logits = multiscale_logits(model, images, scales=tta_scales)
        probs = torch.sigmoid(seg_logits)

        pcfg = cfg.get("postprocess", {})
        use_pp = bool(pcfg.get("enabled", False)) and not args.no_postprocess
        if use_pp:
            pred_mask, stats = postprocess_batch_predictions(images, probs, pcfg, threshold=threshold)
            for key in post_stats:
                post_stats[key] += stats.get(key, 0)
            vis_seg = binary_mask_to_logits(pred_mask)
        else:
            pred_mask = (probs > threshold).float()
            vis_seg = seg_logits

        batch_metrics = segmentation_metrics_per_sample_from_binary(
            preds=pred_mask,
            targets=masks,
            beta=2.0,
        )
        batch_size = images.size(0)
        for local_index in range(batch_size):
            row = {"case_index": case_index}
            for metric_name in METRIC_NAMES:
                value = float(batch_metrics[metric_name][local_index])
                metric_values[metric_name].append(value)
                row[metric_name] = value
            per_case_rows.append(row)
            case_index += 1

        outputs = {"seg": vis_seg, "boundary": None, "deep_outputs": []}
        if first_batch is None:
            first_batch = (
                images.detach().cpu(),
                masks.detach().cpu(),
                {
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in outputs.items()
                },
            )

    params = count_parameters(model)
    flops, _ = estimate_flops(
        model,
        input_shape=(
            1,
            mcfg.get("in_channels", 1),
            tcfg.get("image_size", 256),
            tcfg.get("image_size", 256),
        ),
        device=device,
    )

    metric_statistics = {
        name: summarize_metric_values(values, ddof=args.std_ddof)
        for name, values in metric_values.items()
    }
    results = {}
    for metric_name, stats in metric_statistics.items():
        # Keep the original top-level mean fields for backward compatibility.
        results[metric_name] = stats["mean"]
        results[f"{metric_name}_std"] = stats["std"]
        results[f"{metric_name}_sem"] = stats["sem"]
        results[f"{metric_name}_ci95_half_width"] = stats["ci95_half_width"]
        results[f"{metric_name}_mean_pm_std"] = stats["mean_pm_std"]
        results[f"{metric_name}_nonfinite_count"] = stats["n_nonfinite"]

    results["metric_statistics"] = metric_statistics
    results["statistics_scope"] = "per-case"
    results["statistics_notation"] = "mean ± sample standard deviation" if args.std_ddof == 1 else "mean ± population standard deviation"
    results["std_ddof"] = args.std_ddof
    results["n_cases"] = len(per_case_rows)
    results["params"] = params
    results["gflops"] = None if flops is None else flops / 1e9
    results["threshold"] = threshold
    results["tta_scales"] = tta_scales
    results["postprocess_enabled"] = bool(cfg.get("postprocess", {}).get("enabled", False)) and not args.no_postprocess
    results.update({f"post_{key}": value for key, value in post_stats.items()})
    results["checkpoint_epoch"] = ckpt.get("epoch")
    results["checkpoint_best_dice"] = ckpt.get("best_dice")
    results["checkpoint_best_validation_metrics"] = ckpt.get("best_validation_metrics")
    results["checkpoint_selection_policy"] = ckpt.get("checkpoint_selection_policy")
    results["checkpoint_selection_reason"] = ckpt.get("selection_reason")
    results["validation_metric_policy"] = ckpt.get("validation_metric_policy")
    results["split_sha256"] = ckpt.get("split_sha256")
    results["checkpoint_selection_source"] = "validation only"
    if lock_info is not None:
        results["selection_lock_verified"] = True
        results["selection_lock_selected_variant"] = lock_info.get("selected_variant")
        results["test_interpretation"] = (
            "confirmatory selected-model evaluation"
            if cfg.get("experiment", {}).get("variant") == lock_info.get("selected_variant")
            else "post-lock ablation evaluation; not used for architecture selection"
        )
    else:
        results["selection_lock_verified"] = False

    out_dir = Path(paths.get("output_dir", "outputs")) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, out_dir / f"{args.split}_results.json")
    save_per_case_csv(per_case_rows, out_dir / f"{args.split}_per_case_metrics.csv")

    if first_batch is not None:
        images, masks, cpu_outputs = first_batch
        save_prediction_grid(
            images,
            masks,
            cpu_outputs,
            out_dir / f"{args.split}_predictions.png",
            threshold=threshold,
        )

    compact = {
        metric: results[f"{metric}_mean_pm_std"]
        for metric in METRIC_NAMES
    }
    compact.update({"params": params, "gflops": results["gflops"], "n_cases": results["n_cases"]})
    print(compact)
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
