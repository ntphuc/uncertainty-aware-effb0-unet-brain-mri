import json
from pathlib import Path
from collections import Counter

import cv2
import numpy as np


DATA_ROOT = Path("../brisc2025")

TRAIN_IMAGES = DATA_ROOT / "segmentation_task" / "train" / "images"
TRAIN_MASKS = DATA_ROOT / "segmentation_task" / "train" / "masks"
TEST_IMAGES = DATA_ROOT / "segmentation_task" / "test" / "images"
TEST_MASKS = DATA_ROOT / "segmentation_task" / "test" / "masks"

OUTPUT_JSON = "brisc2025_report.json"


def get_num_channels(arr):
    if arr is None:
        return None
    if len(arr.shape) == 2:
        return 1
    return arr.shape[2]


def classify_mask_values(mask_gray):
    uniq = set(np.unique(mask_gray).tolist())
    if uniq.issubset({0, 1}):
        return "binary_0_1"
    if uniq.issubset({0, 255}):
        return "binary_0_255"
    return "other_values"


def inspect_split(images_dir, masks_dir, split_name):
    image_files = sorted(images_dir.glob("*.jpg"))
    mask_files = sorted(masks_dir.glob("*.png"))

    valid_pairs = 0
    missing_masks = 0
    unreadable_images = 0
    unreadable_masks = 0

    image_shape_counter = Counter()
    mask_shape_counter = Counter()

    image_channel_counter = Counter()
    mask_channel_counter = Counter()

    image_dtype_counter = Counter()
    mask_dtype_counter = Counter()

    image_range_counter = Counter()
    mask_range_counter = Counter()

    mask_type_counter = Counter()

    for img_path in image_files:
        mask_path = masks_dir / f"{img_path.stem}.png"

        if not mask_path.exists():
            missing_masks += 1
            continue

        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img_gray is None:
            unreadable_images += 1
            continue

        if mask_raw is None or mask_gray is None:
            unreadable_masks += 1
            continue

        valid_pairs += 1

        image_shape_counter[str(img_gray.shape)] += 1
        mask_shape_counter[str(mask_gray.shape)] += 1

        image_channel_counter[str(get_num_channels(img_gray))] += 1
        mask_channel_counter[str(get_num_channels(mask_raw))] += 1

        image_dtype_counter[str(img_gray.dtype)] += 1
        mask_dtype_counter[str(mask_gray.dtype)] += 1

        image_range_counter[str((int(img_gray.min()), int(img_gray.max())))] += 1
        mask_range_counter[str((int(mask_gray.min()), int(mask_gray.max())))] += 1

        mask_type_counter[classify_mask_values(mask_gray)] += 1

    return {
        "split_name": split_name,
        "num_images_jpg": len(image_files),
        "num_masks_png": len(mask_files),
        "valid_image_mask_pairs": valid_pairs,
        "missing_masks": missing_masks,
        "unreadable_images": unreadable_images,
        "unreadable_masks": unreadable_masks,
        "image_shapes": dict(image_shape_counter),
        "mask_shapes": dict(mask_shape_counter),
        "image_channels": dict(image_channel_counter),
        "mask_raw_channels": dict(mask_channel_counter),
        "image_dtypes": dict(image_dtype_counter),
        "mask_dtypes": dict(mask_dtype_counter),
        "image_min_max_distribution": dict(image_range_counter),
        "mask_min_max_distribution": dict(mask_range_counter),
        "mask_value_type_distribution": dict(mask_type_counter),
    }


def main():
    report = {
        "dataset_root": str(DATA_ROOT.resolve()),
        "paths_exist": {
            "train_images": TRAIN_IMAGES.exists(),
            "train_masks": TRAIN_MASKS.exists(),
            "test_images": TEST_IMAGES.exists(),
            "test_masks": TEST_MASKS.exists(),
        },
        "notes": {
            "image_read_mode": "grayscale",
            "image_channels_used_in_pipeline": 1,
            "image_normalization_recommendation": "image.astype(np.float32) / 255.0",
            "mask_raw_channel_check": "cv2.IMREAD_UNCHANGED",
            "mask_gray_check": "cv2.IMREAD_GRAYSCALE",
            "mask_normalization_recommendation": "mask.astype(np.float32) / 255.0",
            "mask_binarization_recommendation": "(mask > 0.5).astype(np.float32)",
        },
    }

    if not (TRAIN_IMAGES.exists() and TRAIN_MASKS.exists() and TEST_IMAGES.exists() and TEST_MASKS.exists()):
        report["error"] = (
            "Folder structure not found. Expected:\n"
            "segmentation_task/train/images\n"
            "segmentation_task/train/masks\n"
            "segmentation_task/test/images\n"
            "segmentation_task/test/masks"
        )

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("ERROR:", report["error"])
        print("Saved JSON:", OUTPUT_JSON)
        return

    train_report = inspect_split(TRAIN_IMAGES, TRAIN_MASKS, "train")
    test_report = inspect_split(TEST_IMAGES, TEST_MASKS, "test")

    report["train"] = train_report
    report["test"] = test_report

    report["summary"] = {
        "total_images_jpg": train_report["num_images_jpg"] + test_report["num_images_jpg"],
        "total_masks_png": train_report["num_masks_png"] + test_report["num_masks_png"],
        "total_valid_pairs": train_report["valid_image_mask_pairs"] + test_report["valid_image_mask_pairs"],
        "total_missing_masks": train_report["missing_masks"] + test_report["missing_masks"],
        "total_unreadable_images": train_report["unreadable_images"] + test_report["unreadable_images"],
        "total_unreadable_masks": train_report["unreadable_masks"] + test_report["unreadable_masks"],
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nSaved JSON:", OUTPUT_JSON)


if __name__ == "__main__":
    main()