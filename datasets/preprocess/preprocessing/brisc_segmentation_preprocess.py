from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    SEG_TRAIN_IMAGES,
    SEG_TRAIN_MASKS,
    SEG_TEST_IMAGES,
    SEG_TEST_MASKS,
    IMAGE_SIZE,
    MASK_THRESHOLD,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    PLANES_TRAIN_PATH,
    PLANES_TEST_PATH,
)


def parse_plane_from_filename(file_path: Path) -> str:
    """
    Expected filename example:
        brisc2025_train_00001_gl_ax_t1.jpg

    Returns:
        'ax', 'co', 'sa', or 'unknown'
    """
    parts = file_path.stem.split("_")
    if len(parts) >= 6:
        plane = parts[4].lower()
        if plane in {"ax", "co", "sa"}:
            return plane
    return "unknown"


def summarize_planes(planes: List[str]) -> Dict[str, int]:
    stats = {"ax": 0, "co": 0, "sa": 0, "unknown": 0}
    for p in planes:
        if p in stats:
            stats[p] += 1
        else:
            stats["unknown"] += 1
    return stats


class BRISCSegmentationDataset(Dataset):
    """
    BRISC2025 segmentation dataset.

    Processing:
    - read image in grayscale
    - read mask in grayscale
    - resize image to 512x512
    - resize mask with INTER_NEAREST
    - normalize image to [0,1]
    - normalize mask to [0,1]
    - binarize mask to {0,1}
    - output tensors in shape (1, H, W)
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        image_size: Tuple[int, int] = IMAGE_SIZE,
        mask_threshold: float = MASK_THRESHOLD,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.mask_threshold = mask_threshold

        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        self.valid_samples = []

        for img_file in self.image_files:
            mask_file = self.masks_dir / f"{img_file.stem}.png"
            if mask_file.exists():
                self.valid_samples.append((img_file, mask_file))

        print(f"[INFO] {self.images_dir} -> {len(self.valid_samples)} valid image-mask pairs")

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.valid_samples[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Cannot read image: {img_path}")
        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")

        # Resize
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        # Normalize image -> [0,1]
        image = image.astype(np.float32) / 255.0

        # Normalize mask -> [0,1], then binarize -> {0,1}
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > self.mask_threshold).astype(np.float32)

        # Add channel dim -> (1, H, W)
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        plane = parse_plane_from_filename(img_path)

        return {
            "image": image,
            "mask": mask,
            "plane": plane,
        }


def dataset_to_numpy(dataset: BRISCSegmentationDataset):
    """
    Convert entire dataset to numpy arrays.

    Output shapes:
    - X: (N, 1, H, W)
    - Y: (N, 1, H, W)
    - planes: (N,)
    """
    images = []
    masks = []
    planes = []

    for i in range(len(dataset)):
        sample = dataset[i]

        image = sample["image"].numpy()  # (1, H, W)
        mask = sample["mask"].numpy()    # (1, H, W)

        images.append(image)
        masks.append(mask)
        planes.append(sample["plane"])

    X = np.stack(images, axis=0).astype(np.float32)   # (N, 1, H, W)
    Y = np.stack(masks, axis=0).astype(np.float32)    # (N, 1, H, W)
    planes = np.array(planes, dtype=object)           # (N,)

    return X, Y, planes


def export_split(
    images_dir: Path,
    masks_dir: Path,
    x_path: Path,
    y_path: Path,
    planes_path: Path,
    split_name: str,
):
    dataset = BRISCSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=IMAGE_SIZE,
        mask_threshold=MASK_THRESHOLD,
    )

    X, Y, planes = dataset_to_numpy(dataset)

    np.save(x_path, X)
    np.save(y_path, Y)
    np.save(planes_path, planes)

    print(f"[INFO] Exported {split_name}")
    print(f"       X: {x_path} -> shape={X.shape}, dtype={X.dtype}, range=({X.min()}, {X.max()})")
    print(f"       Y: {y_path} -> shape={Y.shape}, dtype={Y.dtype}, unique={np.unique(Y).tolist()}")
    print(f"       Planes: {planes_path} -> distribution={summarize_planes(planes.tolist())}")


def export_all():
    required_dirs = [
        SEG_TRAIN_IMAGES,
        SEG_TRAIN_MASKS,
        SEG_TEST_IMAGES,
        SEG_TEST_MASKS,
    ]

    for p in required_dirs:
        if not p.exists():
            raise FileNotFoundError(f"Missing folder: {p}")

    export_split(
        images_dir=SEG_TRAIN_IMAGES,
        masks_dir=SEG_TRAIN_MASKS,
        x_path=X_TRAIN_PATH,
        y_path=Y_TRAIN_PATH,
        planes_path=PLANES_TRAIN_PATH,
        split_name="train",
    )

    export_split(
        images_dir=SEG_TEST_IMAGES,
        masks_dir=SEG_TEST_MASKS,
        x_path=X_TEST_PATH,
        y_path=Y_TEST_PATH,
        planes_path=PLANES_TEST_PATH,
        split_name="test",
    )

    print("\n[INFO] Done. Generated 6 files:")
    print(f"  1. {X_TRAIN_PATH}")
    print(f"  2. {Y_TRAIN_PATH}")
    print(f"  3. {X_TEST_PATH}")
    print(f"  4. {Y_TEST_PATH}")
    print(f"  5. {PLANES_TRAIN_PATH}")
    print(f"  6. {PLANES_TEST_PATH}")


if __name__ == "__main__":
    export_all()