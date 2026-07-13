from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from scipy import ndimage
except Exception:  # pragma: no cover
    ndimage = None

import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

class NPYSliceDataset(Dataset):
    """
    Dataset for preprocessed MRI slices stored as .npy.

    Supported X shapes:
        [N,H,W]
        [N,H,W,C]
        [N,C,H,W]

    Supported Y shapes:
        [N,H,W]
        [N,H,W,1]
        [N,1,H,W]

    Output:
        image: [C,H,W], float32
        mask:  [1,H,W], float32 binary

    Important augmentation modes:
        - whole-lesion crop: crop a box covering all foreground pixels.
        - component-aware crop: split the mask into connected lesion components,
          then crop around one component, optionally favoring small components.
        - mixed crop: keep full images sometimes, whole-lesion crop sometimes,
          component crop most of the time.

    Why component-aware crop?
        In multi-lesion slices, a whole-mask bbox can still be very large, so small
        separated lesions remain tiny after resizing. Component-aware crop gives each
        separated lesion a chance to be seen at higher effective resolution during training.
    """

    def __init__(
        self,
        x_path: str,
        y_path: str,
        image_size: Optional[int] = 256,
        in_channels: int = 1,
        normalize: str = "zscore",
        augment: bool = False,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        random_rotate90: bool = True,
        lesion_crop_prob: float = 0.0,
        lesion_crop_scale_min: float = 1.3,
        lesion_crop_scale_max: float = 2.5,
        lesion_crop_min_size: int = 96,
        lesion_crop_mode: str = "whole",  # none | whole | component | mixed
        component_crop_prob: float = 0.75,
        component_min_area: int = 5,
        component_connectivity: int = 8,
        small_component_bias: float = 0.75,
        component_jitter_ratio: float = 0.15,
        intensity_aug: bool = False,
        noise_std: float = 0.03,
        contrast_range: tuple = (0.9, 1.1),
        brightness_range: tuple = (-0.05, 0.05),
        return_index: bool = False,
    ):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        if not self.x_path.exists():
            raise FileNotFoundError(f"X file not found: {self.x_path}")
        if not self.y_path.exists():
            raise FileNotFoundError(f"Y file not found: {self.y_path}")

        self.X = np.load(self.x_path, mmap_mode="r")
        self.Y = np.load(self.y_path, mmap_mode="r")
        if len(self.X) != len(self.Y):
            raise ValueError(f"X and Y length mismatch: {len(self.X)} vs {len(self.Y)}")

        self.image_size = image_size
        self.in_channels = in_channels
        self.normalize = normalize
        self.augment = augment
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.random_rotate90 = random_rotate90

        self.lesion_crop_prob = float(lesion_crop_prob)
        self.lesion_crop_scale_min = float(lesion_crop_scale_min)
        self.lesion_crop_scale_max = float(lesion_crop_scale_max)
        self.lesion_crop_min_size = int(lesion_crop_min_size)
        self.lesion_crop_mode = str(lesion_crop_mode).lower()
        self.component_crop_prob = float(component_crop_prob)
        self.component_min_area = int(component_min_area)
        self.component_connectivity = int(component_connectivity)
        self.small_component_bias = float(small_component_bias)
        self.component_jitter_ratio = float(component_jitter_ratio)

        self.intensity_aug = bool(intensity_aug)
        self.noise_std = float(noise_std)
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.return_index = bool(return_index)

        valid_modes = {"none", "whole", "component", "mixed"}
        if self.lesion_crop_mode not in valid_modes:
            raise ValueError(f"lesion_crop_mode must be one of {valid_modes}, got {self.lesion_crop_mode}")

    def __len__(self) -> int:
        return len(self.X)

    def _to_chw(self, arr: np.ndarray, is_mask: bool) -> np.ndarray:
        arr = np.asarray(arr)

        if arr.ndim == 2:
            arr = arr[None, ...]  # [1,H,W]
        elif arr.ndim == 3:
            # Heuristic: if last dim is small, assume HWC; otherwise CHW.
            if arr.shape[-1] <= 4 and arr.shape[0] > 4:
                arr = np.transpose(arr, (2, 0, 1))
            # else already CHW
        else:
            raise ValueError(f"Each sample must be 2D or 3D, got shape {arr.shape}")

        if is_mask:
            if arr.shape[0] > 1:
                # For multi-class masks encoded as channels, merge foreground quickly for binary segmentation.
                arr = arr.max(axis=0, keepdims=True)
            return arr.astype(np.float32)

        arr = arr.astype(np.float32)
        if arr.shape[0] == 1 and self.in_channels == 3:
            arr = np.repeat(arr, 3, axis=0)
        elif arr.shape[0] != self.in_channels:
            if self.in_channels == 1:
                arr = arr[:1]
            else:
                raise ValueError(f"Image channel mismatch: got {arr.shape[0]}, expected {self.in_channels}")
        return arr

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        if self.normalize == "none":
            return image
        if self.normalize == "minmax":
            mn = image.amin(dim=(-2, -1), keepdim=True)
            mx = image.amax(dim=(-2, -1), keepdim=True)
            return (image - mn) / (mx - mn + 1e-6)
        if self.normalize == "zscore":
            mean = image.mean(dim=(-2, -1), keepdim=True)
            std = image.std(dim=(-2, -1), keepdim=True)
            return (image - mean) / (std + 1e-6)
        raise ValueError(f"Unknown normalize mode: {self.normalize}")

    def _resize(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.image_size is None:
            return image, mask
        size = (self.image_size, self.image_size) if isinstance(self.image_size, int) else tuple(self.image_size)
        image = F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=size, mode="nearest").squeeze(0)
        return image, mask

    def _get_foreground_bbox(self, mask: torch.Tensor):
        fg = (mask[0] > 0.5).nonzero(as_tuple=False)
        if fg.numel() == 0:
            return None
        y1 = int(fg[:, 0].min().item())
        y2 = int(fg[:, 0].max().item()) + 1
        x1 = int(fg[:, 1].min().item())
        x2 = int(fg[:, 1].max().item()) + 1
        return {"y1": y1, "y2": y2, "x1": x1, "x2": x2, "area": int(fg.shape[0])}

    def _get_components(self, mask: torch.Tensor) -> List[Dict[str, int]]:
        """Return connected foreground components from a binary mask."""
        if ndimage is None:
            # scipy is listed in requirements. This fallback keeps the project importable.
            bbox = self._get_foreground_bbox(mask)
            return [] if bbox is None else [bbox]

        binary = (mask[0].detach().cpu().numpy() > 0.5)
        if not binary.any():
            return []

        conn = 2 if self.component_connectivity == 8 else 1
        structure = ndimage.generate_binary_structure(2, conn)
        labeled, num = ndimage.label(binary, structure=structure)

        components: List[Dict[str, int]] = []
        for comp_id in range(1, num + 1):
            ys, xs = np.where(labeled == comp_id)
            area = int(len(xs))
            if area < self.component_min_area:
                continue
            components.append({
                "y1": int(ys.min()),
                "y2": int(ys.max()) + 1,
                "x1": int(xs.min()),
                "x2": int(xs.max()) + 1,
                "area": area,
            })
        return components

    def _select_component(self, components: List[Dict[str, int]]) -> Dict[str, int]:
        """
        Select one component. small_component_bias > 0 gives smaller lesions higher probability.

        Example:
            bias=0.0  -> uniform component sampling
            bias=0.75 -> small components sampled more often
            bias=1.0  -> stronger small-lesion emphasis
        """
        if len(components) == 1:
            return components[0]
        areas = np.asarray([max(1, c["area"]) for c in components], dtype=np.float64)
        if self.small_component_bias <= 0:
            probs = np.ones_like(areas) / len(areas)
        else:
            weights = 1.0 / np.power(areas, self.small_component_bias)
            probs = weights / (weights.sum() + 1e-12)
        idx = int(np.random.choice(len(components), p=probs))
        return components[idx]

    def _crop_from_bbox(self, image: torch.Tensor, mask: torch.Tensor, bbox: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = mask.shape[-2:]
        y1, y2 = bbox["y1"], bbox["y2"]
        x1, x2 = bbox["x1"], bbox["x2"]

        bbox_h = max(1, y2 - y1)
        bbox_w = max(1, x2 - x1)
        max_dim = max(bbox_h, bbox_w)

        scale = self.lesion_crop_scale_min + torch.rand(1).item() * (self.lesion_crop_scale_max - self.lesion_crop_scale_min)
        crop_size = int(max(max_dim * scale, self.lesion_crop_min_size))
        crop_size = int(min(crop_size, h, w))

        cy = (y1 + y2) // 2
        cx = (x1 + x2) // 2

        # Jitter avoids teaching the model that lesions always appear perfectly centered.
        jitter = max(1, int(crop_size * self.component_jitter_ratio))
        cy = int(cy + torch.randint(-jitter, jitter + 1, (1,)).item())
        cx = int(cx + torch.randint(-jitter, jitter + 1, (1,)).item())

        top = max(0, min(h - crop_size, cy - crop_size // 2))
        left = max(0, min(w - crop_size, cx - crop_size // 2))
        bottom = top + crop_size
        right = left + crop_size

        return image[:, top:bottom, left:right], mask[:, top:bottom, left:right]

    def _lesion_aware_crop(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lesion-aware crop used only for training augmentation.

        Modes:
            whole:
                Crop one bbox around all lesion pixels. Good for single-lesion cases.
            component:
                Split mask into connected components and crop one component. Good for multi-lesion cases.
            mixed:
                Use component crop for most lesion-crop samples and whole crop for the rest.
                Full images are still preserved by lesion_crop_prob < 1.
        """
        if (not self.augment) or self.lesion_crop_prob <= 0 or self.lesion_crop_mode == "none":
            return image, mask
        if torch.rand(1).item() > self.lesion_crop_prob:
            return image, mask

        bbox_all = self._get_foreground_bbox(mask)
        if bbox_all is None:
            return image, mask

        mode = self.lesion_crop_mode
        if mode == "mixed":
            mode = "component" if torch.rand(1).item() < self.component_crop_prob else "whole"

        if mode == "component":
            components = self._get_components(mask)
            if len(components) > 0:
                bbox = self._select_component(components)
                return self._crop_from_bbox(image, mask, bbox)
            # Fallback to whole lesion crop if connected components cannot be computed.
            return self._crop_from_bbox(image, mask, bbox_all)

        return self._crop_from_bbox(image, mask, bbox_all)

    def _intensity_augment(self, image: torch.Tensor) -> torch.Tensor:
        if (not self.augment) or (not self.intensity_aug):
            return image
        cmin, cmax = self.contrast_range
        bmin, bmax = self.brightness_range
        contrast = cmin + torch.rand(1).item() * (cmax - cmin)
        brightness = bmin + torch.rand(1).item() * (bmax - bmin)
        image = image * contrast + brightness
        if self.noise_std > 0 and torch.rand(1).item() < 0.5:
            image = image + torch.randn_like(image) * self.noise_std
        return image

    def _augment(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.augment:
            return image, mask
        if self.horizontal_flip and torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        if self.vertical_flip and torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
        if self.random_rotate90:
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                image = torch.rot90(image, k=k, dims=[1, 2])
                mask = torch.rot90(mask, k=k, dims=[1, 2])
        return image, mask

    def __getitem__(self, idx: int):
        image = self._to_chw(self.X[idx], is_mask=False)
        mask = self._to_chw(self.Y[idx], is_mask=True)

        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float()
        mask = (mask > 0.5).float()

        # Important: lesion-aware crop is TRAINING ONLY. Validation/test set augment=False,
        # so no ground-truth mask is used to crop during evaluation or real inference.
        image, mask = self._lesion_aware_crop(image, mask)
        image = self._normalize(image)
        image = self._intensity_augment(image)
        image, mask = self._resize(image, mask)
        image, mask = self._augment(image, mask)

        if self.return_index:
            return image, mask, idx
        return image, mask
