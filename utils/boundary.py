import torch
import torch.nn.functional as F


def mask_to_boundary(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Convert binary mask [B,1,H,W] to hard boundary target [B,1,H,W].
    Boundary = dilation(mask) - erosion(mask).
    """
    mask = (mask > 0.5).float()
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=pad)
    boundary = (dilated - eroded).clamp(0, 1)
    return boundary


def _gaussian_kernel2d(radius: int = 5, sigma: float = 2.0, device=None, dtype=None) -> torch.Tensor:
    size = 2 * radius + 1
    coords = torch.arange(size, device=device, dtype=dtype) - radius
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.max().clamp_min(1e-6)
    return kernel.view(1, 1, size, size)


def mask_to_soft_boundary(
    mask: torch.Tensor,
    kernel_size: int = 3,
    sigma: float = 2.0,
    radius: int = 5,
) -> torch.Tensor:
    """
    Convert binary mask to a soft boundary target.

    Why useful:
    - MRI tumor margins are often ambiguous.
    - A hard 0/1 boundary can be too strict.
    - Soft boundary gives high weight near the contour and gradually lower weight around it.

    Output range: [0, 1]
    """
    hard = mask_to_boundary(mask, kernel_size=kernel_size)
    kernel = _gaussian_kernel2d(radius=radius, sigma=sigma, device=mask.device, dtype=mask.dtype)
    soft = F.conv2d(hard, kernel, padding=radius)
    soft = soft / soft.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return soft.clamp(0, 1)
