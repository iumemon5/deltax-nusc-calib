"""
Loss Functions for Intrinsics Optimization

The main idea: LiDAR points projected with correct intrinsics should land on
image edges (depth discontinuities, object boundaries, etc). We use a distance
transform so the loss is differentiable and smooth.

Author: DeltaX Assignment Submission
"""

from typing import Tuple

import cv2
import numpy as np
import torch


def edge_distance_transform(
    gray: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    normalize: bool = True,
) -> np.ndarray:
    edges = cv2.Canny(gray, canny_low, canny_high)
    inverted = cv2.bitwise_not(edges)
    dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
    if normalize:
        max_dim = max(gray.shape[:2])
        if max_dim > 0:
            dist = dist / float(max_dim)
    return dist.astype(np.float32)


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    abs_x = torch.abs(x)
    quad = torch.minimum(abs_x, torch.tensor(delta, device=x.device))
    lin = abs_x - quad
    return 0.5 * quad * quad + delta * lin


def pixel_grid(height: int, width: int, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    ys = np.arange(0, height, stride, dtype=np.float32)
    xs = np.arange(0, width, stride, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return grid_x, grid_y
