"""
Iterative Intrinsics Refinement

Takes an initial intrinsics estimate and refines it using gradient descent.
The optimization uses multiple frames simultaneously for stability - single-frame
calibration is way too noisy.

Key trick: parameterize fx/fy as log-scale offsets from initial guess, and
cx/cy as bounded tanh offsets. This prevents the optimizer from going crazy.

Author: DeltaX Assignment Submission
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .losses import charbonnier, huber


@dataclass
class FrameData:
    cam_name: str
    height: int
    width: int
    dist_map: torch.Tensor
    lidar_points: torch.Tensor
    depth_map: Optional[torch.Tensor] = None


@dataclass
class RefineConfig:
    iters: int = 1000
    batch_frames: int = 4
    max_points: int = 30000
    lr: float = 1e-2
    edge_weight: float = 1.0
    depth_weight: float = 0.0
    pp_weight: float = 1.0
    aspect_weight: float = 0.5
    f_prior_weight: float = 1.0
    focal_scale_max: float = 4.0
    pp_range: float = 0.125
    pp_center_x: float = 0.5
    pp_center_y: float = 0.5
    depth_delta: float = 1.0


def _project_points(
    points: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2].clamp(min=1e-6)
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return torch.stack([u, v], dim=1), z


def _sample_map(dist_map: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    _, _, height, width = dist_map.shape
    gx = (uv[:, 0] / (width - 1)) * 2 - 1
    gy = (uv[:, 1] / (height - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        dist_map,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.view(-1)


def _valid_mask(uv: torch.Tensor, z: torch.Tensor, width: int, height: int) -> torch.Tensor:
    valid = z > 0
    valid &= uv[:, 0] >= 0
    valid &= uv[:, 0] <= (width - 1)
    valid &= uv[:, 1] >= 0
    valid &= uv[:, 1] <= (height - 1)
    return valid


def _depth_affine_loss(
    depth_map: torch.Tensor,
    uv: torch.Tensor,
    z: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    depth_samples = _sample_map(depth_map, uv)
    valid = depth_samples > 0
    if valid.sum() < 10:
        return torch.tensor(0.0, device=depth_map.device)
    d = depth_samples[valid]
    z_sel = z[valid]
    a = torch.stack([d, torch.ones_like(d)], dim=1)
    try:
        params = torch.linalg.lstsq(a, z_sel).solution
    except AttributeError:
        params = torch.pinverse(a) @ z_sel
    s, t = params[0], params[1]
    resid = s * d + t - z_sel
    return huber(resid, delta=delta).mean()


def refine_intrinsics(
    frames: List[FrameData],
    init_k: np.ndarray,
    config: RefineConfig,
    device: str = "cpu",
) -> Tuple[np.ndarray, List[float]]:
    if not frames:
        raise ValueError("No frames provided for refinement.")

    height = frames[0].height
    width = frames[0].width
    fx0 = float(init_k[0, 0])
    fy0 = float(init_k[1, 1])
    log_scale = float(np.log(max(config.focal_scale_max, 1.01)))

    params = torch.zeros(4, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=config.lr)

    losses: List[float] = []
    rng = np.random.default_rng(0)

    for _ in range(config.iters):
        optimizer.zero_grad()
        batch_idx = rng.choice(len(frames), size=min(config.batch_frames, len(frames)), replace=False)
        total_edge = torch.tensor(0.0, device=device)
        total_depth = torch.tensor(0.0, device=device)

        dx, dy, dcx, dcy = params
        fx = fx0 * torch.exp(log_scale * torch.tanh(dx))
        fy = fy0 * torch.exp(log_scale * torch.tanh(dy))
        cx0 = width * config.pp_center_x
        cy0 = height * config.pp_center_y
        cx = cx0 + (width * config.pp_range) * torch.tanh(dcx)
        cy = cy0 + (height * config.pp_range) * torch.tanh(dcy)

        for idx in batch_idx:
            frame = frames[idx]
            points = frame.lidar_points
            if points.shape[0] > config.max_points:
                choice = rng.choice(points.shape[0], size=config.max_points, replace=False)
                points = points[choice]
            uv, z = _project_points(points, fx, fy, cx, cy)
            valid = _valid_mask(uv, z, frame.width, frame.height)
            if valid.sum() < 50:
                continue
            uv = uv[valid]
            z = z[valid]
            dist = _sample_map(frame.dist_map, uv)
            total_edge = total_edge + charbonnier(dist).mean()
            if frame.depth_map is not None and config.depth_weight > 0:
                total_depth = total_depth + _depth_affine_loss(
                    frame.depth_map, uv, z, delta=config.depth_delta
                )

        l_pp = ((cx - width / 2.0) / width) ** 2 + ((cy - height / 2.0) / height) ** 2
        l_aspect = torch.log(fx / fy) ** 2
        l_focal = torch.log(fx / fx0) ** 2 + torch.log(fy / fy0) ** 2

        loss = (
            config.edge_weight * total_edge
            + config.depth_weight * total_depth
            + config.pp_weight * l_pp
            + config.aspect_weight * l_aspect
            + config.f_prior_weight * l_focal
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    with torch.no_grad():
        dx, dy, dcx, dcy = params
        fx = fx0 * torch.exp(log_scale * torch.tanh(dx))
        fy = fy0 * torch.exp(log_scale * torch.tanh(dy))
        cx0 = width * config.pp_center_x
        cy0 = height * config.pp_center_y
        cx = cx0 + (width * config.pp_range) * torch.tanh(dcx)
        cy = cy0 + (height * config.pp_range) * torch.tanh(dcy)

    k = np.array(
        [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return k, losses
