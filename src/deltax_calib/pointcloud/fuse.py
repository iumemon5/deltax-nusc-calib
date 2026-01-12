"""
Point Cloud Fusion Module

Handles multi-camera depth fusion for pseudo point cloud generation:
1. Read estimated/GT intrinsics
2. Scale monocular depth using sparse LiDAR measurements  
3. Back-project depth maps to 3D points
4. Voxel downsampling for efficiency

The scale fitting uses a robust median approach - we compute scale = median(z_lidar * disparity)
which is more robust to outliers than least-squares fitting.

Author: DeltaX Assignment Submission
"""

import csv
from typing import Dict, Tuple

import numpy as np


def read_intrinsics_csv(path: str) -> Dict[str, np.ndarray]:
    intrinsics: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fx = float(row["fx"])
            fy = float(row["fy"])
            cx = float(row["cx"])
            cy = float(row["cy"])
            k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
            intrinsics[row["cam_name"]] = k
    return intrinsics


def _bilinear_sample(depth_map: np.ndarray, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height, width = depth_map.shape[:2]
    u = uv[:, 0]
    v = uv[:, 1]
    valid = (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
    u = u[valid]
    v = v[valid]
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1
    du = (u - u0).astype(np.float32)
    dv = (v - v0).astype(np.float32)
    d00 = depth_map[v0, u0]
    d10 = depth_map[v0, u1]
    d01 = depth_map[v1, u0]
    d11 = depth_map[v1, u1]
    d0 = d00 * (1 - du) + d10 * du
    d1 = d01 * (1 - du) + d11 * du
    d = d0 * (1 - dv) + d1 * dv
    return d, valid


def fit_depth_affine(disparity_map: np.ndarray, points_cam: np.ndarray, k: np.ndarray) -> Tuple[float, float]:
    """Fit scale to convert disparity to metric depth using robust median fitting.
    
    MiDaS outputs relative disparity d. We fit: depth = scale / disparity
    Using median ratio for robustness against outliers.
    """
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    # Only use points at reasonable depth range for fitting
    valid = (z > 2.0) & (z < 60.0)
    if valid.sum() < 50:
        valid = z > 1.0
    
    u = k[0, 0] * (x[valid] / z[valid]) + k[0, 2]
    v = k[1, 1] * (y[valid] / z[valid]) + k[1, 2]
    uv = np.stack([u, v], axis=1)
    disp_samples, mask = _bilinear_sample(disparity_map, uv)
    
    if disp_samples.size < 50:
        # Fallback
        med_disp = np.median(disparity_map)
        return float(med_disp * 15.0), 0.0
    
    z_samples = z[valid][mask]
    
    # Filter out sky/invalid regions (very low disparity often means sky)
    disp_thresh = np.percentile(disp_samples, 10)
    good = disp_samples > disp_thresh
    if good.sum() < 30:
        good = np.ones_like(disp_samples, dtype=bool)
    
    disp_samples = disp_samples[good]
    z_samples = z_samples[good]
    
    # Compute scale using median of (z * disparity) - robust to outliers
    # depth = scale / disparity => scale = depth * disparity
    scale_samples = z_samples * disp_samples
    scale = float(np.median(scale_samples))
    
    return scale, 0.0  # No shift - simpler model


def backproject_depth(
    image: np.ndarray,
    depth_map: np.ndarray,
    k: np.ndarray,
    stride: int = 3,
    max_depth: float = 80.0,
    min_depth: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = depth_map.shape[:2]
    ys = np.arange(0, height, stride, dtype=np.int32)
    xs = np.arange(0, width, stride, dtype=np.int32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    z = depth_map[grid_y, grid_x].astype(np.float32)
    valid = (z > min_depth) & (z < max_depth)
    z = z[valid]
    u = grid_x[valid].astype(np.float32)
    v = grid_y[valid].astype(np.float32)
    x = (u - k[0, 2]) * z / k[0, 0]
    y = (v - k[1, 2]) * z / k[1, 1]
    points = np.stack([x, y, z], axis=1)
    colors = image[grid_y[valid], grid_x[valid]]
    return points, colors


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, voxel: float
) -> Tuple[np.ndarray, np.ndarray]:
    if voxel <= 0:
        return points, colors
    coords = np.floor(points / voxel).astype(np.int32)
    _, idx = np.unique(coords, axis=0, return_index=True)
    return points[idx], colors[idx]
