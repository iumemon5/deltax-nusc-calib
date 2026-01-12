"""
Camera Intrinsics Estimator

This module estimates camera intrinsics (fx, fy, cx, cy) without using ground truth
calibration. The approach uses LiDAR-to-image edge alignment: we project LiDAR points
onto the image and optimize intrinsics to minimize distance to detected edges.

The key insight is that 3D structure (from LiDAR) should align with 2D edges in 
the image when using correct intrinsics. This is a form of self-supervised calibration.

Author: DeltaX Assignment Submission
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ..nusc.index import CAM_NAMES, build_cam_lidar_index, group_pairs_by_camera, sample_pairs
from ..nusc.loader import load_image, load_lidar_points, load_nuscenes, get_sample_data
from ..nusc.transforms import apply_transform, get_image_size, lidar_to_camera_transform
from .losses import edge_distance_transform
from .refine import FrameData, RefineConfig, refine_intrinsics


@dataclass
class EstimatorConfig:
    n_frames_per_cam: int = 120
    stride: int = 2
    scale: float = 0.5
    init_mode: str = "heuristic"
    init_fov_deg: float = 90.0
    init_fov_map: Optional[Dict[str, float]] = None
    init_model: Optional[str] = None
    max_lidar_points: int = 30000
    batch_frames: int = 4
    iters: int = 1000
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
    save_overlays: bool = False
    overlay_dir: str = "outputs/task1/overlays"


def _init_intrinsics_heuristic(width: int, height: int, fov_deg: float) -> np.ndarray:
    fov_rad = np.deg2rad(fov_deg)
    fx = 0.5 * width / np.tan(0.5 * fov_rad)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _load_torchscript_model(model_path: str, device: str):
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def _predict_intrinsics_model(model, image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    param = next(model.parameters(), None)
    if param is not None:
        tensor = tensor.to(param.device)
    with torch.no_grad():
        pred = model(tensor).squeeze().cpu().numpy()
    if pred.shape[0] != 4:
        raise ValueError("Model output must be 4 values: fx, fy, cx, cy.")
    fx, fy, cx, cy = pred
    height, width = image.shape[:2]
    if max(fx, fy) <= 10.0:
        fx *= width
        fy *= height
        cx *= width
        cy *= height
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _scale_intrinsics(k: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return k
    k_scaled = k.copy()
    k_scaled[0, 0] *= scale
    k_scaled[1, 1] *= scale
    k_scaled[0, 2] *= scale
    k_scaled[1, 2] *= scale
    return k_scaled


def _unscale_intrinsics(k: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return k
    k_full = k.copy()
    k_full[0, 0] /= scale
    k_full[1, 1] /= scale
    k_full[0, 2] /= scale
    k_full[1, 2] /= scale
    return k_full


def estimate_intrinsics(
    dataroot: str,
    version: str,
    config: EstimatorConfig,
    device: str = "cpu",
) -> List[Dict[str, str]]:
    nusc = load_nuscenes(dataroot, version)
    pairs = build_cam_lidar_index(nusc, CAM_NAMES, require_lidar=True, require_all_cams=True)
    grouped = group_pairs_by_camera(pairs)

    results: List[Dict[str, str]] = []
    model = None
    if config.init_mode == "torchscript":
        if not config.init_model:
            raise ValueError("init_model is required when init_mode is torchscript.")
        model = _load_torchscript_model(config.init_model, device)

    for cam_name, cam_pairs in grouped.items():
        selected = sample_pairs(cam_pairs, config.n_frames_per_cam, config.stride)
        if not selected:
            continue

        frames: List[FrameData] = []
        height_full, width_full = get_image_size(nusc, selected[0].cam_sd_token)

        for pair in tqdm(selected, desc=f"Frames {cam_name}"):
            image = load_image(nusc, dataroot, pair.cam_sd_token, scale=config.scale, rgb=True)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            dist = edge_distance_transform(gray)
            dist_tensor = torch.from_numpy(dist).float().unsqueeze(0).unsqueeze(0).to(device)

            points = load_lidar_points(
                nusc, dataroot, pair.lidar_sd_token, max_points=None
            )
            t_cam_from_lidar = lidar_to_camera_transform(nusc, pair.lidar_sd_token, pair.cam_sd_token)
            points_cam = apply_transform(points, t_cam_from_lidar)
            points_tensor = torch.from_numpy(points_cam.astype(np.float32)).to(device)

            height, width = gray.shape[:2]
            frames.append(
                FrameData(
                    cam_name=cam_name,
                    height=height,
                    width=width,
                    dist_map=dist_tensor,
                    lidar_points=points_tensor,
                    depth_map=None,
                )
            )

        if config.init_mode == "torchscript":
            image = load_image(nusc, dataroot, selected[0].cam_sd_token, scale=1.0, rgb=True)
            init_k = _predict_intrinsics_model(model, image)
        else:
            fov = config.init_fov_deg
            if config.init_fov_map and cam_name in config.init_fov_map:
                fov = config.init_fov_map[cam_name]
            init_k = _init_intrinsics_heuristic(width_full, height_full, fov)

        init_k_scaled = _scale_intrinsics(init_k, config.scale)
        refine_config = RefineConfig(
            iters=config.iters,
            batch_frames=config.batch_frames,
            max_points=config.max_lidar_points,
            lr=config.lr,
            edge_weight=config.edge_weight,
            depth_weight=config.depth_weight,
            pp_weight=config.pp_weight,
            aspect_weight=config.aspect_weight,
            f_prior_weight=config.f_prior_weight,
            focal_scale_max=config.focal_scale_max,
            pp_range=config.pp_range,
            pp_center_x=config.pp_center_x,
            pp_center_y=config.pp_center_y,
        )
        refined_k_scaled, _ = refine_intrinsics(frames, init_k_scaled, refine_config, device=device)
        refined_k = _unscale_intrinsics(refined_k_scaled, config.scale)

        cam_sd = get_sample_data(nusc, selected[0].cam_sd_token)
        cam_token = cam_sd["calibrated_sensor_token"]
        result = {
            "cam_name": cam_name,
            "cam_token": cam_token,
            "fx": f"{refined_k[0, 0]:.6f}",
            "fy": f"{refined_k[1, 1]:.6f}",
            "cx": f"{refined_k[0, 2]:.6f}",
            "cy": f"{refined_k[1, 2]:.6f}",
            "intrinsic_matrix": " ".join([f"{v:.6f}" for v in refined_k.reshape(-1)]),
        }
        results.append(result)

        if config.save_overlays:
            _save_overlay(
                nusc,
                dataroot,
                selected[0].cam_sd_token,
                selected[0].lidar_sd_token,
                refined_k,
                cam_name,
                config.overlay_dir,
            )

    return results


def _save_overlay(
    nusc,
    dataroot: str,
    cam_sd_token: str,
    lidar_sd_token: str,
    k: np.ndarray,
    cam_name: str,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    image = load_image(nusc, dataroot, cam_sd_token, scale=1.0, rgb=True)
    height, width = image.shape[:2]
    points = load_lidar_points(nusc, dataroot, lidar_sd_token, max_points=20000)
    t_cam_from_lidar = lidar_to_camera_transform(nusc, lidar_sd_token, cam_sd_token)
    points_cam = apply_transform(points, t_cam_from_lidar)
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    valid_z = z > 0
    u = k[0, 0] * (x[valid_z] / z[valid_z]) + k[0, 2]
    v = k[1, 1] * (y[valid_z] / z[valid_z]) + k[1, 2]
    valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid_uv].astype(int)
    v = v[valid_uv].astype(int)
    overlay = image.copy()
    for px, py in zip(u, v):
        cv2.circle(overlay, (int(px), int(py)), 1, (0, 255, 0), -1)
    out_path = os.path.join(out_dir, f"{cam_name}_overlay.png")
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
