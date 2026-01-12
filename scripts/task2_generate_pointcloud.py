#!/usr/bin/env python3
"""
Task 2: Pseudo Point Cloud Generation

Generates a colored 3D point cloud by fusing monocular depth estimates
from all 6 cameras. Uses Depth Anything V2 for depth estimation and
LiDAR for scale calibration.

Usage:
    python scripts/task2_generate_pointcloud.py --config configs/task2.yaml

Author: DeltaX Assignment Submission
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from deltax_calib.depth.infer import DepthInference
from deltax_calib.nusc.index import CAM_NAMES, find_sample_with_all_cams
from deltax_calib.nusc.loader import load_image, load_lidar_points, load_nuscenes
from deltax_calib.nusc.transforms import apply_transform, lidar_to_camera_transform
from deltax_calib.pointcloud.export import export_pointcloud
from deltax_calib.pointcloud.fuse import backproject_depth, fit_depth_affine, read_intrinsics_csv, voxel_downsample


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _merge_config(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def _str2bool(value):
    if value is None or isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in ("true", "1", "yes", "y"):
        return True
    if val in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2: generate fused pseudo point cloud.")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument("--dataroot", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--intrinsics_csv", default=None)
    parser.add_argument("--sample_token", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--format", default=None)
    parser.add_argument("--pixel_stride", type=int, default=None)
    parser.add_argument("--voxel", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--depth_method", default=None)
    parser.add_argument("--depth_model", default=None)
    parser.add_argument("--scale_depth_with_lidar", action="store_true", default=None)
    parser.add_argument("--max_depth", type=float, default=None)
    parser.add_argument("--min_depth", type=float, default=None)
    parser.add_argument("--use_gt_intrinsics", nargs="?", const=True, default=None, type=_str2bool)
    args = parser.parse_args()

    config = {}
    if args.config:
        config = _load_yaml(args.config)

    overrides = vars(args)
    overrides.pop("config", None)
    merged = _merge_config(config, overrides)

    device = merged.get("device")
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataroot = merged.get("dataroot", "../nuScenes_mini")
    version = merged.get("version", "v1.0-mini")
    intrinsics_csv = merged.get("intrinsics_csv", "outputs/task1/intrinsics_est.csv")
    out_dir = merged.get("out_dir", "deliverables")
    fmt = merged.get("format", "ply")
    pixel_stride = int(merged.get("pixel_stride", 3))
    voxel = float(merged.get("voxel", 0.1))
    depth_method = merged.get("depth_method", "stub")
    depth_model = merged.get("depth_model")
    scale_depth_with_lidar = bool(merged.get("scale_depth_with_lidar", False))
    max_depth = float(merged.get("max_depth", 80.0))
    min_depth = float(merged.get("min_depth", 0.5))
    use_gt_intrinsics = merged.get("use_gt_intrinsics")
    if use_gt_intrinsics is None:
        use_gt_intrinsics = False
    else:
        use_gt_intrinsics = bool(use_gt_intrinsics)

    nusc = load_nuscenes(dataroot, version)
    sample_token = merged.get("sample_token")
    if not sample_token:
        sample_token = find_sample_with_all_cams(nusc)
        if not sample_token:
            raise ValueError("No sample with all cameras + LIDAR_TOP found.")

    sample = nusc.get("sample", sample_token)
    lidar_sd_token = sample["data"]["LIDAR_TOP"]

    intrinsics = read_intrinsics_csv(intrinsics_csv)
    depth_infer = DepthInference(method=depth_method, model_path=depth_model, device=device)

    all_points = []
    all_colors = []
    lidar_points = None

    for cam_name in CAM_NAMES:
        cam_sd_token = sample["data"][cam_name]
        if use_gt_intrinsics:
            cam_sd = nusc.get("sample_data", cam_sd_token)
            calib = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
            k = np.array(calib["camera_intrinsic"], dtype=np.float32)
        else:
            if cam_name not in intrinsics:
                raise ValueError(f"Missing intrinsics for {cam_name}")
            k = intrinsics[cam_name]

        image = load_image(nusc, dataroot, cam_sd_token, scale=1.0, rgb=True)
        disparity = depth_infer.infer(image)

        if scale_depth_with_lidar:
            if lidar_points is None:
                lidar_points = load_lidar_points(nusc, dataroot, lidar_sd_token, max_points=50000)
            t_cam_from_lidar = lidar_to_camera_transform(nusc, lidar_sd_token, cam_sd_token)
            points_cam = apply_transform(lidar_points, t_cam_from_lidar)
            scale, shift = fit_depth_affine(disparity, points_cam, k)
            # Convert disparity to metric depth: depth = scale / disparity
            eps = 1e-6
            depth = scale / (disparity + eps)
            depth = np.clip(depth, min_depth, max_depth)
            
            # Filter sky: set depth to 0 where disparity is very low (sky regions)
            sky_threshold = np.percentile(disparity, 5)  # Bottom 5% disparity = sky
            depth[disparity < sky_threshold] = 0
            
            print(f"{cam_name}: scale={scale:.1f}, depth range={depth[depth>0].min():.1f}-{depth[depth>0].max():.1f}m")
        else:
            # No LiDAR scaling - use disparity as-is (will produce relative depth)
            eps = 1e-6
            depth = 1.0 / (disparity + eps)

        points_cam, colors = backproject_depth(
            image, depth, k, stride=pixel_stride, max_depth=max_depth, min_depth=min_depth
        )
        
        # Filter points at extreme viewing angles (unreliable depth)
        # Camera looks down -Z axis, so points with large X or Y relative to Z are at grazing angles
        if len(points_cam) > 0:
            viewing_angle = np.abs(points_cam[:, :2]) / (points_cam[:, 2:3] + 1e-6)  # |X/Z|, |Y/Z|
            max_angle = np.tan(np.radians(75))  # 75 degrees from optical axis
            angle_mask = np.all(viewing_angle < max_angle, axis=1)
            points_cam = points_cam[angle_mask]
            colors = colors[angle_mask]
        
        # Use lidar frame as reference: invert lidar->camera transform
        t_cam_from_lidar = lidar_to_camera_transform(nusc, lidar_sd_token, cam_sd_token)
        t_lidar_from_cam = np.linalg.inv(t_cam_from_lidar)
        points_ego = apply_transform(points_cam, t_lidar_from_cam)

        all_points.append(points_ego)
        all_colors.append(colors)

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    # Filter by height - remove points too high (likely sky artifacts)
    # In LiDAR frame, Z is up. Typical vehicle height ~2m, buildings ~30m
    max_height = 25.0  # meters above ground
    min_height = -3.0  # below ground level (for slopes)
    height_mask = (points[:, 2] > min_height) & (points[:, 2] < max_height)
    points = points[height_mask]
    colors = colors[height_mask]
    
    # Statistical outlier removal - helps with stitching artifacts
    # Compute local point density using nearest neighbors
    if len(points) > 1000:
        from scipy.spatial import KDTree
        tree = KDTree(points)
        k_neighbors = 20
        distances, _ = tree.query(points, k=k_neighbors)
        mean_distances = distances[:, 1:].mean(axis=1)  # Skip self (distance=0)
        
        # Filter points with unusually large distances to neighbors (isolated/noisy)
        threshold = mean_distances.mean() + 2.5 * mean_distances.std()
        outlier_mask = mean_distances < threshold
        points = points[outlier_mask]
        colors = colors[outlier_mask]
        print(f"Removed {(~outlier_mask).sum()} outliers")

    points, colors = voxel_downsample(points, colors, voxel)
    print(f"Final point cloud: {len(points)} points")

    timestamp = str(sample["timestamp"])
    os.makedirs(out_dir, exist_ok=True)
    # Add suffix to distinguish estimated vs GT intrinsics
    suffix = "_estimated" if not use_gt_intrinsics else ""
    out_path = os.path.join(out_dir, f"{timestamp}{suffix}.{fmt}")
    export_pointcloud(out_path, points, colors, fmt=fmt)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
