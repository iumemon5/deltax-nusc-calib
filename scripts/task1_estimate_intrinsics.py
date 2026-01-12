#!/usr/bin/env python3
"""
Task 1: Camera Intrinsics Estimation

Estimates fx, fy, cx, cy for all 6 nuScenes cameras using LiDAR-image
edge alignment. Does NOT use ground truth intrinsics during estimation.

Usage:
    python scripts/task1_estimate_intrinsics.py --config configs/task1.yaml

Author: DeltaX Assignment Submission
"""

import argparse
import csv
import os
import sys

import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from deltax_calib.calib.estimator import EstimatorConfig, estimate_intrinsics


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _merge_config(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def _write_csv(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["cam_name", "cam_token", "fx", "fy", "cx", "cy", "intrinsic_matrix"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1: estimate camera intrinsics.")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument("--dataroot", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--n_frames_per_cam", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--init_mode", default=None)
    parser.add_argument("--init_fov_deg", type=float, default=None)
    parser.add_argument("--init_model", default=None)
    parser.add_argument("--max_lidar_points", type=int, default=None)
    parser.add_argument("--batch_frames", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--edge_weight", type=float, default=None)
    parser.add_argument("--depth_weight", type=float, default=None)
    parser.add_argument("--pp_weight", type=float, default=None)
    parser.add_argument("--aspect_weight", type=float, default=None)
    parser.add_argument("--f_prior_weight", type=float, default=None)
    parser.add_argument("--focal_scale_max", type=float, default=None)
    parser.add_argument("--pp_range", type=float, default=None)
    parser.add_argument("--pp_center_x", type=float, default=None)
    parser.add_argument("--pp_center_y", type=float, default=None)
    parser.add_argument("--save_overlays", action="store_true")
    parser.add_argument("--device", default=None)
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

    est_config = EstimatorConfig(
        n_frames_per_cam=int(merged.get("n_frames_per_cam", 120)),
        stride=int(merged.get("stride", 2)),
        scale=float(merged.get("scale", 0.5)),
        init_mode=str(merged.get("init_mode", "heuristic")),
        init_fov_deg=float(merged.get("init_fov_deg", 90.0)),
        init_fov_map=merged.get("init_fov_map"),
        init_model=merged.get("init_model"),
        max_lidar_points=int(merged.get("max_lidar_points", 30000)),
        batch_frames=int(merged.get("batch_frames", 4)),
        iters=int(merged.get("iters", 1000)),
        lr=float(merged.get("lr", 1e-2)),
        edge_weight=float(merged.get("edge_weight", 1.0)),
        depth_weight=float(merged.get("depth_weight", 0.0)),
        pp_weight=float(merged.get("pp_weight", 1.0)),
        aspect_weight=float(merged.get("aspect_weight", 0.5)),
        f_prior_weight=float(merged.get("f_prior_weight", 1.0)),
        focal_scale_max=float(merged.get("focal_scale_max", 4.0)),
        pp_range=float(merged.get("pp_range", 0.125)),
        pp_center_x=float(merged.get("pp_center_x", 0.5)),
        pp_center_y=float(merged.get("pp_center_y", 0.5)),
        save_overlays=bool(merged.get("save_overlays", False)),
    )

    dataroot = merged.get("dataroot", "../nuScenes_mini")
    version = merged.get("version", "v1.0-mini")
    out_csv = merged.get("out_csv", "outputs/task1/intrinsics_est.csv")

    rows = estimate_intrinsics(dataroot, version, est_config, device=device)
    _write_csv(out_csv, rows)


if __name__ == "__main__":
    main()
