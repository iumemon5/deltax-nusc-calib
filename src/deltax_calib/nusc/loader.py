"""
nuScenes Data Loader

Provides utilities for loading images, LiDAR points, and calibration data
from the nuScenes dataset. Intentionally avoids using nusc.get_sample_data()
which would leak GT intrinsics during Task 1 estimation.

Author: DeltaX Assignment Submission
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes


@dataclass
class SamplePair:
    cam_name: str
    cam_sd_token: str
    lidar_sd_token: str
    sample_token: str
    timestamp: int


def load_nuscenes(dataroot: str, version: str) -> NuScenes:
    return NuScenes(version=version, dataroot=dataroot, verbose=False)


def get_sample_data(nusc: NuScenes, sd_token: str) -> dict:
    return nusc.get("sample_data", sd_token)


def get_calibrated_sensor_extrinsics(
    nusc: NuScenes, calibrated_sensor_token: str
) -> Tuple[np.ndarray, np.ndarray]:
    rec = nusc.get("calibrated_sensor", calibrated_sensor_token)
    translation = np.asarray(rec["translation"], dtype=np.float32)
    rotation = np.asarray(rec["rotation"], dtype=np.float32)
    return translation, rotation


def get_ego_pose(
    nusc: NuScenes, ego_pose_token: str
) -> Tuple[np.ndarray, np.ndarray]:
    rec = nusc.get("ego_pose", ego_pose_token)
    translation = np.asarray(rec["translation"], dtype=np.float32)
    rotation = np.asarray(rec["rotation"], dtype=np.float32)
    return translation, rotation


def load_image(
    nusc: NuScenes,
    dataroot: str,
    sd_token: str,
    scale: float = 1.0,
    rgb: bool = True,
) -> np.ndarray:
    sd = get_sample_data(nusc, sd_token)
    path = os.path.join(dataroot, sd["filename"])
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if scale != 1.0:
        image = cv2.resize(
            image,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_lidar_points(
    nusc: NuScenes,
    dataroot: str,
    sd_token: str,
    max_points: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    sd = get_sample_data(nusc, sd_token)
    path = os.path.join(dataroot, sd["filename"])
    points = np.fromfile(path, dtype=np.float32)
    if points.size % 5 == 0:
        points = points.reshape(-1, 5)
    elif points.size % 4 == 0:
        points = points.reshape(-1, 4)
    else:
        raise ValueError(f"Unexpected lidar point format in {path}")
    points = points[:, :3]
    if max_points is not None and points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
    return points
