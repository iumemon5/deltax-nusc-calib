"""
nuScenes Sample Indexing

Builds an index of (camera_sample_data, lidar_sample_data) pairs for
efficient iteration. This is used to sample frames for calibration.

Author: DeltaX Assignment Submission
"""

from typing import Dict, Iterable, List, Optional

import numpy as np

from .loader import SamplePair


# All camera names in nuScenes
CAM_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def build_cam_lidar_index(
    nusc,
    camera_names: Iterable[str] = CAM_NAMES,
    require_lidar: bool = True,
    require_all_cams: bool = True,
) -> List[SamplePair]:
    pairs: List[SamplePair] = []
    cam_names = list(camera_names)
    for sample in nusc.sample:
        data = sample["data"]
        if require_lidar and "LIDAR_TOP" not in data:
            continue
        if require_all_cams and any(cam not in data for cam in cam_names):
            continue
        cams_present = cam_names if require_all_cams else [c for c in cam_names if c in data]
        lidar_sd_token = data.get("LIDAR_TOP", "")
        for cam_name in cams_present:
            pairs.append(
                SamplePair(
                    cam_name=cam_name,
                    cam_sd_token=data[cam_name],
                    lidar_sd_token=lidar_sd_token,
                    sample_token=sample["token"],
                    timestamp=sample["timestamp"],
                )
            )
    return pairs


def group_pairs_by_camera(pairs: Iterable[SamplePair]) -> Dict[str, List[SamplePair]]:
    grouped: Dict[str, List[SamplePair]] = {}
    for pair in pairs:
        grouped.setdefault(pair.cam_name, []).append(pair)
    for cam_name in grouped:
        grouped[cam_name].sort(key=lambda p: p.timestamp)
    return grouped


def sample_pairs(
    pairs: List[SamplePair],
    n_frames: int,
    stride: int = 1,
) -> List[SamplePair]:
    if stride < 1:
        stride = 1
    pairs_sorted = sorted(pairs, key=lambda p: p.timestamp)
    pairs_strided = pairs_sorted[::stride]
    if n_frames <= 0 or n_frames >= len(pairs_strided):
        return pairs_strided
    idx = np.linspace(0, len(pairs_strided) - 1, num=n_frames)
    idx = np.unique(np.round(idx).astype(int))
    return [pairs_strided[i] for i in idx]


def find_sample_with_all_cams(
    nusc,
    camera_names: Iterable[str] = CAM_NAMES,
) -> Optional[str]:
    cam_names = list(camera_names)
    for sample in nusc.sample:
        data = sample["data"]
        if "LIDAR_TOP" not in data:
            continue
        if any(cam not in data for cam in cam_names):
            continue
        return sample["token"]
    return None
