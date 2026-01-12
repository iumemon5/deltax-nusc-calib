"""
Coordinate Transform Utilities for nuScenes

Handles the various coordinate frame transformations in nuScenes:
- sensor (camera/lidar) <-> ego vehicle
- ego vehicle <-> global world

Important nuScenes conventions (took me a while to figure out!):
- calibrated_sensor stores the transform FROM sensor TO ego (sensor pose in ego frame)
- ego_pose stores the transform FROM ego TO global (ego pose in world frame)

So to go from lidar to camera:
  lidar -> ego (lidar timestamp) -> global -> ego (cam timestamp) -> camera

Author: DeltaX Assignment Submission
"""

from typing import Tuple

import numpy as np

from .loader import get_calibrated_sensor_extrinsics, get_ego_pose, get_sample_data

try:
    from pyquaternion import Quaternion

    def _quat_to_rot(quat: np.ndarray) -> np.ndarray:
        return Quaternion(quat).rotation_matrix

except ImportError:  # pragma: no cover

    def _quat_to_rot(quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )


def transform_matrix(
    translation: np.ndarray, rotation: np.ndarray, inverse: bool = False
) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float32)
    translation = np.asarray(translation, dtype=np.float32)
    rot = _quat_to_rot(rotation)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rot
    transform[:3, 3] = translation
    if not inverse:
        return transform
    rot_inv = rot.T
    transform_inv = np.eye(4, dtype=np.float32)
    transform_inv[:3, :3] = rot_inv
    transform_inv[:3, 3] = -rot_inv @ translation
    return transform_inv


def ego_to_sensor(nusc, sd_token: str) -> np.ndarray:
    """Transform from ego to sensor coordinates."""
    sd = get_sample_data(nusc, sd_token)
    trans, rot = get_calibrated_sensor_extrinsics(nusc, sd["calibrated_sensor_token"])
    # calibration stores sensor->ego (position of sensor in ego frame)
    # so we need to invert to get ego->sensor
    return transform_matrix(trans, rot, inverse=True)


def sensor_to_ego(nusc, sd_token: str) -> np.ndarray:
    """Transform from sensor to ego coordinates."""
    sd = get_sample_data(nusc, sd_token)
    trans, rot = get_calibrated_sensor_extrinsics(nusc, sd["calibrated_sensor_token"])
    # calibration stores sensor->ego directly
    return transform_matrix(trans, rot, inverse=False)


def global_to_ego(nusc, sd_token: str) -> np.ndarray:
    """Transform from global to ego coordinates."""
    sd = get_sample_data(nusc, sd_token)
    trans, rot = get_ego_pose(nusc, sd["ego_pose_token"])
    # ego_pose stores ego->global (position of ego in global frame)
    # so we need to invert to get global->ego
    return transform_matrix(trans, rot, inverse=True)


def ego_to_global(nusc, sd_token: str) -> np.ndarray:
    """Transform from ego to global coordinates."""
    sd = get_sample_data(nusc, sd_token)
    trans, rot = get_ego_pose(nusc, sd["ego_pose_token"])
    # ego_pose stores ego->global directly
    return transform_matrix(trans, rot, inverse=False)


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    points_t = (transform @ points_h.T).T
    return points_t[:, :3]


def _sensor_to_ego(nusc, sd_token: str) -> np.ndarray:
    return sensor_to_ego(nusc, sd_token)


def _ego_to_global(nusc, sd_token: str) -> np.ndarray:
    return ego_to_global(nusc, sd_token)


def lidar_to_camera_transform(nusc, lidar_sd_token: str, cam_sd_token: str) -> np.ndarray:
    # lidar -> ego (at lidar timestamp)
    t_lidar_to_ego = sensor_to_ego(nusc, lidar_sd_token)
    # ego -> global (at lidar timestamp)
    t_ego_to_global_lidar = ego_to_global(nusc, lidar_sd_token)
    # lidar -> global
    t_lidar_to_global = t_ego_to_global_lidar @ t_lidar_to_ego

    # global -> ego (at camera timestamp)
    t_global_to_ego_cam = global_to_ego(nusc, cam_sd_token)
    # ego -> cam
    t_ego_to_cam = ego_to_sensor(nusc, cam_sd_token)

    # cam <- global <- lidar
    return t_ego_to_cam @ t_global_to_ego_cam @ t_lidar_to_global


def camera_to_ego_lidar_transform(nusc, cam_sd_token: str, lidar_sd_token: str) -> np.ndarray:
    # cam -> ego
    t_cam_to_ego = sensor_to_ego(nusc, cam_sd_token)
    # ego -> global (cam timestamp)
    t_ego_to_global_cam = ego_to_global(nusc, cam_sd_token)
    # global -> ego (lidar timestamp)
    t_global_to_ego_lidar = global_to_ego(nusc, lidar_sd_token)
    # ego -> lidar
    t_ego_to_lidar = ego_to_sensor(nusc, lidar_sd_token)
    return t_ego_to_lidar @ t_global_to_ego_lidar @ t_ego_to_global_cam @ t_cam_to_ego


def get_image_size(nusc, sd_token: str) -> Tuple[int, int]:
    sd = get_sample_data(nusc, sd_token)
    return int(sd["height"]), int(sd["width"])
