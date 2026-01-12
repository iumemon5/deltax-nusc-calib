"""
Point Cloud Export Utilities

Exports point clouds in PLY and PCD formats. PLY is simpler and more
widely supported (works in MeshLab, CloudCompare, etc). PCD is the
PCL native format.

Author: DeltaX Assignment Submission
"""

import os
from typing import Tuple

import numpy as np


def _ensure_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def write_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    _ensure_dir(path)
    if colors is None:
        colors = np.zeros_like(points)
    colors = colors.astype(np.uint8)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            handle.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def _pack_rgb_float(colors: np.ndarray) -> np.ndarray:
    colors = colors.astype(np.uint32)
    rgb = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
    return rgb.view(np.float32)


def write_pcd(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    _ensure_dir(path)
    if colors is None:
        colors = np.zeros_like(points)
    colors = colors.astype(np.uint8)
    rgb = _pack_rgb_float(colors)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# .PCD v0.7 - Point Cloud Data file format\n")
        handle.write("VERSION 0.7\n")
        handle.write("FIELDS x y z rgb\n")
        handle.write("SIZE 4 4 4 4\n")
        handle.write("TYPE F F F F\n")
        handle.write("COUNT 1 1 1 1\n")
        handle.write(f"WIDTH {points.shape[0]}\n")
        handle.write("HEIGHT 1\n")
        handle.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        handle.write(f"POINTS {points.shape[0]}\n")
        handle.write("DATA ascii\n")
        for (x, y, z), c in zip(points, rgb):
            handle.write(f"{x:.6f} {y:.6f} {z:.6f} {float(c)}\n")


def export_pointcloud(
    path: str, points: np.ndarray, colors: np.ndarray, fmt: str = "ply"
) -> None:
    fmt = fmt.lower()
    if fmt == "ply":
        write_ply(path, points, colors)
    elif fmt == "pcd":
        write_pcd(path, points, colors)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
