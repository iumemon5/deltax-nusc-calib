#!/usr/bin/env python3
"""
Task 1: Intrinsics Evaluation

Compares estimated intrinsics against ground truth from nuScenes.
This is the ONLY place where GT intrinsics are read (for evaluation).

Usage:
    python scripts/task1_eval_intrinsics.py \
        --dataroot ../nuScenes_mini \
        --est_csv deliverables/task1/intrinsics_est.csv \
        --out_csv deliverables/task1/task1_abs_error.csv

Author: DeltaX Assignment Submission
"""

import argparse
import csv
import os
import sys

from nuscenes.nuscenes import NuScenes

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def _read_estimates(path: str) -> list:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _write_csv(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["cam_name", "cam_token", "fx_err", "fy_err", "cx_err", "cy_err"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1: evaluate intrinsics vs GT.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--est_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    estimates = _read_estimates(args.est_csv)
    results = []

    for row in estimates:
        cam_token = row.get("cam_token")
        if not cam_token:
            raise ValueError("cam_token missing from estimates CSV.")
        calib = nusc.get("calibrated_sensor", cam_token)
        k_gt = calib["camera_intrinsic"]
        fx_gt, fy_gt = k_gt[0][0], k_gt[1][1]
        cx_gt, cy_gt = k_gt[0][2], k_gt[1][2]
        fx = float(row["fx"])
        fy = float(row["fy"])
        cx = float(row["cx"])
        cy = float(row["cy"])
        results.append(
            {
                "cam_name": row["cam_name"],
                "cam_token": cam_token,
                "fx_err": f"{abs(fx - fx_gt):.6f}",
                "fy_err": f"{abs(fy - fy_gt):.6f}",
                "cx_err": f"{abs(cx - cx_gt):.6f}",
                "cy_err": f"{abs(cy - cy_gt):.6f}",
            }
        )

    _write_csv(args.out_csv, results)


if __name__ == "__main__":
    main()
