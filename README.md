# DeltaX nuScenes Camera Calibration & Point Cloud Generation

This project solves two computer vision tasks on the nuScenes-mini dataset:
- **Task 1**: Estimate camera intrinsics without using ground truth calibration
- **Task 2**: Generate colored pseudo point clouds using monocular depth estimation

## Project Structure

```
deltax-nusc-calib/
├── configs/                   # Configuration files
│   ├── task1.yaml             # Task 1 settings
│   ├── task2.yaml             # Task 2 with GT intrinsics
│   └── task2_estimated.yaml   # Task 2 with estimated intrinsics
├── scripts/                   # Executable scripts
│   ├── task1_estimate_intrinsics.py
│   ├── task1_eval_intrinsics.py
│   └── task2_generate_pointcloud.py
├── src/deltax_calib/          # Source code
│   ├── calib/                 # Calibration module
│   │   ├── estimator.py       # Main estimation logic
│   │   ├── refine.py          # Optimization loop
│   │   └── losses.py          # Loss functions
│   ├── depth/                 # Depth estimation
│   │   └── infer.py           # Depth Anything V2 inference
│   ├── nusc/                  # nuScenes utilities
│   │   ├── loader.py          # Data loading
│   │   ├── index.py           # Sample indexing
│   │   └── transforms.py      # Coordinate transforms
│   └── pointcloud/            # Point cloud utilities
│       ├── fuse.py            # Multi-camera fusion
│       └── export.py          # PLY export
├── deliverables/              # Output files
│   ├── task1/
│   │   ├── intrinsics_est.csv     # Estimated intrinsics
│   │   └── task1_abs_error.csv    # Error vs ground truth
│   └── task2/
│       ├── <timestamp>.ply            # Point cloud (GT intrinsics)
│       └── <timestamp>_estimated.ply  # Point cloud (est. intrinsics)
└── requirements.txt           # Python dependencies
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for faster inference)
- ~4GB GPU memory for Depth Anything V2

## Installation

### 1. Clone and setup environment

```bash
# Create conda environment
conda create -n nuscenes python=3.8 -y
conda activate nuscenes

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

### 2. Download nuScenes-mini dataset

Download from [nuScenes website](https://www.nuscenes.org/download) and extract:

```
parent_folder/
├── nuScenes_mini/          # Dataset folder
│   ├── samples/
│   ├── sweeps/
│   └── v1.0-mini/
└── deltax-nusc-calib/      # This repository
```

The `dataroot` in configs points to `../nuScenes_mini` relative to the repo.

## Usage

### Task 1: Camera Intrinsics Estimation

Estimates intrinsic parameters (fx, fy, cx, cy) for all 6 cameras using LiDAR-image edge alignment.

```bash
cd deltax-nusc-calib

# Run estimation
python scripts/task1_estimate_intrinsics.py --config configs/task1.yaml

# Evaluate against ground truth
python scripts/task1_eval_intrinsics.py \
    --dataroot ../nuScenes_mini \
    --est_csv deliverables/task1/intrinsics_est.csv \
    --out_csv deliverables/task1/task1_abs_error.csv
```

**Output:**
- `deliverables/task1/intrinsics_est.csv` - Estimated intrinsics per camera
- `deliverables/task1/task1_abs_error.csv` - Absolute errors vs ground truth

### Task 2: Pseudo Point Cloud Generation

Generates dense colored point clouds by:
1. Running Depth Anything V2 on each camera image
2. Scaling depth using LiDAR alignment
3. Back-projecting to 3D and fusing all cameras

```bash
# Using ground truth intrinsics
python scripts/task2_generate_pointcloud.py --config configs/task2.yaml

# Using estimated intrinsics from Task 1
python scripts/task2_generate_pointcloud.py --config configs/task2_estimated.yaml
```

**Output:**
- `deliverables/task2/<timestamp>.ply` - Point cloud (GT intrinsics)
- `deliverables/task2/<timestamp>_estimated.ply` - Point cloud (estimated intrinsics)

### Visualizing Point Clouds

```bash
# Using Open3D
python -c "import open3d as o3d; \
    pcd = o3d.io.read_point_cloud('deliverables/task2/1532402927647951.ply'); \
    o3d.visualization.draw_geometries([pcd])"
```

Or use MeshLab, or any PLY viewer.

## Configuration Options

### Task 1 (`configs/task1.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_frames_per_cam` | 120 | Number of frames to sample per camera |
| `stride` | 2 | Frame sampling stride |
| `scale` | 1.0 | Image scale factor (0.5 = half resolution) |
| `iters` | 1000 | Optimization iterations |
| `lr` | 0.01 | Learning rate |
| `pp_weight` | 40.0 | Principal point regularization weight |
| `device` | cuda | Device (cuda/cpu) |

### Task 2 (`configs/task2.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `depth_method` | depth_anything | Depth model (depth_anything/midas) |
| `pixel_stride` | 2 | Pixel sampling stride |
| `voxel` | 0.05 | Voxel size for downsampling (meters) |
| `max_depth` | 50.0 | Maximum depth to keep (meters) |
| `use_gt_intrinsics` | true | Use GT or estimated intrinsics |
| `scale_depth_with_lidar` | true | Align depth scale with LiDAR |

## Approach Summary

### Task 1: Intrinsics Estimation

**Method:** LiDAR-to-image edge alignment optimization

1. Extract edges using Canny edge detector
2. Compute distance transform (distance to nearest edge)
3. Project LiDAR points to image using current intrinsics estimate
4. Minimize distance of projected points to edges (Charbonnier loss)
5. Regularize principal point near image center and enforce square pixels

### Task 2: Point Cloud Generation

**Method:** Monocular depth estimation + LiDAR scale alignment

1. Run Depth Anything V2 on each camera image (relative disparity)
2. Fit scale factor using sparse LiDAR projections: `depth = scale / disparity`
3. Back-project pixels to 3D using camera intrinsics
4. Transform to common LiDAR reference frame
5. Filter outliers (sky, height bounds, statistical)
6. Voxel downsample and export as PLY

## Results

### Task 1: Intrinsics Accuracy

| Camera | fx_err | fy_err | cx_err | cy_err |
|--------|--------|--------|--------|--------|
| CAM_FRONT | 9.9 | 21.7 | 16.3 | 0.7 |
| CAM_FRONT_LEFT | 18.1 | 19.0 | 24.4 | 11.1 |
| CAM_FRONT_RIGHT | 6.6 | 14.3 | 8.1 | 4.5 |
| CAM_BACK | 12.3 | 17.7 | 30.3 | 9.1 |
| CAM_BACK_LEFT | 1.7 | 4.4 | 8.2 | 1.9 |
| CAM_BACK_RIGHT | 7.2 | 10.6 | 7.1 | 10.4 |

Errors are in pixels (image resolution: 1600×900).

### Task 2: Point Cloud

- **Points:** ~850,000 (after filtering)
- **Coverage:** 360° around vehicle
- **Depth range:** 0.5m - 50m
- **Format:** PLY with RGB colors

## Troubleshooting

### CUDA out of memory
- Reduce `batch_frames` in task1.yaml
- Use `scale: 0.5` to process at half resolution
- Set `device: cpu` (slower but works)

### nuScenes not found
- Check `dataroot` path in config files
- Ensure nuScenes_mini is extracted correctly

### Depth Anything download fails
- Model downloads automatically from HuggingFace
- Check internet connection
- Alternatively, pre-download to `~/.cache/huggingface/`

## License

This project uses:
- nuScenes dataset (CC BY-NC-SA 4.0)
- Depth Anything V2 (Apache 2.0)
- Open source libraries (see requirements.txt)

## Author
MEMON IRFANULLAH
DeltaX AI Engineer Assignment Submission
