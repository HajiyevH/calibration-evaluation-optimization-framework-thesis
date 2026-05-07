# Bundle Adjustment Tools

Python utilities for the Thesis Bundle Adjustment project.

## Directory Structure

- `utils.py` — Shared utilities (quaternions, projections, I/O, statistics)
- `synth_gen.py` — Synthetic dataset generator
- `converters/` — Data format converters (COLMAP, sv_utils)
- `analysis/` — Visualization and comparison tools
- `pipelines/` — End-to-end workflow runners
- `diagnostic/` — Debugging and testing utilities

## Workflows

### Real Data: Pre-triangulation Pipeline

Two-step pipeline for real automotive recordings:

```bash
# Step 1: Triangulate from odometry + COLMAP features
.venv/Scripts/python.exe calibri/tools/pipelines/triangulate_from_odometry.py \
  --database     pre-triangulation/<recording>/database_rig.db \
  --images       pre-triangulation/<recording>/images \
  --odometry     pre-triangulation/<recording>/<odometry>.csv \
  --calibs-prior pre-triangulation/<recording>/calibs_prior \
  --output       pre-triangulation/<recording>/reconstruction/odometry_rig_triangulated \
  -v

# Step 2: Run BA pipeline
.venv/Scripts/python.exe calibri/tools/pipelines/colmap_pipeline.py \
  --reconstruction pre-triangulation/<recording>/reconstruction/odometry_rig_triangulated \
  --database       pre-triangulation/<recording>/database_rig.db \
  --output-dir     results/<recording>_ba \
  --use-rig \
  --optimize-intrinsics \
  --optimize-distortion \
  --optimize-extrinsics \
  --max-iterations 200 \
  --loss-type Huber \
  --loss-scale 1.0 \
  -v
```

### Synthetic Data: Experiment Framework

```bash
# Generate 4-camera rig synthetic data
python calibri/tools/synth_gen.py --num-cameras 4 --num-poses 25 --seed 42

# Run via experiment framework
python scripts/run_experiment.py run experiments/<config>.yaml
```

## Individual Tools

### Converters

- **`colmap_to_problem.py`** — COLMAP reconstruction to problem.json (supports multi-camera rigs)
  ```bash
  python calibri/tools/converters/colmap_to_problem.py \
      --reconstruction <path> \
      --database <path> \
      --output data/problem.json \
      --use-rig
  ```

- **`sv_utils_to_ba.py`** — sv_utils calibrations to camera JSON
  ```bash
  python calibri/tools/converters/sv_utils_to_ba.py \
      --calibs-dir <path_to_sv_utils_calibs>  \
      --output cameras.json
  ```

- **`merge_ground_truth_extrinsics.py`** — Merge GT extrinsics into problem.json
  ```bash
  python calibri/tools/converters/merge_ground_truth_extrinsics.py \
      --problem data/problem.json \
      --calibs-dir <path_to_sv_utils_calibs> \
      --output data/problem_with_gt.json
  ```

### Analysis

- **`plot_errors.py`** — Multi-panel error visualization
  ```bash
  python calibri/tools/analysis/plot_errors.py \
      data/result.json \
      --save figures/errors.png
  ```

- **`overlay_frame.py`** — Measured vs reprojected keypoint overlays
  ```bash
  python calibri/tools/analysis/overlay_frame.py \
      data/result.json \
      data/problem.json \
      --frame 0 \
      --save figures/overlay_0.png
  ```

### Diagnostic

- **`test_reprojection.py`** — Basic reprojection validation
  ```bash
  python calibri/tools/diagnostic/test_reprojection.py data/problem.json
  ```

## Coordinate Frame Conventions

**CRITICAL:** This codebase uses the following conventions:

- **Vehicle poses:** `T_wv` (vehicle-to-world transform)
  - `X_vehicle = R_wv^-1 * (X_world - t_wv)`
- **Camera extrinsics:** `T_vc` (vehicle-to-camera transform)
  - `X_camera = R_vc * X_vehicle + t_vc`
- **Quaternions in JSON:** `[w, x, y, z]` (constructor order)
- **Quaternions in scipy:** `[x, y, z, w]` (internal storage)

See `utils.py` for conversion functions.

## Camera Models Supported

- **Pinhole + Radial-Tangential** (5 distortion params: k1, k2, p1, p2, k3)
- **OpenCV Fisheye** (4 distortion params: k1, k2, k3, k4)

See `calibri/engine/cameras/` for C++ implementations.
