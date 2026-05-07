# Bundle Adjustment Pipeline: COLMAP to BA Conversion and Optimization

**Note:** This document focuses on the bundle adjustment conversion (Phase 4) and optimization (Phase 5) steps. The upstream COLMAP database preparation and triangulation (Phases 1-3) are covered by the scripts in `calibri/tools/pipelines/`.

## Overview

This pipeline converts a COLMAP reconstruction (with triangulated 3D points and camera poses) into the bundle adjustment problem format, then optimizes camera poses, landmarks, intrinsics, and extrinsics.

**Supported configurations:**
- Rectified pinhole rigs (e.g. KITTI raw drives, 4-camera stereo).
- Fisheye rigs with the equidistant model (k1–k4).

**Input:** COLMAP reconstruction directory (`*.bin` files) + COLMAP database (`*.db`)
**Output:** `problem.json` → `ba_cli.exe` → `result.json`

---

## PHASE 4: CONVERT TO BUNDLE ADJUSTMENT FORMAT

**Tool:** `colmap_to_problem.py`

**Command:**
```bash
python calibri/tools/converters/colmap_to_problem.py \
  --reconstruction reconstruction/odometry_rig_triangulated \
  --database database_rig.db \
  --output problem.json
```

### 4.1 Read COLMAP Binary Files

**Reads:**
- `cameras.bin` → camera intrinsics
- `images.bin` → camera poses (cam_from_world)
- `points3D.bin` → 3D landmarks
- `frames.bin` → rig frame poses (rig_from_world)
- `rigs.bin` → rig structure

**Extracts:**
- 2D observations: `(image_id, point2D_idx) → (u, v, point3D_id)`
- 3D landmarks: `point3D_id → (X, Y, Z)`
- Camera poses: `image_id → T_cam_from_world`
- Rig frame poses: `frame_id → T_rig_from_world`

### 4.2 Compute Vehicle Poses from Rig Frames

For GLOMAP rig mode:
- Rig frame pose = vehicle pose (rig center = vehicle center)
- `T_world_vehicle = inv(T_rig_from_world)`

Convert to BA format:
- Quaternion: `[w, x, y, z]` in JSON (constructor order)
- Translation: `[x, y, z]`

**Example:** 77 frames → 77 vehicle poses

### 4.3 Compute Camera Extrinsics (sensor_from_rig)

For each camera:

1. Get `cam_from_world` from `images.bin`
2. Get `rig_from_world` from `frames.bin` (for same timestamp)
3. Compute:
   ```
   cam_from_rig = cam_from_world * world_from_rig
                = cam_from_world * inv(rig_from_world)
   ```

This gives `T_camera_vehicle` (vehicle-to-camera extrinsic).

Convert to BA format:
- `CameraExtrinsics.T_vehicle_to_camera = cam_from_rig`
- Store as 4×4 matrix in row-major format

### 4.4 Build problem.json

**Structure:**
```json
{
  "camera_rig": {
    "reference_camera_id": 0,
    "intrinsics": [
      {
        "camera_id": 0,
        "camera_type": "PINHOLE_RADTAN",
        "width": 1242,
        "height": 375,
        "fx": 721.5377, "fy": 721.5377,
        "cx": 609.5593, "cy": 172.854,
        "k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0
      }
    ],
    "extrinsics": [
      {
        "camera_id": 0,
        "T_vehicle_to_camera": [
          R00, R01, R02, tx,
          R10, R11, R12, ty,
          R20, R21, R22, tz,
          0, 0, 0, 1
        ]
      }
    ]
  },

  "vehicle_poses": [
    {
      "timestamp": 0,
      "pose": {
        "q": [qw, qx, qy, qz],
        "t": [x, y, z]
      }
    }
  ],

  "landmarks": [
    {
      "landmark_id": 0,
      "position": [X, Y, Z]
    }
  ],

  "observations": [
    {
      "camera_id": 0,
      "timestamp": 0,
      "landmark_id": 0,
      "u": 234.56,
      "v": 123.45
    }
  ]
}
```

**For a typical KITTI sequence:**
- 4 cameras
- 77 vehicle poses
- 3,934 landmarks
- 13,320 observations
- File size: ~3 MB (compressed JSON)

---

## PHASE 5: BUNDLE ADJUSTMENT OPTIMIZATION

**Tool:** `ba_cli.exe`

**Command:**
```bash
ba_cli.exe problem.json result.json
```

### 5.1 Load Problem (json_codec.cc: loadProblemJson)

Reads `problem.json` and populates:
- `BundleAdjustmentProblem::camera_rig`
- `BundleAdjustmentProblem::vehicle_poses`
- `BundleAdjustmentProblem::landmarks`
- `BundleAdjustmentProblem::observations`
- `BundleAdjustmentParams` (optimization settings)

**Coordinate frame conversions:**
- JSON stores `[qw, qx, qy, qz]` (constructor order)
- Eigen uses `[qx, qy, qz, qw]` internally
- `contracts_bridge.cc` handles the conversion

### 5.2 Setup Optimization (bundle_adjuster.cc: solve)

#### Phase A: Convert to double precision
`problem.json` uses `float` → cast to `double` for Ceres

#### Phase B: Build Ceres problem

For each observation `(camera_id, timestamp, landmark_id, u, v)`:

1. Find vehicle pose `q_wv`, `t_wv`
2. Find landmark position `X_world`
3. Get camera intrinsics `(fx, fy, cx, cy, k1, k2, p1, p2)`
4. Get camera extrinsic `T_vc`

5. **Create ReprojectionCost residual:**
   ```
   residual = measured_uv - project(X_world, q_wv, t_wv, T_vc, intrinsics)
   ```

   **Projection chain:**
   ```
   X_vehicle = R_wv^T * (X_world - t_wv)
   X_camera = R_vc * X_vehicle + t_vc
   X_cam = [X, Y, Z]^T
   Apply distortion: (x', y') = distort(X/Z, Y/Z)
   u_proj = fx * x' + cx
   v_proj = fy * y' + cy
   ```

6. Add residual to Ceres problem with:
   - Parameter blocks: `[q_wv, t_wv, X_world]`
   - Loss function: `HuberLoss(1.0)`
   - Parameterization: `EigenQuaternionManifold` for `q_wv`

**Pruning:**
- Check if point is behind camera (Z < 0) → prune
- Check if initial error > 2 px → prune (likely outlier)

**For a typical KITTI sequence:**
- 13,320 observations → 11,694 residuals (1,626 pruned)

#### Phase C: Set gauge constraints
- Fix first vehicle pose (q and t) to resolve gauge freedom
- Fix reference camera extrinsic (if optimizing extrinsics)

#### Phase D: Configure Ceres solver
- Solver: `SPARSE_SCHUR` (exploits BA structure)
- Linear solver: `EIGEN_SPARSE`
- Ordering: Schur structure `(2,3,3)`
  - Group 0: Camera parameters (fixed in Iteration 1)
  - Group 1: Landmark positions (eliminated first)
  - Group 2: Vehicle poses (solved with reduced system)
- Max iterations: 100
- Convergence tolerances:
  - `function_tolerance`: 1e-6
  - `gradient_tolerance`: 1e-10
  - `parameter_tolerance`: 1e-8

### 5.3 Run Ceres Solver

**Levenberg-Marquardt algorithm:**

```
Initialize λ (damping parameter)
For each iteration:
  1. Compute Jacobian J and residual r
  2. Form normal equations: (J^T J + λI) Δx = -J^T r
  3. Solve using Schur complement:
     • Eliminate landmark parameters first
     • Solve reduced system for poses
     • Back-substitute for landmarks
  4. Compute cost reduction ratio ρ
  5. Update λ based on ρ (increase if bad, decrease if good)
  6. Accept or reject step
```

**For a typical KITTI sequence:**
- 101 iterations
- 96 successful steps, 5 unsuccessful
- Termination: NO_CONVERGENCE (max iterations)
- But still succeeded: 80.4% cost reduction

**Typical iteration (example):**
```
iter=10: cost=963.6, change=7.15, |grad|=93.1, |step|=53.2
         tr_ratio=0.725 (good), increase trust region
```

### 5.4 Compute Statistics and Write Result

For each observation:
- Reproject using optimized parameters
- Compute residual: `err = sqrt((u-u_proj)² + (v-v_proj)²)`
- Mark as inlier if `err < threshold`

**Aggregate statistics:**
- Mean, median, P95 reprojection error
- RMSE
- Inlier count and ratio
- Per-frame statistics
- Per-camera statistics

**Write result.json:**
```json
{
  "solver_status": {
    "success": true,
    "termination_type": "NO_CONVERGENCE",
    "num_iterations": 101,
    "initial_cost": 4703.8,
    "final_cost": 919.7,
    "solver_time_ms": 612.7
  },
  "camera_rig": { ... },
  "vehicle_poses": [ ... ],
  "landmarks": [ ... ],
  "observations": [
    {
      "camera_id": 0,
      "timestamp": 0,
      "landmark_id": 0,
      "u_measured": 234.56,
      "v_measured": 123.45,
      "u_reprojected": 234.58,
      "v_reprojected": 123.47,
      "error_px": 0.028,
      "weight": 0.998,
      "inlier": true
    }
  ],
  "statistics": {
    "mean_error": 0.3,
    "rmse_error": 0.4,
    "inlier_count": 11623,
    "total_count": 11694
  }
}
```

**For a typical KITTI sequence:** `result.json` is 4.1 MB

---

## Fisheye Rigs

The same Phase 4–5 workflow applies to fisheye recordings; the camera entries
in `problem.json` simply use `model: "Fisheye"` and the equidistant
distortion coefficients `k1, k2, k3, k4`. Fisheye datasets are typically
optimised with all parameter blocks free (poses, landmarks, intrinsics,
distortion and extrinsics) because the imagery is not pre-rectified.

---

## SUMMARY: Bundle Adjustment Data Flow

```
COLMAP Reconstruction (*.bin files)
  ├─ cameras.bin                      (intrinsics)
  ├─ images.bin                       (poses + 2D obs)
  ├─ points3D.bin                     (triangulated landmarks)
  ├─ frames.bin                       (rig frame poses)
  └─ rigs.bin                         (rig structure)
       │
       │ colmap_to_problem.py
       ↓
BA Problem (problem.json)
  ├─ camera_rig                       (intrinsics + extrinsics)
  ├─ vehicle_poses                    (odometry-based initial poses)
  ├─ landmarks                        (triangulated 3D points)
  └─ observations                     (2D-3D correspondences)
       │
       │ ba_cli.exe
       ↓
BA Result (result.json)
  ├─ solver_status                    (convergence info)
  ├─ camera_rig                       (optimized intrinsics + extrinsics)
  ├─ vehicle_poses                    (optimized poses)
  ├─ landmarks                        (optimized 3D points)
  ├─ observations                     (with residuals)
  └─ statistics                       (error metrics)
```

---

## Key Coordinate Frames

### Frames:
- **World Frame**: Fixed global coordinate system (odometry reference)
- **Vehicle Frame**: Moving frame attached to vehicle center
- **Camera Frame**: Moving frame attached to each camera

### Transforms:
- `T_wv`: World-to-vehicle (from `odometry.csv`)
- `T_vc`: Vehicle-to-camera (from `calibs_prior/*.json`)
- `T_wc`: World-to-camera (composed: `T_vc * T_vw`)

### Projection Chain:
```
X_world → [T_wv^-1] → X_vehicle → [T_vc] → X_camera → [K, distort] → (u, v)
```

Where:
- `T_wv^-1` transforms from world to vehicle coordinates
- `T_vc` transforms from vehicle to camera coordinates
- `K` is the camera intrinsic matrix
- `distort` applies radial-tangential distortion (k1, k2, p1, p2, k3)
- `(u, v)` are pixel coordinates

---

## See Also

- `docs/ARCHITECTURE.md` — Deep dive into the bundle adjustment solver mathematics and code architecture.
- `docs/THEORY.md` — Pose estimation and BA theoretical background.
- `calibri/tools/pipelines/` — COLMAP database preparation and triangulation scripts (Phases 1–3).
