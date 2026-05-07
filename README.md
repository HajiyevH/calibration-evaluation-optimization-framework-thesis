# Calibri — Bundle Adjustment Framework for Automotive Camera Calibration

A C++/Python framework for targetless multi-camera calibration via bundle
adjustment. Supports pinhole and equidistant fisheye projection models,
joint optimisation of poses, landmarks, intrinsics, distortion and
extrinsics, robust losses with iterative outlier rejection, and a synthetic
data generator for controlled experiments.

This repository accompanies a Bachelor's thesis in Computer Science (BSc)
completed at **ELTE Faculty of Informatics, 2026**.

## Features

- **C++17 solver** built on Ceres, Eigen and OpenCV through vcpkg.
- **Two camera models** — Pinhole + radial-tangential distortion, and
  equidistant fisheye (k1–k4).
- **Joint optimisation** of poses, landmarks, intrinsics, distortion and
  extrinsics, with explicit gauge fixing (first pose + reference camera
  extrinsic).
- **Robust loss** — Huber/Cauchy/SoftL1 with adaptive outlier rejection.
- **Python framework** — synthetic data generation, experiment orchestration,
  ground-truth recovery metrics and trajectory ATE evaluation.
- **Streamlit UI** — interactive synthetic experiments and real-world
  calibration runs.
- **Test suite** — 37 GTest cases plus pytest coverage of the synthetic
  pipeline.

## Prerequisites

- Visual Studio 2022 (MSVC, C++17) on Windows, or any modern C++17 toolchain.
- CMake 3.20+
- vcpkg (with `VCPKG_ROOT` set, or the toolchain at
  `C:/vcpkg/scripts/buildsystems/vcpkg.cmake`).
- Python 3.10+

## Build

```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release --parallel
```

Artifacts:

- `build/calibri/app/Release/ba_cli.exe` — CLI entry point.
- `build/tests/Release/ba_tests.exe` — GTest binary.

The first configure step compiles Ceres, OpenCV and other native
dependencies through vcpkg, which can take 15–30 minutes on a fresh
machine. Subsequent builds reuse the vcpkg binary cache.

## Run Tests

```bash
build/tests/Release/ba_tests.exe
pytest tests/test_synth_pipeline.py
```

The end-to-end GTest cases solve nonlinear problems and may take a few
minutes.

## Python Environment

```bash
python -m venv .venv
.venv/Scripts/activate         # Windows
# source .venv/bin/activate    # Linux/macOS
pip install -e .
```

## Reproduce the Bundled Results

All commands assume the virtual environment is active and the C++ solver
has been built. Scripts take no CLI arguments — every parameter is defined
in source for determinism.

### 1. Synthetic Validation

Five experiments: baseline, intrinsics recovery, joint Pinhole, outlier
robustness, joint Fisheye.

```bash
python -m calibri.tools.analysis.run_synth_experiments
```

Outputs land in `thesis_results/tables/synth_*.csv`.

### 2. Monte Carlo Basin of Attraction

Solver robustness across 12 perturbation levels (0.1°–20°), 100 trials each.

```bash
python -m calibri.tools.analysis.run_monte_carlo
```

Outputs: `thesis_results/tables/monte_carlo_basin.csv`,
`thesis_results/figures/monte_carlo_basin.pdf`.

### 3. KITTI Real-World Sequences

Six sequences from the KITTI raw drives, optimising poses + landmarks +
extrinsics with intrinsics and distortion held fixed (stereo-rectified
imagery).

```bash
python -m calibri.tools.analysis.batch_kitti_run
```

Inputs: `thesis_results/batch_runs/kitti_*/problem.json` (bundled).
Outputs: `thesis_results/batch_runs/kitti_*/result.json` and
`thesis_results/tables/real_world_metrics.csv`.

### 4. Trajectory Plots and ATE

Aligns trajectories via SE(3) and reports the absolute trajectory error.

```bash
python -m calibri.tools.analysis.plot_trajectory
```

Outputs: `thesis_results/figures/trajectory_*.pdf`,
`thesis_results/tables/ate_results.csv`.

### 5. Per-Sequence Error Diagnostics

```bash
python calibri/tools/analysis/plot_errors.py thesis_results/batch_runs/kitti_0117/result.json --save errors.pdf
```

### 6. Streamlit Interface

```bash
pip install streamlit
streamlit run streamlit_app/Home.py
```

Pages:

- **1 Synthetic Experiments** — configure and run synthetic bundle
  adjustment via `ba_cli`.
- **2 Browse Runs** / **3 Compare Runs** — inspect runs stored in `runs/`.
- **4 Real-World Calibration** — drives the COLMAP-to-BA pipeline on
  recordings placed under `pre-triangulation/`. Requires `pycolmap`.

## Solver Configuration

All evaluation scripts use the same solver settings:

| Parameter | Value |
|-----------|-------|
| Loss function | Huber |
| Loss scale | 1.0 px |
| Max iterations | 200 |
| Outlier rejection passes | 3 |
| Outlier threshold | Adaptive (3 × median error) |
| Linear solver | `SPARSE_SCHUR` |
| Trust region | Levenberg–Marquardt |
| Function tolerance | 1e-6 |
| Gradient tolerance | 1e-10 |
| Parameter tolerance | 1e-8 |

## Project Layout

```
calibri/
  app/              CLI entry point (ba_cli.exe)
  configs/          Default problem.json templates
  engine/           C++ solver (cameras, cost functors, solver, I/O, contracts)
  framework/        Python experiment orchestration
  tools/            Python utilities, data generation, analysis, pipelines
streamlit_app/      Interactive web interface
tests/              GTest suite + pytest synthetic pipeline tests
thesis_results/
  batch_runs/       Per-KITTI-sequence problem.json + result.json
  tables/           CSV summary tables
  figures/          PDF figures
docs/               Architecture, theory, pipeline notes
```

## Data Contract

The solver reads `problem.json` and writes `result.json`. The schemas are
documented in `docs/ARCHITECTURE.md`. Every bundled result file can be
recreated from the matching `problem.json` by running the batch scripts
above.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
