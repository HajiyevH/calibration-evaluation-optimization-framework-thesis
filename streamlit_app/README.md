# Calibri Streamlit App

Web UI for running bundle adjustment experiments and processing COLMAP data.

## Quick Start

```bash
# From repo root
cd streamlit_app
../.venv/Scripts/python.exe -m streamlit run Home.py
```

The app will open in your browser at http://localhost:8501

## Pages

### 1. Run Experiment (Synthetic Data)

Generate synthetic multi-camera data and run bundle adjustment experiments.

**Features:**
- Configure scene (number of cameras, poses, landmarks)
- Add observation noise
- Configure solver parameters (optimize poses, landmarks, intrinsics, etc.)
- Generate error plots and frame overlays
- Results saved to `runs/EXPERIMENT_NAME_TIMESTAMP/`

**Use for:**
- Testing solver configurations
- Validating algorithm changes
- Generating thesis figures on controlled data

### 2. Browse Runs

View completed synthetic experiment results.

**Features:**
- Select from completed runs
- View summary metrics (success, iterations, errors)
- Inspect solver logs
- View error distribution plots
- Browse frame overlay images

### 3. Compare Runs

Side-by-side comparison of two experiment runs.

**Features:**
- Compare solver configurations
- Compare error metrics
- Analyze performance differences

### 4. COLMAP Pipeline (Real Data) 🆕

Process COLMAP reconstructions through the custom BA solver.

**Features:**
- Load COLMAP reconstruction (cameras.bin, images.bin, points3D.bin)
- Automatic pose composition for GLOMAP rig mode
- Configure BA solver parameters
- Generate error plots and overlays
- **COLMAP baseline comparison** - Shows improvement over COLMAP's BA
- Results saved to `results/colmap_pipeline_TIMESTAMP/`

**Inputs:**
- **Reconstruction directory** - Path to COLMAP reconstruction (e.g., `results-different/ba/reconstruction/3_rig_glomap/0`)
- **Database file** - Path to COLMAP database.db (needed for rig config)
- **Use rig mode** - Enable for multi-camera rigs (GLOMAP rig reconstructions)

**Outputs:**
- `problem.json` - Converted BA problem
- `result.json` - BA solver output
- `errors.png` - Error distribution plot
- `overlay_frame_*.png` - Frame overlay images
- `colmap_comparison.json` - Comparison with COLMAP baseline (if pycolmap available)

**Use for:**
- Processing real COLMAP data through custom BA solver
- Comparing against COLMAP's built-in BA
- Generating thesis results on real data
- Testing solver on real-world challenges

## Directory Structure

```
streamlit_app/
├── Home.py                    # Landing page
├── ui_backend.py              # Backend logic (config, subprocess management)
├── ui_utils.py                # UI helpers (badges, metrics, etc.)
├── pages/
│   ├── 1_Run_Experiment.py    # Synthetic data experiments
│   ├── 2_Browse_Runs.py       # Browse synthetic results
│   ├── 3_Compare_Runs.py      # Compare synthetic runs
│   └── 4_Real_World_Calibration.py  # Real-world calibration pipeline
└── README.md                  # This file
```

## Quick Access: Common Reconstruction Paths

### GLOMAP no-BA (our test case)
```
Reconstruction: results-different/glomap_no_ba/reconstruction/3_rig_glomap/0
Database:       results-different/glomap_no_ba/reconstruction/database_rig.db
Use rig:        ✓
```

### GLOMAP with BA (baseline comparison)
```
Reconstruction: results-different/ba/reconstruction/3_rig_glomap/0
Database:       results-different/ba/reconstruction/database_rig.db
Use rig:        ✓
```

### COLMAP incremental (single-camera)
```
Reconstruction: results-different/colmap-glomap/reconstruction/1_separate_cams_glomap/0
Database:       results-different/colmap-glomap/reconstruction/database.db
Use rig:        ✗
```

## Example Workflow: Process COLMAP Data

1. **Launch app:** `streamlit run Home.py`
2. **Navigate to "COLMAP Pipeline"** (sidebar)
3. **Configure inputs:**
   - Reconstruction: `results-different/ba/reconstruction/3_rig_glomap/0`
   - Database: `results-different/ba/reconstruction/database_rig.db`
   - Use rig: ✓
   - Max iterations: 200
4. **Click "Run Pipeline"**
5. **View results:**
   - Summary metrics (median error, iterations)
   - COLMAP baseline comparison (improvement %)
   - Error distribution plot
   - Frame overlays

## Troubleshooting

### Pipeline fails with "pycolmap not found"
- Install pycolmap: `.venv/Scripts/pip install pycolmap`
- COLMAP baseline comparison will be skipped without pycolmap

### Reconstruction path not found
- Verify the path exists
- Use forward slashes or raw strings
- Check from repo root

### BA diverges (infinite cost)
- Check coordinate frame conventions (GLOMAP rig mode has known issues)
- Try single-camera COLMAP incremental first
- Verify camera model is supported (OPENCV or OPENCV_FISHEYE)

### Plots not generating
- Check matplotlib is installed: `.venv/Scripts/pip install matplotlib`
- Use `--skip-plots` flag to bypass

## Dependencies

```
streamlit>=1.55.0
pyyaml>=5.4.0
numpy>=1.21.0
matplotlib>=3.4.0  # For plots
pycolmap>=4.0.0    # Optional, for COLMAP baseline comparison
```

Install: `.venv/Scripts/pip install -r calibri/tools/requirements.txt`

## Notes

- **Synthetic experiments** (Pages 1-3) use the `calibri.framework` package and `run_experiment.py`
- **COLMAP pipeline** (Page 4) directly calls `calibri/tools/pipelines/colmap_pipeline.py`
- Results are NOT stored in `runs/` directory but in timestamped `results/colmap_pipeline_*/` directories
- COLMAP baseline comparison requires pycolmap (optional but recommended)
