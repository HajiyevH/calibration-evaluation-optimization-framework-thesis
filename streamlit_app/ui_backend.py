"""Backend adapter between Streamlit UI and calibri.framework.

All framework imports and subprocess management go through this module.
"""

import json
import sys
import time
import queue
import threading
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RUNS_DIR = REPO_ROOT / "runs"

# Platform-aware venv Python detection
_VENV_DIR = REPO_ROOT / ".venv"
if sys.platform == "win32":
    PYTHON_EXE = _VENV_DIR / "Scripts" / "python.exe"
else:
    PYTHON_EXE = _VENV_DIR / "bin" / "python"

RUN_EXPERIMENT_SCRIPT = REPO_ROOT / "scripts" / "run_experiment.py"

from calibri.framework.config import (
    AnalysisConfig,
    DataConfig,
    ExperimentConfig,
    PerturbationConfig,
    RobustConfig,
    SolverConfig,
    SolverFlags,
    config_to_dict,
    load_config,
    validate_config,
)

# ---------------------------------------------------------------------------
# Experiment presets
# ---------------------------------------------------------------------------

EXPERIMENT_PRESETS = OrderedDict([
    ("Quick sanity check", {
        "description": "Fast single-camera baseline test",
        "badge": "Recommended",
        "data": DataConfig(
            scene_type="corridor", trajectory_type="arc",
            camera_model="Pinhole", num_cameras=1,
            num_poses=10, num_landmarks=300, noise_sigma=0.5,
        ),
        "flags": SolverFlags(opt_poses=True, opt_landmarks=True),
    }),
    ("Noise sweep", {
        "description": "Test solver robustness under increasing noise",
        "badge": "Recommended",
        "data": DataConfig(
            scene_type="corridor", trajectory_type="arc",
            camera_model="Pinhole", num_cameras=1,
            num_poses=15, num_landmarks=500, noise_sigma=1.0,
        ),
        "flags": SolverFlags(opt_poses=True, opt_landmarks=True),
    }),
    ("Outlier robustness", {
        "description": "Test robust loss under outlier contamination",
        "badge": "Recommended",
        "data": DataConfig(
            scene_type="corridor", trajectory_type="arc",
            camera_model="Pinhole", num_cameras=1,
            num_poses=15, num_landmarks=500, noise_sigma=0.5,
            outlier_ratio=0.1,
        ),
        "flags": SolverFlags(opt_poses=True, opt_landmarks=True),
    }),
    ("Initialization perturbation", {
        "description": "Test convergence basin under perturbed initial guesses",
        "badge": "Advanced",
        "data": DataConfig(
            scene_type="corridor", trajectory_type="arc",
            camera_model="Pinhole", num_cameras=1,
            num_poses=15, num_landmarks=500, noise_sigma=0.5,
            perturbation=PerturbationConfig(
                pose_rot_sigma_deg=3.0, pose_trans_sigma_m=0.1,
                landmark_sigma_m=0.2,
            ),
        ),
        "flags": SolverFlags(opt_poses=True, opt_landmarks=True),
    }),
    ("4-camera recovery", {
        "description": "Multi-camera extrinsic calibration with surround scene",
        "badge": "Recommended",
        "data": DataConfig(
            scene_type="surround", trajectory_type="arc",
            camera_model="Pinhole", num_cameras=4,
            num_poses=25, num_landmarks=800, noise_sigma=0.5,
            perturbation=PerturbationConfig(
                extrinsic_rot_sigma_deg=2.0, extrinsic_trans_sigma_m=0.05,
            ),
        ),
        "flags": SolverFlags(
            opt_poses=True, opt_landmarks=True, opt_extrinsics=True,
        ),
    }),
    ("Custom", {
        "description": "Configure all parameters manually",
        "badge": None,
        "data": None,
        "flags": None,
    }),
])

# ---------------------------------------------------------------------------
# Comparison presets
# ---------------------------------------------------------------------------

COMPARISON_PRESETS = OrderedDict([
    ("Noise sweep", {
        "x_field": "data.noise_sigma",
        "y_metric": "reproj_rmse",
        "group_field": None,
    }),
    ("Outlier sweep", {
        "x_field": "data.outlier_ratio",
        "y_metric": "reproj_rmse",
        "group_field": None,
    }),
    ("Perturbation sweep", {
        "x_field": "perturbation_mag",
        "y_metric": "extr_rot_err_deg",
        "group_field": None,
    }),
    ("Trajectory comparison", {
        "x_field": "data.trajectory_type",
        "y_metric": "reproj_rmse",
        "group_field": None,
    }),
    ("Custom", None),
])

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def config_to_yaml_str(config: ExperimentConfig) -> str:
    """Serialize an ExperimentConfig to a YAML string for preview."""
    return yaml.dump(config_to_dict(config), default_flow_style=False, sort_keys=False)


def validate_config_safe(config: ExperimentConfig) -> Optional[str]:
    """Validate config, returning error string or None if valid."""
    try:
        validate_config(config)
        return None
    except (ValueError, TypeError) as exc:
        return str(exc)


def build_config_from_ui(
    *,
    name: str,
    description: str,
    # Data generation
    scene_type: str = "corridor",
    trajectory_type: str = "arc",
    camera_model: str = "Pinhole",
    num_cameras: int = 1,
    num_poses: int = 15,
    num_landmarks: int = 500,
    noise_sigma: float = 0.5,
    outlier_ratio: float = 0.0,
    image_width: int = 1280,
    image_height: int = 800,
    seed: int = 42,
    # Corridor geometry
    corridor_length_m: float = 30.0,
    corridor_width_m: float = 8.0,
    corridor_wall_height_m: float = 4.0,
    # Perturbation
    pose_rot_sigma_deg: float = 1.0,
    pose_trans_sigma_m: float = 0.02,
    landmark_sigma_m: float = 0.05,
    extrinsic_rot_sigma_deg: float = 2.0,
    extrinsic_trans_sigma_m: float = 0.05,
    intrinsic_sigma_px: float = 0.0,
    distortion_sigma: float = 0.0,
    # Solver flags
    opt_poses: bool = True,
    opt_landmarks: bool = True,
    opt_intrinsics: bool = False,
    opt_distortion: bool = False,
    opt_extrinsics: bool = False,
    # Robust
    robust_type: str = "Huber",
    robust_scale: float = 1.0,
    timeout: int = 600,
    # Analysis
    plot_errors: bool = True,
    overlay_frames: Optional[List[Union[int, str]]] = None,
    print_summary: bool = True,
    evaluate_gt: bool = True,
) -> ExperimentConfig:
    """Assemble an ExperimentConfig from individual widget values."""
    if overlay_frames is None:
        overlay_frames = [0, "mid", "last"]

    return ExperimentConfig(
        name=name,
        description=description,
        data=DataConfig(
            scene_type=scene_type,
            trajectory_type=trajectory_type,
            camera_model=camera_model,
            num_cameras=num_cameras,
            num_poses=num_poses,
            num_landmarks=num_landmarks,
            noise_sigma=noise_sigma,
            outlier_ratio=outlier_ratio,
            image_width=image_width,
            image_height=image_height,
            seed=seed,
            corridor_length_m=corridor_length_m,
            corridor_width_m=corridor_width_m,
            corridor_wall_height_m=corridor_wall_height_m,
            perturbation=PerturbationConfig(
                pose_rot_sigma_deg=pose_rot_sigma_deg,
                pose_trans_sigma_m=pose_trans_sigma_m,
                landmark_sigma_m=landmark_sigma_m,
                extrinsic_rot_sigma_deg=extrinsic_rot_sigma_deg,
                extrinsic_trans_sigma_m=extrinsic_trans_sigma_m,
                intrinsic_sigma_px=intrinsic_sigma_px,
                distortion_sigma=distortion_sigma,
            ),
        ),
        solver=SolverConfig(
            flags=SolverFlags(
                opt_poses=opt_poses,
                opt_landmarks=opt_landmarks,
                opt_intrinsics=opt_intrinsics,
                opt_distortion=opt_distortion,
                opt_extrinsics=opt_extrinsics,
            ),
            robust=RobustConfig(type=robust_type, scale=robust_scale),
            timeout=timeout,
        ),
        analysis=AnalysisConfig(
            plot_errors=plot_errors,
            overlay_frames=overlay_frames,
            print_summary=print_summary,
            evaluate_gt=evaluate_gt,
        ),
    )


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------

def _reader_thread(pipe, q: queue.Queue):
    """Daemon thread that reads lines from a pipe into a queue.

    The ValueError catch handles the expected case where the pipe is
    closed while readline() is blocked (e.g. process terminated).
    """
    try:
        for line in iter(pipe.readline, ""):
            q.put(line)
    except ValueError:
        # Pipe closed by the subprocess exiting -- expected, not an error.
        pass
    finally:
        pipe.close()


def start_experiment_subprocess(config: ExperimentConfig) -> dict:
    """Launch run_experiment.py as a non-blocking subprocess.

    Returns a tracking dict stored in st.session_state.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    temp_config = RUNS_DIR / f"_pending_{config.name}_{timestamp}.yaml"
    temp_config.write_text(
        yaml.dump(config_to_dict(config), default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    proc = subprocess.Popen(
        [str(PYTHON_EXE), str(RUN_EXPERIMENT_SCRIPT), "run", str(temp_config)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(REPO_ROOT),
    )

    q: queue.Queue = queue.Queue()
    reader = threading.Thread(target=_reader_thread, args=(proc.stdout, q), daemon=True)
    reader.start()

    return {
        "proc": proc,
        "stdout_queue": q,
        "stdout_lines": [],
        "start_time": time.time(),
        "config_name": config.name,
        "temp_config": str(temp_config),
    }


def drain_stdout(tracking: dict) -> str:
    """Drain accumulated stdout from the subprocess queue.

    Appends new lines to tracking["stdout_lines"] and returns the
    full accumulated output as a single string.
    """
    q: queue.Queue = tracking["stdout_queue"]
    while True:
        try:
            line = q.get_nowait()
            tracking["stdout_lines"].append(line)
        except queue.Empty:
            break
    return "".join(tracking["stdout_lines"])


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def list_runs() -> List[dict]:
    """Discover completed runs in runs/, newest first (by modification time).

    Returns list of dicts with keys: name, path, has_summary.
    """
    if not RUNS_DIR.is_dir():
        return []

    entries = []
    for entry in RUNS_DIR.iterdir():
        if not entry.is_dir() or entry.name.startswith("_pending"):
            continue
        entries.append(entry)

    # Sort by modification time, newest first
    entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    runs = []
    for entry in entries:
        summary_path = entry / "summary.yaml"
        runs.append({
            "name": entry.name,
            "path": entry,
            "has_summary": summary_path.exists(),
        })
    return runs


def load_run_summary(run_path: Path) -> Optional[dict]:
    """Load summary.yaml from a run directory."""
    summary_path = run_path / "summary.yaml"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (yaml.YAMLError, OSError):
        return None


def load_run_config(run_path: Path) -> Optional[dict]:
    """Load the frozen config.yaml from a run directory."""
    config_path = run_path / "config.yaml"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (yaml.YAMLError, OSError):
        return None


def load_run_log(run_path: Path, name: str = "cpp_stdout.log") -> str:
    """Load a log file from a run directory."""
    log_path = run_path / name
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="replace")


def get_run_images(run_path: Path) -> Dict[str, Path]:
    """Discover PNG images in a run directory.

    Returns dict mapping filename (without extension) to full path.
    """
    images = {}
    for png in sorted(run_path.glob("*.png")):
        images[png.stem] = png
    return images


def load_run_gt_evaluation(run_path: Path) -> Optional[dict]:
    """Load gt_evaluation.json from a run directory."""
    eval_path = run_path / "gt_evaluation.json"
    if not eval_path.exists():
        return None
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Run table builder
# ---------------------------------------------------------------------------

def build_run_table(runs: List[dict]) -> List[dict]:
    """Build table rows from run list, loading config + summary for each.

    Returns list of dicts with columns suitable for a DataFrame.
    """
    rows = []
    for run in runs:
        run_path = run["path"]
        config = load_run_config(run_path)
        summary = load_run_summary(run_path)
        gt_eval = load_run_gt_evaluation(run_path)

        row = {
            "name": run["name"],
            "path": str(run_path),
        }

        if config:
            data = config.get("data", {})
            row["scene"] = data.get("scene_type", "")
            row["trajectory"] = data.get("trajectory_type", "")
            row["model"] = data.get("camera_model", "")
            row["cameras"] = data.get("num_cameras", 1)
            row["poses"] = data.get("num_poses", 0)
            row["landmarks"] = data.get("num_landmarks", 0)
            row["noise"] = data.get("noise_sigma", 0.0)
            row["outlier_ratio"] = data.get("outlier_ratio", 0.0)
        else:
            row.update({
                "scene": "", "trajectory": "", "model": "",
                "cameras": 0, "poses": 0, "landmarks": 0,
                "noise": 0.0, "outlier_ratio": 0.0,
            })

        if summary:
            solver = summary.get("solver", {})
            stats = summary.get("statistics", {})
            reproj = stats.get("reprojection_error", {})
            row["success"] = solver.get("success", False)
            row["reproj_mean"] = reproj.get("mean")
            row["reproj_median"] = reproj.get("median")
        else:
            row["success"] = None
            row["reproj_mean"] = None
            row["reproj_median"] = None

        if gt_eval:
            s = gt_eval.get("summary", {})
            row["extr_rot_err"] = s.get("mean_extr_rot_err_deg")
            row["extr_trans_err"] = s.get("mean_extr_trans_err_m")
        else:
            row["extr_rot_err"] = None
            row["extr_trans_err"] = None

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Sweep data extraction
# ---------------------------------------------------------------------------

def _get_nested(d: dict, dotted_key: str):
    """Get a value from a nested dict using dot notation."""
    keys = dotted_key.split(".")
    val = d
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return None
    return val


def extract_sweep_data(
    run_paths: List[Path],
    x_field: str,
    y_metric: str,
    group_field: Optional[str] = None,
) -> dict:
    """Extract (x, y, group) data from multiple runs for charting.

    Returns:
        {
            "x": [...], "y": [...], "group": [...],
            "x_label": str, "y_label": str,
            "runs": [run_name, ...],
        }
    """
    x_values = []
    y_values = []
    group_values = []
    run_names = []

    for run_path in run_paths:
        config = load_run_config(run_path)
        summary = load_run_summary(run_path)
        gt_eval = load_run_gt_evaluation(run_path)
        if config is None or summary is None:
            continue

        # Extract x value
        if x_field == "perturbation_mag":
            pert = config.get("data", {}).get("perturbation", {})
            x_val = max(
                pert.get("pose_rot_sigma_deg", 0),
                pert.get("extrinsic_rot_sigma_deg", 0),
            )
        else:
            x_val = _get_nested(config, x_field)

        if x_val is None:
            continue

        # Extract y value
        stats = summary.get("statistics", {})
        reproj = stats.get("reprojection_error", {})
        gt_summary = (gt_eval or {}).get("summary", {})

        metric_map = {
            "reproj_rmse": reproj.get("mean"),
            "reproj_median": reproj.get("median"),
            "reproj_p95": reproj.get("p95"),
            "extr_rot_err_deg": gt_summary.get("mean_extr_rot_err_deg"),
            "extr_trans_err_m": gt_summary.get("mean_extr_trans_err_m"),
            "pose_rot_err_deg": gt_summary.get("mean_pose_rot_err_deg"),
            "pose_trans_err_m": gt_summary.get("mean_pose_trans_err_m"),
            "landmark_err_m": gt_summary.get("mean_landmark_err_m"),
            "final_cost": stats.get("final_cost"),
        }

        y_val = metric_map.get(y_metric)
        if y_val is None:
            continue

        x_values.append(x_val)
        y_values.append(y_val)
        run_names.append(run_path.name)

        # Extract group
        if group_field:
            g_val = _get_nested(config, group_field)
            group_values.append(str(g_val) if g_val is not None else "unknown")
        else:
            group_values.append("all")

    y_labels = {
        "reproj_rmse": "Mean Reproj. Error (px)",
        "reproj_median": "Median Reproj. Error (px)",
        "reproj_p95": "P95 Reproj. Error (px)",
        "extr_rot_err_deg": "Extr. Rotation Error (deg)",
        "extr_trans_err_m": "Extr. Translation Error (m)",
        "pose_rot_err_deg": "Pose Rotation Error (deg)",
        "pose_trans_err_m": "Pose Translation Error (m)",
        "landmark_err_m": "Landmark Error (m)",
        "final_cost": "Final Cost",
    }

    return {
        "x": x_values,
        "y": y_values,
        "group": group_values,
        "x_label": x_field.split(".")[-1].replace("_", " ").title(),
        "y_label": y_labels.get(y_metric, y_metric),
        "runs": run_names,
    }


# ---------------------------------------------------------------------------
# Config diff
# ---------------------------------------------------------------------------

