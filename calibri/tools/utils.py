"""Shared utilities for BA evaluation tools.

All logic lives in importable functions so they can be reused
by pybind11 bindings, a package CLI, or a PySide6 UI later.
"""

import json
import logging
import subprocess
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def load_json(path: Path) -> dict:
    """Load any JSON file and return the parsed dict.

    Args:
        path: Path to a JSON file.

    Returns:
        Parsed JSON as a dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_result(path: Path) -> dict:
    """Load a result.json file and return the parsed dict."""
    return load_json(path)


def compute_error_stats(errors: Sequence[float]) -> Dict[str, float]:
    """Compute mean, median, p95, max from a list of error values.

    Args:
        errors: array-like of reprojection errors (px).

    Returns:
        dict with keys: mean, median, p95, max, std, count.
    """
    errors = np.asarray(errors, dtype=float)
    if len(errors) == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0,
                "max": 0.0, "std": 0.0, "count": 0}
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
        "std": float(np.std(errors)),
        "count": len(errors),
    }


def get_residuals_array(result: dict) -> Dict[str, np.ndarray]:
    """Extract per-observation residuals as structured arrays.

    Note: Pruned observations (rejected before optimization) are not included
    in result.json, so this only returns observations that were actually used.

    Returns:
        dict with numpy arrays: u, v, u_hat, v_hat, du, dv, err,
        timestamp_ns, camera_id, landmark_id, inlier.
    """
    residuals = result.get("residuals", [])
    n = len(residuals)
    out = {
        "u": np.zeros(n),
        "v": np.zeros(n),
        "u_hat": np.zeros(n),
        "v_hat": np.zeros(n),
        "du": np.zeros(n),
        "dv": np.zeros(n),
        "err": np.zeros(n),
        "timestamp_ns": np.zeros(n, dtype=np.uint64),
        "camera_id": np.zeros(n, dtype=np.uint8),
        "landmark_id": np.zeros(n, dtype=np.uint64),
        "inlier": np.zeros(n, dtype=bool),
    }
    for i, r in enumerate(residuals):
        out["u"][i] = r["u"]
        out["v"][i] = r["v"]
        # Handle None/NaN values from diverged BA
        u_hat = r["u_hat"] if r["u_hat"] is not None else np.nan
        v_hat = r["v_hat"] if r["v_hat"] is not None else np.nan
        out["u_hat"][i] = u_hat
        out["v_hat"][i] = v_hat
        out["du"][i] = r["u"] - u_hat if not np.isnan(u_hat) else np.nan
        out["dv"][i] = r["v"] - v_hat if not np.isnan(v_hat) else np.nan
        out["err"][i] = r["err"] if r["err"] is not None else np.nan
        out["timestamp_ns"][i] = r["timestamp_ns"]
        out["camera_id"][i] = r["camera_id"]
        out["landmark_id"][i] = r["landmark_id"]
        out["inlier"][i] = r.get("inlier", True)
    return out


def get_per_frame_errors(result: dict) -> List[dict]:
    """Extract per-frame error statistics.

    Returns:
        list of dicts with timestamp_ns, mean_px, median_px, p95_px.
    """
    return result.get("per_frame", [])


# ============================================================================
# Quaternion and Rotation Utilities
# ============================================================================

def scipy_quat_to_json(q_scipy: np.ndarray) -> List[float]:
    """Convert scipy quaternion [x,y,z,w] to JSON format [w,x,y,z].

    Args:
        q_scipy: Quaternion in scipy/Eigen internal order [x,y,z,w]

    Returns:
        Quaternion in JSON/constructor order [w,x,y,z]
    """
    return [float(q_scipy[3]), float(q_scipy[0]), float(q_scipy[1]), float(q_scipy[2])]


def json_quat_to_scipy(q_json: List[float]) -> np.ndarray:
    """Convert JSON quaternion [w,x,y,z] to scipy format [x,y,z,w].

    Args:
        q_json: Quaternion in JSON/constructor order [w,x,y,z]

    Returns:
        Quaternion in scipy/Eigen internal order [x,y,z,w]
    """
    return np.array([q_json[1], q_json[2], q_json[3], q_json[0]])


# ============================================================================
# Coordinate Frame Transformations
# ============================================================================

# ============================================================================
# Projection Functions (for validation and testing)
# ============================================================================

def project_pinhole_radtan(
    X_cam: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    k1: float, k2: float, p1: float, p2: float, k3: float
) -> Optional[Tuple[float, float]]:
    """Project 3D point in camera frame to 2D using pinhole + radial-tangential distortion.

    Args:
        X_cam: 3D point in camera frame [x, y, z]
        fx, fy, cx, cy: Intrinsic parameters
        k1, k2, p1, p2, k3: Distortion coefficients

    Returns:
        (u, v) pixel coordinates, or None if point is behind camera
    """
    x, y, z = X_cam
    if z <= 0:
        return None

    # Normalized coordinates
    xn = x / z
    yn = y / z

    # Radial distortion
    r2 = xn**2 + yn**2
    radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3

    # Tangential distortion
    xd = xn * radial + 2*p1*xn*yn + p2*(r2 + 2*xn**2)
    yd = yn * radial + p1*(r2 + 2*yn**2) + 2*p2*xn*yn

    # Project to pixel coordinates
    u = fx * xd + cx
    v = fy * yd + cy

    return (float(u), float(v))


def project_pinhole_fisheye(
    X_cam: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    k1: float, k2: float, k3: float, k4: float,
) -> Optional[Tuple[float, float]]:
    """Project 3D point in camera frame to 2D using pinhole + OpenCV fisheye distortion.

    Implements the equidistant (theta) fisheye model:
      r     = sqrt(x^2 + y^2)
      theta = atan(r)
      theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
      scale = theta_d / r      (limit = 1 at r -> 0)
      u = fx * scale * x + cx
      v = fy * scale * y + cy

    Matches C++ projectPinholeFisheyeT in calibri/engine/cameras/pinhole_fisheye.h.

    Args:
        X_cam: 3D point in camera frame [x, y, z]
        fx, fy, cx, cy: Intrinsic parameters
        k1, k2, k3, k4: Fisheye distortion coefficients

    Returns:
        (u, v) pixel coordinates, or None if point is behind camera
    """
    x, y, z = float(X_cam[0]), float(X_cam[1]), float(X_cam[2])
    if z <= 0:
        return None

    # Normalized coordinates
    xn = x / z
    yn = y / z

    r_sq = xn * xn + yn * yn
    r = np.sqrt(r_sq)

    theta = np.arctan(r)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4

    theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)

    # Safe division matching C++ eps guard: sqrt(r_sq + eps^2)
    eps = 1e-10
    r_safe = np.sqrt(r_sq + eps * eps)
    scale = theta_d / r_safe

    x_d = scale * xn
    y_d = scale * yn

    u = fx * x_d + cx
    v = fy * y_d + cy

    return (float(u), float(v))


# ============================================================================
# Logging Helpers
# ============================================================================

def setup_logging(verbose: bool = False):
    """Configure logging for CLI tools.

    Args:
        verbose: If True, set logging level to DEBUG, otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


# ============================================================================
# BA CLI Helpers
# ============================================================================

def find_ba_cli(cli_path: Optional[Path] = None) -> Path:
    """Locate ba_cli.exe, using explicit path or auto-detection.

    Args:
        cli_path: Explicit path to ba_cli.exe. If None, auto-detects
                  from the repository build directory.

    Returns:
        Path to ba_cli.exe.

    Raises:
        FileNotFoundError: If ba_cli.exe cannot be found.
    """
    if cli_path is not None:
        cli = Path(cli_path)
        if not cli.exists():
            raise FileNotFoundError(f"ba_cli.exe not found at {cli}")
        return cli

    repo_root = Path(__file__).resolve().parent.parent.parent
    default = repo_root / "build" / "calibri" / "app" / "Release" / "ba_cli.exe"
    if default.exists():
        return default

    raise FileNotFoundError(
        f"ba_cli.exe not found at {default}. "
        "Build the project first, or pass cli_path explicitly."
    )


def run_ba_cli(
    problem_path: Path,
    result_path: Path,
    cli_path: Optional[Path] = None,
    timeout: int = 600,
    extra_args: Optional[List[str]] = None,
) -> Tuple[bool, str, Optional[dict]]:
    """Run ba_cli.exe on a problem.json and return results.

    Args:
        problem_path: Path to input problem.json.
        result_path: Path where result.json will be written.
        cli_path: Explicit path to ba_cli.exe (auto-detected if None).
        timeout: Subprocess timeout in seconds.
        extra_args: Additional CLI arguments (e.g. --loss-type Huber).

    Returns:
        Tuple of (success, combined_output, parsed_result_or_None).
    """
    cli = find_ba_cli(cli_path)
    cmd = [str(cli), str(problem_path), str(result_path)]
    if extra_args:
        cmd.extend(extra_args)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout + proc.stderr
        success = proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", None
    except Exception as e:
        return False, f"ERROR: {e}", None

    parsed = None
    if success:
        try:
            parsed = load_json(result_path)
        except Exception:
            pass

    return success, output, parsed
