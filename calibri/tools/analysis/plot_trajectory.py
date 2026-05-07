"""Trajectory alignment and ATE (Absolute Trajectory Error) evaluation.

For each real-world sequence:
  1. Extract optimized vehicle poses (T_wv) from result.json
  2. Extract initial poses from problem.json as "ground truth" / reference
  3. Align using Umeyama (SE(3)) alignment
  4. Compute translational RMSE (ATE)
  5. Generate bird's-eye view (X-Z plane) plots

Usage:
    python -m calibri.tools.analysis.plot_trajectory
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from calibri.tools.analysis.sequences import (
    OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, BATCH_DIR,
    ALL_SEQUENCES,
)


def extract_poses_from_problem(problem_path: Path) -> Dict[int, np.ndarray]:
    """Extract initial vehicle poses from problem.json.

    Returns dict mapping timestamp_ns -> translation [x, y, z].
    Poses are T_wv (vehicle-to-world), so t is the vehicle position in world frame.
    """
    with open(problem_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    poses = {}
    for p in data["poses"]:
        ts = p["timestamp_ns"]
        q = p["q"]  # [w,x,y,z]
        t = p["t"]  # [x,y,z]
        poses[ts] = np.array(t, dtype=np.float64)
    return poses


def extract_poses_from_result(result_path: Path) -> Dict[int, np.ndarray]:
    """Extract optimized vehicle poses from result.json.

    Returns dict mapping timestamp_ns -> translation [x, y, z].
    """
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    poses = {}
    for p in data["final_params"]["poses"]:
        ts = p["timestamp_ns"]
        t = p["t"]
        poses[ts] = np.array(t, dtype=np.float64)
    return poses


def umeyama_alignment(
    source: np.ndarray, target: np.ndarray, with_scale: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Umeyama alignment (SE(3) or Sim(3)).

    Finds R, t, s such that target ≈ s * R @ source + t

    Args:
        source: (N, 3) source points
        target: (N, 3) target points
        with_scale: if True, estimate scale (Sim(3)); else s=1 (SE(3))

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation
        s: scale factor
    """
    assert source.shape == target.shape
    n, dim = source.shape

    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)

    src_centered = source - mu_s
    tgt_centered = target - mu_t

    sigma_s_sq = np.sum(src_centered ** 2) / n

    Sigma = tgt_centered.T @ src_centered / n

    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[dim - 1, dim - 1] = -1

    R = U @ S @ Vt

    if with_scale:
        s = np.trace(np.diag(D) @ S) / sigma_s_sq
    else:
        s = 1.0

    t = mu_t - s * R @ mu_s

    return R, t, s


def compute_ate(
    initial_positions: np.ndarray,
    optimized_positions: np.ndarray,
    align: bool = True,
) -> Tuple[float, np.ndarray]:
    """Compute Absolute Trajectory Error (translational RMSE).

    Args:
        initial_positions: (N, 3) reference trajectory
        optimized_positions: (N, 3) estimated trajectory
        align: whether to SE(3)-align before computing error

    Returns:
        rmse: translational RMSE in meters
        aligned_opt: (N, 3) aligned optimized positions
    """
    if align and len(initial_positions) >= 3:
        R, t, s = umeyama_alignment(optimized_positions, initial_positions, with_scale=False)
        aligned = (R @ optimized_positions.T).T + t
    else:
        aligned = optimized_positions

    errors = np.linalg.norm(aligned - initial_positions, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return rmse, aligned


def plot_birdseye(
    initial_pos: np.ndarray,
    optimized_pos: np.ndarray,
    aligned_pos: np.ndarray,
    seq_name: str,
    ate_rmse: float,
    save_path: Path,
):
    """Generate a bird's-eye view (top-down X-Z plane) trajectory plot."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot initial (reference) trajectory
    ax.plot(
        initial_pos[:, 0], initial_pos[:, 2],
        "o-", color="#2196F3", markersize=4, linewidth=1.8,
        label="Initial / Reference", alpha=0.8,
    )

    # Plot aligned optimized trajectory
    ax.plot(
        aligned_pos[:, 0], aligned_pos[:, 2],
        "s-", color="#FF5722", markersize=4, linewidth=1.8,
        label="Optimized (aligned)", alpha=0.8,
    )

    # Mark start and end
    ax.plot(initial_pos[0, 0], initial_pos[0, 2], "^", color="green", markersize=12, label="Start")
    ax.plot(initial_pos[-1, 0], initial_pos[-1, 2], "v", color="red", markersize=12, label="End")

    ax.set_xlabel("X (meters)", fontsize=16)
    ax.set_ylabel("Z (meters)", fontsize=16)
    ax.set_title(f"Bird's-Eye View: {seq_name}\nATE RMSE = {ate_rmse:.4f} m", fontsize=18)
    ax.legend(fontsize=14, loc="best")

    # Use equal aspect only when the trajectory has comparable X and Z extent.
    # For straight-line highway sequences the Z variation is tiny compared to X,
    # and equal aspect creates a thin line with vast empty space.
    x_range = float(np.ptp(initial_pos[:, 0]))
    z_range = float(np.ptp(initial_pos[:, 2]))
    aspect_ratio = x_range / max(z_range, 1e-6)
    if aspect_ratio < 10:
        ax.set_aspect("equal", adjustable="datalim")

    ax.tick_params(labelsize=13)

    plt.tight_layout()
    plt.savefig(str(save_path), format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved: {save_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    ate_results = {}

    print("=" * 70)
    print("TRAJECTORY ALIGNMENT & ATE EVALUATION")
    print("=" * 70)

    for seq_name, problem_path in ALL_SEQUENCES.items():
        print(f"\n  Sequence: {seq_name}")

        result_path = BATCH_DIR / seq_name / "result.json"
        if not result_path.exists():
            print(f"    [SKIP] No result.json found at {result_path}")
            continue

        try:
            # Extract poses
            init_poses = extract_poses_from_problem(problem_path)
            opt_poses = extract_poses_from_result(result_path)

            # Match timestamps
            common_ts = sorted(set(init_poses.keys()) & set(opt_poses.keys()))
            if len(common_ts) < 3:
                print(f"    [SKIP] Only {len(common_ts)} common timestamps")
                continue

            init_pos = np.array([init_poses[ts] for ts in common_ts])
            opt_pos = np.array([opt_poses[ts] for ts in common_ts])

            # Compute ATE with SE(3) alignment
            ate_rmse, aligned_pos = compute_ate(init_pos, opt_pos, align=True)
            ate_results[seq_name] = ate_rmse
            print(f"    ATE RMSE: {ate_rmse:.4f} m ({len(common_ts)} poses)")

            # Generate bird's-eye view plot
            plot_path = FIGURES_DIR / f"trajectory_{seq_name}.pdf"
            plot_birdseye(init_pos, opt_pos, aligned_pos, seq_name, ate_rmse, plot_path)

        except Exception as e:
            print(f"    [ERROR] {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── Update CSV with ATE column ─────────────────────────────────────
    csv_path = TABLES_DIR / "real_world_metrics.csv"
    if csv_path.exists() and ate_results:
        print(f"\n  Updating {csv_path} with ATE column...")
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            for row in reader:
                seq = row["sequence"]
                if seq in ate_results:
                    row["ate_rmse_m"] = round(ate_results[seq], 4)
                rows.append(row)

        if "ate_rmse_m" not in fieldnames:
            fieldnames.append("ate_rmse_m")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV updated with ATE for {len(ate_results)} sequences.")

    # ── Save standalone ATE table ──────────────────────────────────────
    ate_csv = TABLES_DIR / "ate_results.csv"
    with open(ate_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "ate_rmse_m", "num_poses"])
        for seq_name in ALL_SEQUENCES:
            if seq_name in ate_results:
                # Re-count poses
                result_path = BATCH_DIR / seq_name / "result.json"
                init_poses = extract_poses_from_problem(ALL_SEQUENCES[seq_name])
                opt_poses = extract_poses_from_result(result_path)
                n = len(set(init_poses.keys()) & set(opt_poses.keys()))
                writer.writerow([seq_name, round(ate_results[seq_name], 4), n])

    print(f"\n  ATE table written to: {ate_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
