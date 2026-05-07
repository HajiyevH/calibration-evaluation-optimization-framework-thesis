"""Monte Carlo Basin of Attraction analysis.

Tests solver robustness to progressively larger initial extrinsic perturbations.
Generates a basin-of-attraction curve: perturbation magnitude vs success rate.

Usage:
    python -m calibri.tools.analysis.run_monte_carlo
"""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from calibri.tools.utils import load_json, json_quat_to_scipy, run_ba_cli
from calibri.tools.analysis.sequences import (
    REPO_ROOT, BA_CLI, OUTPUT_DIR, TABLES_DIR, FIGURES_DIR,
)

SYNTH_GEN = REPO_ROOT / "calibri" / "tools" / "synth_gen.py"
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

# Monte Carlo parameters
PERTURBATION_LEVELS_DEG = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 17.0, 20.0]
TRIALS_PER_LEVEL = 100
SUCCESS_THRESHOLD_DEG = 0.5  # max extrinsic recovery error to count as success
SEED_BASE = 42


def generate_baseline_scene(output_dir: Path, seed: int = 42) -> Tuple[Path, Path]:
    """Generate a multi-camera synthetic scene as baseline."""
    problem_path = output_dir / "problem_baseline.json"
    gt_path = output_dir / "gt_baseline.json"

    cmd = [
        str(PYTHON_EXE), str(SYNTH_GEN),
        "--num-cameras", "4",
        "--num-poses", "25",
        "--num-landmarks", "700",
        "--noise-sigma", "0.5",
        "--seed", str(seed),
        "--output", str(problem_path),
        "--gt", str(gt_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"synth_gen failed: {proc.stderr}")

    return problem_path, gt_path


def save_json(data: Dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def perturb_extrinsics_custom(
    problem: Dict,
    perturbation_deg: float,
    rng: np.random.Generator,
    reference_camera_id: int = 0,
) -> Dict:
    """Apply a controlled rotation perturbation to non-reference camera extrinsics.

    The perturbation magnitude is exactly `perturbation_deg` degrees (applied as a
    random-axis rotation).
    """
    perturbed = json.loads(json.dumps(problem))  # deep copy

    for cam in perturbed["cameras"]:
        cam_id = cam["id"]
        if cam_id == reference_camera_id:
            continue

        # Get current extrinsic rotation
        ext = cam.get("extrinsics", cam.get("extrinsic", {}))
        tvc = ext.get("T_vehicle_to_camera", ext)
        q = tvc["q"]  # [w,x,y,z]
        t = tvc["t"]

        R_orig = Rotation.from_quat(json_quat_to_scipy(q))

        # Generate random rotation axis, fixed magnitude
        axis = rng.standard_normal(3)
        axis = axis / np.linalg.norm(axis)
        angle_rad = np.radians(perturbation_deg)
        R_perturb = Rotation.from_rotvec(axis * angle_rad)

        R_new = R_perturb * R_orig
        q_new = R_new.as_quat()  # [x,y,z,w]

        tvc["q"] = [float(q_new[3]), float(q_new[0]), float(q_new[1]), float(q_new[2])]

        # Also perturb translation proportionally (0.01m per degree)
        trans_sigma = perturbation_deg * 0.01
        t_perturb = rng.standard_normal(3) * trans_sigma
        tvc["t"] = [float(t[i] + t_perturb[i]) for i in range(3)]

    return perturbed


def compute_extrinsic_recovery_error(
    gt: Dict, result: Dict, reference_camera_id: int = 0
) -> float:
    """Compute max rotation error across non-reference cameras."""
    gt_cameras = {c["id"]: c for c in gt["cameras"]}
    result_cameras = {c["id"]: c for c in result["final_params"]["cameras"]}

    max_rot_err = 0.0
    for cid, gt_cam in gt_cameras.items():
        if cid == reference_camera_id:
            continue
        if cid not in result_cameras:
            continue

        # GT extrinsic
        gt_ext = gt_cam.get("extrinsics", gt_cam.get("extrinsic", {}))
        gt_tvc = gt_ext.get("T_vehicle_to_camera", gt_ext)
        gq = gt_tvc["q"]  # [w,x,y,z]
        R_gt = Rotation.from_quat(json_quat_to_scipy(gq))

        # Optimized extrinsic
        opt_cam = result_cameras[cid]
        oq = opt_cam["extrinsic"]["q"]  # [w,x,y,z]
        R_opt = Rotation.from_quat(json_quat_to_scipy(oq))

        R_delta = R_gt.inv() * R_opt
        rot_err_deg = np.degrees(R_delta.magnitude())
        max_rot_err = max(max_rot_err, rot_err_deg)

    return max_rot_err


def run_single_trial(
    problem_path: Path,
    result_path: Path,
    timeout_s: int = 120,
) -> Tuple[bool, Optional[Dict]]:
    """Run solver on a single perturbed problem. Returns (cli_success, parsed_result)."""
    extra_args = [
        "--loss-type", "Huber",
        "--loss-scale", "1.0",
        "--max-iterations", "200",
    ]
    try:
        success, _, result = run_ba_cli(
            problem_path, result_path,
            timeout=timeout_s, extra_args=extra_args,
        )
        if not success or result is None:
            return False, None
        return result["summary"].get("success", False), result
    except (KeyError, Exception):
        return False, None


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    mc_dir = OUTPUT_DIR / "monte_carlo"
    mc_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MONTE CARLO BASIN OF ATTRACTION ANALYSIS")
    print(f"  Perturbation levels: {PERTURBATION_LEVELS_DEG}")
    print(f"  Trials per level: {TRIALS_PER_LEVEL}")
    print(f"  Success threshold: < {SUCCESS_THRESHOLD_DEG} deg")
    print("=" * 70)

    # Generate baseline scene
    print("\n  Generating baseline synthetic scene...")
    baseline_problem, baseline_gt = generate_baseline_scene(mc_dir, seed=SEED_BASE)
    gt_data = load_json(baseline_gt)
    print(f"  GT baseline: {baseline_gt}")

    # Set flags to isolate extrinsic recovery:
    # - opt_extrinsics=True: the parameter we're testing
    # - opt_poses=False, opt_landmarks=False: keep at GT to prevent the solver
    #   from absorbing extrinsic errors into pose/landmark adjustments
    gt_data.setdefault("flags", {})
    gt_data["flags"]["opt_extrinsics"] = True
    gt_data["flags"]["opt_poses"] = False
    gt_data["flags"]["opt_landmarks"] = False
    if "camera_rig" not in gt_data:
        gt_data["camera_rig"] = {"reference_camera_id": 0}

    # Results storage
    level_results: Dict[float, Dict[str, Any]] = {}

    for level_deg in PERTURBATION_LEVELS_DEG:
        print(f"\n{'-' * 60}")
        print(f"  Perturbation: {level_deg}°  ({TRIALS_PER_LEVEL} trials)")

        successes = 0
        convergences = 0
        recovery_errors = []

        level_idx = PERTURBATION_LEVELS_DEG.index(level_deg)
        for trial in range(TRIALS_PER_LEVEL):
            rng = np.random.default_rng(SEED_BASE * 100_000 + level_idx * 1000 + trial)

            # Start from GROUND TRUTH — only extrinsics will be perturbed.
            # Poses and landmarks remain at GT values, isolating the
            # extrinsic recovery capability of the solver.
            perturbed = perturb_extrinsics_custom(gt_data, level_deg, rng, reference_camera_id=0)

            # Write perturbed problem
            trial_problem = mc_dir / f"trial_{level_deg:.1f}_{trial:03d}_problem.json"
            trial_result = mc_dir / f"trial_{level_deg:.1f}_{trial:03d}_result.json"
            save_json(perturbed, trial_problem)

            # Run solver
            cli_ok, result = run_single_trial(trial_problem, trial_result, timeout_s=120)

            if cli_ok and result is not None:
                convergences += 1
                recovery_err = compute_extrinsic_recovery_error(gt_data, result, reference_camera_id=0)
                recovery_errors.append(recovery_err)
                if recovery_err < SUCCESS_THRESHOLD_DEG:
                    successes += 1

            # Clean up trial files to save disk
            trial_problem.unlink(missing_ok=True)
            trial_result.unlink(missing_ok=True)

            # Progress
            if (trial + 1) % 20 == 0:
                print(f"    Trial {trial+1}/{TRIALS_PER_LEVEL}: "
                      f"{successes} successes, {convergences} converged")

        success_rate = successes / TRIALS_PER_LEVEL * 100
        convergence_rate = convergences / TRIALS_PER_LEVEL * 100
        mean_err = float(np.mean(recovery_errors)) if recovery_errors else float("nan")
        median_err = float(np.median(recovery_errors)) if recovery_errors else float("nan")

        level_results[level_deg] = {
            "perturbation_deg": level_deg,
            "trials": TRIALS_PER_LEVEL,
            "successes": successes,
            "convergences": convergences,
            "success_rate": success_rate,
            "convergence_rate": convergence_rate,
            "mean_recovery_error_deg": mean_err,
            "median_recovery_error_deg": median_err,
        }

        print(f"  Result: {success_rate:.0f}% success, {convergence_rate:.0f}% converged, "
              f"mean err={mean_err:.3f}°, median err={median_err:.3f}°")

    # ── Save CSV ───────────────────────────────────────────────────────
    csv_path = TABLES_DIR / "monte_carlo_basin.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "perturbation_deg", "trials", "successes", "convergences",
            "success_rate", "convergence_rate",
            "mean_recovery_error_deg", "median_recovery_error_deg",
        ])
        writer.writeheader()
        for level_deg in PERTURBATION_LEVELS_DEG:
            writer.writerow(level_results[level_deg])

    print(f"\n  CSV written to: {csv_path}")

    # ── Plot basin of attraction ───────────────────────────────────────
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    levels = [level_results[d]["perturbation_deg"] for d in PERTURBATION_LEVELS_DEG]
    success_rates = [level_results[d]["success_rate"] for d in PERTURBATION_LEVELS_DEG]
    convergence_rates = [level_results[d]["convergence_rate"] for d in PERTURBATION_LEVELS_DEG]
    mean_errors = [level_results[d]["mean_recovery_error_deg"] for d in PERTURBATION_LEVELS_DEG]

    # Success rate curve
    ax1.plot(levels, success_rates, "o-", color="#2196F3", linewidth=2.5,
             markersize=7, label=f"Success Rate (< {SUCCESS_THRESHOLD_DEG}°)", zorder=3)
    ax1.plot(levels, convergence_rates, "s--", color="#4CAF50", linewidth=1.5,
             markersize=5, label="Convergence Rate", alpha=0.7, zorder=2)

    ax1.set_xlabel("Initial Perturbation Magnitude (degrees)", fontsize=13)
    ax1.set_ylabel("Rate (%)", fontsize=13)
    ax1.set_ylim(-5, 105)
    ax1.tick_params(labelsize=11)

    # Mean recovery error on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(levels, mean_errors, "D-", color="#FF5722", linewidth=1.5,
             markersize=5, label="Mean Recovery Error", alpha=0.8)
    ax2.set_ylabel("Mean Recovery Error (degrees)", fontsize=13, color="#FF5722")
    ax2.tick_params(axis="y", labelcolor="#FF5722", labelsize=11)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="center right")

    ax1.set_title("Basin of Attraction: Solver Robustness to Extrinsic Perturbations", fontsize=14)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = FIGURES_DIR / "monte_carlo_basin.pdf"
    plt.savefig(str(plot_path), format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Plot saved to: {plot_path}")

    # ── Also plot recovery error distribution for selected levels ──────
    fig2, ax = plt.subplots(figsize=(8, 5))
    # Re-run a few key levels to capture error distributions
    # (we already have the summary stats, just plot those)
    selected = [1.0, 5.0, 10.0, 20.0]
    x_pos = range(len(selected))
    success_vals = [level_results[d]["success_rate"] for d in selected if d in level_results]
    selected_actual = [d for d in selected if d in level_results]

    bars = ax.bar(range(len(selected_actual)), success_vals, color=["#4CAF50", "#2196F3", "#FF9800", "#F44336"],
                  edgecolor="black", linewidth=0.5, width=0.6)
    ax.set_xticks(range(len(selected_actual)))
    ax.set_xticklabels([f"{d}°" for d in selected_actual], fontsize=12)
    ax.set_xlabel("Initial Perturbation", fontsize=13)
    ax.set_ylabel("Success Rate (%)", fontsize=13)
    ax.set_title("Solver Recovery Success at Key Perturbation Levels", fontsize=14)
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, success_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    bar_path = FIGURES_DIR / "monte_carlo_bars.pdf"
    plt.savefig(str(bar_path), format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print(f"  Bar plot saved to: {bar_path}")

    print("\n" + "=" * 70)
    print("MONTE CARLO ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
