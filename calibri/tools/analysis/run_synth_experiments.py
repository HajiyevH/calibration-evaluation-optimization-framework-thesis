"""Synthetic validation experiments for thesis results.

Runs five experiments that validate the solver's parameter recovery.

Code-to-thesis mapping (code is 0-indexed, thesis is 1-indexed):
  Code Exp 0 = Thesis Exp 1: Baseline pose+landmark recovery (single-camera)
  Code Exp 1 = Thesis Exp 2: Intrinsics recovery (single-camera, Pinhole)
  Code Exp 2 = Thesis Exp 3: Joint recovery (4-camera, Pinhole)
  Code Exp 3 = Thesis Exp 4: Outlier robustness comparison
  Code Exp 4 = Thesis Exp 5: Fisheye joint recovery (4-camera)

All results are saved as thesis-ready CSV tables in thesis_results/tables/.

Usage:
    python -m calibri.tools.analysis.run_synth_experiments
"""

import copy
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from calibri.framework.config import (
    ExperimentConfig,
    DataConfig,
    PerturbationConfig,
    SolverConfig,
    SolverFlags,
    RobustConfig,
    AnalysisConfig,
)
from calibri.framework.runner import run_experiment
from calibri.framework.evaluate import evaluate_against_gt
from calibri.framework.solver import run_solver
from calibri.tools.utils import load_json
from calibri.tools.analysis.sequences import TABLES_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------


def _exp0_baseline() -> ExperimentConfig:
    """Experiment 0: single-camera baseline (poses + landmarks only)."""
    return ExperimentConfig(
        name="synth-baseline",
        description="Single-camera baseline: pose and landmark optimization with fixed intrinsics",
        data=DataConfig(
            scene_type="corridor",
            num_landmarks=600,
            num_poses=25,
            trajectory_type="arc",
            camera_model="Pinhole",
            num_cameras=1,
            image_width=1280,
            image_height=800,
            noise_sigma=0.5,
            perturbation=PerturbationConfig(
                pose_rot_sigma_deg=1.0,
                pose_trans_sigma_m=0.02,
                landmark_sigma_m=0.05,
            ),
            seed=42,
        ),
        solver=SolverConfig(
            flags=SolverFlags(
                opt_poses=True,
                opt_landmarks=True,
            ),
            robust=RobustConfig(type="Huber", scale=1.0),
        ),
        analysis=AnalysisConfig(
            plot_errors=False,
            overlay_frames=[],
            print_summary=True,
            evaluate_gt=True,
        ),
    )


def _exp1_intrinsics_recovery() -> ExperimentConfig:
    """Experiment 1: single-camera intrinsics recovery."""
    return ExperimentConfig(
        name="synth-intrinsics-recovery",
        description="Single-camera intrinsics recovery (fx, fy, cx, cy) from perturbed initial guess",
        data=DataConfig(
            scene_type="corridor",
            num_landmarks=600,
            num_poses=25,
            trajectory_type="arc",
            camera_model="Pinhole",
            num_cameras=1,
            image_width=1280,
            image_height=800,
            noise_sigma=0.5,
            perturbation=PerturbationConfig(
                pose_rot_sigma_deg=1.0,
                pose_trans_sigma_m=0.02,
                landmark_sigma_m=0.05,
                intrinsic_sigma_px=20.0,
            ),
            seed=42,
        ),
        solver=SolverConfig(
            flags=SolverFlags(
                opt_poses=True,
                opt_landmarks=True,
                opt_intrinsics=True,
            ),
            robust=RobustConfig(type="Huber", scale=1.0),
        ),
        analysis=AnalysisConfig(
            plot_errors=False,
            overlay_frames=[],
            print_summary=True,
            evaluate_gt=True,
        ),
    )


def _exp2_joint_recovery() -> ExperimentConfig:
    """Experiment 2: 4-camera joint intrinsics + extrinsics + distortion recovery."""
    return ExperimentConfig(
        name="synth-joint-recovery",
        description="4-camera joint intrinsics, extrinsics, and distortion recovery",
        data=DataConfig(
            scene_type="surround",
            num_landmarks=800,
            num_poses=30,
            trajectory_type="slalom",
            camera_model="Pinhole",
            num_cameras=4,
            image_width=1280,
            image_height=800,
            noise_sigma=0.5,
            perturbation=PerturbationConfig(
                pose_rot_sigma_deg=1.0,
                pose_trans_sigma_m=0.02,
                landmark_sigma_m=0.05,
                extrinsic_rot_sigma_deg=2.0,
                extrinsic_trans_sigma_m=0.05,
                intrinsic_sigma_px=15.0,
                distortion_sigma=0.01,
            ),
            seed=42,
        ),
        solver=SolverConfig(
            flags=SolverFlags(
                opt_poses=True,
                opt_landmarks=True,
                opt_intrinsics=True,
                opt_distortion=True,
                opt_extrinsics=True,
            ),
            robust=RobustConfig(type="Huber", scale=1.0),
        ),
        analysis=AnalysisConfig(
            plot_errors=False,
            overlay_frames=[],
            print_summary=True,
            evaluate_gt=True,
        ),
    )


def _exp3_outlier_base_config() -> ExperimentConfig:
    """Experiment 3 base: data generation config for outlier robustness comparison."""
    return ExperimentConfig(
        name="synth-outlier-base",
        description="Outlier robustness comparison base data (10% outliers)",
        data=DataConfig(
            scene_type="surround",
            num_landmarks=800,
            num_poses=30,
            trajectory_type="slalom",
            camera_model="Pinhole",
            num_cameras=4,
            image_width=1280,
            image_height=800,
            noise_sigma=0.5,
            outlier_ratio=0.10,
            perturbation=PerturbationConfig(
                pose_rot_sigma_deg=1.0,
                pose_trans_sigma_m=0.02,
                landmark_sigma_m=0.05,
                extrinsic_rot_sigma_deg=2.0,
                extrinsic_trans_sigma_m=0.05,
            ),
            seed=42,
        ),
        solver=SolverConfig(
            flags=SolverFlags(
                opt_poses=True,
                opt_landmarks=True,
                opt_extrinsics=True,
            ),
            robust=RobustConfig(type="Huber", scale=1.0),
        ),
        analysis=AnalysisConfig(
            plot_errors=False,
            overlay_frames=[],
            print_summary=True,
            evaluate_gt=True,
        ),
    )


def _exp4_fisheye_joint_recovery() -> ExperimentConfig:
    """Experiment 4: 4-camera fisheye joint intrinsics + extrinsics + distortion recovery."""
    return ExperimentConfig(
        name="synth-fisheye-joint",
        description="4-camera fisheye joint intrinsics, extrinsics, and distortion recovery",
        data=DataConfig(
            scene_type="surround",
            num_landmarks=800,
            num_poses=30,
            trajectory_type="slalom",
            camera_model="Fisheye",
            num_cameras=4,
            image_width=1280,
            image_height=800,
            noise_sigma=0.5,
            perturbation=PerturbationConfig(
                pose_rot_sigma_deg=1.0,
                pose_trans_sigma_m=0.02,
                landmark_sigma_m=0.05,
                extrinsic_rot_sigma_deg=2.0,
                extrinsic_trans_sigma_m=0.05,
                intrinsic_sigma_px=15.0,
                distortion_sigma=0.01,
            ),
            seed=42,
        ),
        solver=SolverConfig(
            flags=SolverFlags(
                opt_poses=True,
                opt_landmarks=True,
                opt_intrinsics=True,
                opt_distortion=True,
                opt_extrinsics=True,
            ),
            robust=RobustConfig(type="Huber", scale=1.0),
        ),
        analysis=AnalysisConfig(
            plot_errors=False,
            overlay_frames=[],
            print_summary=True,
            evaluate_gt=True,
        ),
    )


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _write_intrinsics_csv(gt_eval: Dict, csv_path: Path) -> None:
    """Write per-camera intrinsic recovery errors to CSV."""
    rows = gt_eval.get("intrinsics", [])
    if not rows:
        logger.warning("No intrinsic data to write to %s", csv_path)
        return

    fieldnames = ["camera_id", "fx_err", "fy_err", "cx_err", "cy_err"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"  CSV: {csv_path}")


def _write_joint_csv(gt_eval: Dict, csv_path: Path, fisheye: bool = False) -> None:
    """Write per-camera extrinsic + intrinsic + distortion errors to CSV."""
    extr_map = {e["camera_id"]: e for e in gt_eval.get("extrinsics", [])}
    intr_map = {i["camera_id"]: i for i in gt_eval.get("intrinsics", [])}
    dist_map = {d["camera_id"]: d for d in gt_eval.get("distortion", [])}

    cam_ids = sorted(set(list(extr_map) + list(intr_map) + list(dist_map)))
    if not cam_ids:
        logger.warning("No joint data to write to %s", csv_path)
        return

    if fisheye:
        dist_keys = ["k1_err", "k2_err", "k3_err", "k4_err"]
    else:
        dist_keys = ["k1_err", "k2_err", "p1_err", "p2_err", "k3_err"]

    fieldnames = [
        "camera_id",
        "extr_rot_err_deg", "extr_trans_err_m",
        "fx_err", "fy_err", "cx_err", "cy_err",
    ] + dist_keys

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for cid in cam_ids:
            row = {"camera_id": cid}
            if cid in extr_map:
                row["extr_rot_err_deg"] = extr_map[cid].get("rot_err_deg", "")
                row["extr_trans_err_m"] = extr_map[cid].get("trans_err_m", "")
            if cid in intr_map:
                row["fx_err"] = intr_map[cid].get("fx_err", "")
                row["fy_err"] = intr_map[cid].get("fy_err", "")
                row["cx_err"] = intr_map[cid].get("cx_err", "")
                row["cy_err"] = intr_map[cid].get("cy_err", "")
            if cid in dist_map:
                coeffs = dist_map[cid].get("coeffs", {})
                for dk in dist_keys:
                    row[dk] = coeffs.get(dk, "")
            writer.writerow(row)

    print(f"  CSV: {csv_path}")


def _write_outlier_csv(variant_results: List[Dict], csv_path: Path) -> None:
    """Write outlier robustness comparison to CSV (one row per variant)."""
    fieldnames = [
        "variant",
        "robust_type", "outlier_rejection_passes",
        "rmse_reproj_px", "mean_reproj_px", "median_reproj_px", "p95_reproj_px",
        "num_inliers", "num_outliers", "inlier_ratio",
        "mean_extr_rot_err_deg", "mean_extr_trans_err_m",
        "mean_pose_rot_err_deg", "mean_pose_trans_err_m",
        "mean_landmark_err_m",
        "iterations", "solver_success",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for vr in variant_results:
            writer.writerow(vr)

    print(f"  CSV: {csv_path}")


def _write_summary_csv(all_rows: List[Dict], csv_path: Path) -> None:
    """Write aggregated summary table across all synthetic experiments."""
    fieldnames = [
        "experiment", "camera_model", "num_cameras", "scene_type",
        "mean_extr_rot_err_deg", "mean_extr_trans_err_m",
        "mean_intr_err_px", "mean_dist_err",
        "rmse_reproj_px",
        "mean_pose_rot_err_deg", "mean_pose_trans_err_m",
        "mean_landmark_err_m",
        "solver_time_s",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"  CSV: {csv_path}")


# ---------------------------------------------------------------------------
# Experiment 3 helpers
# ---------------------------------------------------------------------------


def _patch_problem_variant(
    problem: Dict,
    robust_type: str,
    robust_scale: float,
    max_passes: int,
) -> Dict:
    """Return a patched copy of problem.json for an outlier robustness variant."""
    patched = copy.deepcopy(problem)
    patched["robust"] = {"type": robust_type, "scale": robust_scale}
    patched["outlier_rejection"] = {
        "max_passes": max_passes,
        "threshold_px": 0.0,
        "multiplier": 3.0,
    }
    return patched


def _run_outlier_variant(
    variant_name: str,
    problem_path: Path,
    result_path: Path,
    gt_path: Path,
    robust_type: str,
    robust_scale: float,
    max_passes: int,
) -> Dict[str, Any]:
    """Run one outlier robustness variant and return summary row."""
    # Load and patch problem
    problem = load_json(problem_path)
    patched = _patch_problem_variant(problem, robust_type, robust_scale, max_passes)

    patched_path = result_path.parent / f"problem_{variant_name}.json"
    with open(patched_path, "w", encoding="utf-8") as f:
        json.dump(patched, f, indent=2)

    variant_result_path = result_path.parent / f"result_{variant_name}.json"

    print(f"    Running variant: {variant_name} (robust={robust_type}, passes={max_passes})")
    solver_result = run_solver(
        problem_path=patched_path,
        result_path=variant_result_path,
        timeout=600,
    )

    row: Dict[str, Any] = {
        "variant": variant_name,
        "robust_type": robust_type,
        "outlier_rejection_passes": max_passes,
        "solver_success": solver_result.success,
    }

    if solver_result.success and variant_result_path.exists():
        result_data = load_json(variant_result_path)
        summary = result_data.get("summary", {})

        row["rmse_reproj_px"] = round(summary.get("rmse_reproj_px", 0), 4)
        row["mean_reproj_px"] = round(summary.get("mean_reproj_px", 0), 4)
        row["median_reproj_px"] = round(summary.get("median_reproj_px", 0), 4)
        row["p95_reproj_px"] = round(summary.get("p95_reproj_px", 0), 4)
        row["num_inliers"] = summary.get("num_inliers", 0)
        row["num_outliers"] = summary.get("num_outliers", 0)
        row["iterations"] = summary.get("iterations", 0)

        num_obs = summary.get("num_observations", 0)
        row["inlier_ratio"] = round(
            row["num_inliers"] / num_obs if num_obs > 0 else 0.0, 4
        )

        # Evaluate against ground truth
        gt_eval = evaluate_against_gt(variant_result_path, gt_path)
        gt_summary = gt_eval.get("summary", {})
        row["mean_extr_rot_err_deg"] = gt_summary.get("mean_extr_rot_err_deg", "")
        row["mean_extr_trans_err_m"] = gt_summary.get("mean_extr_trans_err_m", "")
        row["mean_pose_rot_err_deg"] = gt_summary.get("mean_pose_rot_err_deg", "")
        row["mean_pose_trans_err_m"] = gt_summary.get("mean_pose_trans_err_m", "")
        row["mean_landmark_err_m"] = gt_summary.get("mean_landmark_err_m", "")

        print(f"      RMSE={row['rmse_reproj_px']} px, "
              f"inliers={row['inlier_ratio']*100:.1f}%, "
              f"extr_rot={row.get('mean_extr_rot_err_deg', '?')}")
    else:
        print(f"      FAILED (exit code {solver_result.exit_code})")

    return row


# ---------------------------------------------------------------------------
# Summary row builder
# ---------------------------------------------------------------------------


def _build_summary_row(
    experiment_name: str,
    config: ExperimentConfig,
    summary_dict: Dict,
    solver_elapsed_s: float = 0.0,
) -> Dict[str, Any]:
    """Build one row for the aggregated summary table."""
    gt = summary_dict.get("gt_evaluation", {})
    stats = summary_dict.get("statistics", {})

    # Use C++ solver's RMSE (excludes pruned observations)
    rmse = ""
    run_dir = _find_run_dir(summary_dict)
    if run_dir:
        result_path = run_dir / "result.json"
        if result_path.exists():
            result_data = load_json(result_path)
            solver_summary = result_data.get("summary", {})
            rmse = solver_summary.get("rmse_reproj_px", "")

    if not rmse:
        reproj = stats.get("reprojection_error", {})
        rmse = reproj.get("rmse", reproj.get("mean", ""))

    return {
        "experiment": experiment_name,
        "camera_model": config.data.camera_model,
        "num_cameras": config.data.num_cameras,
        "scene_type": config.data.scene_type,
        "mean_extr_rot_err_deg": gt.get("mean_extr_rot_err_deg", ""),
        "mean_extr_trans_err_m": gt.get("mean_extr_trans_err_m", ""),
        "mean_intr_err_px": gt.get("mean_intrinsic_err_px", ""),
        "mean_dist_err": gt.get("mean_distortion_err", ""),
        "rmse_reproj_px": rmse,
        "mean_pose_rot_err_deg": gt.get("mean_pose_rot_err_deg", ""),
        "mean_pose_trans_err_m": gt.get("mean_pose_trans_err_m", ""),
        "mean_landmark_err_m": gt.get("mean_landmark_err_m", ""),
        "solver_time_s": round(solver_elapsed_s, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict] = []

    print("=" * 70)
    print("SYNTHETIC VALIDATION EXPERIMENTS")
    print("=" * 70)

    # ── Experiment 0: Baseline ─────────────────────────────────────────
    print("\n" + "-" * 70)
    print("EXPERIMENT 0: Single-Camera Baseline (poses + landmarks only)")
    print("-" * 70)

    exp0_config = _exp0_baseline()
    exp0_summary = run_experiment(exp0_config)

    summary_rows.append(_build_summary_row(
        "baseline", exp0_config, exp0_summary,
        solver_elapsed_s=exp0_summary.get("solver", {}).get("elapsed_seconds", 0),
    ))

    # ── Experiment 1: Intrinsics Recovery ──────────────────────────────
    print("\n" + "-" * 70)
    print("EXPERIMENT 1: Intrinsics Recovery (single-camera)")
    print("-" * 70)

    exp1_config = _exp1_intrinsics_recovery()
    exp1_summary = run_experiment(exp1_config)

    gt_eval_1 = exp1_summary.get("gt_evaluation", {})
    _write_intrinsics_csv(
        # The runner's summary only has the summary sub-dict of gt_evaluation.
        # Re-evaluate from the run directory to get full per-camera data.
        _get_full_gt_eval(exp1_summary),
        TABLES_DIR / "synth_intrinsics_recovery.csv",
    )
    summary_rows.append(_build_summary_row(
        "intrinsics_recovery", exp1_config, exp1_summary,
        solver_elapsed_s=exp1_summary.get("solver", {}).get("elapsed_seconds", 0),
    ))

    # ── Experiment 2: Joint Recovery ──────────────────────────────────
    print("\n" + "-" * 70)
    print("EXPERIMENT 2: Joint Intrinsics + Extrinsics + Distortion Recovery (4-camera)")
    print("-" * 70)

    exp2_config = _exp2_joint_recovery()
    exp2_summary = run_experiment(exp2_config)

    _write_joint_csv(
        _get_full_gt_eval(exp2_summary),
        TABLES_DIR / "synth_joint_recovery.csv",
    )
    summary_rows.append(_build_summary_row(
        "joint_recovery", exp2_config, exp2_summary,
        solver_elapsed_s=exp2_summary.get("solver", {}).get("elapsed_seconds", 0),
    ))

    # ── Experiment 3: Outlier Robustness Comparison ───────────────────
    print("\n" + "-" * 70)
    print("EXPERIMENT 3: Outlier Robustness Comparison (L2 vs Huber vs Huber+Rejection)")
    print("-" * 70)

    # Step 1: generate data once using the framework
    exp3_config = _exp3_outlier_base_config()
    exp3_summary = run_experiment(exp3_config)

    # Find the run directory from the summary
    run_dir = _find_run_dir(exp3_summary)
    if run_dir is None:
        print("  [ERROR] Cannot locate experiment 3 run directory, skipping variants")
    else:
        problem_path = run_dir / "problem.json"
        gt_path = run_dir / "ground_truth.json"

        # Step 2: run three variants on the same data
        variants = [
            ("L2", "None", 1.0, 0),
            ("Huber", "Huber", 1.0, 0),
            ("Huber_Rejection", "Huber", 1.0, 3),
        ]

        variant_results = []
        for vname, rtype, rscale, passes in variants:
            vr = _run_outlier_variant(
                variant_name=vname,
                problem_path=problem_path,
                result_path=run_dir / "result.json",  # parent dir for variant files
                gt_path=gt_path,
                robust_type=rtype,
                robust_scale=rscale,
                max_passes=passes,
            )
            variant_results.append(vr)

        _write_outlier_csv(variant_results, TABLES_DIR / "synth_outlier_comparison.csv")

        # Add each variant as a summary row
        for vr in variant_results:
            summary_rows.append({
                "experiment": f"outlier_{vr['variant']}",
                "camera_model": exp3_config.data.camera_model,
                "num_cameras": exp3_config.data.num_cameras,
                "scene_type": exp3_config.data.scene_type,
                "mean_extr_rot_err_deg": vr.get("mean_extr_rot_err_deg", ""),
                "mean_extr_trans_err_m": vr.get("mean_extr_trans_err_m", ""),
                "mean_intr_err_px": "",
                "mean_dist_err": "",
                "rmse_reproj_px": vr.get("rmse_reproj_px", ""),
                "mean_pose_rot_err_deg": vr.get("mean_pose_rot_err_deg", ""),
                "mean_pose_trans_err_m": vr.get("mean_pose_trans_err_m", ""),
                "mean_landmark_err_m": vr.get("mean_landmark_err_m", ""),
                "solver_time_s": "",
            })

    # ── Experiment 4: Fisheye Joint Recovery ─────────────────────────
    print("\n" + "-" * 70)
    print("EXPERIMENT 4: Fisheye Joint Recovery (4-camera)")
    print("-" * 70)

    exp4_config = _exp4_fisheye_joint_recovery()
    exp4_summary = run_experiment(exp4_config)

    _write_joint_csv(
        _get_full_gt_eval(exp4_summary),
        TABLES_DIR / "synth_fisheye_joint_recovery.csv",
        fisheye=True,
    )
    summary_rows.append(_build_summary_row(
        "fisheye_joint_recovery", exp4_config, exp4_summary,
        solver_elapsed_s=exp4_summary.get("solver", {}).get("elapsed_seconds", 0),
    ))

    # ── Aggregated summary ────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("AGGREGATED SUMMARY")
    print("-" * 70)

    _write_summary_csv(summary_rows, TABLES_DIR / "synth_summary.csv")

    # Print table
    print(f"\n{'Experiment':<25} {'Cameras':<8} {'RMSE(px)':<10} "
          f"{'Extr Rot':<10} {'Intr Err':<10} {'Dist Err':<10}")
    print("-" * 73)
    for row in summary_rows:
        print(
            f"{row['experiment']:<25} "
            f"{row.get('num_cameras', '?'):<8} "
            f"{_fmt(row.get('rmse_reproj_px')):<10} "
            f"{_fmt(row.get('mean_extr_rot_err_deg')):<10} "
            f"{_fmt(row.get('mean_intr_err_px')):<10} "
            f"{_fmt(row.get('mean_dist_err')):<10}"
        )

    print("\n" + "=" * 70)
    print("SYNTHETIC EXPERIMENTS COMPLETE")
    print(f"  Tables in: {TABLES_DIR}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(val: Any) -> str:
    """Format a value for the summary table."""
    if val is None or val == "":
        return "-"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _get_full_gt_eval(summary_dict: Dict) -> Dict:
    """Re-load the full gt_evaluation.json from the run directory.

    The runner's summary.yaml only stores the 'summary' sub-dict of gt_evaluation.
    We need the full per-camera breakdown for CSV output.
    """
    run_dir = _find_run_dir(summary_dict)
    if run_dir is None:
        return {}
    eval_path = run_dir / "gt_evaluation.json"
    if eval_path.exists():
        return load_json(eval_path)
    return {}


def _find_run_dir(summary_dict: Dict) -> Path:
    """Find the run directory from a summary dict returned by run_experiment().

    The summary contains an 'artifacts' list whose entries are filenames relative
    to the run directory. We can find the run directory by looking for the
    config.yaml path in the config dict.
    """
    # The config.name gives us the experiment name prefix used in the directory
    config = summary_dict.get("config", {})
    exp_name = config.get("name", "")

    # Search in the runs/ directory for the most recent matching directory
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    runs_dir = repo_root / "runs"

    if not runs_dir.exists():
        return None

    candidates = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(exp_name)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if candidates:
        return candidates[0]
    return None


if __name__ == "__main__":
    main()
