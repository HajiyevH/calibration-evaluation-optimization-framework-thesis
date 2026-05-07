"""Summary YAML generation for experiment runs."""

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .solver import SolverResult


def write_summary(
    run_dir: Path,
    config_dict: dict,
    solver_result: SolverResult,
    statistics: dict,
    artifacts: List[str],
    gt_evaluation: Optional[Dict] = None,
) -> Path:
    """Write summary.yaml to the run directory.

    Args:
        run_dir: Path to the run output directory.
        config_dict: Experiment config as a plain dict.
        solver_result: Result from the solver subprocess.
        statistics: Dict from compute_run_statistics().
        artifacts: List of artifact filenames produced.
        gt_evaluation: Optional GT evaluation results from evaluate_against_gt().

    Returns:
        Path to the written summary.yaml.
    """
    summary = {
        "experiment": config_dict.get("name", "unknown"),
        "description": config_dict.get("description", ""),
        "config": config_dict,
        "solver": {
            "success": solver_result.success,
            "exit_code": solver_result.exit_code,
            "elapsed_seconds": round(solver_result.elapsed_seconds, 3),
        },
        "statistics": statistics,
        "artifacts": artifacts,
    }

    if gt_evaluation:
        summary["gt_evaluation"] = gt_evaluation.get("summary", {})

    summary_path = run_dir / "summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    return summary_path


def format_summary_text(summary_path: Path) -> str:
    """Format a summary.yaml as human-readable text for console output."""
    with open(summary_path) as f:
        summary = yaml.safe_load(f)

    lines = []
    lines.append(f"Experiment: {summary.get('experiment', '?')}")
    if summary.get("description"):
        lines.append(f"  {summary['description']}")
    lines.append("")

    solver = summary.get("solver", {})
    status = "SUCCESS" if solver.get("success") else "FAILED"
    lines.append(f"Solver: {status} (exit code {solver.get('exit_code', '?')})")
    lines.append(f"  Elapsed: {solver.get('elapsed_seconds', '?')}s")
    lines.append("")

    stats = summary.get("statistics", {})
    reproj = stats.get("reprojection_error", {})
    if reproj:
        lines.append("Reprojection error (px):")
        lines.append(f"  Mean:   {reproj.get('mean', 0):.4f}")
        lines.append(f"  Median: {reproj.get('median', 0):.4f}")
        lines.append(f"  P95:    {reproj.get('p95', 0):.4f}")
        lines.append(f"  Max:    {reproj.get('max', 0):.4f}")
        lines.append(f"  Count:  {reproj.get('count', 0)}")
    lines.append("")

    lines.append(f"Iterations: {stats.get('iterations', '?')}")
    initial = stats.get("initial_cost")
    final = stats.get("final_cost")
    if initial is not None:
        lines.append(f"Initial cost: {initial:.6f}")
    if final is not None:
        lines.append(f"Final cost:   {final:.6f}")
    lines.append("")

    # GT evaluation summary
    gt = summary.get("gt_evaluation", {})
    if gt:
        lines.append("GT recovery:")
        if "mean_pose_rot_err_deg" in gt:
            lines.append(f"  Pose rot err:  {gt['mean_pose_rot_err_deg']:.4f} deg (mean)")
        if "mean_pose_trans_err_m" in gt:
            lines.append(f"  Pose trans err: {gt['mean_pose_trans_err_m']:.4f} m (mean)")
        if "mean_extr_rot_err_deg" in gt:
            lines.append(f"  Extr rot err:  {gt['mean_extr_rot_err_deg']:.4f} deg (mean)")
        if "mean_extr_trans_err_m" in gt:
            lines.append(f"  Extr trans err: {gt['mean_extr_trans_err_m']:.4f} m (mean)")
        if "mean_intrinsic_err_px" in gt:
            lines.append(f"  Intr err:      {gt['mean_intrinsic_err_px']:.4f} px (mean)")
        if "mean_landmark_err_m" in gt:
            lines.append(f"  Landmark err:  {gt['mean_landmark_err_m']:.4f} m (mean)")
        lines.append("")

    artifacts = summary.get("artifacts", [])
    if artifacts:
        lines.append(f"Artifacts ({len(artifacts)}):")
        for a in artifacts:
            lines.append(f"  - {a}")

    return "\n".join(lines)
