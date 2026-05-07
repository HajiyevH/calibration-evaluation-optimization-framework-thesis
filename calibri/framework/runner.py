"""Experiment orchestrator -- end-to-end run from config to summary."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)

from .config import ExperimentConfig, config_to_dict
from .data_gen import generate_synthetic_data
from .solver import run_solver
from .analysis import generate_error_plot, generate_overlay, compute_run_statistics
from .evaluate import evaluate_against_gt, format_evaluation_text
from .report import write_summary, format_summary_text


def run_experiment(
    config: ExperimentConfig,
    base_dir: Optional[Path] = None,
    cli_path: Optional[Path] = None,
) -> Dict:
    """Run a complete experiment: generate data, solve, analyze, evaluate, report.

    Args:
        config: Validated experiment config.
        base_dir: Parent directory for run output (default: <repo>/runs/).
        cli_path: Explicit path to ba_cli.exe (auto-detected if None).

    Returns:
        Summary dict with all results.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    if base_dir is None:
        base_dir = repo_root / "runs"

    # 1. Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{config.name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run directory: %s", run_dir)

    # 2. Save frozen config
    config_dict = config_to_dict(config)
    frozen_config_path = run_dir / "config.yaml"
    with open(frozen_config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    artifacts = ["config.yaml"]

    # 3. Generate synthetic data
    logger.info("Generating synthetic data...")
    problem_path, gt_path = generate_synthetic_data(
        output_dir=run_dir,
        data_config=config.data,
        solver_flags=config.solver.flags,
        robust_config=config.solver.robust,
    )
    artifacts.extend(["problem.json", "ground_truth.json"])
    logger.info("  problem.json: %s", problem_path)
    logger.info("  ground_truth.json: %s", gt_path)

    # 4. Run solver
    result_path = run_dir / "result.json"
    logger.info("Running solver...")
    solver_result = run_solver(
        problem_path=problem_path,
        result_path=result_path,
        cli_path=cli_path,
        timeout=config.solver.timeout,
    )

    # Save stdout/stderr logs
    stdout_log = run_dir / "cpp_stdout.log"
    stderr_log = run_dir / "cpp_stderr.log"
    stdout_log.write_text(solver_result.stdout)
    stderr_log.write_text(solver_result.stderr)
    artifacts.extend(["result.json", "cpp_stdout.log", "cpp_stderr.log"])

    if solver_result.success:
        logger.info("  Solver succeeded (exit code %d)", solver_result.exit_code)
    else:
        logger.error("  Solver FAILED (exit code %d)", solver_result.exit_code)
        if solver_result.stderr:
            logger.error("  stderr: %s", solver_result.stderr[:500])

    # 5. Analysis (only if solver succeeded and result exists)
    statistics = {}
    gt_evaluation = {}
    if solver_result.success and result_path.exists():
        statistics = compute_run_statistics(result_path)

        if config.analysis.plot_errors:
            logger.info("Generating error plots...")
            plot_path = run_dir / "plot_errors.png"
            generate_error_plot(result_path, plot_path)
            if plot_path.exists():
                artifacts.append("plot_errors.png")

        if config.analysis.overlay_frames:
            logger.info("Generating overlay frames...")
            overlay_paths = generate_overlay(
                result_path, run_dir, config.analysis.overlay_frames
            )
            for p in overlay_paths:
                artifacts.append(p.name)

        # 6. GT evaluation
        if config.analysis.evaluate_gt and gt_path.exists():
            logger.info("Evaluating against ground truth...")
            gt_evaluation = evaluate_against_gt(result_path, gt_path)
            # Save detailed evaluation
            eval_path = run_dir / "gt_evaluation.json"
            import json
            with open(eval_path, "w") as f:
                json.dump(gt_evaluation, f, indent=2)
            artifacts.append("gt_evaluation.json")
            logger.info("  %s", format_evaluation_text(gt_evaluation).replace("\n", "\n  "))

    # 7. Write summary
    artifacts.append("summary.yaml")
    summary_path = write_summary(
        run_dir=run_dir,
        config_dict=config_dict,
        solver_result=solver_result,
        statistics=statistics,
        artifacts=artifacts,
        gt_evaluation=gt_evaluation,
    )

    # 8. Log summary if configured
    if config.analysis.print_summary:
        logger.info("")
        logger.info("=" * 60)
        logger.info(format_summary_text(summary_path))
        logger.info("=" * 60)

    logger.info("Run complete: %s", run_dir)

    # Return summary for programmatic use
    with open(summary_path) as f:
        return yaml.safe_load(f)
