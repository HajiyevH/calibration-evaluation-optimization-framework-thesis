#!/usr/bin/env python3
"""
End-to-end pipeline: COLMAP reconstruction → BA → Analysis

Usage:
    python pipelines/colmap_pipeline.py --reconstruction <dir> --database <db> --output-dir results/

Steps:
    1. Convert COLMAP reconstruction to problem.json
    2. Run BA solver (ba_cli.exe)
    3. Generate error plots (plot_errors.py)
    4. Create frame overlays (overlay_frame.py)
    5. Compare with COLMAP baseline (using pycolmap)
"""

import argparse
import subprocess
import logging
import json
from pathlib import Path
import sys
import numpy as np

from calibri.tools.utils import setup_logging, load_result

# Check for optional dependencies
try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False


def run_colmap_pipeline(
    reconstruction_path: Path,
    database_path: Path,
    output_dir: Path,
    use_rig: bool = False,
    max_iterations: int = 200,
    loss_type: str = "Huber",
    loss_scale: float = 1.0,
    skip_ba: bool = False,
    skip_plots: bool = False,
    optimize_intrinsics: bool = False,
    optimize_distortion: bool = False,
    optimize_extrinsics: bool = False,
    perturb_poses: float = 0.0,
    perturb_landmarks: float = 0.0,
    outlier_passes: int = 0,
    outlier_threshold: float = 0.0,
    verbose: bool = False
):
    """Run complete COLMAP → BA → Analysis pipeline."""

    output_dir.mkdir(parents=True, exist_ok=True)
    problem_path = output_dir / "problem.json"
    result_path = output_dir / "result.json"

    # Step 1: Convert COLMAP to problem.json
    logging.info("Step 1: Converting COLMAP reconstruction to problem.json...")
    converter_cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "converters" / "colmap_to_problem.py"),
        "--reconstruction", str(reconstruction_path),
        "--database", str(database_path),
        "--output", str(problem_path)
    ]
    if use_rig:
        converter_cmd.append("--use-rig")
    if optimize_intrinsics:
        converter_cmd.append("--optimize-intrinsics")
    if optimize_distortion:
        converter_cmd.append("--optimize-distortion")
    if optimize_extrinsics:
        converter_cmd.append("--optimize-extrinsics")
    if perturb_poses > 0.0:
        converter_cmd.extend(["--perturb-poses", str(perturb_poses)])
    if perturb_landmarks > 0.0:
        converter_cmd.extend(["--perturb-landmarks", str(perturb_landmarks)])
    if verbose:
        converter_cmd.append("-v")

    conv_result = subprocess.run(converter_cmd, capture_output=True, text=True)
    if conv_result.returncode != 0:
        logging.error("COLMAP converter failed (exit code %d):\n%s",
                       conv_result.returncode, conv_result.stderr)
        raise RuntimeError(f"COLMAP converter failed with exit code {conv_result.returncode}")
    if conv_result.stdout:
        print(conv_result.stdout, flush=True)
    logging.info(f"Created problem.json: {problem_path}")

    # Step 2: Run BA solver
    if not skip_ba:
        logging.info(f"Step 2: Running BA solver (max {max_iterations} iterations, loss={loss_type}, scale={loss_scale})...")
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        ba_cli = repo_root / "build" / "calibri" / "app" / "Release" / "ba_cli.exe"
        if not ba_cli.exists():
            raise FileNotFoundError(f"BA CLI not found at {ba_cli}. Build the project first.")

        cmd = [str(ba_cli), str(problem_path), str(result_path)]

        # Add solver parameter overrides
        if max_iterations:
            cmd.extend(["--max-iterations", str(max_iterations)])
        if loss_type:
            cmd.extend(["--loss-type", loss_type])
        if loss_scale:
            cmd.extend(["--loss-scale", str(loss_scale)])
        if outlier_passes > 0:
            cmd.extend(["--outlier-passes", str(outlier_passes)])
        if outlier_threshold > 0.0:
            cmd.extend(["--outlier-threshold", str(outlier_threshold)])

        ba_result = subprocess.run(cmd, capture_output=True, text=True)
        if ba_result.returncode != 0:
            logging.error("BA solver failed (exit code %d):\n%s",
                           ba_result.returncode, ba_result.stderr)
            raise RuntimeError(f"BA solver failed with exit code {ba_result.returncode}")
        if ba_result.stdout:
            print(ba_result.stdout, flush=True)
        logging.info(f"BA completed. Result: {result_path}")
    else:
        logging.info("Step 2: Skipping BA execution (--skip-ba)")

    # Step 3: Generate plots
    if not skip_plots and result_path.exists():
        logging.info("Step 3: Generating error plots...")

        # Error distribution plots
        plot_cmd = [
            sys.executable,
            str(Path(__file__).parent.parent / "analysis" / "plot_errors.py"),
            str(result_path),
            "--save", str(output_dir / "errors.png")
        ]
        plot_result = subprocess.run(plot_cmd, capture_output=True, text=True)
        if plot_result.returncode != 0:
            logging.warning("Plot generation failed (exit code %d):\n%s",
                            plot_result.returncode, plot_result.stderr)

        # Frame overlays (first 5 frames)
        result = load_result(result_path)
        num_frames = len(result.get('per_frame', []))
        for i in range(min(5, num_frames)):
            overlay_cmd = [
                sys.executable,
                str(Path(__file__).parent.parent / "analysis" / "overlay_frame.py"),
                str(result_path),
                "--frame-index", str(i),
                "--save", str(output_dir / f"overlay_frame_{i}.png")
            ]
            overlay_result = subprocess.run(overlay_cmd, capture_output=True, text=True)
            if overlay_result.returncode != 0:
                logging.warning("Overlay frame %d failed (exit code %d):\n%s",
                                i, overlay_result.returncode, overlay_result.stderr)
        logging.info(f"Generated {min(5, num_frames)} frame overlays")
    else:
        logging.info("Step 3: Skipping plots (--skip-plots or no result.json)")

    # Step 4: Compare with COLMAP baseline (using pycolmap)
    if HAS_PYCOLMAP and result_path.exists():
        logging.info("Step 4: Comparing with COLMAP baseline...")
        try:
            compare_with_colmap(
                reconstruction_path=reconstruction_path,
                result=load_result(result_path),
                output_path=output_dir / "colmap_comparison.json"
            )
        except Exception as e:
            logging.warning("COLMAP comparison failed (non-fatal): %s", e)
    elif not HAS_PYCOLMAP:
        logging.warning("Step 4: Skipping COLMAP comparison (pycolmap not installed)")
    else:
        logging.info("Step 4: Skipping COLMAP comparison (no result.json)")

    logging.info(f"Pipeline complete. Results in {output_dir}")


def compare_with_colmap(reconstruction_path: Path, result: dict, output_path: Path):
    """Compare BA result with COLMAP reconstruction errors.

    NOTE: COLMAP point3D.error is the mean reprojection error per 3D point
    (averaged across its track), while BA residuals are per-observation.
    These are not directly comparable. The comparison is approximate and
    should be interpreted with caution.
    """
    # Handle COLMAP numbered sub-model directory (e.g. 0/)
    sub0 = reconstruction_path / "0"
    recon_dir = sub0 if sub0.is_dir() and (sub0 / "cameras.bin").exists() else reconstruction_path
    reconstruction = pycolmap.Reconstruction(str(recon_dir))

    # Compute COLMAP per-point track errors
    colmap_errors = []
    for point3D_id, point3D in reconstruction.points3D.items():
        if point3D.error >= 0:
            colmap_errors.append(point3D.error)

    # Get our BA per-observation errors
    ba_errors = [obs['err'] for obs in result.get('residuals', [])
                 if obs.get('err') is not None and np.isfinite(obs['err'])]

    if not colmap_errors or not ba_errors:
        logging.warning("No errors to compare")
        return

    # Get BA summary values with null safety
    ba_mean = result.get('summary', {}).get('mean_reproj_px')
    ba_median = result.get('summary', {}).get('median_reproj_px')

    # Detect solver divergence
    diverged = (ba_mean is None or ba_median is None
                or not np.isfinite(ba_mean) or not np.isfinite(ba_median)
                or ba_median > 100.0)

    # Use computed values if summary is null
    if ba_mean is None or not np.isfinite(ba_mean):
        ba_mean = float(np.mean(ba_errors)) if ba_errors else None
    if ba_median is None or not np.isfinite(ba_median):
        ba_median = float(np.median(ba_errors)) if ba_errors else None

    comparison = {
        "note": "COLMAP errors are per-3D-point track averages; BA errors are per-observation. Not directly comparable.",
        "colmap": {
            "num_points": len(colmap_errors),
            "mean_px": float(np.mean(colmap_errors)),
            "median_px": float(np.median(colmap_errors))
        },
        "ba": {
            "num_residuals": len(ba_errors),
            "mean_px": ba_mean,
            "median_px": ba_median,
            "diverged": diverged
        }
    }

    # Calculate approximate improvement (with caveats)
    colmap_median = comparison['colmap']['median_px']
    if colmap_median > 0 and ba_median is not None and np.isfinite(ba_median):
        comparison['improvement_pct'] = (1 - ba_median / colmap_median) * 100
    else:
        comparison['improvement_pct'] = None

    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    if diverged:
        logging.warning(f"BA solver appears to have diverged (median={ba_median} px)")
    logging.info(f"COLMAP baseline (per-point): {comparison['colmap']['median_px']:.2f} px")
    logging.info(f"BA result (per-observation): {ba_median:.2f} px" if ba_median else "BA result: N/A")
    if comparison['improvement_pct'] is not None:
        logging.info(f"Approximate improvement: {comparison['improvement_pct']:.1f}% (caveat: different error definitions)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="COLMAP -> BA -> Analysis pipeline")
    parser.add_argument('--reconstruction', type=Path, required=True,
                       help="Path to COLMAP reconstruction directory")
    parser.add_argument('--database', type=Path, required=True,
                       help="Path to COLMAP database file")
    parser.add_argument('--output-dir', type=Path, default=Path('results'),
                       help="Output directory for results")
    parser.add_argument('--use-rig', action='store_true',
                       help="Use multi-camera rig mode")
    parser.add_argument('--max-iterations', type=int, default=200,
                       help="Maximum BA iterations")
    parser.add_argument('--loss-type', type=str, default='Huber',
                       choices=['Huber', 'Cauchy', 'SoftL1', 'None'],
                       help='Robust loss function type')
    parser.add_argument('--loss-scale', type=float, default=1.0,
                       help='Robust loss scale parameter')
    parser.add_argument('--outlier-passes', type=int, default=0,
                       help="Iterative outlier rejection passes (0=disabled)")
    parser.add_argument('--outlier-threshold', type=float, default=0.0,
                       help="Fixed outlier threshold in px (0=adaptive k*median)")
    parser.add_argument('--skip-ba', action='store_true',
                       help="Skip BA execution (analysis only)")
    parser.add_argument('--skip-plots', action='store_true',
                       help="Skip plot generation")

    # Option A: Refinement mode
    parser.add_argument('--optimize-intrinsics', action='store_true',
                       help="Optimize camera intrinsics (fx, fy, cx, cy)")
    parser.add_argument('--optimize-distortion', action='store_true',
                       help="Optimize distortion coefficients")
    parser.add_argument('--optimize-extrinsics', action='store_true',
                       help="Optimize camera extrinsics (T_vehicle_to_camera)")

    # Option B: Validation mode
    parser.add_argument('--perturb-poses', type=float, default=0.0, metavar='SIGMA',
                       help="Add Gaussian noise to poses (meters)")
    parser.add_argument('--perturb-landmarks', type=float, default=0.0, metavar='SIGMA',
                       help="Add Gaussian noise to landmarks (meters)")

    parser.add_argument('-v', '--verbose', action='store_true',
                       help="Verbose logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    run_colmap_pipeline(
        reconstruction_path=args.reconstruction,
        database_path=args.database,
        output_dir=args.output_dir,
        use_rig=args.use_rig,
        max_iterations=args.max_iterations,
        loss_type=args.loss_type,
        loss_scale=args.loss_scale,
        skip_ba=args.skip_ba,
        skip_plots=args.skip_plots,
        optimize_intrinsics=args.optimize_intrinsics,
        optimize_distortion=args.optimize_distortion,
        optimize_extrinsics=args.optimize_extrinsics,
        perturb_poses=args.perturb_poses,
        perturb_landmarks=args.perturb_landmarks,
        outlier_passes=args.outlier_passes,
        outlier_threshold=args.outlier_threshold,
        verbose=args.verbose
    )
