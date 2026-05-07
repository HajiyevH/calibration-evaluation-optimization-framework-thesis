"""Wraps plot_errors, overlay_frame, and utils for headless analysis."""

from pathlib import Path
from typing import Dict, List, Optional, Union

# Force headless backend before any matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibri.tools import utils as ba_utils
from calibri.tools.analysis import plot_errors
from calibri.tools.analysis import overlay_frame as overlay_mod


def generate_error_plot(result_path: Path, output_path: Path) -> Path:
    """Generate the 2x2 diagnostic plot and save as PNG.

    Panels:
      [0,0] Per-frame reprojection mean/P95 bars
      [0,1] Residual error histogram
      [1,0] du vs dv residual scatter / hexbin
      [1,1] Convergence curve (or summary text if unavailable)

    Returns the output path.
    """
    result_path = Path(result_path)
    output_path = Path(output_path)

    result = ba_utils.load_result(result_path)
    residuals = ba_utils.get_residuals_array(result)
    per_frame = ba_utils.get_per_frame_errors(result)
    convergence = result.get("convergence_curve", [])

    if len(residuals["err"]) == 0:
        return output_path  # nothing to plot

    summary = result.get("summary", {})
    stats = {
        "mean": summary.get("mean_reproj_px", 0.0),
        "median": summary.get("median_reproj_px", 0.0),
        "p95": summary.get("p95_reproj_px", 0.0),
        "max": summary.get("max_reproj_px", 0.0),
        "count": summary.get("num_observations", 0),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_errors.plot_per_frame_bars(per_frame, axes[0, 0])
    plot_errors.plot_error_histogram(residuals["err"], axes[0, 1], stats=stats)
    plot_errors.plot_residual_scatter(residuals["du"], residuals["dv"], axes[1, 0])

    if convergence:
        plot_errors.plot_convergence_curve(convergence, axes[1, 1])
    else:
        ax_text = axes[1, 1]
        ax_text.axis("off")
        info_lines = [
            f"Mean:     {stats['mean']:.3f} px",
            f"Median:   {stats['median']:.3f} px",
            f"P95:      {stats['p95']:.3f} px",
            f"Max:      {stats['max']:.3f} px",
            f"Count:    {stats['count']}",
        ]
        ax_text.text(0.1, 0.5, "\n".join(info_lines), transform=ax_text.transAxes,
                     fontsize=12, verticalalignment="center", fontfamily="monospace")
        ax_text.set_title("Summary")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)

    return output_path


def _resolve_frame_indices(result: dict, frame_specs: List[Union[int, str]]) -> List[int]:
    """Resolve frame specs (int, 'mid', 'last') to concrete 0-based indices."""
    residuals = result.get("residuals", [])
    timestamps = sorted(set(r["timestamp_ns"] for r in residuals))
    n = len(timestamps)
    if n == 0:
        return []

    indices = []
    for spec in frame_specs:
        if isinstance(spec, int):
            if 0 <= spec < n:
                indices.append(spec)
        elif spec == "mid":
            indices.append(n // 2)
        elif spec == "last":
            indices.append(n - 1)
    return sorted(set(indices))


def generate_overlay(
    result_path: Path,
    output_dir: Path,
    frame_specs: List[Union[int, str]],
) -> List[Path]:
    """Generate overlay PNGs for the specified frames.

    Returns list of paths to generated PNGs.
    """
    result_path = Path(result_path)
    output_dir = Path(output_dir)

    result = ba_utils.load_result(result_path)
    residuals = result.get("residuals", [])
    timestamps = sorted(set(r["timestamp_ns"] for r in residuals))

    if not timestamps:
        return []

    indices = _resolve_frame_indices(result, frame_specs)
    generated = []

    for idx in indices:
        ts = timestamps[idx]
        out_path = output_dir / f"overlay_frame_{idx}.png"

        fig, ax = plt.subplots(figsize=(10, 6))
        ax = overlay_mod.overlay_frame(result, ts, ax=ax)
        if ax is not None:
            fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
            generated.append(out_path)
        plt.close(fig)

    return generated


def compute_run_statistics(result_path: Path) -> Dict:
    """Compute summary statistics from a result.json.

    Returns dict with error stats, observation count, and solver info.
    Solver metadata lives under result["summary"] in the C++ output.
    """
    result_path = Path(result_path)
    result = ba_utils.load_result(result_path)
    residuals = ba_utils.get_residuals_array(result)

    stats = ba_utils.compute_error_stats(residuals["err"])

    # Solver metadata is nested under "summary" in result.json
    solver_summary = result.get("summary", {})

    return {
        "reprojection_error": stats,
        "solver_success": solver_summary.get("success", False),
        "iterations": solver_summary.get("iterations", 0),
        "initial_cost": solver_summary.get("initial_cost", None),
        "final_cost": solver_summary.get("final_cost", None),
        "num_observations": len(residuals["err"]),
    }
