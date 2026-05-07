"""Plot reprojection error analysis from result.json.

Usage:
    python plot_errors.py <result.json>

Produces:
    1. Per-frame mean reprojection error bar chart
    2. Residual error histogram (clipped to p99 for readability)
    3. du vs dv scatter / density plot (clipped to p99 for readability)
    4. Solver convergence curve (if convergence_curve present)
"""

import argparse
import math
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import numpy as np

from calibri.tools.utils import load_result, compute_error_stats, get_residuals_array, get_per_frame_errors

# ---------------------------------------------------------------------------
# Style configuration — thesis-quality defaults
# ---------------------------------------------------------------------------
STYLE_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.titlesize": 16,
    "axes.titleweight": "semibold",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "#cccccc",
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Calibri", "DejaVu Sans", "Arial"],
}

# Colour palette (colour-blind-safe tones)
C_PRIMARY = "#3574b0"       # blue — main data
C_SECONDARY = "#e8873a"     # orange — secondary data / P95
C_ACCENT_RED = "#d44a3f"    # red — mean line
C_ACCENT_GREEN = "#4a9a5b"  # green — median line
C_GRADIENT = "#6a4c93"      # purple — gradient norm


def _apply_style():
    """Apply the custom rcParams once."""
    plt.rcParams.update(STYLE_RC)


def _format_suptitle(stats, summary):
    """Build a one-line figure super-title with key metrics."""
    parts = []
    parts.append(f"Mean {stats['mean']:.2f} px")
    parts.append(f"Median {stats['median']:.2f} px")
    parts.append(f"P95 {stats['p95']:.2f} px")
    if "rmse_reproj_px" in summary:
        parts.append(f"RMSE {summary['rmse_reproj_px']:.2f} px")
    if "std_reproj_px" in summary:
        parts.append(f"Std {summary['std_reproj_px']:.2f} px")
    parts.append(f"n={stats['count']}")
    if "total_time_ms" in summary:
        parts.append(f"{summary['total_time_ms']:.0f} ms")
    if "termination_type" in summary:
        parts.append(summary["termination_type"])
    return "  |  ".join(parts)


def plot_per_frame_bars(per_frame, ax):
    """Plot per-frame mean reprojection error as a bar chart."""
    if len(per_frame) == 0:
        ax.text(0.5, 0.5, "No per-frame data available",
                ha='center', va='center', transform=ax.transAxes, fontsize=12,
                color="#888888")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Reprojection error (px)")
        ax.set_title("Per-Frame Reprojection Error")
        return ax

    frames = sorted(per_frame, key=lambda f: f["timestamp_ns"])
    means = np.array([f["mean_px"] if (f["mean_px"] is not None and math.isfinite(f["mean_px"])) else 0 for f in frames])
    p95s = np.array([f["p95_px"] if (f["p95_px"] is not None and math.isfinite(f["p95_px"])) else 0 for f in frames])

    x = np.arange(len(frames))
    width = 1.0 if len(frames) > 30 else 0.7

    ax.bar(x, p95s, width=width, label="P95", alpha=0.30, color=C_SECONDARY,
           edgecolor="none")
    ax.bar(x, means, width=width, label="Mean", alpha=0.85, color=C_PRIMARY,
           edgecolor="none")

    # Overall mean line
    overall_mean = float(np.mean(means[means > 0])) if np.any(means > 0) else 0
    if overall_mean > 0:
        ax.axhline(overall_mean, color=C_ACCENT_RED, linewidth=0.8,
                   linestyle=":", alpha=0.7,
                   label=f"Overall mean: {overall_mean:.2f} px")

    ax.set_ylabel("Reprojection error (px)")
    ax.set_title(f"Per-Frame Reprojection Error  ({len(frames)} frames)")

    if len(frames) <= 20:
        ax.set_xticks(x)
        ax.set_xticklabels([str(f["timestamp_ns"]) for f in frames],
                           rotation=45, ha="right", fontsize=10)
        ax.set_xlabel("Frame (timestamp)")
    else:
        step = max(1, len(frames) // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([str(i) for i in x[::step]])
        ax.set_xlabel("Frame index")

    ax.legend(loc="upper right")
    ax.set_xlim(-0.5, len(frames) - 0.5)
    return ax


def plot_error_histogram(errors, ax, stats=None, bins=60):
    """Plot a histogram of reprojection errors, clipped at p99 for readability.

    Args:
        errors: Array of error values to plot
        ax: Matplotlib axis
        stats: Optional dict with pre-computed mean/median/p95/max from solver
        bins: Number of histogram bins
    """
    valid_errors = errors[np.isfinite(errors)]

    if len(valid_errors) == 0:
        ax.text(0.5, 0.5, "No valid errors (BA diverged)",
                ha='center', va='center', transform=ax.transAxes, fontsize=12,
                color="#888888")
        ax.set_xlabel("Reprojection error (px)")
        ax.set_ylabel("Count")
        ax.set_title("Residual Error Distribution (DIVERGED)")
        return ax

    # Use solver-computed stats if provided, otherwise compute from data
    if stats is None:
        stats = compute_error_stats(valid_errors)

    p99 = float(np.percentile(valid_errors, 99))
    clip_max = max(p99, stats["p95"] * 1.5)
    clipped = valid_errors[valid_errors <= clip_max]
    n_outliers = len(valid_errors) - len(clipped)

    ax.hist(clipped, bins=bins, alpha=0.75, color=C_PRIMARY,
            edgecolor="white", linewidth=0.4)
    ax.axvline(stats["mean"], color=C_ACCENT_RED, linestyle="--", linewidth=1.3,
               label=f"Mean: {stats['mean']:.2f} px")
    ax.axvline(stats["median"], color=C_ACCENT_GREEN, linestyle="--", linewidth=1.3,
               label=f"Median: {stats['median']:.2f} px")
    ax.axvline(stats["p95"], color=C_SECONDARY, linestyle="--", linewidth=1.3,
               label=f"P95: {stats['p95']:.2f} px")

    ax.set_xlabel("Reprojection error (px)")
    ax.set_ylabel("Count")
    title = "Residual Error Distribution"
    if n_outliers > 0:
        title += f"  ({n_outliers} outliers > {clip_max:.1f} px hidden)"
    ax.set_title(title)
    ax.legend(loc="upper right")
    return ax


def plot_residual_scatter(du, dv, ax):
    """Plot du vs dv as a density hexbin for large datasets, scatter for small."""
    valid_mask = np.isfinite(du) & np.isfinite(dv)
    du_valid = du[valid_mask]
    dv_valid = dv[valid_mask]

    if len(du_valid) == 0:
        ax.text(0.5, 0.5, "No valid residuals (BA diverged)",
                ha='center', va='center', transform=ax.transAxes, fontsize=12,
                color="#888888")
        ax.set_xlabel("du (px)")
        ax.set_ylabel("dv (px)")
        ax.set_title("Residual Scatter (DIVERGED)")
        return ax

    abs_vals = np.sqrt(du_valid**2 + dv_valid**2)
    p99 = float(np.percentile(abs_vals, 99))
    clip_radius = max(p99 * 1.3, 0.5)
    inlier_mask = abs_vals <= clip_radius
    n_hidden = int(np.sum(~inlier_mask))

    du_plot = du_valid[inlier_mask]
    dv_plot = dv_valid[inlier_mask]

    # Use hexbin density for large datasets, plain scatter for small
    if len(du_plot) > 2000:
        gridsize = min(60, max(25, int(np.sqrt(len(du_plot)) / 3)))
        hb = ax.hexbin(du_plot, dv_plot, gridsize=gridsize,
                        cmap="Blues", mincnt=1,
                        extent=[-clip_radius, clip_radius,
                                -clip_radius, clip_radius],
                        linewidths=0.2, edgecolors="white")
        cb = plt.colorbar(hb, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label("Count", fontsize=12)
        cb.ax.tick_params(labelsize=10)
    else:
        ax.scatter(du_plot, dv_plot, s=4, alpha=0.4, color=C_PRIMARY,
                   edgecolors="none", rasterized=True)

    ax.set_xlabel("$\\Delta u$ (px)")
    ax.set_ylabel("$\\Delta v$ (px)")
    title = "Residual Scatter ($\\Delta u$ vs $\\Delta v$)"
    if n_hidden > 0:
        title += f"\n({n_hidden} outliers hidden)"
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-clip_radius, clip_radius)
    ax.set_ylim(-clip_radius, clip_radius)
    ax.axhline(0, color="#999999", linewidth=0.5)
    ax.axvline(0, color="#999999", linewidth=0.5)
    return ax


def plot_convergence_curve(convergence, ax):
    """Plot per-iteration cost and gradient norm from convergence_curve data.

    Only accepted iterations (non-negative relative_decrease) are plotted.
    Rejected LM steps produce cost spikes that obscure the actual convergence
    trend and make the plot unreadable.
    """
    if not convergence:
        ax.text(0.5, 0.5, "No convergence data available",
                ha='center', va='center', transform=ax.transAxes, fontsize=12,
                color="#888888")
        ax.set_title("Solver Convergence")
        return ax

    # Filter to accepted steps only (relative_decrease >= 0 or iteration 0)
    accepted = [r for r in convergence
                if r.get("relative_decrease", 0) >= 0 or r["iteration"] == 0]
    if not accepted:
        accepted = convergence  # fallback if no relative_decrease field

    iters = list(range(len(accepted)))  # re-index for clean x-axis
    costs = [r["cost"] for r in accepted]
    grad_norms = [r["gradient_norm"] for r in accepted]

    ax.semilogy(iters, costs, 'o-', color=C_PRIMARY, markersize=4,
                linewidth=1.5, label="Cost", markeredgecolor="white",
                markeredgewidth=0.3)
    ax.set_xlabel("Iteration (accepted)")
    ax.set_ylabel("Cost (log scale)", color=C_PRIMARY)
    ax.tick_params(axis='y', labelcolor=C_PRIMARY)
    ax.set_title("Solver Convergence")

    ax2 = ax.twinx()
    ax2.semilogy(iters, grad_norms, 's--', color=C_GRADIENT, markersize=3,
                 alpha=0.8, linewidth=1.2, label="Gradient norm",
                 markeredgecolor="white", markeredgewidth=0.3)
    ax2.set_ylabel("Gradient norm (log scale)", color=C_GRADIENT)
    ax2.tick_params(axis='y', labelcolor=C_GRADIENT)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Integer ticks on x-axis
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    return ax


def main():
    parser = argparse.ArgumentParser(description="Plot BA result errors")
    parser.add_argument("result_json", help="Path to result.json")
    parser.add_argument("--save", default=None,
                        help="Save figure to path instead of showing")
    args = parser.parse_args()

    _apply_style()

    result = load_result(args.result_json)
    residuals = get_residuals_array(result)

    # Filter to inlier observations only — outlier-rejected observations
    # (including behind-camera penalties with 1000+ px errors) distort axis
    # limits and make the diagnostic plots unreadable.
    raw_residuals = result.get("residuals", [])
    if raw_residuals and "inlier" in raw_residuals[0]:
        inlier_mask = np.array([r.get("inlier", True) for r in raw_residuals])
        for key in residuals:
            if isinstance(residuals[key], np.ndarray) and len(residuals[key]) == len(inlier_mask):
                residuals[key] = residuals[key][inlier_mask]
        # Keep only inlier raw_residuals for subsequent filtering
        raw_residuals = [r for r, m in zip(raw_residuals, inlier_mask) if m]

    # Filter out residuals from landmarks with only 2 observations.
    # A 3-DOF landmark with 4 constraints (2 obs × 2D) is nearly
    # unconstrained; the solver drives their errors to near-zero,
    # creating a degenerate spike at bin 0 of the histogram.
    if raw_residuals and "landmark_id" in raw_residuals[0]:
        from collections import Counter
        obs_count = Counter(r["landmark_id"] for r in raw_residuals)
        well_obs_mask = np.array([obs_count[r["landmark_id"]] >= 3
                                  for r in raw_residuals])
        n_removed = int(np.sum(~well_obs_mask))
        if n_removed > 0:
            print(f"  Excluded {n_removed} residuals from 2-observation landmarks")
            for key in residuals:
                if isinstance(residuals[key], np.ndarray) and len(residuals[key]) == len(well_obs_mask):
                    residuals[key] = residuals[key][well_obs_mask]

    per_frame = get_per_frame_errors(result)
    convergence = result.get("convergence_curve", [])

    if len(residuals["err"]) == 0:
        print("No residuals found in result.json")
        sys.exit(1)

    # Use statistics from C++ solver (already excludes pruned observations)
    summary = result.get("summary", {})
    stats = {
        "mean": summary.get("mean_reproj_px", 0.0),
        "median": summary.get("median_reproj_px", 0.0),
        "p95": summary.get("p95_reproj_px", 0.0),
        "max": summary.get("max_reproj_px", 0.0),
        "count": summary.get("num_observations", 0)
    }
    print(f"Residual stats: mean={stats['mean']:.4f} median={stats['median']:.4f} "
          f"p95={stats['p95']:.4f} max={stats['max']:.4f} n={stats['count']}")
    if "rmse_reproj_px" in summary:
        print(f"  RMSE={summary['rmse_reproj_px']:.4f}  "
              f"std={summary.get('std_reproj_px', 0):.4f}")
    if "total_time_ms" in summary:
        print(f"  Time={summary['total_time_ms']:.1f} ms  "
              f"iterations={summary.get('iterations', '?')}  "
              f"termination={summary.get('termination_type', '?')}")

    # --- Layout: 2x2 grid ---
    has_convergence = len(convergence) > 0
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    plot_per_frame_bars(per_frame, axes[0, 0])
    plot_error_histogram(residuals["err"], axes[0, 1], stats=stats)
    plot_residual_scatter(residuals["du"], residuals["dv"], axes[1, 0])

    if has_convergence:
        plot_convergence_curve(convergence, axes[1, 1])
    else:
        # Summary text box when no convergence data
        ax_text = axes[1, 1]
        ax_text.axis("off")
        info_lines = [
            f"Mean:     {stats['mean']:.3f} px",
            f"Median:   {stats['median']:.3f} px",
            f"P95:      {stats['p95']:.3f} px",
            f"Max:      {stats['max']:.3f} px",
            f"Count:    {stats['count']}",
        ]
        if "rmse_reproj_px" in summary:
            info_lines.append(f"RMSE:     {summary['rmse_reproj_px']:.3f} px")
        if "total_time_ms" in summary:
            info_lines.append(f"Time:     {summary['total_time_ms']:.1f} ms")
        if "termination_type" in summary:
            info_lines.append(f"Status:   {summary['termination_type']}")
        if "iterations" in summary:
            info_lines.append(f"Iters:    {summary['iterations']}")
        ax_text.text(0.1, 0.95, "Solver Summary",
                     transform=ax_text.transAxes,
                     fontsize=13, fontweight="semibold", va="top")
        ax_text.text(0.1, 0.82, "\n".join(info_lines),
                     transform=ax_text.transAxes,
                     fontsize=10, va="top", family="monospace",
                     bbox=dict(boxstyle="round,pad=0.5",
                               facecolor="#f0f0f0", edgecolor="#cccccc",
                               alpha=0.9))

    # Super-title
    suptitle = _format_suptitle(stats, summary)
    fig.suptitle(suptitle, fontsize=13, color="#333333", y=0.99)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.save:
        fig.savefig(args.save, dpi=180, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
