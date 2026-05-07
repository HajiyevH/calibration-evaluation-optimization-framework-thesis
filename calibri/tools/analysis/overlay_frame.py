"""Overlay measured vs reprojected points for a single frame.

Usage:
    python overlay_frame.py <result.json> --timestamp <timestamp_ns>
    python overlay_frame.py <result.json> --frame-index <0-based index>

Draws measured observations (blue dots) vs reprojected positions (red crosses)
on a blank canvas of the camera's image dimensions.
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from calibri.tools.utils import load_result, get_residuals_array


def overlay_frame(result, timestamp_ns, ax=None):
    """Plot measured vs reprojected points for one timestamp.

    Args:
        result: parsed result.json dict.
        timestamp_ns: the frame timestamp to filter on.
        ax: optional matplotlib Axes.

    Returns:
        The matplotlib Axes.
    """
    residuals = get_residuals_array(result)
    mask = residuals["timestamp_ns"] == timestamp_ns

    if not np.any(mask):
        print(f"No observations found for timestamp {timestamp_ns}")
        return None

    u = residuals["u"][mask]
    v = residuals["v"][mask]
    u_hat = residuals["u_hat"][mask]
    v_hat = residuals["v_hat"][mask]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))

    ax.scatter(u, v, c="blue", s=14, marker="o", label="Measured", alpha=0.6)
    ax.scatter(u_hat, v_hat, c="red", s=28, marker="x", label="Reprojected", alpha=0.6)

    # Draw lines connecting measured to reprojected
    for i in range(len(u)):
        ax.plot([u[i], u_hat[i]], [v[i], v_hat[i]], "gray", linewidth=0.3, alpha=0.4)

    ax.set_xlabel("u (px)", fontsize=16)
    ax.set_ylabel("v (px)", fontsize=16)
    ax.set_title(f"Frame {timestamp_ns}: Measured vs Reprojected ({np.sum(mask)} points)", fontsize=18)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=13)
    ax.invert_yaxis()  # image convention: y increases downward
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    return ax


def main():
    parser = argparse.ArgumentParser(description="Overlay measured vs reprojected for one frame")
    parser.add_argument("result_json", help="Path to result.json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--timestamp", type=int, help="Frame timestamp_ns")
    group.add_argument("--frame-index", type=int, help="0-based frame index")
    parser.add_argument("--save", default=None, help="Save figure to path instead of showing")
    args = parser.parse_args()

    result = load_result(args.result_json)

    if args.timestamp is not None:
        ts = args.timestamp
    else:
        # Get unique timestamps from residuals, sorted
        residuals = result.get("residuals", [])
        timestamps = sorted(set(r["timestamp_ns"] for r in residuals))
        if args.frame_index < 0 or args.frame_index >= len(timestamps):
            print(f"Frame index {args.frame_index} out of range (have {len(timestamps)} frames)")
            sys.exit(1)
        ts = timestamps[args.frame_index]

    ax = overlay_frame(result, ts)
    if ax is None:
        sys.exit(1)

    if args.save:
        ax.figure.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
