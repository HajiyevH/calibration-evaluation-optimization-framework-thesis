"""Shared Streamlit display helpers.

No calibri.framework dependencies -- only streamlit, matplotlib, pandas,
and standard library.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


SCENE_DESCRIPTIONS = {
    "corridor": "Forward-facing structured street -- Recommended for single-camera",
    "surround": "360-degree structured scene -- Recommended for 4-camera rigs",
    "random": "Uniform random point cloud -- Debug only",
}

TRAJECTORY_DESCRIPTIONS = {
    "arc": "Gentle forward arc with yaw variation -- Recommended",
    "straight": "Pure forward motion -- Degenerate for testing",
    "circular": "3/4 circle, radius 10m -- Good lateral variation",
    "slalom": "S-curve oscillation -- Good for observability",
}
def render_summary_metrics(summary: dict) -> None:
    """Display key solver metrics as st.metric widgets."""
    solver = summary.get("solver", {})
    stats = summary.get("statistics", {})
    reproj = stats.get("reprojection_error", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        success = solver.get("success", False)
        st.metric("Solver", "Success" if success else "FAILED")
    with col2:
        st.metric("Iterations", stats.get("iterations", "N/A"))
    with col3:
        elapsed = solver.get("elapsed_seconds")
        st.metric("Elapsed", f"{elapsed:.1f}s" if elapsed is not None else "N/A")
    with col4:
        st.metric("Observations", stats.get("num_observations", "N/A"))

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        mean = reproj.get("mean")
        st.metric("Mean Error (px)", f"{mean:.4f}" if mean is not None else "N/A")
    with col6:
        median = reproj.get("median")
        st.metric("Median Error (px)", f"{median:.4f}" if median is not None else "N/A")
    with col7:
        p95 = reproj.get("p95")
        st.metric("P95 Error (px)", f"{p95:.4f}" if p95 is not None else "N/A")
    with col8:
        mx = reproj.get("max")
        st.metric("Max Error (px)", f"{mx:.4f}" if mx is not None else "N/A")

    col9, col10, _, _ = st.columns(4)
    with col9:
        ic = stats.get("initial_cost")
        st.metric("Initial Cost", f"{ic:.6f}" if ic is not None else "N/A")
    with col10:
        fc = stats.get("final_cost")
        st.metric("Final Cost", f"{fc:.6f}" if fc is not None else "N/A")


def render_gt_recovery_metrics(gt_eval: dict) -> None:
    """Show GT recovery summary metrics as st.metric cards."""
    s = gt_eval.get("summary", {})
    if not s:
        st.info("No ground-truth recovery data available.")
        return

    cols = st.columns(4)
    with cols[0]:
        v = s.get("mean_extr_rot_err_deg")
        st.metric("Extr. Rot. Err", f"{v:.4f} deg" if v is not None else "N/A")
    with cols[1]:
        v = s.get("mean_extr_trans_err_m")
        st.metric("Extr. Trans. Err", f"{v:.4f} m" if v is not None else "N/A")
    with cols[2]:
        v = s.get("mean_pose_rot_err_deg")
        st.metric("Pose Rot. Err", f"{v:.4f} deg" if v is not None else "N/A")
    with cols[3]:
        v = s.get("mean_pose_trans_err_m")
        st.metric("Pose Trans. Err", f"{v:.4f} m" if v is not None else "N/A")

    cols2 = st.columns(4)
    with cols2[0]:
        v = s.get("mean_intrinsic_err_px")
        st.metric("Intrinsic Err", f"{v:.4f} px" if v is not None else "N/A")
    with cols2[1]:
        v = s.get("mean_distortion_err")
        st.metric("Distortion Err", f"{v:.6f}" if v is not None else "N/A")
    with cols2[2]:
        v = s.get("mean_landmark_err_m")
        st.metric("Landmark Err", f"{v:.4f} m" if v is not None else "N/A")

def render_gt_recovery_tables(gt_eval: dict) -> None:
    """Show detailed per-camera and per-pose recovery tables."""
    # Extrinsics table
    extr = gt_eval.get("extrinsics", [])
    if extr:
        st.markdown("**Per-camera extrinsic errors**")
        df = pd.DataFrame(extr)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Intrinsics table
    intr = gt_eval.get("intrinsics", [])
    if intr:
        st.markdown("**Per-camera intrinsic errors**")
        df = pd.DataFrame(intr)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Distortion table
    dist = gt_eval.get("distortion", [])
    if dist:
        st.markdown("**Per-camera distortion errors**")
        rows = []
        for entry in dist:
            row = {"camera_id": entry["camera_id"]}
            row.update(entry.get("coeffs", {}))
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Pose error summary
    poses = gt_eval.get("poses", [])
    if poses:
        rot_errs = [p["rot_err_deg"] for p in poses]
        trans_errs = [p["trans_err_m"] for p in poses]
        st.markdown("**Pose error summary**")
        cols = st.columns(4)
        cols[0].metric("Mean rot (deg)", f"{np.mean(rot_errs):.4f}")
        cols[1].metric("Max rot (deg)", f"{np.max(rot_errs):.4f}")
        cols[2].metric("Mean trans (m)", f"{np.mean(trans_errs):.4f}")
        cols[3].metric("Max trans (m)", f"{np.max(trans_errs):.4f}")

    # Landmark stats
    lm = gt_eval.get("landmarks", {})
    if lm:
        st.markdown("**Landmark error statistics**")
        cols = st.columns(5)
        cols[0].metric("Mean (m)", f"{lm.get('mean_err_m', 0):.4f}")
        cols[1].metric("Median (m)", f"{lm.get('median_err_m', 0):.4f}")
        cols[2].metric("P95 (m)", f"{lm.get('p95_err_m', 0):.4f}")
        cols[3].metric("Max (m)", f"{lm.get('max_err_m', 0):.4f}")
        cols[4].metric("Count", lm.get("count", 0))

def render_scene_preview(
    scene_type: str,
    trajectory_type: str,
    num_poses: int,
    num_landmarks: int,
    num_cameras: int,
    seed: int = 42,
    corridor_length_m: float = 30.0,
    corridor_width_m: float = 8.0,
    corridor_wall_height_m: float = 4.0,
) -> None:
    """Render a simple top-down matplotlib scatter preview of scene + trajectory.

    X-Z plane: X = lateral, Z = forward/depth.
    """
    rng = np.random.default_rng(seed)
    multicam = num_cameras > 1

    # Generate landmarks (simplified inline, no framework import)
    if scene_type == "corridor":
        lm_x, lm_z = _generate_corridor_preview(
            num_landmarks, corridor_length_m, corridor_width_m, rng,
        )
    elif scene_type == "surround":
        lm_x, lm_z = _generate_surround_preview(num_landmarks, rng)
    else:  # random
        lm_x = rng.uniform(-8, 8, num_landmarks)
        lm_z = rng.uniform(5, 30, num_landmarks)

    # Generate trajectory (simplified inline)
    traj_x, traj_z = _generate_trajectory_preview(
        trajectory_type, num_poses, multicam,
    )

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(lm_x, lm_z, s=2, c="silver", alpha=0.5, label="Landmarks")
    ax.plot(traj_x, traj_z, "b-o", markersize=3, linewidth=1.5, label="Trajectory")

    # Direction arrow on trajectory
    if len(traj_x) > 1:
        mid = len(traj_x) // 2
        dx = traj_x[min(mid + 1, len(traj_x) - 1)] - traj_x[mid]
        dz = traj_z[min(mid + 1, len(traj_z) - 1)] - traj_z[mid]
        ax.annotate("", xy=(traj_x[mid] + dx, traj_z[mid] + dz),
                     xytext=(traj_x[mid], traj_z[mid]),
                     arrowprops=dict(arrowstyle="->", color="blue", lw=2))

    # Camera frustum indicators at first pose
    if num_cameras > 1:
        colors = ["tab:red", "tab:green", "tab:orange", "tab:purple"]
        cam_labels = ["front", "left", "right", "rear"]
        fov_len = 1.5
        angles_deg = [0, 90, -90, 180]  # front, left, right, rear
        for ci in range(min(num_cameras, 4)):
            angle = np.radians(angles_deg[ci])
            fx = traj_x[0] + fov_len * np.sin(angle)
            fz = traj_z[0] + fov_len * np.cos(angle)
            ax.annotate("", xy=(fx, fz), xytext=(traj_x[0], traj_z[0]),
                         arrowprops=dict(arrowstyle="->", color=colors[ci], lw=1.5))

    ax.set_xlabel("X (lateral, m)")
    ax.set_ylabel("Z (forward, m)")
    ax.set_title(f"{scene_type} / {trajectory_type}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _generate_corridor_preview(n, length, width, rng):
    """Generate corridor-like landmark positions for preview (X-Z only)."""
    n_ground = n // 3
    n_wall = n // 3
    n_far = n - n_ground - n_wall

    x = np.concatenate([
        rng.uniform(-width / 2, width / 2, n_ground),  # ground
        np.concatenate([
            np.full(n_wall // 2, -width / 2) + rng.normal(0, 0.1, n_wall // 2),
            np.full(n_wall - n_wall // 2, width / 2) + rng.normal(0, 0.1, n_wall - n_wall // 2),
        ]),  # walls
        rng.uniform(-width / 2, width / 2, n_far),  # far
    ])
    z = np.concatenate([
        rng.uniform(2, length, n_ground),
        rng.uniform(2, length, n_wall),
        rng.uniform(length * 0.7, length * 1.2, n_far),
    ])
    return x, z


def _generate_surround_preview(n, rng):
    """Generate surround-like landmark positions (X-Z only)."""
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = rng.uniform(8, 25, n)
    x = radii * np.cos(angles)
    z = radii * np.sin(angles)
    return x, z


def _generate_trajectory_preview(trajectory_type, num_poses, multicam):
    """Generate simplified trajectory points (X-Z) for preview."""
    xs, zs = [], []
    for i in range(num_poses):
        frac = i / max(num_poses - 1, 1)
        if trajectory_type == "straight":
            xs.append(0.0)
            zs.append(i * 1.0)
        elif trajectory_type == "circular":
            radius = 10.0
            angle = frac * 2.0 * np.pi * 0.75
            xs.append(radius * np.sin(angle))
            zs.append(radius * (1.0 - np.cos(angle)))
        elif trajectory_type == "slalom":
            amplitude = 3.0
            frequency = 2.0
            xs.append(amplitude * np.sin(frac * np.pi * frequency * 2.0))
            zs.append(i * 0.8)
        else:  # arc
            if multicam:
                xs.append(3.0 * np.sin(frac * np.pi))
                zs.append(i * 0.8)
            else:
                xs.append(0.3 * np.sin(frac * np.pi * 0.5))
                zs.append(i * 0.5)
    return np.array(xs), np.array(zs)

def render_preset_card(
    preset_name: str,
    data_config,
    solver_flags,
    badge: Optional[str] = None,
) -> None:
    """Render a compact summary card for a preset configuration."""
    if badge:
        badge_colors = {
            "Recommended": ":green[Recommended]",
            "Advanced": ":orange[Advanced]",
            "Debug": ":red[Debug]",
        }
        st.caption(badge_colors.get(badge, f":gray[{badge}]"))

    if data_config is None:
        st.caption("Custom: all parameters configurable")
        return

    cols = st.columns(3)
    with cols[0]:
        st.markdown(f"**Scene:** {data_config.scene_type}")
        st.markdown(f"**Trajectory:** {data_config.trajectory_type}")
    with cols[1]:
        st.markdown(f"**Cameras:** {data_config.num_cameras}")
        st.markdown(f"**Model:** {data_config.camera_model}")
    with cols[2]:
        st.markdown(f"**Poses:** {data_config.num_poses}")
        st.markdown(f"**Landmarks:** {data_config.num_landmarks}")

    details = []
    details.append(f"Noise: {data_config.noise_sigma} px")
    if data_config.outlier_ratio > 0:
        details.append(f"Outliers: {data_config.outlier_ratio * 100:.0f}%")
    pert = data_config.perturbation
    if pert.pose_rot_sigma_deg > 1.0 or pert.extrinsic_rot_sigma_deg > 0:
        details.append("Perturbation: enabled")
    if solver_flags:
        active = []
        if solver_flags.opt_poses:
            active.append("poses")
        if solver_flags.opt_landmarks:
            active.append("landmarks")
        if solver_flags.opt_intrinsics:
            active.append("intrinsics")
        if solver_flags.opt_distortion:
            active.append("distortion")
        if solver_flags.opt_extrinsics:
            active.append("extrinsics")
        details.append(f"Optimize: {', '.join(active)}")

    st.caption(" | ".join(details))

def render_sweep_chart(sweep_data: dict, x_label: str, y_label: str) -> None:
    """Render a line chart (numeric x) or bar chart (categorical x) from sweep data."""
    x = sweep_data["x"]
    y = sweep_data["y"]
    groups = sweep_data["group"]

    if not x:
        st.info("No data points to chart.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    is_numeric = all(isinstance(v, (int, float)) for v in x)

    unique_groups = sorted(set(groups))

    if is_numeric:
        for group in unique_groups:
            gx = [xi for xi, gi in zip(x, groups) if gi == group]
            gy = [yi for yi, gi in zip(y, groups) if gi == group]
            # Sort by x
            pairs = sorted(zip(gx, gy))
            gx = [p[0] for p in pairs]
            gy = [p[1] for p in pairs]
            label = group if group != "all" else None
            ax.plot(gx, gy, "-o", markersize=5, label=label)
    else:
        # Categorical x -- bar chart
        x_cats = sorted(set(x), key=str)
        width = 0.8 / max(len(unique_groups), 1)
        for gi, group in enumerate(unique_groups):
            gx = [xi for xi, ggi in zip(x, groups) if ggi == group]
            gy = [yi for yi, ggi in zip(y, groups) if ggi == group]
            cat_vals = {xi: yi for xi, yi in zip(gx, gy)}
            bar_y = [cat_vals.get(c, 0) for c in x_cats]
            positions = np.arange(len(x_cats)) + gi * width
            label = group if group != "all" else None
            ax.bar(positions, bar_y, width, label=label)
        ax.set_xticks(np.arange(len(x_cats)) + width * (len(unique_groups) - 1) / 2)
        ax.set_xticklabels(x_cats, rotation=45, ha="right")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    if len(unique_groups) > 1 or (len(unique_groups) == 1 and unique_groups[0] != "all"):
        ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def render_log_viewer(text: str, max_lines: int = 200) -> None:
    """Show log text in a code block, with expander for long logs."""
    if not text.strip():
        st.info("Log is empty.")
        return
    lines = text.splitlines()
    if len(lines) <= max_lines:
        st.code(text, language="text")
    else:
        st.code("\n".join(lines[:max_lines]), language="text")
        with st.expander(f"Show remaining {len(lines) - max_lines} lines"):
            st.code("\n".join(lines[max_lines:]), language="text")


def render_image_gallery(images: Dict[str, Path]) -> None:
    """Display PNG images with st.image, two per row."""
    if not images:
        st.info("No images found.")
        return
    names = list(images.keys())
    for i in range(0, len(names), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(names):
                with col:
                    st.image(str(images[names[idx]]), caption=names[idx], use_container_width=True)


# Phase detection patterns (ordered by specificity)
_PHASE_PATTERNS = [
    ("Run complete:", "Complete"),
    ("Generating overlay frames...", "Analyzing (overlays)"),
    ("Generating error plots...", "Analyzing (plots)"),
    ("Solver succeeded", "Solver finished"),
    ("Solver FAILED", "Solver finished"),
    ("Running solver...", "Solving"),
    ("Generating synthetic data...", "Generating data"),
]


def detect_experiment_phase(stdout: str) -> str:
    """Parse runner.py stdout to detect the current experiment phase."""
    if not stdout:
        return "Starting..."
    # Check patterns in reverse priority (last match in stdout wins)
    for pattern, phase in _PHASE_PATTERNS:
        if pattern in stdout:
            return phase
    return "Running..."

def status_badge(success: Optional[bool]) -> str:
    """Return a compact status label for run summaries."""
    if success is True:
        return "✅ Success"
    if success is False:
        return "❌ Failed"
    return "⏳ Unknown"


def parse_run_dirname(run_name: str) -> Tuple[str, str]:
    """Split <experiment>_<YYYYMMDD_HHMMSS> into (experiment, timestamp)."""
    if len(run_name) >= 16 and run_name[-15] == "_" and run_name[-8] == "_":
        ts = run_name[-15:]
        exp = run_name[:-16]
        return (exp if exp else run_name, ts)
    return run_name, "-"


def render_config_viewer(config_dict: dict) -> None:
    """Render run config as formatted JSON in the UI."""
    if not config_dict:
        st.info("No config available.")
        return
    st.json(config_dict, expanded=2)


def render_artifact_downloads(run_path: Path) -> None:
    """Render downloadable run artifacts from a run directory."""
    run_path = Path(run_path)
    if not run_path.exists() or not run_path.is_dir():
        st.info("Run directory not found.")
        return

    artifacts = sorted(
        [
            p for p in run_path.iterdir()
            if p.is_file() and p.suffix.lower() in {
                ".json", ".yaml", ".yml", ".log", ".txt", ".png", ".jpg", ".jpeg", ".pdf", ".csv"
            }
        ],
        key=lambda p: p.name.lower(),
    )

    if not artifacts:
        st.info("No downloadable artifacts found.")
        return

    st.markdown("**Artifacts**")
    for artifact in artifacts:
        mime = "application/octet-stream"
        if artifact.suffix.lower() in {".json"}:
            mime = "application/json"
        elif artifact.suffix.lower() in {".yaml", ".yml"}:
            mime = "text/yaml"
        elif artifact.suffix.lower() in {".log", ".txt"}:
            mime = "text/plain"
        elif artifact.suffix.lower() in {".png"}:
            mime = "image/png"
        elif artifact.suffix.lower() in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif artifact.suffix.lower() in {".pdf"}:
            mime = "application/pdf"
        elif artifact.suffix.lower() in {".csv"}:
            mime = "text/csv"

        st.download_button(
            label=f"Download {artifact.name}",
            data=artifact.read_bytes(),
            file_name=artifact.name,
            mime=mime,
            key=f"download_{run_path.name}_{artifact.name}",
            use_container_width=True,
        )
