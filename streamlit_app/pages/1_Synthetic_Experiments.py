"""Page 1: Configure and run synthetic bundle adjustment experiments."""

import re
import time
from pathlib import Path

import streamlit as st

from ui_backend import (
    EXPERIMENT_PRESETS,
    RUNS_DIR,
    build_config_from_ui,
    config_to_yaml_str,
    drain_stdout,
    get_run_images,
    list_runs,
    load_run_gt_evaluation,
    load_run_summary,
    start_experiment_subprocess,
    validate_config_safe,
)
from ui_utils import (
    SCENE_DESCRIPTIONS,
    TRAJECTORY_DESCRIPTIONS,
    detect_experiment_phase,
    render_artifact_downloads,
    render_gt_recovery_metrics,
    render_gt_recovery_tables,
    render_image_gallery,
    render_log_viewer,
    render_preset_card,
    render_scene_preview,
    render_summary_metrics,
    status_badge,
)

st.set_page_config(page_title="Synthetic Experiments", layout="wide")
st.title("Synthetic Experiments")

# ------------------------------------------------------------------
# Initialize session state
# ------------------------------------------------------------------
if "last_run_path" not in st.session_state:
    st.session_state.last_run_path = None

# ------------------------------------------------------------------
# 1. Preset selector
# ------------------------------------------------------------------
preset_names = list(EXPERIMENT_PRESETS.keys())
selected_preset = st.selectbox(
    "Experiment preset",
    preset_names,
    help="Choose a preset to auto-fill settings, or select Custom for full control.",
)

preset = EXPERIMENT_PRESETS[selected_preset]
preset_data = preset["data"]
preset_flags = preset["flags"]
is_custom = selected_preset == "Custom"

# Show preset summary card
render_preset_card(selected_preset, preset_data, preset_flags, preset["badge"])

# Default values: from preset or dataclass defaults
if preset_data is not None:
    d = preset_data
    f = preset_flags
else:
    # Check if we have a copied config from Browse Runs
    copied = st.session_state.get("copied_config")
    if copied:
        d_raw = copied.get("data", {})
        from calibri.framework.config import DataConfig, SolverFlags, PerturbationConfig
        pert_raw = d_raw.pop("perturbation", {})
        d = DataConfig(**{k: v for k, v in d_raw.items() if k in DataConfig.__dataclass_fields__},
                       perturbation=PerturbationConfig(**{k: v for k, v in pert_raw.items() if k in PerturbationConfig.__dataclass_fields__}))
        f_raw = copied.get("solver", {}).get("flags", {})
        f = SolverFlags(**{k: v for k, v in f_raw.items() if k in SolverFlags.__dataclass_fields__})
        st.success("Config loaded from Browse Runs. Edit as needed.")
        st.session_state.copied_config = None  # consume it
    else:
        from calibri.framework.config import DataConfig, SolverFlags
        d = DataConfig()
        f = SolverFlags()

# ------------------------------------------------------------------
# 2. Two-column layout: controls left, card + preview right
# ------------------------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    # --- Experiment name ---
    default_name = re.sub(r"[^a-zA-Z0-9_-]", "_", selected_preset.lower().replace(" ", "_"))
    cfg_name = st.text_input("Experiment name", value=default_name)
    cfg_desc = st.text_input("Description", value=preset.get("description", ""))

    # --- Scenario section ---
    with st.expander("Scenario", expanded=is_custom):
        scene_types = ["corridor", "surround", "random"]
        scene_idx = scene_types.index(d.scene_type) if d.scene_type in scene_types else 0
        cfg_scene_type = st.selectbox(
            "Scene type", scene_types, index=scene_idx,
            help="\n".join(f"- **{k}**: {v}" for k, v in SCENE_DESCRIPTIONS.items()),
        )

        traj_types = ["arc", "straight", "circular", "slalom"]
        traj_idx = traj_types.index(d.trajectory_type) if d.trajectory_type in traj_types else 0
        cfg_trajectory = st.selectbox(
            "Trajectory type", traj_types, index=traj_idx,
            help="\n".join(f"- **{k}**: {v}" for k, v in TRAJECTORY_DESCRIPTIONS.items()),
        )

        sc1, sc2 = st.columns(2)
        cfg_num_poses = sc1.number_input("Number of poses", min_value=2, value=d.num_poses)
        cfg_num_landmarks = sc2.number_input("Number of landmarks", min_value=10, value=d.num_landmarks)
        cfg_seed = st.number_input("Random seed", min_value=0, value=d.seed)

        # Corridor geometry (only if corridor)
        if cfg_scene_type == "corridor":
            gc1, gc2, gc3 = st.columns(3)
            cfg_corridor_length = gc1.number_input("Corridor length (m)", min_value=5.0, value=d.corridor_length_m, step=5.0)
            cfg_corridor_width = gc2.number_input("Corridor width (m)", min_value=2.0, value=d.corridor_width_m, step=1.0)
            cfg_corridor_wall_height = gc3.number_input("Wall height (m)", min_value=1.0, value=d.corridor_wall_height_m, step=0.5)
        else:
            cfg_corridor_length = d.corridor_length_m
            cfg_corridor_width = d.corridor_width_m
            cfg_corridor_wall_height = d.corridor_wall_height_m

    # --- Camera Rig section ---
    with st.expander("Camera Rig", expanded=is_custom):
        cr1, cr2 = st.columns(2)
        cfg_num_cameras = cr1.radio("Number of cameras", [1, 4],
                                     index=[1, 4].index(d.num_cameras), horizontal=True)
        camera_models = ["Pinhole", "Fisheye"]
        cfg_camera_model = cr2.selectbox("Camera model", camera_models,
                                          index=camera_models.index(d.camera_model) if d.camera_model in camera_models else 0)
        ir1, ir2 = st.columns(2)
        cfg_image_width = ir1.number_input("Image width (px)", min_value=100, value=d.image_width)
        cfg_image_height = ir2.number_input("Image height (px)", min_value=100, value=d.image_height)

        # Compatibility warnings
        if cfg_scene_type == "corridor" and cfg_num_cameras == 4:
            st.warning("Corridor scene with 4 cameras: only the front camera will see most landmarks.")
        if cfg_scene_type == "surround" and cfg_num_cameras == 1:
            st.info("Surround scene with 1 camera: consider using corridor for better coverage.")

    # --- Observation Corruption section ---
    with st.expander("Observation Corruption", expanded=is_custom):
        nc1, nc2 = st.columns(2)
        cfg_noise_sigma = nc1.number_input("Noise sigma (px)", min_value=0.0, value=d.noise_sigma, step=0.1, format="%.2f")
        cfg_outlier_ratio = nc2.number_input("Outlier ratio", min_value=0.0, max_value=1.0,
                                              value=d.outlier_ratio, step=0.05, format="%.2f")

    # --- Perturbation section ---
    with st.expander("Perturbation", expanded=is_custom):
        pert = d.perturbation
        st.caption("Sigmas applied to initial guesses (ground truth is preserved).")
        pc1, pc2 = st.columns(2)
        cfg_pose_rot_sigma = pc1.number_input("Pose rot sigma (deg)", min_value=0.0, value=pert.pose_rot_sigma_deg, step=0.5, format="%.1f")
        cfg_pose_trans_sigma = pc2.number_input("Pose trans sigma (m)", min_value=0.0, value=pert.pose_trans_sigma_m, step=0.01, format="%.3f")
        pc3, pc4 = st.columns(2)
        cfg_landmark_sigma = pc3.number_input("Landmark sigma (m)", min_value=0.0, value=pert.landmark_sigma_m, step=0.01, format="%.3f")
        cfg_extr_rot_sigma = pc4.number_input("Extrinsic rot sigma (deg)", min_value=0.0, value=pert.extrinsic_rot_sigma_deg, step=0.5, format="%.1f")
        pc5, pc6, pc7 = st.columns(3)
        cfg_extr_trans_sigma = pc5.number_input("Extrinsic trans sigma (m)", min_value=0.0, value=pert.extrinsic_trans_sigma_m, step=0.01, format="%.3f")
        cfg_intr_sigma = pc6.number_input("Intrinsic sigma (px)", min_value=0.0, value=pert.intrinsic_sigma_px, step=1.0, format="%.1f")
        cfg_dist_sigma = pc7.number_input("Distortion sigma", min_value=0.0, value=pert.distortion_sigma, step=0.001, format="%.4f")

    # --- Solver Options section ---
    with st.expander("Solver Options", expanded=is_custom):
        st.markdown("**Optimization flags**")
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        cfg_opt_poses = fc1.checkbox("Poses", value=f.opt_poses)
        cfg_opt_landmarks = fc2.checkbox("Landmarks", value=f.opt_landmarks)
        cfg_opt_intrinsics = fc3.checkbox("Intrinsics", value=f.opt_intrinsics)
        cfg_opt_distortion = fc4.checkbox("Distortion", value=f.opt_distortion)
        cfg_opt_extrinsics = fc5.checkbox("Extrinsics", value=f.opt_extrinsics)

        st.markdown("**Robust loss**")
        rc1, rc2 = st.columns(2)
        robust_options = ["Huber", "Cauchy", "None"]
        cfg_robust_type = rc1.selectbox("Loss type", options=robust_options)
        cfg_robust_scale = rc2.number_input("Loss scale", min_value=0.01, value=1.0, step=0.1, format="%.2f")

        cfg_timeout = st.number_input("Timeout (seconds)", min_value=1, value=600)

# ------------------------------------------------------------------
# Right column: preview + action buttons
# ------------------------------------------------------------------
with col_right:
    st.markdown("**Scene Preview**")
    render_scene_preview(
        scene_type=cfg_scene_type if 'cfg_scene_type' in dir() else d.scene_type,
        trajectory_type=cfg_trajectory if 'cfg_trajectory' in dir() else d.trajectory_type,
        num_poses=cfg_num_poses if 'cfg_num_poses' in dir() else d.num_poses,
        num_landmarks=cfg_num_landmarks if 'cfg_num_landmarks' in dir() else d.num_landmarks,
        num_cameras=cfg_num_cameras if 'cfg_num_cameras' in dir() else d.num_cameras,
        seed=cfg_seed if 'cfg_seed' in dir() else d.seed,
        corridor_length_m=cfg_corridor_length if 'cfg_corridor_length' in dir() else d.corridor_length_m,
        corridor_width_m=cfg_corridor_width if 'cfg_corridor_width' in dir() else d.corridor_width_m,
        corridor_wall_height_m=cfg_corridor_wall_height if 'cfg_corridor_wall_height' in dir() else d.corridor_wall_height_m,
    )

    # Output info
    st.markdown("**Output files**")
    st.caption("problem.json, ground_truth.json, result.json, gt_evaluation.json, plots")
    st.caption("Ground truth: Yes (synthetic)")

# ------------------------------------------------------------------
# 3. Build config
# ------------------------------------------------------------------
config = build_config_from_ui(
    name=cfg_name,
    description=cfg_desc,
    scene_type=cfg_scene_type,
    trajectory_type=cfg_trajectory,
    camera_model=cfg_camera_model,
    num_cameras=cfg_num_cameras,
    num_poses=cfg_num_poses,
    num_landmarks=cfg_num_landmarks,
    noise_sigma=cfg_noise_sigma,
    outlier_ratio=cfg_outlier_ratio,
    image_width=cfg_image_width,
    image_height=cfg_image_height,
    seed=cfg_seed,
    corridor_length_m=cfg_corridor_length,
    corridor_width_m=cfg_corridor_width,
    corridor_wall_height_m=cfg_corridor_wall_height,
    pose_rot_sigma_deg=cfg_pose_rot_sigma,
    pose_trans_sigma_m=cfg_pose_trans_sigma,
    landmark_sigma_m=cfg_landmark_sigma,
    extrinsic_rot_sigma_deg=cfg_extr_rot_sigma,
    extrinsic_trans_sigma_m=cfg_extr_trans_sigma,
    intrinsic_sigma_px=cfg_intr_sigma,
    distortion_sigma=cfg_dist_sigma,
    opt_poses=cfg_opt_poses,
    opt_landmarks=cfg_opt_landmarks,
    opt_intrinsics=cfg_opt_intrinsics,
    opt_distortion=cfg_opt_distortion,
    opt_extrinsics=cfg_opt_extrinsics,
    robust_type=cfg_robust_type,
    robust_scale=cfg_robust_scale,
    timeout=cfg_timeout,
)

# ------------------------------------------------------------------
# 4. Validation + config preview
# ------------------------------------------------------------------
with st.expander("Config preview (YAML)"):
    st.code(config_to_yaml_str(config), language="yaml")

validation_error = validate_config_safe(config)
if validation_error:
    st.error(f"Validation error: {validation_error}")

# ------------------------------------------------------------------
# 5. Action buttons
# ------------------------------------------------------------------
is_running = "running_experiment" in st.session_state and st.session_state.running_experiment is not None

btn_col1, btn_col2, btn_col3 = st.columns(3)

with btn_col1:
    run_clicked = st.button(
        "Generate & Run",
        disabled=bool(validation_error) or is_running,
        type="primary",
        use_container_width=True,
    )

with btn_col2:
    gen_only_clicked = st.button(
        "Generate Only",
        disabled=bool(validation_error) or is_running,
        use_container_width=True,
    )

with btn_col3:
    eval_gt_clicked = st.button(
        "Evaluate GT Only",
        disabled=is_running,
        use_container_width=True,
        help="Run GT evaluation on the last completed run.",
    )

# ------------------------------------------------------------------
# 6. Handle Generate & Run (subprocess)
# ------------------------------------------------------------------
if run_clicked:
    tracking = start_experiment_subprocess(config)
    st.session_state.running_experiment = tracking
    st.rerun()

# ------------------------------------------------------------------
# 7. Handle Generate Only (in-process)
# ------------------------------------------------------------------
if gen_only_clicked:
    from calibri.framework.data_gen import generate_synthetic_data
    from calibri.framework.config import config_to_dict
    import yaml as _yaml
    from datetime import datetime

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{config.name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save frozen config
    frozen_config = run_dir / "config.yaml"
    frozen_config.write_text(
        _yaml.dump(config_to_dict(config), default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    with st.spinner("Generating synthetic data..."):
        problem_path, gt_path = generate_synthetic_data(
            output_dir=run_dir,
            data_config=config.data,
            solver_flags=config.solver.flags,
            robust_config=config.solver.robust,
        )

    st.success(f"Data generated in `{run_dir}`")
    st.caption(f"problem.json: `{problem_path}`")
    st.caption(f"ground_truth.json: `{gt_path}`")
    st.session_state.last_run_path = run_dir

# ------------------------------------------------------------------
# 8. Handle Evaluate GT Only (in-process)
# ------------------------------------------------------------------
if eval_gt_clicked:
    # Find the last run
    last_path = st.session_state.get("last_run_path")
    if last_path is None:
        runs = list_runs()
        if runs:
            last_path = runs[0]["path"]

    if last_path is None:
        st.error("No completed runs found to evaluate.")
    else:
        result_json = Path(last_path) / "result.json"
        gt_json = Path(last_path) / "ground_truth.json"
        if not result_json.exists():
            st.error(f"No result.json in `{last_path}`. Run the solver first.")
        elif not gt_json.exists():
            st.error(f"No ground_truth.json in `{last_path}`.")
        else:
            import json
            from calibri.framework.evaluate import evaluate_against_gt

            with st.spinner("Evaluating against ground truth..."):
                gt_eval = evaluate_against_gt(result_json, gt_json)
                eval_path = Path(last_path) / "gt_evaluation.json"
                with open(eval_path, "w") as ef:
                    json.dump(gt_eval, ef, indent=2)

            st.success(f"GT evaluation saved to `{eval_path}`")
            render_gt_recovery_metrics(gt_eval)

# ------------------------------------------------------------------
# 9. Progress display (while experiment is running)
# ------------------------------------------------------------------
if is_running:
    tracking = st.session_state.running_experiment
    proc = tracking["proc"]
    stdout_text = drain_stdout(tracking)
    phase = detect_experiment_phase(stdout_text)
    elapsed = time.time() - tracking["start_time"]

    st.divider()
    st.subheader(f"Experiment: {tracking['config_name']}")

    col_a, col_b = st.columns(2)
    col_a.metric("Phase", phase)
    col_b.metric("Elapsed", f"{elapsed:.0f}s")

    lines = stdout_text.splitlines()
    tail = "\n".join(lines[-40:]) if lines else ""
    st.code(tail, language="text")

    poll = proc.poll()
    if poll is not None:
        try:
            proc.terminate()
        except OSError:
            pass
        final_stdout = drain_stdout(tracking)
        st.session_state.running_experiment = None

        # Find the run directory that was just created
        runs = list_runs()
        if runs:
            st.session_state.last_run_path = runs[0]["path"]

        if poll == 0:
            st.success("Experiment completed successfully.")
        else:
            st.error(f"Experiment failed (exit code {poll}).")

        if final_stdout.strip():
            with st.expander("Full output"):
                st.code(final_stdout, language="text")

        st.rerun()
    else:
        time.sleep(2)
        st.rerun()

# ------------------------------------------------------------------
# 10. Post-run results (full width, 4 tabs)
# ------------------------------------------------------------------
last_run = st.session_state.get("last_run_path")
if last_run and not is_running:
    last_run = Path(last_run)
    if last_run.exists():
        summary = load_run_summary(last_run)
        gt_eval = load_run_gt_evaluation(last_run)

        if summary or gt_eval:
            st.divider()
            st.subheader(f"Results: {last_run.name}")

            if summary:
                solver = summary.get("solver", {})
                st.caption(f"Status: {status_badge(solver.get('success', False))}")

            tab_summary, tab_recovery, tab_diag, tab_files = st.tabs(
                ["Summary", "Recovery", "Diagnostics", "Files"]
            )

            with tab_summary:
                if summary:
                    render_summary_metrics(summary)
                    if gt_eval:
                        st.markdown("**Ground-Truth Recovery**")
                        render_gt_recovery_metrics(gt_eval)
                else:
                    st.info("No summary available yet (Generate Only was used).")

            with tab_recovery:
                if gt_eval:
                    render_gt_recovery_metrics(gt_eval)
                    render_gt_recovery_tables(gt_eval)
                else:
                    st.info("No GT evaluation available. Click 'Evaluate GT Only' after solving.")

            with tab_diag:
                # Error plot
                images = get_run_images(last_run)
                if "plot_errors" in images:
                    st.image(str(images["plot_errors"]), caption="Reprojection Error Distribution", use_container_width=True)

                # Overlays
                overlay_images = {k: v for k, v in images.items() if k.startswith("overlay_frame")}
                if overlay_images:
                    st.markdown("**Overlay Frames**")
                    render_image_gallery(overlay_images)

                # Solver log
                from ui_backend import load_run_log
                log_text = load_run_log(last_run, "cpp_stdout.log")
                if log_text:
                    with st.expander("Solver Log"):
                        render_log_viewer(log_text)

            with tab_files:
                render_artifact_downloads(last_run)
