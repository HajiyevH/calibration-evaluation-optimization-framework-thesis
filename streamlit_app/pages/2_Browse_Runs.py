"""Page 2: Browse and inspect completed experiment runs."""

from pathlib import Path

import pandas as pd
import streamlit as st

from ui_backend import (
    build_run_table,
    get_run_images,
    list_runs,
    load_run_config,
    load_run_gt_evaluation,
    load_run_log,
    load_run_summary,
)
from ui_utils import (
    parse_run_dirname,
    render_artifact_downloads,
    render_config_viewer,
    render_gt_recovery_metrics,
    render_gt_recovery_tables,
    render_image_gallery,
    render_log_viewer,
    render_summary_metrics,
    status_badge,
)

st.set_page_config(page_title="Browse Runs", layout="wide")
st.title("Browse Runs")

# ------------------------------------------------------------------
# 1. Load runs + build table
# ------------------------------------------------------------------
col_hdr, col_refresh = st.columns([4, 1])
with col_refresh:
    if st.button("Refresh"):
        st.rerun()

runs = list_runs()
if not runs:
    st.info("No runs found in the runs/ directory.")
    st.stop()

# Build enriched table
run_rows = build_run_table(runs)

# ------------------------------------------------------------------
# 2. Sidebar filters
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filters")
    filter_success = st.checkbox("Success only", value=False)
    all_scenes = sorted(set(r["scene"] for r in run_rows if r["scene"]))
    filter_scenes = st.multiselect("Scene type", all_scenes, default=all_scenes) if all_scenes else all_scenes
    all_models = sorted(set(r["model"] for r in run_rows if r["model"]))
    filter_models = st.multiselect("Camera model", all_models, default=all_models) if all_models else all_models
    all_cams = sorted(set(r["cameras"] for r in run_rows if r["cameras"]))
    filter_cams = st.multiselect("Cameras", all_cams, default=all_cams) if all_cams else all_cams

# Apply filters
filtered = run_rows
if filter_success:
    filtered = [r for r in filtered if r.get("success") is True]
if filter_scenes:
    filtered = [r for r in filtered if r["scene"] in filter_scenes or not r["scene"]]
if filter_models:
    filtered = [r for r in filtered if r["model"] in filter_models or not r["model"]]
if filter_cams:
    filtered = [r for r in filtered if r["cameras"] in filter_cams or not r["cameras"]]

# ------------------------------------------------------------------
# 3. Display run table
# ------------------------------------------------------------------
if not filtered:
    st.info("No runs match the current filters.")
    st.stop()

display_cols = ["name", "scene", "cameras", "model", "trajectory", "noise",
                "outlier_ratio", "success", "reproj_mean", "extr_rot_err", "extr_trans_err"]
df = pd.DataFrame(filtered)
display_df = df[[c for c in display_cols if c in df.columns]].copy()

# Format success column
if "success" in display_df.columns:
    display_df["success"] = display_df["success"].apply(
        lambda x: "Yes" if x is True else ("No" if x is False else "-")
    )

# Format numeric columns
for col in ["reproj_mean", "extr_rot_err", "extr_trans_err", "noise", "outlier_ratio"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x:.4f}" if x is not None and x == x else "-"
        )

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------
# 4. Run selector
# ------------------------------------------------------------------
run_names = [r["name"] for r in filtered]
selected_name = st.selectbox("Select run to inspect", run_names)

selected_run = next((r for r in filtered if r["name"] == selected_name), None)
if selected_run is None:
    st.stop()

run_path = Path(selected_run["path"])
exp_name, ts = parse_run_dirname(selected_name)
st.caption(f"Experiment: **{exp_name}** | Timestamp: {ts} | Path: `{run_path}`")

# ------------------------------------------------------------------
# 5. Copy config button
# ------------------------------------------------------------------
config_dict = load_run_config(run_path)
if config_dict:
    if st.button("Copy config to new experiment"):
        st.session_state.copied_config = config_dict
        st.success("Config copied. Navigate to Synthetic Experiments and select 'Custom' preset.")

# ------------------------------------------------------------------
# 6. Tabbed detail view (4 tabs)
# ------------------------------------------------------------------
summary = load_run_summary(run_path)
gt_eval = load_run_gt_evaluation(run_path)

if summary:
    solver = summary.get("solver", {})
    st.subheader(f"Result: {status_badge(solver.get('success', False))}")

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
        st.warning("No summary.yaml found for this run.")

    if config_dict:
        with st.expander("Config"):
            render_config_viewer(config_dict)

with tab_recovery:
    if gt_eval:
        render_gt_recovery_metrics(gt_eval)
        render_gt_recovery_tables(gt_eval)
    else:
        st.info("No GT evaluation (gt_evaluation.json) found for this run.")

with tab_diag:
    # Error plot
    images = get_run_images(run_path)
    if "plot_errors" in images:
        st.image(str(images["plot_errors"]), caption="Reprojection Error Distribution", use_container_width=True)
    else:
        st.info("No error plot found.")

    # Overlays
    overlay_images = {k: v for k, v in images.items() if k.startswith("overlay_frame")}
    if overlay_images:
        st.markdown("**Overlay Frames**")
        render_image_gallery(overlay_images)

    # Solver log
    log_text = load_run_log(run_path, "cpp_stdout.log")
    if log_text:
        with st.expander("Solver Log"):
            render_log_viewer(log_text)

    stderr_text = load_run_log(run_path, "cpp_stderr.log")
    if stderr_text.strip():
        with st.expander("Solver stderr"):
            render_log_viewer(stderr_text)

with tab_files:
    render_artifact_downloads(run_path)
