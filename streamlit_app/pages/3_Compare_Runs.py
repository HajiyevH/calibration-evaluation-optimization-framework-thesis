"""Page 3: Compare experiment runs with sweep charts and thesis presets."""

from pathlib import Path

import pandas as pd
import streamlit as st

from ui_backend import (
    COMPARISON_PRESETS,
    extract_sweep_data,
    list_runs,
    load_run_config,
    load_run_summary,
    load_run_gt_evaluation,
)
from ui_utils import render_sweep_chart

st.set_page_config(page_title="Compare Runs", layout="wide")
st.title("Compare Runs")

# ------------------------------------------------------------------
# 1. Load available runs
# ------------------------------------------------------------------
runs = list_runs()
if len(runs) < 1:
    st.info("Need at least 1 completed run to compare.")
    st.stop()

run_names = [r["name"] for r in runs]
run_paths = {r["name"]: r["path"] for r in runs}

# ------------------------------------------------------------------
# 2. Thesis comparison presets
# ------------------------------------------------------------------
st.subheader("Comparison Mode")

preset_names = list(COMPARISON_PRESETS.keys())
selected_preset = st.radio(
    "Quick thesis presets",
    preset_names,
    horizontal=True,
    help="Pre-configured axis mappings for common thesis comparisons.",
)

preset_config = COMPARISON_PRESETS[selected_preset]
is_custom = preset_config is None

# ------------------------------------------------------------------
# 3. Run selection
# ------------------------------------------------------------------
st.subheader("Select Runs")
selected_runs = st.multiselect(
    "Runs to compare",
    run_names,
    default=run_names[:min(5, len(run_names))],
    help="Select multiple runs to include in the comparison.",
)

if not selected_runs:
    st.info("Select at least one run to compare.")
    st.stop()

selected_paths = [run_paths[n] for n in selected_runs]

# ------------------------------------------------------------------
# 4. Axis configuration
# ------------------------------------------------------------------
x_field_options = [
    "data.noise_sigma",
    "data.outlier_ratio",
    "data.num_poses",
    "data.num_landmarks",
    "data.trajectory_type",
    "data.scene_type",
    "perturbation_mag",
]

y_metric_options = [
    "reproj_rmse",
    "reproj_median",
    "reproj_p95",
    "extr_rot_err_deg",
    "extr_trans_err_m",
    "pose_rot_err_deg",
    "pose_trans_err_m",
    "landmark_err_m",
    "final_cost",
]

group_field_options = [
    None,
    "data.scene_type",
    "data.trajectory_type",
    "data.camera_model",
    "data.num_cameras",
    "solver.robust.type",
]

with st.expander("Axis Configuration", expanded=is_custom):
    if is_custom:
        ac1, ac2, ac3 = st.columns(3)
        x_field = ac1.selectbox("X-axis", x_field_options)
        y_metric = ac2.selectbox("Y-axis metric", y_metric_options)
        group_field = ac3.selectbox(
            "Group by",
            group_field_options,
            format_func=lambda x: "None" if x is None else x,
        )
    else:
        x_field = preset_config["x_field"]
        y_metric = preset_config["y_metric"]
        group_field = preset_config["group_field"]
        st.caption(f"X-axis: `{x_field}` | Y-axis: `{y_metric}` | Group: `{group_field or 'None'}`")

        # Allow override
        if st.checkbox("Override axes"):
            ac1, ac2, ac3 = st.columns(3)
            x_idx = x_field_options.index(x_field) if x_field in x_field_options else 0
            y_idx = y_metric_options.index(y_metric) if y_metric in y_metric_options else 0
            g_idx = group_field_options.index(group_field) if group_field in group_field_options else 0
            x_field = ac1.selectbox("X-axis", x_field_options, index=x_idx)
            y_metric = ac2.selectbox("Y-axis metric", y_metric_options, index=y_idx)
            group_field = ac3.selectbox(
                "Group by",
                group_field_options,
                index=g_idx,
                format_func=lambda x: "None" if x is None else x,
            )

# ------------------------------------------------------------------
# 5. Extract sweep data and render chart
# ------------------------------------------------------------------
st.divider()
st.subheader("Chart")

sweep_data = extract_sweep_data(
    run_paths=selected_paths,
    x_field=x_field,
    y_metric=y_metric,
    group_field=group_field,
)

if sweep_data["x"]:
    render_sweep_chart(sweep_data, sweep_data["x_label"], sweep_data["y_label"])
else:
    st.warning(
        "No data points could be extracted. "
        "Check that selected runs have the required config fields and summary metrics."
    )

# ------------------------------------------------------------------
# 6. Comparison table
# ------------------------------------------------------------------
st.divider()
st.subheader("Comparison Table")

if sweep_data["x"]:
    table_data = {
        "Run": sweep_data["runs"],
        sweep_data["x_label"]: sweep_data["x"],
        sweep_data["y_label"]: sweep_data["y"],
    }
    if any(g != "all" for g in sweep_data["group"]):
        table_data["Group"] = sweep_data["group"]

    comp_df = pd.DataFrame(table_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # CSV export
    csv = comp_df.to_csv(index=False)
    st.download_button(
        "Export CSV",
        data=csv,
        file_name="comparison.csv",
        mime="text/csv",
    )

# ------------------------------------------------------------------
# 7. Gauge invariance note
# ------------------------------------------------------------------
st.caption(
    "Note: Only gauge-invariant quantities (reprojection errors, relative poses) "
    "are meaningful to compare across runs. Absolute pose values depend on gauge "
    "constraints (which pose/extrinsic is fixed) and may differ between runs "
    "without indicating a real difference."
)
