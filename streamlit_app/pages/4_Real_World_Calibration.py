"""Page 4: Real-World Calibration Pipeline.

Runs colmap_pipeline.py on a pre-triangulated COLMAP reconstruction:
  COLMAP reconstruction -> problem.json -> BA solver -> plots + comparison
"""

from __future__ import annotations

import sys
import time
import json
import queue
import subprocess
from pathlib import Path
from datetime import datetime

import streamlit as st

from ui_backend import _reader_thread, drain_stdout

st.set_page_config(page_title="Real-World Calibration", layout="wide")
st.title("Real-World Calibration")

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
_APP_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _APP_DIR.parent
_PRETRI_DIR = _REPO_ROOT / "pre-triangulation"

# Platform-aware venv Python detection
_VENV_DIR = _REPO_ROOT / ".venv"
if sys.platform == "win32":
    _PYTHON_EXE = _VENV_DIR / "Scripts" / "python.exe"
else:
    _PYTHON_EXE = _VENV_DIR / "bin" / "python"

_PIPELINE_SCRIPT = _REPO_ROOT / "calibri" / "tools" / "pipelines" / "colmap_pipeline.py"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _discover_recordings() -> list[str]:
    """Return recording names from pre-triangulation/.

    Supports both flat (sequence_name/) and nested (kitti/kitti_01/) layouts.
    A directory is a recording if it contains an images/ subdirectory.
    """
    if not _PRETRI_DIR.is_dir():
        return []
    results = []
    for d in _PRETRI_DIR.iterdir():
        if not d.is_dir():
            continue
        if (d / "images").is_dir():
            results.append(d.name)
        else:
            # Check one level deeper for nested recordings
            for sub in d.iterdir():
                if sub.is_dir() and (sub / "images").is_dir():
                    results.append(f"{d.name}/{sub.name}")
    return sorted(results)


def _find_reconstruction(recording_name: str) -> Path | None:
    """Return the single reconstruction directory under a recording, or None."""
    recon_root = _PRETRI_DIR / recording_name / "reconstruction"
    if not recon_root.is_dir():
        return None
    subdirs = [d for d in recon_root.iterdir() if d.is_dir()]
    return subdirs[0] if subdirs else None


def _detect_pipeline_phase(stdout: str) -> str:
    """Detect current pipeline phase from stdout text."""
    if not stdout:
        return "Starting..."
    patterns = [
        ("Pipeline complete.", "Complete"),
        ("Approximate improvement:", "Complete"),
        ("Generating error plots", "Generating plots"),
        ("BA completed.", "BA finished"),
        ("Running BA solver", "Solving"),
        ("Converting COLMAP", "Converting"),
        ("Created problem.json", "Converting"),
    ]
    for pattern, phase in patterns:
        if pattern in stdout:
            return phase
    return "Running..."


def _start_pipeline(cmd: list[str], output_dir: Path) -> dict:
    """Launch the pipeline subprocess with non-blocking stdout."""
    import threading

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(_REPO_ROOT),
    )
    q: queue.Queue = queue.Queue()
    reader = threading.Thread(target=_reader_thread, args=(proc.stdout, q), daemon=True)
    reader.start()
    return {
        "proc": proc,
        "stdout_queue": q,
        "stdout_lines": [],
        "start_time": time.time(),
        "output_dir": output_dir,
    }


# ------------------------------------------------------------------
# 1. Data Source
# ------------------------------------------------------------------
recordings = _discover_recordings()

with st.expander("Data Source", expanded=True):
    if not recordings:
        st.warning("No recordings found in `pre-triangulation/`")
        st.stop()

    recording = st.selectbox(
        "Recording",
        options=recordings,
        help="Select a recording from pre-triangulation/",
    )

    recording_dir = _PRETRI_DIR / recording
    recon_path = _find_reconstruction(recording)
    reconstruction_path = str(recon_path) if recon_path else ""
    database_path = str(recording_dir / "database_rig.db")

    st.caption(f"Reconstruction: `{reconstruction_path}`")
    st.caption(f"Database: `{database_path}`")

# ------------------------------------------------------------------
# 2. Solver Settings
# ------------------------------------------------------------------
with st.expander("Solver Settings", expanded=True):
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        max_iterations = st.number_input(
            "Max iterations", min_value=1, max_value=1000, value=200,
        )
    with sc2:
        loss_type = st.selectbox(
            "Loss function",
            options=["Huber", "Cauchy", "SoftL1", "None"],
        )
    with sc3:
        loss_scale = st.number_input(
            "Loss scale", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
        )

# ------------------------------------------------------------------
# 3. Outlier Rejection
# ------------------------------------------------------------------
with st.expander("Outlier Rejection", expanded=True):
    enable_rejection = st.checkbox(
        "Enable iterative outlier rejection",
        value=False,
        help="3 passes of solve-reject-resolve with adaptive threshold "
             "(3 × median reprojection error).",
    )

    if enable_rejection:
        outlier_passes = 3
        st.caption("3 passes, adaptive threshold: 3 × median error")
    else:
        outlier_passes = 0

# ------------------------------------------------------------------
# 4. Optimization Flags
# ------------------------------------------------------------------
is_kitti = recording.lower().startswith("kitti")

with st.expander("Optimization Flags", expanded=True):
    if is_kitti:
        st.info(
            "KITTI images are stereo-rectified. Their intrinsics and distortion "
            "coefficients are optimized for the rectified geometry and should not "
            "be refined by the solver."
        )
    fc1, fc2, fc3 = st.columns(3)
    opt_intrinsics = fc1.checkbox("Optimize intrinsics", value=not is_kitti)
    opt_distortion = fc2.checkbox("Optimize distortion", value=not is_kitti)
    opt_extrinsics = fc3.checkbox("Optimize extrinsics", value=True)

# ------------------------------------------------------------------
# 5. Validation
# ------------------------------------------------------------------
errors = []
if not recon_path:
    errors.append("No reconstruction found -- run triangulation first")
elif not recon_path.exists():
    errors.append(f"Reconstruction not found: `{reconstruction_path}`")
if not Path(database_path).exists():
    errors.append(f"Database not found: `{database_path}`")

if errors:
    for err in errors:
        st.error(err)
else:
    st.success("Paths validated. Ready to run.")

can_run = not errors

# ------------------------------------------------------------------
# 6. Run button
# ------------------------------------------------------------------
st.divider()

is_running = st.session_state.get("pipeline_tracking") is not None

if st.button("Run Pipeline", disabled=(not can_run or is_running), type="primary"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = recording.replace("/", "_").replace("\\", "_")
    output_dir = _REPO_ROOT / "results" / f"{safe_name}_ba_{timestamp}"

    cmd = [
        str(_PYTHON_EXE),
        str(_PIPELINE_SCRIPT),
        "--reconstruction", reconstruction_path,
        "--database", database_path,
        "--output-dir", str(output_dir),
        "--max-iterations", str(max_iterations),
        "--loss-type", loss_type,
        "--loss-scale", str(loss_scale),
        "--use-rig",
        "-v",
    ]
    if opt_intrinsics:
        cmd.append("--optimize-intrinsics")
    if opt_distortion:
        cmd.append("--optimize-distortion")
    if opt_extrinsics:
        cmd.append("--optimize-extrinsics")
    if outlier_passes > 0:
        cmd.extend(["--outlier-passes", str(outlier_passes)])

    tracking = _start_pipeline(cmd, output_dir)
    st.session_state.pipeline_tracking = tracking
    st.rerun()

# ------------------------------------------------------------------
# 7. Progress display (while running)
# ------------------------------------------------------------------
if is_running:
    tracking = st.session_state.pipeline_tracking
    proc = tracking["proc"]
    stdout_text = drain_stdout(tracking)
    phase = _detect_pipeline_phase(stdout_text)
    elapsed = time.time() - tracking["start_time"]

    st.subheader(f"Pipeline: {recording}")

    col_p1, col_p2 = st.columns(2)
    col_p1.metric("Phase", phase)
    col_p2.metric("Elapsed", f"{elapsed:.0f}s")

    lines = stdout_text.splitlines()
    tail = "\n".join(lines[-40:]) if lines else ""
    st.code(tail, language="text")

    poll = proc.poll()
    if poll is not None:
        # Process finished — ensure cleanup
        try:
            proc.terminate()
        except OSError:
            pass
        final_stdout = drain_stdout(tracking)
        output_dir = tracking["output_dir"]
        st.session_state.pipeline_tracking = None
        st.session_state.pipeline_result = {
            "returncode": poll,
            "output": final_stdout,
            "elapsed": time.time() - tracking["start_time"],
            "output_dir": output_dir,
        }
        st.rerun()
    else:
        time.sleep(2)
        st.rerun()

# ------------------------------------------------------------------
# 8. Results display (after completion)
# ------------------------------------------------------------------
if "pipeline_result" in st.session_state:
    result_state = st.session_state.pipeline_result
    returncode = result_state["returncode"]
    elapsed = result_state["elapsed"]
    output_dir = Path(result_state["output_dir"])

    st.divider()
    st.subheader("Results")

    # --- Status bar ---
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if returncode == 0:
            st.success("Pipeline completed successfully")
        else:
            st.error(f"Pipeline failed (exit code {returncode})")
    with col_s2:
        st.metric("Elapsed", f"{elapsed:.1f}s")

    # --- Tabbed results ---
    result_json = output_dir / "result.json"
    problem_json = output_dir / "problem.json"
    comparison_json = output_dir / "colmap_comparison.json"
    error_plot = output_dir / "errors.png"

    has_result = result_json.exists()

    tab_names = ["Summary"]
    if has_result and comparison_json.exists():
        tab_names.append("Comparison")
    tab_names.append("Diagnostics")
    tab_names.append("Log")

    tabs = st.tabs(tab_names)
    tab_idx = 0

    # --- Tab: Metrics ---
    with tabs[tab_idx]:
        tab_idx += 1
        if has_result:
            with open(result_json) as f:
                result_data = json.load(f)

            summary = result_data.get("summary", {})

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Solver", "Success" if summary.get("success") else "FAILED")
            mc2.metric("Iterations", summary.get("iterations", "N/A"))
            median_err = summary.get("median_reproj_px")
            mean_err = summary.get("mean_reproj_px")
            mc3.metric("Median Error", f"{median_err:.2f} px" if median_err is not None else "N/A")
            mc4.metric("Mean Error", f"{mean_err:.2f} px" if mean_err is not None else "N/A")

            p95_err = summary.get("p95_reproj_px")
            max_err = summary.get("max_reproj_px")
            initial_cost = summary.get("initial_cost")
            final_cost = summary.get("final_cost")

            mc5, mc6, mc7, mc8 = st.columns(4)
            mc5.metric("P95 Error", f"{p95_err:.2f} px" if p95_err is not None else "N/A")
            mc6.metric("Max Error", f"{max_err:.2f} px" if max_err is not None else "N/A")
            mc7.metric("Initial Cost", f"{initial_cost:.2f}" if initial_cost is not None else "N/A")
            mc8.metric("Final Cost", f"{final_cost:.2f}" if final_cost is not None else "N/A")

            termination_msg = summary.get("termination_message", "")
            if termination_msg:
                st.caption(f"Termination: {termination_msg}")

            if median_err is not None and median_err > 50.0:
                st.warning("Median error is very high -- solver may have diverged")

            # Outlier rejection stats (if rejection was used)
            total_rejected = summary.get("total_rejected", 0)
            if total_rejected > 0:
                st.markdown("#### Outlier Rejection")
                rr1, rr2, rr3 = st.columns(3)
                rr1.metric("Passes Used", summary.get("outlier_rejection_passes_used", 0))
                rr2.metric("Total Rejected", total_rejected)
                num_obs = summary.get("num_observations", 0)
                rej_pct = (total_rejected / (num_obs + total_rejected) * 100
                           if (num_obs + total_rejected) > 0 else 0)
                rr3.metric("Rejection Rate", f"{rej_pct:.1f}%")
                rejected_per_pass = summary.get("rejected_per_pass", [])
                if rejected_per_pass:
                    pass_cols = st.columns(len(rejected_per_pass))
                    for pi, (col, cnt) in enumerate(zip(pass_cols, rejected_per_pass)):
                        col.metric(f"Pass {pi + 1}", cnt)

            # Problem statistics
            if problem_json.exists():
                st.markdown("#### Problem Size")
                with open(problem_json) as f:
                    problem_data = json.load(f)
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("Cameras", len(problem_data.get("cameras", [])))
                pc2.metric("Poses", len(problem_data.get("poses", [])))
                pc3.metric("Landmarks", len(problem_data.get("landmarks", [])))
                pc4.metric("Observations", len(problem_data.get("observations", [])))

                # Show active flags
                flags = problem_data.get("flags", {})
                if flags:
                    st.markdown("#### Active Flags")
                    flag_cols = st.columns(5)
                    flag_cols[0].metric("Poses", "ON" if flags.get("opt_poses") else "OFF")
                    flag_cols[1].metric("Landmarks", "ON" if flags.get("opt_landmarks") else "OFF")
                    flag_cols[2].metric("Intrinsics", "ON" if flags.get("opt_intrinsics") else "OFF")
                    flag_cols[3].metric("Distortion", "ON" if flags.get("opt_distortion") else "OFF")
                    flag_cols[4].metric("Extrinsics", "ON" if flags.get("opt_extrinsics") else "OFF")
        else:
            st.warning("result.json not found in output directory")

    # --- Tab: Comparison ---
    if has_result and comparison_json.exists():
        with tabs[tab_idx]:
            tab_idx += 1
            with open(comparison_json) as f:
                comp_data = json.load(f)

            note = comp_data.get("note")
            if note:
                st.caption(f"Note: {note}")

            if comp_data.get("ba", {}).get("diverged", False):
                st.error("BA solver appears to have diverged. Comparison may be unreliable.")

            cc1, cc2, cc3 = st.columns(3)
            colmap_median = comp_data.get("colmap", {}).get("median_px")
            ba_median = comp_data.get("ba", {}).get("median_px")
            improvement = comp_data.get("improvement_pct")

            cc1.metric(
                "COLMAP Baseline (per-point)",
                f"{colmap_median:.2f} px" if colmap_median is not None else "N/A",
            )
            cc2.metric(
                "Our BA (per-obs)",
                f"{ba_median:.2f} px" if ba_median is not None else "N/A",
            )
            cc3.metric(
                "Improvement",
                f"{improvement:+.1f}%" if improvement is not None else "N/A",
                delta=f"{improvement:.1f}%" if improvement is not None else None,
            )

            # Per-source detail
            with st.expander("Detailed Numbers"):
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.markdown("**COLMAP**")
                    colmap_data = comp_data.get("colmap", {})
                    st.metric("Points", colmap_data.get("num_points", "N/A"))
                    colmap_mean = colmap_data.get("mean_px")
                    st.metric("Mean", f"{colmap_mean:.2f} px" if colmap_mean is not None else "N/A")
                    st.metric("Median", f"{colmap_median:.2f} px" if colmap_median is not None else "N/A")
                with dc2:
                    st.markdown("**Bundle Adjustment**")
                    ba_data = comp_data.get("ba", {})
                    st.metric("Residuals", ba_data.get("num_residuals", "N/A"))
                    ba_mean = ba_data.get("mean_px")
                    st.metric("Mean", f"{ba_mean:.2f} px" if ba_mean is not None else "N/A")
                    st.metric("Median", f"{ba_median:.2f} px" if ba_median is not None else "N/A")

    # --- Tab: Plots ---
    with tabs[tab_idx]:
        tab_idx += 1
        if has_result:
            st.markdown("#### Error Distribution")
            if error_plot.exists():
                st.image(str(error_plot), use_container_width=True)
            else:
                st.info("No error plot generated")

            st.markdown("#### Frame Overlays")
            overlay_files = sorted(output_dir.glob("overlay_frame_*.png"))
            if overlay_files:
                for i in range(0, len(overlay_files), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(overlay_files):
                            with col:
                                f = overlay_files[idx]
                                st.image(str(f), caption=f"Frame {f.stem.split('_')[-1]}", use_container_width=True)
            else:
                st.info("No overlay images generated")
        else:
            st.info("No plots available")

    # --- Tab: Log ---
    with tabs[tab_idx]:
        pipeline_output = result_state.get("output", "")
        if pipeline_output.strip():
            log_lines = pipeline_output.splitlines()
            if len(log_lines) <= 200:
                st.code(pipeline_output, language="text")
            else:
                st.code("\n".join(log_lines[:200]), language="text")
                with st.expander(f"Show remaining {len(log_lines) - 200} lines"):
                    st.code("\n".join(log_lines[200:]), language="text")
        else:
            st.info("Log is empty.")

    # --- Clear ---
    if st.button("Clear Results"):
        st.session_state.pop("pipeline_result", None)
        st.rerun()
