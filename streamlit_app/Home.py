"""Calibri -- Bundle Adjustment Experiment Framework (Streamlit UI)."""

import streamlit as st

st.set_page_config(page_title="Calibri -- BA Experiments", layout="wide")

st.title("Calibri -- Bundle Adjustment Experiment Framework")

st.markdown(
    """
This application provides a graphical interface for the thesis experiment
framework.  Use the sidebar to navigate between pages:

- **Synthetic Experiments** -- Configure and run synthetic BA experiments with presets, scene previews, and GT evaluation.
- **Browse Runs** -- Inspect completed runs with metrics, ground-truth recovery tables, diagnostics, and file downloads.
- **Compare Runs** -- Plot sweep comparisons across runs with thesis presets and CSV export.
- **Real-World Calibration** -- Run bundle adjustment on real automotive recordings (pre-triangulated COLMAP reconstructions).

**Synthetic experiments** live in the `runs/` directory. The UI wraps
the `calibri.framework` library and launches experiments via `scripts/run_experiment.py`.

**Calibration pipeline** outputs go to timestamped `results/<recording>_ba_YYYYMMDD_HHMMSS/` directories.
"""
)
