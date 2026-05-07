"""Shared sequence registry and output paths for evaluation scripts.

Evaluation scripts (batch_kitti_run, plot_trajectory, etc.) import the
sequence definitions from here to avoid duplication.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

OUTPUT_DIR = REPO_ROOT / "thesis_results"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
BATCH_DIR = OUTPUT_DIR / "batch_runs"

BA_CLI = REPO_ROOT / "build" / "calibri" / "app" / "Release" / "ba_cli.exe"

KITTI_SEQUENCES = {
    "kitti_0002": BATCH_DIR / "kitti_0002" / "problem.json",
    "kitti_0009": BATCH_DIR / "kitti_0009" / "problem.json",
    "kitti_0019": BATCH_DIR / "kitti_0019" / "problem.json",
    "kitti_0039": BATCH_DIR / "kitti_0039" / "problem.json",
    "kitti_0051": BATCH_DIR / "kitti_0051" / "problem.json",
    "kitti_0117": BATCH_DIR / "kitti_0117" / "problem.json",
}

ALL_SEQUENCES = dict(KITTI_SEQUENCES)
