"""Batch runner for the bundled KITTI sequences.

Runs ``ba_cli`` on the 6 KITTI problem.json files shipped under
``thesis_results/batch_runs/`` with stereo-rectified settings: poses,
landmarks and extrinsics are optimised; intrinsics and distortion are kept
fixed.

Usage:
    python -m calibri.tools.analysis.batch_kitti_run
"""

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from scipy.spatial.transform import Rotation

from calibri.tools.utils import json_quat_to_scipy, load_json, run_ba_cli
from calibri.tools.analysis.sequences import (
    OUTPUT_DIR, TABLES_DIR, FIGURES_DIR,
    KITTI_SEQUENCES,
)

FLAGS = {
    "opt_poses": True,
    "opt_landmarks": True,
    "opt_intrinsics": False,
    "opt_distortion": False,
    "opt_extrinsics": True,
}


def prepare_problem(problem_path: Path, work_dir: Path) -> Path:
    with open(problem_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "camera_rig" not in data or "reference_camera_id" not in data.get("camera_rig", {}):
        cam_ids = [c["id"] for c in data["cameras"]]
        data["camera_rig"] = {"reference_camera_id": min(cam_ids) if cam_ids else 0}
    data["flags"] = FLAGS
    out = work_dir / "problem.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return out


def run_solver(problem_path: Path, result_path: Path, timeout_s: int = 1200) -> Tuple[bool, str]:
    extra_args = [
        "--loss-type", "Huber",
        "--loss-scale", "1.0",
        "--max-iterations", "200",
        "--outlier-passes", "3",
    ]
    success, output, _ = run_ba_cli(problem_path, result_path, timeout=timeout_s, extra_args=extra_args)
    return success, output


def parse_result(result_path: Path) -> Optional[Dict[str, Any]]:
    if not result_path.exists():
        return None
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        s = data["summary"]
        return {
            "success": s.get("success", False),
            "iterations": s.get("iterations", 0),
            "initial_cost": s.get("initial_cost", 0.0),
            "final_cost": s.get("final_cost", 0.0),
            "mean_reproj_px": s.get("mean_reproj_px", 0.0),
            "median_reproj_px": s.get("median_reproj_px", 0.0),
            "rmse_reproj_px": s.get("rmse_reproj_px", 0.0),
            "p95_reproj_px": s.get("p95_reproj_px", 0.0),
            "max_reproj_px": s.get("max_reproj_px", 0.0),
            "num_observations": s.get("num_observations", 0),
            "num_inliers": s.get("num_inliers", 0),
            "total_time_ms": s.get("total_time_ms", 0.0),
            "termination_type": s.get("termination_type", ""),
            "final_params": data.get("final_params", {}),
        }
    except Exception as e:
        print(f"  [WARN] Failed to parse {result_path}: {e}")
        return None


def compute_extrinsic_delta(initial_cameras, optimized_cameras):
    deltas = []
    init_map = {c["id"]: c for c in initial_cameras}
    for opt_cam in optimized_cameras:
        cid = opt_cam["id"]
        if cid not in init_map:
            continue
        init_cam = init_map[cid]
        iq = init_cam["extrinsics"]["T_vehicle_to_camera"]["q"]
        it = init_cam["extrinsics"]["T_vehicle_to_camera"]["t"]
        oq = opt_cam["extrinsic"]["q"]
        ot = opt_cam["extrinsic"]["t"]
        R_init = Rotation.from_quat(json_quat_to_scipy(iq))
        R_opt = Rotation.from_quat(json_quat_to_scipy(oq))
        R_delta = R_init.inv() * R_opt
        deltas.append({
            "camera_id": cid,
            "rot_deg": np.degrees(R_delta.magnitude()),
            "trans_m": np.linalg.norm(np.array(ot) - np.array(it)),
        })
    return deltas


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUTPUT_DIR / "batch_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    print("=" * 70)
    print("BATCH RUNNER — KITTI (6 sequences)")
    print("Flags: poses/landmarks/extrinsics = true, intrinsics/distortion = false")
    print("=" * 70)

    for seq_name, problem_path in KITTI_SEQUENCES.items():
        print(f"\n{'-' * 60}")
        print(f"  Sequence: {seq_name}")
        print(f"  Problem:  {problem_path}")

        if not problem_path.exists():
            print("  [SKIP] problem.json not found")
            rows.append({"sequence": seq_name, "dataset": "KITTI", "success": False})
            continue

        try:
            initial_cameras = [c for c in json.load(open(problem_path, encoding="utf-8"))["cameras"]]
        except Exception:
            initial_cameras = []

        seq_dir = run_dir / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)
        result_path = seq_dir / "result.json"

        actual_problem = prepare_problem(problem_path, seq_dir)
        print("  [PATCH] poses, landmarks, extrinsics = true; intrinsics, distortion = false")

        print("  Running solver (Huber, scale=1.0, max_iter=200, outlier_passes=3)...")
        t0 = time.time()
        success, stdout = run_solver(actual_problem, result_path)
        wall_time = time.time() - t0
        print(f"  Solver finished in {wall_time:.1f}s (success={success})")

        (seq_dir / "solver_stdout.log").write_text(stdout, encoding="utf-8")

        if not success:
            print("  [FAIL]")
            rows.append({"sequence": seq_name, "dataset": "KITTI", "success": False})
            continue

        metrics = parse_result(result_path)
        if metrics is None:
            rows.append({"sequence": seq_name, "dataset": "KITTI", "success": False})
            continue

        inlier_ratio = metrics["num_inliers"] / metrics["num_observations"] if metrics["num_observations"] > 0 else 0.0

        row = {
            "sequence": seq_name, "dataset": "KITTI", "success": metrics["success"],
            "iterations": metrics["iterations"],
            "initial_cost": round(metrics["initial_cost"], 2),
            "final_cost": round(metrics["final_cost"], 2),
            "mean_reproj_px": round(metrics["mean_reproj_px"], 4),
            "median_reproj_px": round(metrics["median_reproj_px"], 4),
            "rmse_reproj_px": round(metrics["rmse_reproj_px"], 4),
            "p95_reproj_px": round(metrics["p95_reproj_px"], 4),
            "max_reproj_px": round(metrics["max_reproj_px"], 4),
            "num_observations": metrics["num_observations"],
            "num_inliers": metrics["num_inliers"],
            "inlier_ratio": round(inlier_ratio, 4),
            "total_time_ms": round(metrics["total_time_ms"], 1),
            "termination": metrics["termination_type"],
        }

        if metrics["final_params"] and "cameras" in metrics["final_params"]:
            deltas = compute_extrinsic_delta(initial_cameras, metrics["final_params"]["cameras"])
            if deltas:
                problem_data = load_json(problem_path)
                ref_id = problem_data.get("camera_rig", {}).get("reference_camera_id", 0)
                non_ref = [d for d in deltas if d["camera_id"] != ref_id]
                if non_ref:
                    row["avg_extr_rot_deg"] = round(np.mean([d["rot_deg"] for d in non_ref]), 4)
                    row["avg_extr_trans_m"] = round(np.mean([d["trans_m"] for d in non_ref]), 4)
                    row["max_extr_rot_deg"] = round(np.max([d["rot_deg"] for d in non_ref]), 4)
                    row["max_extr_trans_m"] = round(np.max([d["trans_m"] for d in non_ref]), 4)

        rows.append(row)
        print(f"  RMSE: {row['rmse_reproj_px']} px | Inliers: {row['inlier_ratio']*100:.1f}% | Time: {row['total_time_ms']} ms")

    # ── Merge into real_world_metrics.csv ──────────────────────────────
    csv_path = TABLES_DIR / "real_world_metrics.csv"
    existing: Dict[str, Dict] = {}
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing[r["sequence"]] = r
    for row in rows:
        existing[row["sequence"]] = row

    ordered = [existing[s] for s in KITTI_SEQUENCES.keys() if s in existing]

    fieldnames = [
        "sequence", "dataset", "success", "iterations",
        "initial_cost", "final_cost",
        "mean_reproj_px", "median_reproj_px", "rmse_reproj_px", "p95_reproj_px", "max_reproj_px",
        "num_observations", "num_inliers", "inlier_ratio",
        "total_time_ms", "termination",
        "avg_extr_rot_deg", "avg_extr_trans_m",
        "max_extr_rot_deg", "max_extr_trans_m",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            writer.writerow(row)

    print(f"\n{'=' * 70}")
    print(f"CSV merged into: {csv_path}")
    print(f"{'=' * 70}")

    print(f"\n{'Sequence':<15} {'RMSE(px)':<10} {'Inlier%':<10} {'Time(ms)':<10} {'OK'}")
    print("-" * 55)
    for row in rows:
        if row.get("success"):
            print(f"{row['sequence']:<15} {row.get('rmse_reproj_px',0):<10.4f} "
                  f"{row.get('inlier_ratio',0)*100:<10.1f} {row.get('total_time_ms',0):<10.1f} YES")
        else:
            print(f"{row['sequence']:<15} FAILED")


if __name__ == "__main__":
    main()
