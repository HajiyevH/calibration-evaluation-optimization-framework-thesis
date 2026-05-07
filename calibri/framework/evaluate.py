"""Ground-truth evaluation for synthetic experiments (A11).

Compares solver output (result.json) against ground truth (ground_truth.json)
and computes parameter-level recovery errors for:
  - Extrinsic rotation / translation per camera
  - Intrinsic parameters per camera (fx, fy, cx, cy)
  - Distortion coefficients per camera
  - Per-pose rotation / translation
  - Landmark position error (aggregate)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy.spatial.transform import Rotation

from calibri.tools.utils import json_quat_to_scipy, load_json


def _rotation_error_deg(q1_wxyz: List[float], q2_wxyz: List[float]) -> float:
    """Compute rotation error in degrees between two quaternions [w,x,y,z]."""
    R1 = Rotation.from_quat(json_quat_to_scipy(q1_wxyz))
    R2 = Rotation.from_quat(json_quat_to_scipy(q2_wxyz))
    dR = R1.inv() * R2
    angle_rad = dR.magnitude()
    return float(np.degrees(angle_rad))


def _translation_error_m(t1: List[float], t2: List[float]) -> float:
    """Compute L2 translation error in meters."""
    return float(np.linalg.norm(np.array(t1) - np.array(t2)))


def evaluate_against_gt(
    result_path: Path,
    gt_path: Path,
) -> Dict:
    """Compare solver output to ground truth and compute parameter-level errors.

    Args:
        result_path: Path to result.json (solver output)
        gt_path: Path to ground_truth.json (unperturbed state)

    Returns:
        Dict with structured evaluation results:
        {
            "extrinsics": [{camera_id, rot_err_deg, trans_err_m}, ...],
            "intrinsics": [{camera_id, fx_err, fy_err, cx_err, cy_err}, ...],
            "distortion": [{camera_id, coeffs: {k1_err, ...}}, ...],
            "poses": [{timestamp_ns, rot_err_deg, trans_err_m}, ...],
            "landmarks": {mean_err_m, median_err_m, p95_err_m, count},
            "summary": {
                mean_extr_rot_err_deg, mean_extr_trans_err_m,
                mean_pose_rot_err_deg, mean_pose_trans_err_m,
                mean_intrinsic_err_px, mean_landmark_err_m,
            }
        }
    """
    result = load_json(result_path)
    gt = load_json(gt_path)

    eval_result = {
        "extrinsics": [],
        "intrinsics": [],
        "distortion": [],
        "poses": [],
        "landmarks": {},
        "summary": {},
    }

    # --- Extrinsic evaluation ---
    gt_cameras = {c["id"]: c for c in gt.get("cameras", [])}
    final_cameras = {}
    final_params = result.get("final_params", {})
    for cam in final_params.get("cameras", []):
        final_cameras[cam["id"]] = cam

    extr_rot_errors = []
    extr_trans_errors = []
    for cam_id, gt_cam in gt_cameras.items():
        opt_cam = final_cameras.get(cam_id)
        if opt_cam is None:
            continue

        gt_ext = gt_cam.get("extrinsic", {})
        opt_ext = opt_cam.get("extrinsic", {})
        if not gt_ext.get("q") or not opt_ext.get("q"):
            continue

        rot_err = _rotation_error_deg(gt_ext["q"], opt_ext["q"])
        trans_err = _translation_error_m(gt_ext["t"], opt_ext["t"])
        eval_result["extrinsics"].append({
            "camera_id": cam_id,
            "rot_err_deg": round(rot_err, 6),
            "trans_err_m": round(trans_err, 6),
        })
        extr_rot_errors.append(rot_err)
        extr_trans_errors.append(trans_err)

    # --- Intrinsic evaluation ---
    intr_errors_all = []
    for cam_id, gt_cam in gt_cameras.items():
        opt_cam = final_cameras.get(cam_id)
        if opt_cam is None or "intr" not in opt_cam:
            continue

        gt_intr = gt_cam.get("intrinsics", {})
        opt_intr = opt_cam["intr"]  # [fx, fy, cx, cy, dist...]

        fx_err = abs(opt_intr[0] - gt_intr.get("fx", 0))
        fy_err = abs(opt_intr[1] - gt_intr.get("fy", 0))
        cx_err = abs(opt_intr[2] - gt_intr.get("cx", 0))
        cy_err = abs(opt_intr[3] - gt_intr.get("cy", 0))

        eval_result["intrinsics"].append({
            "camera_id": cam_id,
            "fx_err": round(fx_err, 6),
            "fy_err": round(fy_err, 6),
            "cx_err": round(cx_err, 6),
            "cy_err": round(cy_err, 6),
        })
        intr_errors_all.extend([fx_err, fy_err, cx_err, cy_err])

    # --- Distortion evaluation ---
    dist_errors_all = []
    for cam_id, gt_cam in gt_cameras.items():
        opt_cam = final_cameras.get(cam_id)
        if opt_cam is None or "intr" not in opt_cam:
            continue

        gt_dist = gt_cam.get("distortion", {})
        gt_model = gt_cam.get("model", "Pinhole")
        opt_intr = opt_cam["intr"]  # [fx, fy, cx, cy, dist_coeffs...]

        # Distortion coefficients start at index 4
        opt_dist = opt_intr[4:]

        if gt_model == "Fisheye":
            dist_keys = ["k1", "k2", "k3", "k4"]
        else:
            dist_keys = ["k1", "k2", "p1", "p2", "k3"]

        coeffs = {}
        for i, key in enumerate(dist_keys):
            gt_val = gt_dist.get(key, 0.0)
            opt_val = opt_dist[i] if i < len(opt_dist) else 0.0
            err = abs(opt_val - gt_val)
            coeffs[f"{key}_err"] = round(err, 8)
            dist_errors_all.append(err)

        eval_result["distortion"].append({
            "camera_id": cam_id,
            "coeffs": coeffs,
        })

    # --- Pose evaluation ---
    gt_poses = {p["timestamp_ns"]: p for p in gt.get("poses", [])}
    opt_poses = {p["timestamp_ns"]: p for p in final_params.get("poses", [])}

    pose_rot_errors = []
    pose_trans_errors = []
    for ts, gt_pose in gt_poses.items():
        opt_pose = opt_poses.get(ts)
        if opt_pose is None:
            continue

        rot_err = _rotation_error_deg(gt_pose["q"], opt_pose["q"])
        trans_err = _translation_error_m(gt_pose["t"], opt_pose["t"])
        eval_result["poses"].append({
            "timestamp_ns": ts,
            "rot_err_deg": round(rot_err, 6),
            "trans_err_m": round(trans_err, 6),
        })
        pose_rot_errors.append(rot_err)
        pose_trans_errors.append(trans_err)

    # --- Landmark evaluation ---
    gt_lm = {lm["id"]: lm for lm in gt.get("landmarks", [])}
    opt_lm_list = final_params.get("landmarks", [])
    opt_lm = {lm["id"]: lm for lm in opt_lm_list}

    lm_errors = []
    for lm_id, gt_l in gt_lm.items():
        opt_l = opt_lm.get(lm_id)
        if opt_l is None:
            continue
        err = np.linalg.norm([
            opt_l["X"] - gt_l["X"],
            opt_l["Y"] - gt_l["Y"],
            opt_l["Z"] - gt_l["Z"],
        ])
        lm_errors.append(float(err))

    if lm_errors:
        lm_arr = np.array(lm_errors)
        eval_result["landmarks"] = {
            "mean_err_m": round(float(np.mean(lm_arr)), 6),
            "median_err_m": round(float(np.median(lm_arr)), 6),
            "p95_err_m": round(float(np.percentile(lm_arr, 95)), 6),
            "max_err_m": round(float(np.max(lm_arr)), 6),
            "count": len(lm_errors),
        }

    # --- Summary ---
    summary = {}
    if extr_rot_errors:
        summary["mean_extr_rot_err_deg"] = round(float(np.mean(extr_rot_errors)), 6)
        summary["max_extr_rot_err_deg"] = round(float(np.max(extr_rot_errors)), 6)
    if extr_trans_errors:
        summary["mean_extr_trans_err_m"] = round(float(np.mean(extr_trans_errors)), 6)
        summary["max_extr_trans_err_m"] = round(float(np.max(extr_trans_errors)), 6)
    if pose_rot_errors:
        summary["mean_pose_rot_err_deg"] = round(float(np.mean(pose_rot_errors)), 6)
        summary["max_pose_rot_err_deg"] = round(float(np.max(pose_rot_errors)), 6)
    if pose_trans_errors:
        summary["mean_pose_trans_err_m"] = round(float(np.mean(pose_trans_errors)), 6)
        summary["max_pose_trans_err_m"] = round(float(np.max(pose_trans_errors)), 6)
    if intr_errors_all:
        summary["mean_intrinsic_err_px"] = round(float(np.mean(intr_errors_all)), 6)
    if dist_errors_all:
        summary["mean_distortion_err"] = round(float(np.mean(dist_errors_all)), 8)
    if lm_errors:
        summary["mean_landmark_err_m"] = round(float(np.mean(lm_errors)), 6)

    eval_result["summary"] = summary
    return eval_result


def format_evaluation_text(eval_result: Dict) -> str:
    """Format GT evaluation results as human-readable text."""
    lines = []
    lines.append("Ground-Truth Evaluation")
    lines.append("-" * 40)

    summary = eval_result.get("summary", {})

    # Poses
    if "mean_pose_rot_err_deg" in summary:
        lines.append(f"Pose rotation error:    mean={summary['mean_pose_rot_err_deg']:.4f} deg"
                     f"  max={summary.get('max_pose_rot_err_deg', 0):.4f} deg")
    if "mean_pose_trans_err_m" in summary:
        lines.append(f"Pose translation error: mean={summary['mean_pose_trans_err_m']:.4f} m"
                     f"  max={summary.get('max_pose_trans_err_m', 0):.4f} m")

    # Extrinsics
    if "mean_extr_rot_err_deg" in summary:
        lines.append(f"Extr rotation error:    mean={summary['mean_extr_rot_err_deg']:.4f} deg"
                     f"  max={summary.get('max_extr_rot_err_deg', 0):.4f} deg")
    if "mean_extr_trans_err_m" in summary:
        lines.append(f"Extr translation error: mean={summary['mean_extr_trans_err_m']:.4f} m"
                     f"  max={summary.get('max_extr_trans_err_m', 0):.4f} m")

    # Intrinsics
    if "mean_intrinsic_err_px" in summary:
        lines.append(f"Intrinsic error:        mean={summary['mean_intrinsic_err_px']:.4f} px")

    # Distortion
    if "mean_distortion_err" in summary:
        lines.append(f"Distortion error:       mean={summary['mean_distortion_err']:.6f}")

    # Landmarks
    if "mean_landmark_err_m" in summary:
        lines.append(f"Landmark error:         mean={summary['mean_landmark_err_m']:.4f} m")

    lm = eval_result.get("landmarks", {})
    if lm:
        lines.append(f"  (median={lm.get('median_err_m', 0):.4f}  "
                     f"p95={lm.get('p95_err_m', 0):.4f}  "
                     f"max={lm.get('max_err_m', 0):.4f}  "
                     f"n={lm.get('count', 0)})")

    if not summary:
        lines.append("(No parameter recovery data available)")

    return "\n".join(lines)
