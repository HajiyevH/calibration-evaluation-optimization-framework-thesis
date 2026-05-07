#!/usr/bin/env python3
"""
COLMAP to problem.json converter

Converts COLMAP rig reconstruction outputs using pycolmap
and database metadata to the bundle adjustment problem.json format.

Usage:
    python colmap_to_problem.py --reconstruction <path/to/reconstruction> --database <path/to/database.db> --output problem.json
"""

import argparse
import json
import struct
import sqlite3
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from calibri.tools.utils import (
    setup_logging,
)

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False


def quaternion_to_rotation_matrix(qw, qx, qy, qz) -> np.ndarray:
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix."""
    q = np.array([qw, qx, qy, qz], dtype=float)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def rotation_matrix_to_quaternion(R: np.ndarray) -> List[float]:
    """Convert 3x3 rotation matrix to quaternion [w,x,y,z]."""
    t = float(np.trace(R))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    q = q / (np.linalg.norm(q) + 1e-12)
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def colmap_to_world_to_vehicle(qw, qx, qy, qz, tx, ty, tz) -> Tuple[List[float], List[float]]:
    """
    Convert COLMAP cam_from_world pose to vehicle-to-world (T_wv) pose.

    COLMAP stores: X_cam = R_cw * X_world + t_cw  (cam-from-world)
    BA expects:    X_world = R_wv * X_vehicle + t_wv  (vehicle-to-world)

    For the reference camera (identity extrinsic), vehicle = camera, so:
      T_wv = inverse(T_cw)
      R_wv = R_cw^T
      t_wv = -R_cw^T * t_cw

    Returns (quaternion [w,x,y,z], translation [x,y,z])
    """
    R_cw = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    R_wv = R_cw.T
    t_cw = np.array([tx, ty, tz])
    t_wv = -R_cw.T @ t_cw

    q_wv = rotation_matrix_to_quaternion(R_wv)

    return (list(q_wv), [float(t_wv[0]), float(t_wv[1]), float(t_wv[2])])


def _perturb_pose_rodrigues(
    quat: List[float],
    trans: List[float],
    sigma: float,
    rng: np.random.Generator,
) -> Tuple[List[float], List[float]]:
    """Apply Gaussian perturbation to an SE(3) pose using Rodrigues rotation.

    Args:
        quat: Quaternion [w,x,y,z].
        trans: Translation [x,y,z].
        sigma: Noise standard deviation (meters for translation,
               scaled by 0.01 for rotation in radians).
        rng: Seeded numpy random generator.

    Returns:
        (perturbed_quat, perturbed_trans) both in the same format.
    """
    trans = [
        trans[0] + rng.normal(0, sigma),
        trans[1] + rng.normal(0, sigma),
        trans[2] + rng.normal(0, sigma),
    ]
    axis = rng.standard_normal(3)
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    angle = rng.normal(0, sigma * 0.01)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R_noise = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    q_noise = rotation_matrix_to_quaternion(R_noise)
    q_orig = np.array(quat)
    quat = [
        q_noise[0] * q_orig[0] - q_noise[1] * q_orig[1] - q_noise[2] * q_orig[2] - q_noise[3] * q_orig[3],
        q_noise[0] * q_orig[1] + q_noise[1] * q_orig[0] + q_noise[2] * q_orig[3] - q_noise[3] * q_orig[2],
        q_noise[0] * q_orig[2] - q_noise[1] * q_orig[3] + q_noise[2] * q_orig[0] + q_noise[3] * q_orig[1],
        q_noise[0] * q_orig[3] + q_noise[1] * q_orig[2] - q_noise[2] * q_orig[1] + q_noise[3] * q_orig[0],
    ]
    norm = np.sqrt(sum(q**2 for q in quat))
    quat = [q / norm for q in quat]
    return quat, trans


def _convert_rig_reconstruction(
    reconstruction_path: Path,
    database_path: Optional[Path],
    output_path: Path,
    use_rig: bool = False,
    optimize_intrinsics: bool = False,
    optimize_distortion: bool = False,
    optimize_extrinsics: bool = False,
    add_noise: bool = False,
    rotation_noise: float = 0.0,
    translation_noise: float = 0.0,
    perturb_poses: float = 0.0,
    perturb_landmarks: float = 0.0
):
    """Convert a pycolmap rig reconstruction (with frames.bin) to problem.json."""

    # Seeded RNG for reproducible perturbations
    _rng = np.random.default_rng(seed=42)

    # Load reconstruction with pycolmap
    rec = pycolmap.Reconstruction(str(reconstruction_path))

    logging.info(f"  Cameras: {len(rec.cameras)}")
    logging.info(f"  Images: {len(rec.images)}")
    logging.info(f"  3D Points: {len(rec.points3D)}")
    logging.info(f"  Frames (rig poses): {len(rec.frames)}")

    # Get rig configuration if available
    rig_config = None
    if database_path and use_rig:
        rig_config = get_camera_rig_from_db(database_path)
        if rig_config:
            logging.info(f"  Rig detected: reference_camera_id={rig_config['reference_camera_id']}")

    # Build problem.json structure
    problem = {
        "cameras": [],
        "poses": [],
        "landmarks": [],
        "observations": [],
        "flags": {
            "opt_intrinsics": optimize_intrinsics,
            "opt_distortion": optimize_distortion,
            "opt_extrinsics": optimize_extrinsics,
            "opt_poses": True,
            "opt_landmarks": True
        },
        "robust": {
            "type": "Huber",
            "scale": 1.0
        }
    }

    if rig_config:
        problem["camera_rig"] = {
            "reference_camera_id": rig_config['reference_camera_id']
        }

    # Compute cam_from_rig extrinsics for each camera (from first valid frame)
    camera_extrinsics = {}
    for frame in rec.frames.values():
        if not frame.has_pose:
            continue
        world_from_rig = frame.rig_from_world.inverse()
        for data in frame.image_ids:
            img_id = data.id
            cam_id_ex = data.sensor_id.id
            if cam_id_ex in camera_extrinsics:
                continue
            img = rec.images[img_id]
            cfr = img.cam_from_world() * world_from_rig
            q = cfr.rotation.quat  # pycolmap: [x,y,z,w]
            t = cfr.translation
            
            if add_noise and (rotation_noise > 0.0 or translation_noise > 0.0):
                q_wxyz = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
                t_list = [float(t[0]), float(t[1]), float(t[2])]
                q_wxyz, t_list = _perturb_pose_rodrigues(q_wxyz, t_list, translation_noise, _rng)
                q = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
                t = t_list

            camera_extrinsics[cam_id_ex] = {
                "q": [float(q[3]), float(q[0]), float(q[1]), float(q[2])],
                "t": [float(t[0]), float(t[1]), float(t[2])]
            }
        if len(camera_extrinsics) == len(rec.cameras):
            break
    logging.info(f"  Computed cam_from_rig for cameras: {sorted(camera_extrinsics.keys())}")

    if optimize_intrinsics or optimize_distortion or optimize_extrinsics:
        logging.info(f"  Optimization: intrinsics={optimize_intrinsics}, distortion={optimize_distortion}, extrinsics={optimize_extrinsics}")
    if perturb_poses > 0.0 or perturb_landmarks > 0.0:
        logging.info(f"  Perturbation: poses={perturb_poses}m, landmarks={perturb_landmarks}m")

    # Convert cameras
    for cam_id, camera in rec.cameras.items():
        model_name = camera.model_name
        params = camera.params

        if model_name == "OPENCV":
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            k1, k2, p1, p2 = params[4], params[5], params[6], params[7]
            k3 = 0.0
            distortion_model = "Pinhole"
        elif model_name == "OPENCV_FISHEYE":
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            k1, k2, k3, k4 = params[4], params[5], params[6], params[7]
            p1, p2 = 0.0, 0.0
            distortion_model = "Fisheye"
        elif model_name == "RADIAL_FISHEYE":
            fx = fy = params[0]
            cx, cy = params[1], params[2]
            k1 = params[3] if len(params) > 3 else 0.0
            k2 = params[4] if len(params) > 4 else 0.0
            k3, k4 = 0.0, 0.0
            p1, p2 = 0.0, 0.0
            distortion_model = "Fisheye"
        else:
            logging.warning(f"Camera {cam_id} uses unsupported model {model_name}, skipping")
            continue

        if distortion_model == "Fisheye":
            distortion_dict = {
                "k1": float(k1), "k2": float(k2), "k3": float(k3), "k4": float(k4)
            }
        else:
            distortion_dict = {
                "k1": float(k1), "k2": float(k2), "p1": float(p1), "p2": float(p2), "k3": float(k3)
            }

        problem["cameras"].append({
            "id": cam_id,
            "width": camera.width,
            "height": camera.height,
            "model": distortion_model,
            "intrinsics": {
                "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)
            },
            "distortion": distortion_dict,
            "extrinsics": {
                "T_vehicle_to_camera": camera_extrinsics.get(cam_id, {
                    "q": [1.0, 0.0, 0.0, 0.0],
                    "t": [0.0, 0.0, 0.0]
                })
            }
        })

    frame_num_to_images = defaultdict(list)
    for img_id, image in rec.images.items():
        name = image.name
        try:
            frame_num = int(name.split('frame_')[1].split('.')[0])
            frame_num_to_images[frame_num].append((img_id, image))
        except (IndexError, ValueError):
            logging.warning(f"Could not parse frame number from {name}, using image_id")
            frame_num_to_images[img_id].append((img_id, image))

    for frame_num in sorted(frame_num_to_images.keys()):
        img_list = frame_num_to_images[frame_num]

        if rig_config:
            ref_cam_id = rig_config['reference_camera_id']
            ref_img = None
            for img_id, image in img_list:
                if image.camera_id == ref_cam_id:
                    ref_img = image
                    break

            if not ref_img:
                ref_img = img_list[0][1]
        else:
            ref_img = img_list[0][1]

        frame = rec.frames[ref_img.frame_id]
        rig_from_world = frame.rig_from_world

        quat_cw = rig_from_world.rotation.quat  # pycolmap: [x,y,z,w]
        trans_cw = rig_from_world.translation

        quat, trans = colmap_to_world_to_vehicle(
            quat_cw[3], quat_cw[0], quat_cw[1], quat_cw[2],
            trans_cw[0], trans_cw[1], trans_cw[2]
        )

        if perturb_poses > 0.0:
            quat, trans = _perturb_pose_rodrigues(quat, trans, perturb_poses, _rng)

        problem["poses"].append({
            "timestamp_ns": frame_num * 1_000_000_000,
            "q": quat,
            "t": trans
        })

    for pt_id, point3D in rec.points3D.items():
        xyz = [float(point3D.xyz[0]), float(point3D.xyz[1]), float(point3D.xyz[2])]

        if perturb_landmarks > 0.0:
            xyz = [
                xyz[0] + _rng.normal(0, perturb_landmarks),
                xyz[1] + _rng.normal(0, perturb_landmarks),
                xyz[2] + _rng.normal(0, perturb_landmarks),
            ]

        problem["landmarks"].append({
            "id": pt_id,
            "position": xyz
        })

    obs_id = 0
    for img_id, image in rec.images.items():
        name = image.name
        try:
            frame_num = int(name.split('frame_')[1].split('.')[0])
            timestamp_ns = frame_num * 1_000_000_000
        except (IndexError, ValueError):
            timestamp_ns = img_id * 1_000_000_000

        camera_id = image.camera_id

        _INVALID_POINT3D_ID = 2**64 - 1
        for point2D_idx, point2D in enumerate(image.points2D):
            if point2D.point3D_id == _INVALID_POINT3D_ID or point2D.point3D_id < 0:
                continue

            problem["observations"].append({
                "id": obs_id,
                "timestamp_ns": timestamp_ns,
                "camera_id": camera_id,
                "landmark_id": point2D.point3D_id,
                "u": float(point2D.xy[0]),
                "v": float(point2D.xy[1])
            })
            obs_id += 1

    logging.info(f"Converted problem:")
    logging.info(f"  Cameras: {len(problem['cameras'])}")
    logging.info(f"  Poses: {len(problem['poses'])}")
    logging.info(f"  Landmarks: {len(problem['landmarks'])}")
    logging.info(f"  Observations: {len(problem['observations'])}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(problem, f, indent=2)

    logging.info(f"Wrote problem to {output_path}")


def get_camera_rig_from_db(database_path: Path) -> Optional[Dict]:
    """Extract camera rig configuration from COLMAP database"""
    if not database_path.exists():
        return None

    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rigs'")
    if not cursor.fetchone():
        conn.close()
        return None

    cursor.execute("SELECT COUNT(*) FROM rigs")
    if cursor.fetchone()[0] == 0:
        conn.close()
        return None

    cursor.execute("SELECT rig_id, ref_sensor_id FROM rigs")
    rig_data = cursor.fetchone()

    if not rig_data:
        conn.close()
        return None

    rig_id, ref_sensor_id = rig_data

    cursor.execute("SELECT sensor_id, sensor_from_rig FROM rig_sensors WHERE rig_id=?", (rig_id,))
    sensor_transforms = {}
    for row in cursor.fetchall():
        sensor_id, transform_blob = row
        if transform_blob:
            if len(transform_blob) >= 56:
                vals = struct.unpack('<7d', transform_blob[:56])
                sensor_transforms[sensor_id] = {
                    'quaternion': list(vals[0:4]),
                    'translation': list(vals[4:7])
                }

    conn.close()

    return {
        'rig_id': rig_id,
        'reference_camera_id': ref_sensor_id,
        'sensor_transforms': sensor_transforms
    }


def convert_colmap_to_problem(
    reconstruction_path: Path,
    database_path: Optional[Path],
    output_path: Path,
    use_rig: bool = False,
    optimize_intrinsics: bool = False,
    optimize_distortion: bool = False,
    optimize_extrinsics: bool = False,
    perturb_poses: float = 0.0,
    perturb_landmarks: float = 0.0
):
    """Convert COLMAP reconstruction to problem.json format"""

    logging.info(f"Reading COLMAP reconstruction from {reconstruction_path}")

    if not HAS_PYCOLMAP:
        raise ImportError("pycolmap is required for COLMAP conversion")

    frames_bin = reconstruction_path / "frames.bin"
    if not frames_bin.exists():
        sub0 = reconstruction_path / "0"
        if sub0.is_dir() and (sub0 / "frames.bin").exists():
            reconstruction_path = sub0
            logging.info(f"Using COLMAP sub-model: {reconstruction_path}")
        else:
            raise FileNotFoundError(f"Missing frames.bin in {reconstruction_path} (or subdir 0)")

    logging.info("  Rig reconstruction detected, using pycolmap for pose composition")
    return _convert_rig_reconstruction(
        reconstruction_path, database_path, output_path, use_rig,
        optimize_intrinsics, optimize_distortion, optimize_extrinsics,
        perturb_poses, perturb_landmarks
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert COLMAP reconstruction to bundle adjustment problem.json"
    )
    parser.add_argument(
        '--reconstruction',
        type=Path,
        required=True,
        help="Path to COLMAP reconstruction directory (containing cameras.bin, images.bin, points3D.bin)"
    )
    parser.add_argument(
        '--database',
        type=Path,
        help="Path to COLMAP database.db (optional, for rig configuration)"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/problem_from_colmap.json'),
        help="Output path for problem.json"
    )
    parser.add_argument(
        '--use-rig',
        action='store_true',
        help="Use rig configuration from database (for multi-camera)"
    )
    parser.add_argument(
        '--optimize-intrinsics',
        action='store_true',
        help="Enable intrinsics refinement (fx, fy, cx, cy)"
    )
    parser.add_argument(
        '--optimize-distortion',
        action='store_true',
        help="Enable distortion refinement (k1, k2, p1, p2, k3 or k1-k4 for fisheye)"
    )
    parser.add_argument(
        '--optimize-extrinsics',
        action='store_true',
        help="Enable extrinsics refinement (T_vehicle_to_camera)"
    )
    parser.add_argument(
        '--perturb-poses',
        type=float,
        default=0.0,
        metavar='SIGMA',
        help="Add Gaussian noise to poses (meters) for validation mode"
    )
    parser.add_argument(
        '--perturb-landmarks',
        type=float,
        default=0.0,
        metavar='SIGMA',
        help="Add Gaussian noise to landmarks (meters) for validation mode"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Verbose logging"
    )


    args = parser.parse_args()
    setup_logging(args.verbose)

    convert_colmap_to_problem(
        reconstruction_path=args.reconstruction,
        database_path=args.database,
        output_path=args.output,
        use_rig=args.use_rig,
        optimize_intrinsics=args.optimize_intrinsics,
        optimize_distortion=args.optimize_distortion,
        optimize_extrinsics=args.optimize_extrinsics,
        perturb_poses=args.perturb_poses,
        perturb_landmarks=args.perturb_landmarks
    )


if __name__ == '__main__':
    main()
