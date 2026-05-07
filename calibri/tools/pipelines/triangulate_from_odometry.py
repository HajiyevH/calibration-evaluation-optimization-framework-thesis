#!/usr/bin/env python3
"""
Triangulate 3D points from known odometry poses and a COLMAP feature database.

Supports both trivial-rig databases (1 camera per rig, as produced by COLMAP)
and multi-camera rig databases (N cameras per rig, with sensor_from_rig transforms).

Pipeline:
    1. Load odometry CSV -> vehicle-to-world poses (T_wv)
    2. Load calibs_prior JSON -> vehicle-to-camera transforms (T_cv)
    3. Build a pycolmap Reconstruction with cameras + posed images (no 3D points)
    4. Call pycolmap.triangulate_points() to triangulate landmarks from 2D matches
    5. Export the reconstruction for downstream BA

Usage:
    python triangulate_from_odometry.py \
        --database     data/sequence/database_rig.db \
        --images       data/sequence/images \
        --odometry     data/sequence/odometry.csv \
        --calibs-prior data/sequence/calibs_prior \
        --output       results/triangulated/reconstruction
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pycolmap
from scipy.spatial.transform import Rotation as R

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def build_cam_folder_mappings(
    db_path: Path, images_path: Path
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Auto-detect camera folder <-> camera_id mapping from database images.

    Reads image names (e.g. "front/frame_0.png" or "cam00/frame_0.png"),
    groups by folder, and maps each folder to its camera_id.

    Returns (folder_to_id, id_to_folder).
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT camera_id, name FROM images ORDER BY camera_id")
    rows = cur.fetchall()
    conn.close()

    # Extract folder name from image path
    folder_to_id = {}
    for camera_id, name in rows:
        folder = name.split("/")[0]
        if folder not in folder_to_id:
            folder_to_id[folder] = camera_id

    id_to_folder = {v: k for k, v in folder_to_id.items()}
    log.info(f"Camera mapping (auto-detected): {folder_to_id}")
    return folder_to_id, id_to_folder


# -- helpers ----------------------------------------------------------------

def load_odometry(csv_path: Path) -> pd.DataFrame:
    """Load odometry CSV and return DataFrame with pose columns."""
    df = pd.read_csv(csv_path)
    required = ["timestamp", "xPosition_m", "yPosition_m", "elevation_m",
                 "rollAngle_rad", "pitchAngle_rad", "yawAngle_rad"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in odometry CSV")
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info(f"Loaded odometry: {len(df)} samples, "
             f"ts range [{df.timestamp.iloc[0]} .. {df.timestamp.iloc[-1]}]")
    return df


def interpolate_pose(df: pd.DataFrame, ts: int) -> dict:
    """Interpolate odometry at a given timestamp.

    Uses linear interpolation for position and approximate SLERP
    (rotvec interpolation) for orientation.
    Falls back to nearest-neighbour if timestamp is outside range.
    """
    timestamps = df["timestamp"].values

    if ts <= timestamps[0]:
        return _row_to_pose(df.iloc[0])
    if ts >= timestamps[-1]:
        return _row_to_pose(df.iloc[-1])

    idx = np.searchsorted(timestamps, ts, side="right") - 1
    t0, t1 = timestamps[idx], timestamps[idx + 1]
    alpha = (ts - t0) / (t1 - t0)

    row0 = df.iloc[idx]
    row1 = df.iloc[idx + 1]

    pos = (1 - alpha) * np.array([row0.xPosition_m, row0.yPosition_m, row0.elevation_m]) \
        + alpha * np.array([row1.xPosition_m, row1.yPosition_m, row1.elevation_m])

    r0 = R.from_euler("xyz", [row0.rollAngle_rad, row0.pitchAngle_rad, row0.yawAngle_rad])
    r1 = R.from_euler("xyz", [row1.rollAngle_rad, row1.pitchAngle_rad, row1.yawAngle_rad])
    rot = R.from_rotvec(
        (1 - alpha) * r0.as_rotvec() + alpha * r1.as_rotvec()
    )

    return {"position": pos, "rotation": rot}


def _row_to_pose(row) -> dict:
    pos = np.array([row.xPosition_m, row.yPosition_m, row.elevation_m])
    rot = R.from_euler("xyz", [row.rollAngle_rad, row.pitchAngle_rad, row.yawAngle_rad])
    return {"position": pos, "rotation": rot}


def load_calib_prior(json_path: Path) -> dict:
    """Load a calibs_prior JSON and return T_camera_vehicle as a dict."""
    with open(json_path) as f:
        data = json.load(f)

    position = np.array([data["positionX"], data["positionY"], data["positionZ"]])
    rotation = R.from_euler("xyz", [
        data["rotationRoll"], data["rotationPitch"], data["rotationYaw"]
    ])

    return {"position": position, "rotation": rotation, "raw": data}


def compute_cam_from_world(
    vehicle_pose: dict, calib: dict
) -> pycolmap.Rigid3d:
    """Compute cam_from_world = T_cv * T_vw = T_cv * inv(T_wv).

    vehicle_pose: {position: t_wv, rotation: R_wv}  (vehicle-to-world)
    calib: {position: t_cv, rotation: R_cv}  (vehicle-to-camera)

    T_wv = [R_wv | t_wv]  ->  T_vw = [R_wv^T | -R_wv^T * t_wv]
    T_cv = [R_cv | t_cv]
    T_cw = T_cv * T_vw = [R_cv * R_wv^T | R_cv * (-R_wv^T * t_wv) + t_cv]
    """
    R_wv = vehicle_pose["rotation"].as_matrix()
    t_wv = vehicle_pose["position"]
    R_cv = calib["rotation"].as_matrix()
    t_cv = calib["position"]

    R_vw = R_wv.T
    t_vw = -R_vw @ t_wv

    R_cw = R_cv @ R_vw
    t_cw = R_cv @ t_vw + t_cv

    quat_xyzw = R.from_matrix(R_cw).as_quat()  # scipy returns [x,y,z,w]
    return pycolmap.Rigid3d(
        rotation=pycolmap.Rotation3d(xyzw=quat_xyzw),
        translation=t_cw,
    )


def build_seq_to_timestamp_map(images_path: Path, cam_folder: str) -> Dict[int, int]:
    """Build mapping from sequential index to microsecond timestamp.

    Reads actual filenames from disk (frame_<timestamp>.png), sorts them
    chronologically, and assigns sequential indices 0, 1, 2, ...
    """
    cam_dir = images_path / cam_folder
    if not cam_dir.is_dir():
        return {}
    timestamps = sorted(
        int(f.stem.replace("frame_", ""))
        for f in cam_dir.glob("frame_*.png")
    )
    return {i: ts for i, ts in enumerate(timestamps)}


def detect_db_type(cur: sqlite3.Cursor) -> str:
    """Detect whether the DB uses trivial rigs (1 cam/rig) or a multi-camera rig."""
    cur.execute("SELECT COUNT(*) FROM rigs")
    num_rigs = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM frames")
    num_frames = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM images")
    num_images = cur.fetchone()[0]

    images_per_frame = num_images / max(num_frames, 1)

    if num_rigs == 1 and images_per_frame > 1.5:
        return "multi_cam_rig"
    else:
        return "trivial_rigs"


def read_rig_sensors(cur: sqlite3.Cursor) -> Dict[Tuple[int, int], pycolmap.Rigid3d]:
    """Read sensor_from_rig transforms from the rig_sensors table.

    Returns dict of (rig_id, sensor_id) -> Rigid3d.
    The reference sensor is NOT in this table (its transform is identity).
    """
    cur.execute("SELECT rig_id, sensor_id, sensor_type, sensor_from_rig FROM rig_sensors")
    result = {}
    for rig_id, sensor_id, sensor_type, blob in cur.fetchall():
        vals = np.frombuffer(blob, dtype=np.float64)
        if len(vals) != 7:
            log.warning(f"Unexpected rig_sensor blob size {len(vals)} for "
                        f"rig={rig_id}, sensor={sensor_id}")
            continue
        qw, qx, qy, qz = vals[0:4]
        tx, ty, tz = vals[4:7]
        rigid = pycolmap.Rigid3d(
            rotation=pycolmap.Rotation3d(xyzw=np.array([qx, qy, qz, qw])),
            translation=np.array([tx, ty, tz]),
        )
        result[(rig_id, sensor_id)] = rigid
        rot_euler = R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=True)
        log.info(f"  Rig sensor {rig_id}/{sensor_id}: "
                 f"t=[{tx:.3f}, {ty:.3f}, {tz:.3f}], "
                 f"euler=[{rot_euler[0]:.1f}, {rot_euler[1]:.1f}, {rot_euler[2]:.1f}] deg")
    return result


# -- main pipeline ----------------------------------------------------------

def triangulate(
    database_path: Path,
    images_path: Path,
    odometry_path: Path,
    calibs_prior_path: Path,
    output_path: Path,
    min_tri_angle: float = 1.5,
) -> pycolmap.Reconstruction:
    """Build posed reconstruction and triangulate 3D points."""

    # 1. Load inputs
    odom_df = load_odometry(odometry_path)

    # Auto-detect camera folder <-> camera_id mapping from the database
    CAM_FOLDER_TO_ID, CAM_ID_TO_FOLDER = build_cam_folder_mappings(database_path, images_path)

    calibs = {}
    for cam_name in CAM_FOLDER_TO_ID:
        calib_file = calibs_prior_path / f"{cam_name}.json"
        if calib_file.exists():
            calibs[cam_name] = load_calib_prior(calib_file)
            log.info(f"  {cam_name}: pos={calibs[cam_name]['position']}, "
                     f"rot_euler={calibs[cam_name]['rotation'].as_euler('xyz', degrees=True)}")
        else:
            log.warning(f"No calibration for {cam_name}, skipping")

    # 2. Read database structure
    conn = sqlite3.connect(str(database_path))
    cur = conn.cursor()

    db_type = detect_db_type(cur)
    log.info(f"Database type: {db_type}")

    # Read cameras
    cur.execute("SELECT camera_id, model, width, height, params FROM cameras")
    db_cameras = {}
    for cam_id, model, w, h, params_blob in cur.fetchall():
        params = np.frombuffer(params_blob, dtype=np.float64)
        db_cameras[cam_id] = {"model": model, "width": w, "height": h, "params": params}
        log.info(f"  DB camera {cam_id}: model={model}, {w}x{h}, "
                 f"fx={params[0]:.1f}, fy={params[1]:.1f}")

    # Read rigs
    cur.execute("SELECT rig_id, ref_sensor_id FROM rigs")
    db_rigs = {row[0]: row[1] for row in cur.fetchall()}
    log.info(f"  DB rigs: {db_rigs}")

    # Read rig sensor transforms (non-reference cameras)
    db_rig_sensors = read_rig_sensors(cur)

    # Read frames
    cur.execute("SELECT frame_id, rig_id FROM frames")
    db_frames = {row[0]: row[1] for row in cur.fetchall()}

    # Read frame_data
    cur.execute("SELECT frame_id, data_id, sensor_id FROM frame_data")
    db_frame_data = {}
    for frame_id, data_id, sensor_id in cur.fetchall():
        db_frame_data.setdefault(frame_id, []).append((data_id, sensor_id))

    # Read images
    cur.execute("SELECT image_id, camera_id, name FROM images ORDER BY image_id")
    db_images = cur.fetchall()
    conn.close()
    log.info(f"  DB images: {len(db_images)}, frames: {len(db_frames)}")

    # 3. Handle sequential image names -> timestamp mapping
    # Check if image names are sequential (frame_0.png) or timestamped (frame_1763028984167804.png)
    sample_name = db_images[0][2].split("/")[1]  # e.g. "frame_0.png"
    sample_idx = sample_name.replace("frame_", "").replace(".png", "")
    is_sequential = sample_idx.isdigit() and int(sample_idx) < 10000

    seq_to_ts = {}
    if is_sequential:
        log.info("Image names are sequential -- building index-to-timestamp map from disk")
        # Use first available camera folder as reference (all cameras are synchronized)
        ref_cam_folder = next(iter(CAM_FOLDER_TO_ID))
        seq_to_ts = build_seq_to_timestamp_map(images_path, ref_cam_folder)
        log.info(f"  Mapped {len(seq_to_ts)} sequential indices to timestamps")
        if seq_to_ts:
            log.info(f"  Example: frame_0 -> ts {seq_to_ts[0]}, "
                     f"frame_{len(seq_to_ts)-1} -> ts {seq_to_ts[len(seq_to_ts)-1]}")

    # Build image lookup: image_id -> (camera_id, name, cam_folder, timestamp)
    image_info = {}
    for image_id, camera_id, name in db_images:
        parts = name.split("/")
        cam_folder = parts[0]
        fname = parts[1]
        raw_id = int(fname.replace("frame_", "").replace(".png", ""))
        if is_sequential:
            ts = seq_to_ts.get(raw_id)
            if ts is None:
                log.warning(f"No timestamp for sequential index {raw_id}, skipping image {name}")
                continue
        else:
            ts = raw_id
        image_info[image_id] = (camera_id, name, cam_folder, ts)

    # 4. Build pycolmap Reconstruction
    rec = pycolmap.Reconstruction()

    # Add cameras
    for cam_id, cam_data in db_cameras.items():
        cam = pycolmap.Camera(
            camera_id=cam_id,
            model=pycolmap.CameraModelId(cam_data["model"]),
            width=cam_data["width"],
            height=cam_data["height"],
            params=cam_data["params"],
        )
        rec.add_camera(cam)
        log.info(f"  Added camera {cam_id}")

    # Add rigs
    sensors = {}  # sensor_id -> sensor_t
    for rig_id, ref_sensor_id in db_rigs.items():
        sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=ref_sensor_id)
        sensors[ref_sensor_id] = sensor
        rig = pycolmap.Rig()
        rig.rig_id = rig_id
        rig.add_ref_sensor(sensor)

        # Add non-reference sensor transforms
        for (rid, sid), rigid in db_rig_sensors.items():
            if rid == rig_id:
                s = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=sid)
                sensors[sid] = s
                rig.add_sensor(s, rigid)
                log.info(f"  Added rig sensor {sid} to rig {rig_id}")

        rec.add_rig(rig)
        log.info(f"  Added rig {rig_id} (ref_sensor={ref_sensor_id}, "
                 f"total_sensors={1 + sum(1 for r, _ in db_rig_sensors if r == rig_id)})")

    # Add frames with poses, then images
    if db_type == "multi_cam_rig":
        skipped, added = _build_multi_cam_rig(
            rec, db_frames, db_frame_data, db_rigs, image_info,
            calibs, sensors, odom_df, CAM_ID_TO_FOLDER
        )
    else:
        skipped, added = _build_trivial_rigs(
            rec, db_frames, db_frame_data, db_rigs, image_info,
            calibs, sensors, odom_df
        )

    log.info(f"  Added {added} posed images, skipped {skipped} frames")
    log.info(f"  Registered images: {rec.num_reg_images()}, frames: {rec.num_reg_frames()}")

    # 5. Triangulate points
    log.info("Triangulating 3D points from 2D matches + known poses...")

    options = pycolmap.IncrementalPipelineOptions()
    options.triangulation.min_angle = min_tri_angle

    output_path.mkdir(parents=True, exist_ok=True)
    rec_out = pycolmap.triangulate_points(
        reconstruction=rec,
        database_path=str(database_path),
        image_path=str(images_path),
        output_path=str(output_path),
        clear_points=True,
        options=options,
        refine_intrinsics=False,
    )

    log.info(f"Triangulated {rec_out.num_points3D()} 3D points")
    log.info(f"Output written to {output_path}")

    # 6. Summary statistics
    errors = []
    for pt_id in rec_out.point3D_ids():
        pt = rec_out.point3D(pt_id)
        if pt.error >= 0:
            errors.append(pt.error)

    if errors:
        errors = np.array(errors)
        log.info(f"Reprojection error stats (per-point track average):")
        log.info(f"  Mean:   {errors.mean():.2f} px")
        log.info(f"  Median: {np.median(errors):.2f} px")
        log.info(f"  P95:    {np.percentile(errors, 95):.2f} px")
        log.info(f"  Max:    {errors.max():.2f} px")

    return rec_out


def _build_multi_cam_rig(
    rec: pycolmap.Reconstruction,
    db_frames: dict,
    db_frame_data: dict,
    db_rigs: dict,
    image_info: dict,
    calibs: dict,
    sensors: dict,
    odom_df: pd.DataFrame,
    cam_id_to_folder: Dict[int, str] = None,
) -> Tuple[int, int]:
    """Build frames for a multi-camera rig DB.

    In a multi-camera rig, each frame contains N images (one per camera).
    rig_from_world is the reference camera's cam_from_world transform.
    All other cameras are positioned via sensor_from_rig transforms stored
    in the rig.
    """
    skipped = 0
    added = 0

    # Identify reference camera for computing rig_from_world
    rig_id = next(iter(db_rigs))
    ref_sensor_id = db_rigs[rig_id]
    ref_cam_folder = cam_id_to_folder.get(ref_sensor_id) if cam_id_to_folder else None
    if ref_cam_folder is None or ref_cam_folder not in calibs:
        raise ValueError(f"Reference sensor {ref_sensor_id} has no calibration")
    log.info(f"Multi-cam rig: ref_sensor={ref_sensor_id} ({ref_cam_folder})")

    for frame_id, frig_id in sorted(db_frames.items()):
        frame_images = db_frame_data.get(frame_id, [])
        if not frame_images:
            skipped += 1
            continue

        # Find the reference sensor's image to get the timestamp
        ref_data_id = None
        for data_id, sensor_id in frame_images:
            if sensor_id == ref_sensor_id and data_id in image_info:
                ref_data_id = data_id
                break

        if ref_data_id is None:
            # Fall back to any image in the frame
            for data_id, sensor_id in frame_images:
                if data_id in image_info:
                    ref_data_id = data_id
                    break

        if ref_data_id is None:
            skipped += 1
            continue

        _, _, _, ts = image_info[ref_data_id]

        # Interpolate vehicle pose at this timestamp
        vehicle_pose = interpolate_pose(odom_df, ts)

        # rig_from_world = ref_cam_from_world = T_cv_ref * inv(T_wv)
        rig_from_world = compute_cam_from_world(vehicle_pose, calibs[ref_cam_folder])

        # Create frame
        frame = pycolmap.Frame()
        frame.frame_id = frame_id
        frame.rig_id = frig_id
        frame.rig_from_world = rig_from_world

        # Link all images in this frame
        for data_id, sensor_id in frame_images:
            if data_id not in image_info:
                continue
            sensor = sensors.get(sensor_id)
            if sensor is None:
                sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=sensor_id)
                sensors[sensor_id] = sensor
            data = pycolmap.data_t(sensor_id=sensor, id=data_id)
            frame.add_data_id(data)

        rec.add_frame(frame)

        # Add all images
        for data_id, sensor_id in frame_images:
            if data_id not in image_info:
                continue
            camera_id, name, cam_folder, _ = image_info[data_id]
            img = pycolmap.Image(name=name, camera_id=camera_id, image_id=data_id)
            img.frame_id = frame_id
            rec.add_image(img)

        rec.register_frame(frame_id)
        added += 1

    return skipped, added


def _build_trivial_rigs(
    rec: pycolmap.Reconstruction,
    db_frames: dict,
    db_frame_data: dict,
    db_rigs: dict,
    image_info: dict,
    calibs: dict,
    sensors: dict,
    odom_df: pd.DataFrame,
) -> Tuple[int, int]:
    """Build frames for a trivial-rig DB (1 camera per rig, as from COLMAP).

    Each frame has exactly one image. rig_from_world = cam_from_world.
    """
    skipped = 0
    added = 0

    for frame_id, rig_id in sorted(db_frames.items()):
        frame_images = db_frame_data.get(frame_id, [])
        if not frame_images:
            skipped += 1
            continue

        data_id, sensor_id = frame_images[0]
        if data_id not in image_info:
            skipped += 1
            continue

        camera_id, name, cam_folder, ts = image_info[data_id]
        if cam_folder not in calibs:
            skipped += 1
            continue

        vehicle_pose = interpolate_pose(odom_df, ts)
        cam_from_world = compute_cam_from_world(vehicle_pose, calibs[cam_folder])

        frame = pycolmap.Frame()
        frame.frame_id = frame_id
        frame.rig_id = rig_id
        frame.rig_from_world = cam_from_world

        sensor = sensors.get(sensor_id)
        if sensor is None:
            sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=sensor_id)
        data = pycolmap.data_t(sensor_id=sensor, id=data_id)
        frame.add_data_id(data)

        rec.add_frame(frame)

        img = pycolmap.Image(name=name, camera_id=camera_id, image_id=data_id)
        img.frame_id = frame_id
        rec.add_image(img)

        rec.register_frame(frame_id)
        added += 1

    return skipped, added


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Triangulate 3D points from odometry poses + COLMAP feature DB"
    )
    parser.add_argument("--database", type=Path, required=True,
                        help="Path to COLMAP database.db")
    parser.add_argument("--images", type=Path, required=True,
                        help="Path to images directory")
    parser.add_argument("--odometry", type=Path, required=True,
                        help="Path to odometry CSV file")
    parser.add_argument("--calibs-prior", type=Path, required=True,
                        help="Path to calibs_prior directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for COLMAP reconstruction")
    parser.add_argument("--min-tri-angle", type=float, default=1.5,
                        help="Minimum triangulation angle in degrees (default: 1.5)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    triangulate(
        database_path=args.database,
        images_path=args.images,
        odometry_path=args.odometry,
        calibs_prior_path=args.calibs_prior,
        output_path=args.output,
        min_tri_angle=args.min_tri_angle,
    )
