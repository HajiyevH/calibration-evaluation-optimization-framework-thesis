#!/usr/bin/env python3
"""
Preprocess KITTI raw data for the bundle adjustment pipeline.

Takes raw KITTI-style multi-camera data and produces a fully populated
COLMAP rig database ready for triangulate_from_odometry.py.

Input structure (raw KITTI download):
    <recording>/
        image_00/data/0000000000.png ...   (raw KITTI synced+rectified)
        image_01/data/0000000000.png ...
        image_02/data/0000000000.png ...
        image_03/data/0000000000.png ...
        image_00/timestamps.txt            (optional real timestamps)
        calibs_prior/
            cam00.json  cam01.json  cam02.json  cam03.json
        odometry.csv
        database.db          (COLMAP DB with cameras + images, no features)

After prepare_images():
    <recording>/
        images/
            cam00/  frame_0000000000.png  ...
            cam01/  frame_0000000000.png  ...

Output:
    <recording>/
        database_rig.db      (full rig DB: features, matches, rigs, frames)

Pipeline:
    0. Prepare images: copy from image_XX/data/ to images/camXX/frame_*.png
    1. Copy database.db -> database_rig.db, prune orphan images
    2. Extract SIFT features via pycolmap
    3. Match features (exhaustive or sequential)
    4. Build rig structure (rigs, rig_sensors, frames, frame_data)

Usage:
    python kitti_preprocess.py --recording pre-triangulation/kitti/kitti_01
    python kitti_preprocess.py --recording pre-triangulation/kitti/kitti_01 --matching sequential --overlap 5
"""

import argparse
import json
import logging
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False


KITTI_CAM_MAP = {
    "image_00": "cam00",
    "image_01": "cam01",
    "image_02": "cam02",
    "image_03": "cam03",
}


def prepare_images(recording_dir: Path, raw_base_dir: Optional[Path] = None) -> int:
    """Copy raw KITTI images from image_XX/data/ to images/camXX/frame_*.png.

    Detects raw KITTI directories (image_00/, image_01/, ...) and copies
    their contents into the expected images/camXX/ layout with frame_ prefix.

    Args:
        recording_dir: The recording directory (destination for images/).
        raw_base_dir: If provided, look for image_XX/data/ under this directory
                      instead of recording_dir. Useful when raw images live in a
                      shared sibling directory (e.g. 2011_09_26_drive_XXXX_sync/).

    Returns the number of available frames (from the synced data).
    """
    images_dir = recording_dir / "images"
    source_dir = raw_base_dir if raw_base_dir else recording_dir
    available_frames = None

    for kitti_name, cam_name in sorted(KITTI_CAM_MAP.items()):
        raw_dir = source_dir / kitti_name / "data"
        if not raw_dir.is_dir():
            continue

        target_dir = images_dir / cam_name
        target_dir.mkdir(parents=True, exist_ok=True)

        raw_files = sorted(raw_dir.glob("*.png"))
        if not raw_files:
            log.warning(f"No PNG files in {raw_dir}")
            continue

        if available_frames is None:
            available_frames = len(raw_files)
        else:
            available_frames = min(available_frames, len(raw_files))

        # Check how many already exist as real images (not stubs)
        existing_real = 0
        for rf in raw_files:
            target = target_dir / f"frame_{rf.name}"
            if target.exists() and target.stat().st_size > 1000:
                existing_real += 1

        if existing_real == len(raw_files):
            log.info(f"  {cam_name}: all {len(raw_files)} images already present, skipping copy")
            continue

        log.info(f"  {cam_name}: copying {len(raw_files)} images from {raw_dir}")
        for rf in raw_files:
            target = target_dir / f"frame_{rf.name}"
            shutil.copy2(rf, target)

    if available_frames is not None:
        log.info(f"Prepared {available_frames} frames from raw KITTI data")
    else:
        log.info("No raw KITTI image_XX/data/ directories found, using existing images/")

    return available_frames


def parse_kitti_calib(calib_path: Path) -> dict:
    """Parse a KITTI calib_cam_to_cam.txt file.

    Returns a dict with keys like 'S_00', 'K_00', 'P_rect_00', etc.
    Values are numpy arrays.
    """
    data = {}
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                data[key] = value
    return data


def update_cameras_for_rectified(db_path: Path, calib_data: dict, num_cameras: int = 4):
    """Update database camera entries to match rectified image dimensions and intrinsics.

    For KITTI rectified images, all cameras share fx=fy and have zero distortion.
    Uses P_rect_XX from the calibration file.
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    for cam_idx in range(num_cameras):
        s_rect = calib_data.get(f"S_rect_{cam_idx:02d}")
        p_rect = calib_data.get(f"P_rect_{cam_idx:02d}")
        if s_rect is None or p_rect is None:
            log.warning(f"Missing S_rect/P_rect for camera {cam_idx:02d}")
            continue

        width = int(s_rect[0])
        height = int(s_rect[1])

        # P_rect is 3x4: [fx 0 cx tx; 0 fy cy ty; 0 0 1 tz]
        P = p_rect.reshape(3, 4)
        fx = P[0, 0]
        fy = P[1, 1]
        cx = P[0, 2]
        cy = P[1, 2]

        # OPENCV model (id=4): [fx, fy, cx, cy, k1, k2, p1, p2]
        # Rectified images have zero distortion
        params = np.array([fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        cam_id = cam_idx + 1  # 1-indexed
        cur.execute(
            "UPDATE cameras SET width=?, height=?, params=? WHERE camera_id=?",
            (width, height, params.tobytes(), cam_id)
        )
        log.info(f"  Camera {cam_id} (cam{cam_idx:02d}): {width}x{height}, "
                 f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}, distortion=0")

    conn.commit()
    conn.close()


def discover_cameras(images_dir: Path) -> List[str]:
    """Auto-discover camera subdirectories sorted alphabetically."""
    cams = sorted(
        d.name for d in images_dir.iterdir()
        if d.is_dir() and any(d.glob("*.png"))
    )
    if not cams:
        raise ValueError(f"No camera subdirectories with .png images found in {images_dir}")
    return cams


def load_calib_prior(json_path: Path) -> dict:
    """Load a calibs_prior JSON and return intrinsic + extrinsic data."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def compute_sensor_from_rig(ref_calib: dict, sensor_calib: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Compute sensor_from_rig transform.

    If rig frame = reference camera frame, then:
        sensor_from_rig = T_sensor_camera * inv(T_ref_camera)
        = T_sensor_vehicle * inv(T_ref_vehicle)

    Both calibs have (roll, pitch, yaw) Euler angles and (X, Y, Z) position
    representing T_camera_vehicle (camera-from-vehicle).

    sensor_from_rig = T_cv_sensor * inv(T_cv_ref)
    """
    from scipy.spatial.transform import Rotation as R

    # Reference camera
    R_ref = R.from_euler("xyz", [
        ref_calib["rotationRoll"], ref_calib["rotationPitch"], ref_calib["rotationYaw"]
    ]).as_matrix()
    t_ref = np.array([ref_calib["positionX"], ref_calib["positionY"], ref_calib["positionZ"]])

    # Sensor camera
    R_sen = R.from_euler("xyz", [
        sensor_calib["rotationRoll"], sensor_calib["rotationPitch"], sensor_calib["rotationYaw"]
    ]).as_matrix()
    t_sen = np.array([sensor_calib["positionX"], sensor_calib["positionY"], sensor_calib["positionZ"]])

    # T_sensor_from_rig = T_cv_sensor * inv(T_cv_ref)
    # inv(T_cv_ref) = [R_ref^T | -R_ref^T * t_ref]
    R_rel = R_sen @ R_ref.T
    t_rel = t_sen - R_rel @ t_ref

    # Convert to quaternion [w, x, y, z] for COLMAP blob storage
    quat_xyzw = R.from_matrix(R_rel).as_quat()  # scipy: [x,y,z,w]
    qw = quat_xyzw[3]
    qx, qy, qz = quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]

    return np.array([qw, qx, qy, qz]), t_rel


def setup_rig_database(
    src_db: Path,
    dst_db: Path,
    cameras: List[str],
    calibs_dir: Path,
    images_dir: Path,
    ref_camera_idx: int = 0,
) -> dict:
    """Copy source DB and add rig structure (rigs, rig_sensors, frames, frame_data).

    Also prunes orphan image entries (images in DB with no file on disk).
    Returns mapping of camera_folder -> camera_id from the database.
    """
    log.info(f"Copying {src_db} -> {dst_db}")
    shutil.copy2(src_db, dst_db)

    conn = sqlite3.connect(str(dst_db))
    cur = conn.cursor()

    # Prune orphan images (handles synced vs unsynced frame count mismatch)
    cur.execute("SELECT image_id, name FROM images")
    all_db_images = cur.fetchall()
    orphans = []
    for image_id, name in all_db_images:
        img_path = images_dir / name
        if not img_path.exists() or img_path.stat().st_size < 1000:
            orphans.append((image_id, name))
    if orphans:
        log.info(f"Pruning {len(orphans)} orphan images from database")
        orphan_ids = [o[0] for o in orphans]
        placeholders = ",".join("?" * len(orphan_ids))
        cur.execute(f"DELETE FROM images WHERE image_id IN ({placeholders})", orphan_ids)
        conn.commit()
        log.info(f"  Remaining images: {len(all_db_images) - len(orphans)}")

    # Read existing camera IDs and image names
    cur.execute("SELECT camera_id FROM cameras ORDER BY camera_id")
    cam_ids = [row[0] for row in cur.fetchall()]
    if len(cam_ids) != len(cameras):
        raise ValueError(
            f"Database has {len(cam_ids)} cameras but found {len(cameras)} camera folders. "
            f"Camera IDs: {cam_ids}, Folders: {cameras}"
        )

    # Map folder name -> camera_id (by position)
    cam_folder_to_id = {cam: cam_ids[i] for i, cam in enumerate(cameras)}
    log.info(f"Camera mapping: {cam_folder_to_id}")

    ref_cam = cameras[ref_camera_idx]
    ref_cam_id = cam_folder_to_id[ref_cam]
    log.info(f"Reference camera: {ref_cam} (id={ref_cam_id})")

    # Load calibrations for rig_sensor transforms
    calibs = {}
    for cam in cameras:
        calib_file = calibs_dir / f"{cam}.json"
        if calib_file.exists():
            calibs[cam] = load_calib_prior(calib_file)
        else:
            log.warning(f"No calibration for {cam}")

    # Clear existing rig data
    cur.execute("DELETE FROM rigs")
    cur.execute("DELETE FROM rig_sensors")
    cur.execute("DELETE FROM frames")
    cur.execute("DELETE FROM frame_data")

    # Create rig (rig_id=1, ref_sensor_id = ref camera's camera_id, ref_sensor_type=0 (CAMERA))
    rig_id = 1
    cur.execute(
        "INSERT INTO rigs (rig_id, ref_sensor_id, ref_sensor_type) VALUES (?, ?, ?)",
        (rig_id, ref_cam_id, 0)
    )
    log.info(f"Created rig {rig_id} with ref_sensor={ref_cam_id}")

    # Add non-reference sensors to rig_sensors
    ref_calib = calibs.get(ref_cam)
    for cam in cameras:
        cam_id = cam_folder_to_id[cam]
        if cam_id == ref_cam_id:
            continue  # Reference sensor is implicit

        sensor_calib = calibs.get(cam)
        if ref_calib is None or sensor_calib is None:
            log.warning(f"Skipping rig_sensor for {cam} (missing calibration)")
            continue

        quat_wxyz, t_rel = compute_sensor_from_rig(ref_calib, sensor_calib)
        # COLMAP stores sensor_from_rig as a blob: [qw, qx, qy, qz, tx, ty, tz]
        blob = np.array([quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3],
                         t_rel[0], t_rel[1], t_rel[2]], dtype=np.float64).tobytes()

        cur.execute(
            "INSERT INTO rig_sensors (rig_id, sensor_id, sensor_type, sensor_from_rig) VALUES (?, ?, ?, ?)",
            (rig_id, cam_id, 0, blob)  # sensor_type 0 = CAMERA
        )
        log.info(f"  Added rig_sensor: cam={cam} (id={cam_id}), "
                 f"t=[{t_rel[0]:.3f}, {t_rel[1]:.3f}, {t_rel[2]:.3f}]")

    # Build frames from images
    # Each frame = one timestamp with N images (one per camera)
    cur.execute("SELECT image_id, camera_id, name FROM images ORDER BY image_id")
    all_images = cur.fetchall()

    # Group images by frame index (extracted from filename)
    frame_groups = {}
    for image_id, camera_id, name in all_images:
        parts = name.split("/")
        fname = parts[-1]
        # Extract frame index: frame_0000000000.png -> 0000000000
        frame_idx = fname.replace("frame_", "").replace(".png", "")
        frame_groups.setdefault(frame_idx, []).append((image_id, camera_id, name))

    log.info(f"Found {len(frame_groups)} unique timestamps across {len(all_images)} images")

    # Create frames and frame_data
    for frame_id, (frame_idx, images) in enumerate(sorted(frame_groups.items()), start=1):
        cur.execute(
            "INSERT INTO frames (frame_id, rig_id) VALUES (?, ?)",
            (frame_id, rig_id)
        )
        for image_id, camera_id, name in images:
            cur.execute(
                "INSERT INTO frame_data (frame_id, data_id, sensor_id, sensor_type) VALUES (?, ?, ?, ?)",
                (frame_id, image_id, camera_id, 0)  # sensor_type 0 = CAMERA
            )

    log.info(f"Created {len(frame_groups)} frames with {len(all_images)} frame_data entries")

    conn.commit()
    conn.close()
    return cam_folder_to_id


def extract_features(db_path: Path, images_dir: Path):
    """Extract SIFT features using pycolmap."""
    if not HAS_PYCOLMAP:
        raise ImportError("pycolmap is required for feature extraction")

    log.info("Extracting SIFT features...")
    extraction_opts = pycolmap.FeatureExtractionOptions()
    extraction_opts.sift.max_num_features = 8192
    extraction_opts.sift.first_octave = -1  # Use upsampled image for more features

    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(images_dir),
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
        extraction_options=extraction_opts,
    )

    # Count features
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM keypoints")
    n_kp = cur.fetchone()[0]
    cur.execute("SELECT SUM(rows) FROM keypoints")
    total_features = cur.fetchone()[0] or 0
    conn.close()
    log.info(f"Extracted features for {n_kp} images ({total_features} total keypoints)")


def match_features(db_path: Path, matching: str = "exhaustive", overlap: int = 10):
    """Match features between image pairs."""
    if not HAS_PYCOLMAP:
        raise ImportError("pycolmap is required for feature matching")

    matching_opts = pycolmap.FeatureMatchingOptions()
    matching_opts.max_num_matches = 32768

    if matching == "exhaustive":
        log.info("Running exhaustive matching...")
        pycolmap.match_exhaustive(
            database_path=str(db_path),
            matching_options=matching_opts,
        )
    elif matching == "sequential":
        log.info(f"Running sequential matching (overlap={overlap})...")
        pairing_opts = pycolmap.SequentialPairingOptions()
        pairing_opts.overlap = overlap
        pycolmap.match_sequential(
            database_path=str(db_path),
            matching_options=matching_opts,
            pairing_options=pairing_opts,
        )
    else:
        raise ValueError(f"Unknown matching type: {matching}")

    # Count matches
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM matches")
    n_pairs = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM two_view_geometries")
    n_verified = cur.fetchone()[0]
    conn.close()
    log.info(f"Matched {n_pairs} image pairs, {n_verified} verified geometries")


def _rebuild_rig_structure(
    dst_db: Path,
    cameras: List[str],
    calibs_dir: Path,
    images_dir: Path,
    ref_camera_idx: int = 0,
):
    """Rebuild rig structure (rigs, rig_sensors, frames, frame_data) in-place.

    Used when --skip-features is set and database_rig.db already has features.
    Does NOT re-copy the database, preserving existing keypoints/descriptors/matches.
    """
    conn = sqlite3.connect(str(dst_db))
    cur = conn.cursor()

    # Read camera IDs
    cur.execute("SELECT camera_id FROM cameras ORDER BY camera_id")
    cam_ids = [row[0] for row in cur.fetchall()]
    cam_folder_to_id = {cam: cam_ids[i] for i, cam in enumerate(cameras)}
    log.info(f"Camera mapping: {cam_folder_to_id}")

    ref_cam = cameras[ref_camera_idx]
    ref_cam_id = cam_folder_to_id[ref_cam]
    log.info(f"Reference camera: {ref_cam} (id={ref_cam_id})")

    # Load calibrations
    calibs = {}
    for cam in cameras:
        calib_file = calibs_dir / f"{cam}.json"
        if calib_file.exists():
            calibs[cam] = load_calib_prior(calib_file)

    # Clear and rebuild rig structure
    cur.execute("DELETE FROM rigs")
    cur.execute("DELETE FROM rig_sensors")
    cur.execute("DELETE FROM frames")
    cur.execute("DELETE FROM frame_data")

    rig_id = 1
    cur.execute(
        "INSERT INTO rigs (rig_id, ref_sensor_id, ref_sensor_type) VALUES (?, ?, ?)",
        (rig_id, ref_cam_id, 0)
    )
    log.info(f"Created rig {rig_id} with ref_sensor={ref_cam_id}")

    ref_calib = calibs.get(ref_cam)
    for cam in cameras:
        cam_id = cam_folder_to_id[cam]
        if cam_id == ref_cam_id:
            continue
        sensor_calib = calibs.get(cam)
        if ref_calib is None or sensor_calib is None:
            continue
        quat_wxyz, t_rel = compute_sensor_from_rig(ref_calib, sensor_calib)
        blob = np.array([quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3],
                         t_rel[0], t_rel[1], t_rel[2]], dtype=np.float64).tobytes()
        cur.execute(
            "INSERT INTO rig_sensors (rig_id, sensor_id, sensor_type, sensor_from_rig) VALUES (?, ?, ?, ?)",
            (rig_id, cam_id, 0, blob)
        )
        log.info(f"  Added rig_sensor: cam={cam} (id={cam_id}), "
                 f"t=[{t_rel[0]:.3f}, {t_rel[1]:.3f}, {t_rel[2]:.3f}]")

    # Build frames from images
    cur.execute("SELECT image_id, camera_id, name FROM images ORDER BY image_id")
    all_images = cur.fetchall()
    frame_groups = {}
    for image_id, camera_id, name in all_images:
        frame_idx = name.split("/")[-1].replace("frame_", "").replace(".png", "")
        frame_groups.setdefault(frame_idx, []).append((image_id, camera_id, name))

    log.info(f"Found {len(frame_groups)} unique timestamps across {len(all_images)} images")

    for frame_id, (frame_idx, images) in enumerate(sorted(frame_groups.items()), start=1):
        cur.execute("INSERT INTO frames (frame_id, rig_id) VALUES (?, ?)", (frame_id, rig_id))
        for image_id, camera_id, name in images:
            cur.execute(
                "INSERT INTO frame_data (frame_id, data_id, sensor_id, sensor_type) VALUES (?, ?, ?, ?)",
                (frame_id, image_id, camera_id, 0)
            )

    log.info(f"Created {len(frame_groups)} frames with {len(all_images)} frame_data entries")
    conn.commit()
    conn.close()


def _find_kitti_calib(recording_dir: Path) -> Optional[Path]:
    """Find calib_cam_to_cam.txt in the recording directory."""
    for d in recording_dir.glob("2011_*"):
        candidate = d / "calib_cam_to_cam.txt"
        if candidate.exists():
            return candidate
    candidate = recording_dir / "calib_cam_to_cam.txt"
    if candidate.exists():
        return candidate
    return None


def preprocess_kitti(
    recording_dir: Path,
    matching: str = "sequential",
    overlap: int = 10,
    ref_camera_idx: int = 0,
    skip_features: bool = False,
    skip_matching: bool = False,
    raw_images_dir: Optional[Path] = None,
):
    """Run the full KITTI preprocessing pipeline."""

    images_dir = recording_dir / "images"
    calibs_dir = recording_dir / "calibs_prior"
    src_db = recording_dir / "database.db"
    dst_db = recording_dir / "database_rig.db"
    odometry_csv = recording_dir / "odometry.csv"

    # Validate inputs
    if not calibs_dir.is_dir():
        raise FileNotFoundError(f"Calibs directory not found: {calibs_dir}")
    if not src_db.exists():
        raise FileNotFoundError(f"Source database not found: {src_db}")
    if not odometry_csv.exists():
        raise FileNotFoundError(f"Odometry CSV not found: {odometry_csv}")

    # Step 0: Prepare images from raw KITTI layout
    log.info("Step 0: Preparing images from raw KITTI data...")
    prepare_images(recording_dir, raw_base_dir=raw_images_dir)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Step 1: Discover cameras
    cameras = discover_cameras(images_dir)
    log.info(f"Discovered {len(cameras)} cameras: {cameras}")

    # Step 2: Setup rig database
    # When skipping features, reuse existing database_rig.db if it has features
    if skip_features and dst_db.exists():
        conn = sqlite3.connect(str(dst_db))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM keypoints")
        has_features = cur.fetchone()[0] > 0
        conn.close()
        if has_features:
            log.info("Step 2: Reusing existing database_rig.db (has features)")
            # Just rebuild rig structure in-place without re-copying
            _rebuild_rig_structure(dst_db, cameras, calibs_dir, images_dir, ref_camera_idx)
        else:
            log.info("Step 2: database_rig.db has no features, re-copying from database.db")
            cam_mapping = setup_rig_database(
                src_db=src_db, dst_db=dst_db, cameras=cameras,
                calibs_dir=calibs_dir, images_dir=images_dir,
                ref_camera_idx=ref_camera_idx,
            )
    else:
        cam_mapping = setup_rig_database(
            src_db=src_db, dst_db=dst_db, cameras=cameras,
            calibs_dir=calibs_dir, images_dir=images_dir,
            ref_camera_idx=ref_camera_idx,
        )

    # Step 2c: Update camera intrinsics for rectified images if calib file exists
    calib_file = _find_kitti_calib(recording_dir)
    if calib_file is not None:
        log.info(f"Step 2c: Updating camera intrinsics from {calib_file.name}...")
        calib_data = parse_kitti_calib(calib_file)
        update_cameras_for_rectified(dst_db, calib_data, num_cameras=len(cameras))
    else:
        log.warning("No calib_cam_to_cam.txt found — camera intrinsics not updated for rectified images")

    # Step 3: Extract features
    if not skip_features:
        extract_features(dst_db, images_dir)
    else:
        log.info("Skipping feature extraction (--skip-features)")

    # Step 4: Match features
    if not skip_matching:
        match_features(dst_db, matching=matching, overlap=overlap)
    else:
        log.info("Skipping feature matching (--skip-matching)")

    log.info(f"KITTI preprocessing complete. Output: {dst_db}")
    log.info(f"Next step: run triangulate_from_odometry.py with:")
    log.info(f"  --database     {dst_db}")
    log.info(f"  --images       {images_dir}")
    log.info(f"  --odometry     {odometry_csv}")
    log.info(f"  --calibs-prior {calibs_dir}")
    log.info(f"  --output       {recording_dir / 'reconstruction' / 'odometry_rig_triangulated'}")

    return dst_db


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess KITTI raw data for the BA pipeline"
    )
    parser.add_argument("--recording", type=Path, required=True,
                        help="Path to recording directory (e.g. pre-triangulation/kitti/kitti_01)")
    parser.add_argument("--matching", choices=["exhaustive", "sequential"], default="sequential",
                        help="Feature matching strategy (default: sequential)")
    parser.add_argument("--overlap", type=int, default=10,
                        help="Sequential matching overlap (default: 10)")
    parser.add_argument("--ref-camera", type=int, default=0,
                        help="Reference camera index (default: 0 = first camera)")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature extraction (use existing)")
    parser.add_argument("--skip-matching", action="store_true",
                        help="Skip feature matching (use existing)")
    parser.add_argument("--raw-images-dir", type=Path, default=None,
                        help="Directory containing image_XX/data/ subdirs (if not inside recording)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    preprocess_kitti(
        recording_dir=args.recording,
        matching=args.matching,
        overlap=args.overlap,
        ref_camera_idx=args.ref_camera,
        skip_features=args.skip_features,
        skip_matching=args.skip_matching,
        raw_images_dir=args.raw_images_dir,
    )
