"""Generate synthetic problem.json + ground_truth.json for BA testing.

This module is the single source of truth for synthetic scene generation.
It can be used directly via CLI or imported by the experiment framework.

Usage:
    python -m calibri.tools.synth_gen --output data/problem.json --gt data/ground_truth.json
    python -m calibri.tools.synth_gen --num-cameras 4 --scene-type corridor --trajectory circular
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation

from calibri.tools.utils import (
    scipy_quat_to_json, json_quat_to_scipy,
    project_pinhole_radtan, project_pinhole_fisheye,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_WIDTH = 1280
DEFAULT_IMAGE_HEIGHT = 800
DEFAULT_CX = 640.0
DEFAULT_CY = 400.0
DEFAULT_TIMESTAMP_STEP_NS = 1_000_000_000  # 1 second between frames
MIN_DEPTH_M = 0.1  # behind-camera cull threshold (meters)


# ============================================================================
# Camera Generation (A4 / A6)
# ============================================================================

def generate_camera(
    model: str = "Pinhole",
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
) -> Dict:
    """Single camera with model-appropriate default parameters.

    Args:
        model: "Pinhole" or "Fisheye"
        width: Image width in pixels
        height: Image height in pixels
    """
    cx = width / 2.0
    cy = height / 2.0
    cam = {
        "id": 0,
        "model": model,
        "width": width,
        "height": height,
        "intrinsics": {"fx": 950.0, "fy": 950.0, "cx": cx, "cy": cy},
        "extrinsic": {"q": [1, 0, 0, 0], "t": [0, 0, 0], "fixed": True},
    }
    if model == "Fisheye":
        cam["intrinsics"]["fx"] = 400.0
        cam["intrinsics"]["fy"] = 400.0
        cam["distortion"] = {"k1": 0.05, "k2": -0.02, "k3": 0.005, "k4": -0.001}
    else:
        cam["distortion"] = {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0}
    return cam


def generate_rig(
    n_cameras: int = 4,
    model: str = "Pinhole",
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
) -> List[Dict]:
    """Generate an N-camera automotive rig.

    Layout (for n_cameras=4):
      Camera 0 (front, reference): identity rotation, t=[0,0,0]
      Camera 1 (left):  90 deg yaw left,  t=[-0.8, 0, 1.0]
      Camera 2 (right): 90 deg yaw right, t=[0.8, 0, 1.0]
      Camera 3 (rear):  180 deg yaw,      t=[0, 0.3, -2.0]
    """
    cx = width / 2.0
    cy = height / 2.0

    if model == "Fisheye":
        cam_defs = [
            {"id": 0, "yaw_deg": 0.0,   "t": [0.0, 0.0, 0.0],  "fx": 400.0, "fy": 400.0,
             "dist": {"k1": 0.05, "k2": -0.02, "k3": 0.005, "k4": -0.001}},
            {"id": 1, "yaw_deg": 90.0,  "t": [-0.8, 0.0, 1.0],  "fx": 350.0, "fy": 350.0,
             "dist": {"k1": 0.04, "k2": -0.01, "k3": 0.003, "k4": -0.0005}},
            {"id": 2, "yaw_deg": -90.0, "t": [0.8, 0.0, 1.0],   "fx": 350.0, "fy": 350.0,
             "dist": {"k1": 0.04, "k2": -0.01, "k3": 0.003, "k4": -0.0005}},
            {"id": 3, "yaw_deg": 180.0, "t": [0.0, 0.3, -2.0],  "fx": 300.0, "fy": 300.0,
             "dist": {"k1": 0.06, "k2": -0.03, "k3": 0.007, "k4": -0.002}},
        ]
    else:
        cam_defs = [
            {"id": 0, "yaw_deg": 0.0,   "t": [0.0, 0.0, 0.0],  "fx": 950.0, "fy": 950.0,
             "dist": {"k1": -0.10, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0}},
            {"id": 1, "yaw_deg": 90.0,  "t": [-0.8, 0.0, 1.0],  "fx": 800.0, "fy": 800.0,
             "dist": {"k1": -0.12, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0}},
            {"id": 2, "yaw_deg": -90.0, "t": [0.8, 0.0, 1.0],   "fx": 800.0, "fy": 800.0,
             "dist": {"k1": -0.12, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0}},
            {"id": 3, "yaw_deg": 180.0, "t": [0.0, 0.3, -2.0],  "fx": 700.0, "fy": 700.0,
             "dist": {"k1": -0.15, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0}},
        ]

    cameras = []
    for cd in cam_defs[:n_cameras]:
        rot = Rotation.from_euler("y", cd["yaw_deg"], degrees=True)
        q_wxyz = scipy_quat_to_json(rot.as_quat())

        cameras.append({
            "id": cd["id"],
            "model": model,
            "width": width,
            "height": height,
            "intrinsics": {
                "fx": cd["fx"], "fy": cd["fy"],
                "cx": cx, "cy": cy,
            },
            "distortion": cd["dist"],
            "extrinsic": {
                "q": q_wxyz,
                "t": cd["t"],
                "fixed": cd["id"] == 0,
            },
        })

    return cameras


# ============================================================================
# Trajectory Generation (A5)
# ============================================================================

def generate_poses(
    num_poses: int = 15,
    trajectory_type: str = "arc",
    multicam: bool = False,
) -> List[Tuple[int, List[float], List[float]]]:
    """Generate vehicle poses along a synthetic trajectory.

    Convention: vehicle-to-world (T_wv).
    Returns list of (timestamp_ns, quaternion [w,x,y,z], translation [x,y,z]).

    Supported trajectory_type values:
      - "arc": forward motion with sinusoidal yaw and lateral variation
      - "straight": pure forward Z translation, no rotation (critical motion)
      - "circular": constant-radius circular motion
      - "slalom": S-curve lateral oscillation with forward motion
    """
    poses = []
    for i in range(num_poses):
        frac = i / max(num_poses - 1, 1)

        if trajectory_type == "straight":
            tx, ty, tz = 0.0, 0.0, i * 1.0
            yaw = 0.0

        elif trajectory_type == "circular":
            radius = 10.0
            angle = frac * 2.0 * np.pi * 0.75  # 3/4 circle
            tx = radius * np.sin(angle)
            ty = 0.0
            tz = radius * (1.0 - np.cos(angle))
            yaw = np.degrees(angle)

        elif trajectory_type == "slalom":
            amplitude = 3.0
            frequency = 2.0  # full S-curves along the trajectory
            tx = amplitude * np.sin(frac * np.pi * frequency * 2.0)
            ty = 0.0
            tz = i * 0.8
            yaw = np.degrees(np.arctan2(
                amplitude * np.pi * frequency * 2.0 * np.cos(frac * np.pi * frequency * 2.0),
                0.8 * max(num_poses - 1, 1),
            )) if num_poses > 1 else 0.0

        else:  # "arc" (default)
            if multicam:
                yaw = np.sin(frac * np.pi * 2.0) * 15.0
                tx = 3.0 * np.sin(frac * np.pi)
                ty = 1.0 * np.sin(frac * np.pi * 1.5)
                tz = i * 0.8
            else:
                tx = 0.3 * np.sin(frac * np.pi * 0.5)
                ty = 0.0
                tz = i * 0.5
                yaw = frac * 5.0

        rot = Rotation.from_euler("y", yaw, degrees=True)
        q_wxyz = scipy_quat_to_json(rot.as_quat())
        t = [float(tx), float(ty), float(tz)]
        timestamp_ns = (i + 1) * DEFAULT_TIMESTAMP_STEP_NS
        poses.append((timestamp_ns, q_wxyz, t))
    return poses


# ============================================================================
# Scene / Landmark Generation (A3)
# ============================================================================

def generate_landmarks(
    num_landmarks: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Generate random 3D landmarks in front of the camera at Z=5..30m."""
    if rng is None:
        rng = np.random.default_rng(42)
    landmarks = []
    for i in range(num_landmarks):
        x = rng.uniform(-8.0, 8.0)
        y = rng.uniform(-4.0, 4.0)
        z = rng.uniform(5.0, 30.0)
        landmarks.append({"id": i + 1, "X": float(x), "Y": float(y), "Z": float(z)})
    return landmarks


def generate_corridor_landmarks(
    num_landmarks: int = 500,
    length_m: float = 30.0,
    width_m: float = 8.0,
    wall_height_m: float = 4.0,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Generate structured automotive corridor scene landmarks.

    Creates a street-like corridor with:
      - Ground plane points (y ~ 0, spread across corridor floor)
      - Left wall points  (x ~ -width/2, vertical surface)
      - Right wall points (x ~ +width/2, vertical surface)
      - Far-end points (distributed at various depths for triangulation)

    The scene is arranged in the vehicle coordinate frame:
      X = lateral, Y = vertical (up), Z = forward (depth)

    Args:
        num_landmarks: Total number of landmarks to generate
        length_m: Corridor length along Z axis (meters)
        width_m: Corridor width along X axis (meters)
        wall_height_m: Wall height along Y axis (meters)
        rng: Reproducible random generator
    """
    if rng is None:
        rng = np.random.default_rng(42)

    half_w = width_m / 2.0

    # Distribute landmarks across structural elements:
    # 30% ground, 25% left wall, 25% right wall, 20% distributed volume
    n_ground = int(num_landmarks * 0.30)
    n_left = int(num_landmarks * 0.25)
    n_right = int(num_landmarks * 0.25)
    n_volume = num_landmarks - n_ground - n_left - n_right

    landmarks = []
    next_id = 1

    def _add(x, y, z):
        nonlocal next_id
        landmarks.append({"id": next_id, "X": float(x), "Y": float(y), "Z": float(z)})
        next_id += 1

    # Ground plane: y near 0 (slight vertical jitter for texture), spread across floor
    for _ in range(n_ground):
        x = rng.uniform(-half_w + 0.5, half_w - 0.5)
        y = rng.normal(0.0, 0.05)  # nearly flat
        z = rng.uniform(3.0, length_m)
        _add(x, y, z)

    # Left wall: x near -half_w, vertical distribution
    for _ in range(n_left):
        x = -half_w + rng.normal(0.0, 0.1)  # slight depth jitter
        y = rng.uniform(-0.5, wall_height_m)
        z = rng.uniform(3.0, length_m)
        _add(x, y, z)

    # Right wall: x near +half_w, vertical distribution
    for _ in range(n_right):
        x = half_w + rng.normal(0.0, 0.1)
        y = rng.uniform(-0.5, wall_height_m)
        z = rng.uniform(3.0, length_m)
        _add(x, y, z)

    # Volume points: distributed for depth variation and triangulation
    for _ in range(n_volume):
        x = rng.uniform(-half_w, half_w)
        y = rng.uniform(-1.0, wall_height_m)
        z = rng.uniform(3.0, length_m)
        _add(x, y, z)

    return landmarks


def generate_multicam_landmarks(
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Generate landmarks distributed across all 4 camera FOVs + overlap regions.

    Returns ~700 landmarks covering front, left, right, rear sectors.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    landmarks = []
    next_id = 1

    def add_sector(count, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi):
        nonlocal next_id
        for _ in range(count):
            x = rng.uniform(x_lo, x_hi)
            y = rng.uniform(y_lo, y_hi)
            z = rng.uniform(z_lo, z_hi)
            landmarks.append({"id": next_id, "X": float(x), "Y": float(y), "Z": float(z)})
            next_id += 1

    add_sector(200, -8.0,  8.0, -3.0, 3.0,   5.0,  30.0)  # front
    add_sector(150, -30.0, -5.0, -3.0, 3.0,  -5.0,   5.0)  # left
    add_sector(150,  5.0,  30.0, -3.0, 3.0,  -5.0,   5.0)  # right
    add_sector(150, -8.0,   8.0, -3.0, 3.0, -30.0,  -5.0)  # rear
    add_sector(50,  -12.0, -3.0, -3.0, 3.0,   3.0,  12.0)  # front-left overlap
    add_sector(50,   3.0,  12.0, -3.0, 3.0,   3.0,  12.0)  # front-right overlap

    return landmarks


def generate_surround_landmarks(
    num_landmarks: int = 800,
    radius_m: float = 15.0,
    wall_height_m: float = 4.0,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Generate structured 360-degree surround-view scene for a multi-camera rig.

    Creates an automotive urban-block environment around the vehicle origin with:
      - Ground plane points in a ring around the vehicle (20%)
      - Front facade / wall at +Z (15%)
      - Rear facade / wall at -Z (15%)
      - Left facade / wall at -X (15%)
      - Right facade / wall at +X (15%)
      - Volume points at mixed near/mid/far depths (20%)

    All four rig cameras (front, left, right, rear) observe meaningful
    structured geometry rather than random point clouds.

    Coordinate frame: X = lateral, Y = vertical (up), Z = forward.
    Vehicle is at the origin; walls are placed at approximately ±radius_m
    along the relevant axis.

    Args:
        num_landmarks: Total number of landmarks to generate
        radius_m: Distance from origin to facade surfaces (meters)
        wall_height_m: Height of vertical facade surfaces (meters)
        rng: Reproducible random generator
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Budget: ground 20%, front 15%, rear 15%, left 15%, right 15%, volume 20%
    n_ground = int(num_landmarks * 0.20)
    n_front = int(num_landmarks * 0.15)
    n_rear = int(num_landmarks * 0.15)
    n_left = int(num_landmarks * 0.15)
    n_right = int(num_landmarks * 0.15)
    n_volume = num_landmarks - n_ground - n_front - n_rear - n_left - n_right

    landmarks = []
    next_id = 1

    def _add(x, y, z):
        nonlocal next_id
        landmarks.append({"id": next_id, "X": float(x), "Y": float(y), "Z": float(z)})
        next_id += 1

    # Ground plane: ring around vehicle, y near 0
    for _ in range(n_ground):
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(3.0, radius_m)
        x = dist * np.cos(angle)
        z = dist * np.sin(angle)
        y = rng.normal(0.0, 0.05)
        _add(x, y, z)

    # Front facade: z near +radius_m, spread in x, vertical in y
    for _ in range(n_front):
        z = radius_m + rng.normal(0.0, 0.3)
        x = rng.uniform(-radius_m * 0.6, radius_m * 0.6)
        y = rng.uniform(-0.5, wall_height_m)
        _add(x, y, z)

    # Rear facade: z near -radius_m
    for _ in range(n_rear):
        z = -radius_m + rng.normal(0.0, 0.3)
        x = rng.uniform(-radius_m * 0.6, radius_m * 0.6)
        y = rng.uniform(-0.5, wall_height_m)
        _add(x, y, z)

    # Left facade: x near -radius_m
    for _ in range(n_left):
        x = -radius_m + rng.normal(0.0, 0.3)
        z = rng.uniform(-radius_m * 0.6, radius_m * 0.6)
        y = rng.uniform(-0.5, wall_height_m)
        _add(x, y, z)

    # Right facade: x near +radius_m
    for _ in range(n_right):
        x = radius_m + rng.normal(0.0, 0.3)
        z = rng.uniform(-radius_m * 0.6, radius_m * 0.6)
        y = rng.uniform(-0.5, wall_height_m)
        _add(x, y, z)

    # Volume points: distributed at mixed depths around the vehicle
    for _ in range(n_volume):
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(3.0, radius_m * 1.2)
        x = dist * np.cos(angle)
        z = dist * np.sin(angle)
        y = rng.uniform(-1.0, wall_height_m)
        _add(x, y, z)

    return landmarks


def generate_scene_landmarks(
    scene_type: str,
    num_landmarks: int,
    num_cameras: int,
    rng: np.random.Generator,
    corridor_length_m: float = 30.0,
    corridor_width_m: float = 8.0,
    corridor_wall_height_m: float = 4.0,
) -> List[Dict]:
    """Dispatch to the appropriate landmark generator based on scene_type.

    Args:
        scene_type: "random", "corridor", or "surround"
        num_landmarks: Total number of landmarks
        num_cameras: Number of cameras (>1 triggers multicam sector layout for random scenes)
        rng: Reproducible random generator
        corridor_*: Parameters for corridor scene geometry
    """
    if scene_type == "corridor":
        return generate_corridor_landmarks(
            num_landmarks=num_landmarks,
            length_m=corridor_length_m,
            width_m=corridor_width_m,
            wall_height_m=corridor_wall_height_m,
            rng=rng,
        )
    elif scene_type == "surround":
        return generate_surround_landmarks(
            num_landmarks=num_landmarks,
            rng=rng,
        )
    else:  # "random"
        if num_cameras > 1:
            return generate_multicam_landmarks(rng=rng)
        else:
            return generate_landmarks(num_landmarks, rng=rng)


# ============================================================================
# Projection Engine (A6 — critical fix)
# ============================================================================

def project_point(
    Xw: np.ndarray,
    pose_q_wxyz: List[float],
    pose_t: List[float],
    ext_q_wxyz: List[float],
    ext_t: List[float],
    camera: Dict,
) -> Optional[Tuple[float, float]]:
    """Project a 3D world point through vehicle pose + extrinsic + distortion to pixels.

    Dispatches to the correct projection model (Pinhole+RadTan or Fisheye)
    based on camera["model"]. Applies full distortion — NOT ideal pinhole.

    Returns (u, v) or None if behind camera or projection fails.
    """
    # Vehicle-to-world rotation
    R_wv = Rotation.from_quat(json_quat_to_scipy(pose_q_wxyz))
    t_wv = np.array(pose_t)

    # Vehicle-to-camera rotation
    R_vc = Rotation.from_quat(json_quat_to_scipy(ext_q_wxyz))
    t_vc = np.array(ext_t)

    # World to vehicle: X_v = R_wv^(-1) * (X_w - t_wv)
    X_v = R_wv.inv().apply(Xw - t_wv)

    # Vehicle to camera: X_c = R_vc * X_v + t_vc
    X_c = R_vc.apply(X_v) + t_vc

    if X_c[2] <= MIN_DEPTH_M:
        return None

    intr = camera["intrinsics"]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    dist = camera.get("distortion", {})
    model = camera.get("model", "Pinhole")

    if model == "Fisheye":
        k1 = dist.get("k1", 0.0)
        k2 = dist.get("k2", 0.0)
        k3 = dist.get("k3", 0.0)
        k4 = dist.get("k4", 0.0)
        return project_pinhole_fisheye(X_c, fx, fy, cx, cy, k1, k2, k3, k4)
    else:
        # Pinhole + RadTan
        k1 = dist.get("k1", 0.0)
        k2 = dist.get("k2", 0.0)
        p1 = dist.get("p1", 0.0)
        p2 = dist.get("p2", 0.0)
        k3 = dist.get("k3", 0.0)
        return project_pinhole_radtan(X_c, fx, fy, cx, cy, k1, k2, p1, p2, k3)


# ============================================================================
# Observation Generation
# ============================================================================

def generate_observations(
    poses: List[Tuple[int, List[float], List[float]]],
    landmarks: List[Dict],
    camera: Dict,
    noise_sigma: float = 0.5,
    outlier_ratio: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Project landmarks through all poses for a single camera.

    Applies full camera model (including distortion) during projection.
    Culls out-of-frame and behind-camera points.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    ext = camera["extrinsic"]
    ext_q = ext["q"]
    ext_t = ext["t"]
    width = camera["width"]
    height = camera["height"]
    cam_id = camera["id"]

    observations = []
    for ts, pose_q, pose_t in poses:
        for lm in landmarks:
            Xw = np.array([lm["X"], lm["Y"], lm["Z"]])
            uv = project_point(Xw, pose_q, pose_t, ext_q, ext_t, camera)
            if uv is None:
                continue
            u, v = uv
            # Add noise
            u_noisy = u + rng.normal(0, noise_sigma)
            v_noisy = v + rng.normal(0, noise_sigma)
            # Cull out of frame
            if u_noisy < 0 or u_noisy >= width or v_noisy < 0 or v_noisy >= height:
                continue
            observations.append({
                "landmark_id": lm["id"],
                "camera_id": cam_id,
                "timestamp_ns": ts,
                "u": float(u_noisy),
                "v": float(v_noisy),
            })

    # Apply outlier corruption if requested
    if outlier_ratio > 0.0:
        n_outliers = int(len(observations) * outlier_ratio)
        if n_outliers > 0:
            outlier_indices = rng.choice(len(observations), size=n_outliers, replace=False)
            for idx in outlier_indices:
                observations[idx]["u"] = float(rng.uniform(0, width))
                observations[idx]["v"] = float(rng.uniform(0, height))

    return observations


def generate_multicam_observations(
    poses: List[Tuple[int, List[float], List[float]]],
    landmarks: List[Dict],
    cameras: List[Dict],
    noise_sigma: float = 0.5,
    outlier_ratio: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Project landmarks through all poses and all cameras."""
    if rng is None:
        rng = np.random.default_rng(42)

    all_observations = []
    for camera in cameras:
        obs = generate_observations(poses, landmarks, camera,
                                    noise_sigma=noise_sigma,
                                    outlier_ratio=outlier_ratio,
                                    rng=rng)
        all_observations.extend(obs)

    return all_observations


# ============================================================================
# Perturbation Engine (A9)
# ============================================================================

def perturb_poses(
    poses: List[Tuple[int, List[float], List[float]]],
    rot_sigma_deg: float = 1.0,
    trans_sigma_m: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[int, List[float], List[float]]]:
    """Perturb pose estimates (skip first pose -- gauge-fixed)."""
    if rng is None:
        rng = np.random.default_rng(123)
    perturbed = []
    for i, (ts, q_wxyz, t) in enumerate(poses):
        if i == 0:
            perturbed.append((ts, q_wxyz, t))
            continue
        # Perturb rotation
        axis = rng.normal(0, 1, 3)
        axis /= np.linalg.norm(axis) + 1e-10
        angle_deg = rng.normal(0, rot_sigma_deg)
        dR = Rotation.from_rotvec(axis * np.radians(angle_deg))
        R_orig = Rotation.from_quat(json_quat_to_scipy(q_wxyz))
        R_pert = R_orig * dR
        q_pert_wxyz = scipy_quat_to_json(R_pert.as_quat())
        # Perturb translation
        dt = rng.normal(0, trans_sigma_m, 3)
        t_pert = [float(t[j] + dt[j]) for j in range(3)]
        perturbed.append((ts, q_pert_wxyz, t_pert))
    return perturbed


def perturb_landmarks(
    landmarks: List[Dict],
    sigma_m: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Perturb landmark positions."""
    if rng is None:
        rng = np.random.default_rng(456)
    perturbed = []
    for lm in landmarks:
        dx = rng.normal(0, sigma_m, 3)
        perturbed.append({
            "id": lm["id"],
            "X": float(lm["X"] + dx[0]),
            "Y": float(lm["Y"] + dx[1]),
            "Z": float(lm["Z"] + dx[2]),
        })
    return perturbed


def perturb_extrinsics(
    cameras: List[Dict],
    rot_sigma_deg: float = 2.0,
    trans_sigma_m: float = 0.05,
    reference_id: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Perturb camera extrinsics (skip reference camera).

    Returns cameras with perturbed extrinsics for non-reference cameras.
    Intrinsics and distortion are copied unchanged.
    """
    if rng is None:
        rng = np.random.default_rng(789)

    perturbed = []
    for cam in cameras:
        cam_copy = json.loads(json.dumps(cam))  # deep copy
        if cam["id"] == reference_id:
            perturbed.append(cam_copy)
            continue

        ext = cam_copy["extrinsic"]
        q_wxyz = ext["q"]

        # Perturb rotation
        axis = rng.normal(0, 1, 3)
        axis /= np.linalg.norm(axis) + 1e-10
        angle_deg = rng.normal(0, rot_sigma_deg)
        dR = Rotation.from_rotvec(axis * np.radians(angle_deg))
        R_orig = Rotation.from_quat(json_quat_to_scipy(q_wxyz))
        R_pert = R_orig * dR
        ext["q"] = scipy_quat_to_json(R_pert.as_quat())

        # Perturb translation
        dt = rng.normal(0, trans_sigma_m, 3)
        ext["t"] = [float(ext["t"][j] + dt[j]) for j in range(3)]

        perturbed.append(cam_copy)

    return perturbed


def perturb_intrinsics(
    cameras: List[Dict],
    sigma_px: float = 20.0,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Perturb camera intrinsic parameters (fx, fy, cx, cy).

    Modifies cameras in-place on deep copies. Returns perturbed copies.
    """
    if rng is None:
        rng = np.random.default_rng(321)

    perturbed = []
    for cam in cameras:
        cam_copy = json.loads(json.dumps(cam))
        intr = cam_copy["intrinsics"]
        intr["fx"] += float(rng.normal(0, sigma_px))
        intr["fy"] += float(rng.normal(0, sigma_px))
        intr["cx"] += float(rng.normal(0, sigma_px * 0.25))
        intr["cy"] += float(rng.normal(0, sigma_px * 0.25))
        perturbed.append(cam_copy)
    return perturbed


def perturb_distortion(
    cameras: List[Dict],
    sigma: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Perturb camera distortion coefficients.

    Modifies cameras in-place on deep copies. Returns perturbed copies.
    """
    if rng is None:
        rng = np.random.default_rng(654)

    perturbed = []
    for cam in cameras:
        cam_copy = json.loads(json.dumps(cam))
        dist = cam_copy.get("distortion", {})
        for key in dist:
            dist[key] = float(dist[key] + rng.normal(0, sigma))
        cam_copy["distortion"] = dist
        perturbed.append(cam_copy)
    return perturbed


# ============================================================================
# Serialization (A10)
# ============================================================================

def build_problem_json(
    cameras: List[Dict],
    poses: List[Tuple[int, List[float], List[float]]],
    landmarks: List[Dict],
    observations: List[Dict],
    opt_extrinsics: bool = False,
    reference_camera_id: int = 0,
) -> Dict:
    """Assemble a problem.json dict."""
    return {
        "cameras": cameras,
        "reference_camera_id": reference_camera_id,
        "poses": [
            {"timestamp_ns": ts, "q": q, "t": t} for ts, q, t in poses
        ],
        "landmarks": landmarks,
        "observations": observations,
        "flags": {
            "opt_intrinsics": False,
            "opt_extrinsics": opt_extrinsics,
            "opt_poses": True,
            "opt_landmarks": True,
        },
        "robust": {"type": "Huber", "scale": 1.0},
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic BA problem")
    parser.add_argument("--output", default="data/problem.json",
                        help="Output problem.json path")
    parser.add_argument("--gt", default="data/ground_truth.json",
                        help="Output ground_truth.json path")
    parser.add_argument("--num-poses", type=int, default=15)
    parser.add_argument("--num-landmarks", type=int, default=500)
    parser.add_argument("--num-cameras", type=int, default=1,
                        help="Number of cameras (1=single, 4=surround rig)")
    parser.add_argument("--noise-sigma", type=float, default=0.5)
    parser.add_argument("--outlier-ratio", type=float, default=0.0,
                        help="Fraction of observations to corrupt (0.0 to 1.0)")
    parser.add_argument("--scene-type", type=str, default="random",
                        choices=["random", "corridor", "surround"],
                        help="Scene type: 'random', 'corridor', or 'surround'")
    parser.add_argument("--trajectory", type=str, default="arc",
                        choices=["arc", "straight", "circular", "slalom"],
                        help="Trajectory type")
    parser.add_argument("--camera-model", type=str, default="Pinhole",
                        choices=["Pinhole", "Fisheye"],
                        help="Camera model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0.0 <= args.outlier_ratio <= 1.0):
        parser.error("--outlier-ratio must be between 0.0 and 1.0")

    rng = np.random.default_rng(args.seed)
    multicam = args.num_cameras > 1

    if multicam:
        cameras = generate_rig(args.num_cameras, model=args.camera_model)
        gt_poses = generate_poses(args.num_poses, trajectory_type=args.trajectory, multicam=True)
        gt_landmarks = generate_scene_landmarks(
            scene_type=args.scene_type,
            num_landmarks=args.num_landmarks,
            num_cameras=args.num_cameras,
            rng=rng,
        )
        observations = generate_multicam_observations(
            gt_poses, gt_landmarks, cameras,
            noise_sigma=args.noise_sigma, outlier_ratio=args.outlier_ratio, rng=rng)

        print(f"Generated {len(cameras)}-camera rig ({args.camera_model}), "
              f"{len(gt_poses)} poses ({args.trajectory}), "
              f"{len(gt_landmarks)} landmarks ({args.scene_type}), "
              f"{len(observations)} observations")

        gt_json = build_problem_json(cameras, gt_poses, gt_landmarks, observations,
                                     opt_extrinsics=True, reference_camera_id=0)
        gt_json["seed"] = args.seed

        pert_poses = perturb_poses(gt_poses, rng=rng)
        pert_landmarks = perturb_landmarks(gt_landmarks, rng=rng)
        pert_cameras = perturb_extrinsics(cameras, rng=rng)
        problem_json = build_problem_json(
            pert_cameras, pert_poses, pert_landmarks, observations,
            opt_extrinsics=True, reference_camera_id=0)
    else:
        camera = generate_camera(model=args.camera_model)
        cameras = [camera]
        gt_poses = generate_poses(args.num_poses, trajectory_type=args.trajectory)
        gt_landmarks = generate_scene_landmarks(
            scene_type=args.scene_type,
            num_landmarks=args.num_landmarks,
            num_cameras=1,
            rng=rng,
        )
        observations = generate_observations(gt_poses, gt_landmarks, camera,
                                             noise_sigma=args.noise_sigma,
                                             outlier_ratio=args.outlier_ratio,
                                             rng=rng)

        print(f"Generated {len(gt_poses)} poses ({args.trajectory}), "
              f"{len(gt_landmarks)} landmarks ({args.scene_type}), "
              f"{len(observations)} observations ({args.camera_model})")

        gt_json = build_problem_json(cameras, gt_poses, gt_landmarks, observations)
        gt_json["seed"] = args.seed

        pert_poses = perturb_poses(gt_poses, rng=rng)
        pert_landmarks = perturb_landmarks(gt_landmarks, rng=rng)
        problem_json = build_problem_json(cameras, pert_poses, pert_landmarks, observations)

    out_path = Path(args.output)
    gt_path = Path(args.gt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(problem_json, f, indent=2)
    print(f"Wrote problem: {out_path}")

    with open(gt_path, "w") as f:
        json.dump(gt_json, f, indent=2)
    print(f"Wrote ground truth: {gt_path}")


if __name__ == "__main__":
    main()
