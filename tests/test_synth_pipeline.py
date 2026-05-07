"""Tests for the synthetic data generation pipeline.

Covers:
  - Config parsing and validation (A1)
  - Scene generation (A3)
  - Trajectory generation (A5)
  - Projection with distortion (A6)
  - Fisheye projection (A6)
  - Perturbation separation (A9)
  - GT evaluation (A11)
  - Integration: generate -> evaluate roundtrip
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

# ---------------------------------------------------------------------------
# A1: Config layer
# ---------------------------------------------------------------------------

from calibri.framework.config import (
    DataConfig, PerturbationConfig, ExperimentConfig,
    SolverFlags, RobustConfig, SolverConfig, AnalysisConfig,
    validate_config, load_config, config_to_dict,
    VALID_SCENE_TYPES, VALID_TRAJECTORY_TYPES, VALID_CAMERA_MODELS,
)


class TestConfigValidation:
    def _make_config(self, **data_overrides):
        dc = DataConfig(**data_overrides)
        return ExperimentConfig(name="test", data=dc)

    def test_defaults_valid(self):
        validate_config(self._make_config())

    def test_scene_type_valid(self):
        for st in VALID_SCENE_TYPES:
            validate_config(self._make_config(scene_type=st))

    def test_scene_type_invalid(self):
        with pytest.raises(ValueError, match="scene_type"):
            validate_config(self._make_config(scene_type="forest"))

    def test_trajectory_type_valid(self):
        for tt in VALID_TRAJECTORY_TYPES:
            validate_config(self._make_config(trajectory_type=tt))

    def test_trajectory_type_invalid(self):
        with pytest.raises(ValueError, match="trajectory_type"):
            validate_config(self._make_config(trajectory_type="zigzag"))

    def test_camera_model_valid(self):
        for cm in VALID_CAMERA_MODELS:
            validate_config(self._make_config(camera_model=cm))

    def test_camera_model_invalid(self):
        with pytest.raises(ValueError, match="camera_model"):
            validate_config(self._make_config(camera_model="InvalidModel"))

    def test_outlier_ratio_range(self):
        validate_config(self._make_config(outlier_ratio=0.0))
        validate_config(self._make_config(outlier_ratio=0.5))
        with pytest.raises(ValueError, match="outlier_ratio"):
            validate_config(self._make_config(outlier_ratio=-0.1))
        with pytest.raises(ValueError, match="outlier_ratio"):
            validate_config(self._make_config(outlier_ratio=1.5))

    def test_perturbation_sigmas_nonneg(self):
        with pytest.raises(ValueError, match="perturbation"):
            p = PerturbationConfig(pose_rot_sigma_deg=-1.0)
            validate_config(self._make_config(perturbation=p))

    def test_config_yaml_roundtrip(self):
        import yaml
        cfg = self._make_config(
            scene_type="corridor",
            trajectory_type="circular",
            camera_model="Fisheye",
            outlier_ratio=0.1,
            perturbation=PerturbationConfig(intrinsic_sigma_px=10.0),
        )
        d = config_to_dict(cfg)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(d, f)
            f.flush()
            loaded = load_config(f.name)
        assert loaded.data.scene_type == "corridor"
        assert loaded.data.trajectory_type == "circular"
        assert loaded.data.camera_model == "Fisheye"
        assert loaded.data.outlier_ratio == 0.1
        assert loaded.data.perturbation.intrinsic_sigma_px == 10.0


# ---------------------------------------------------------------------------
# A3: Scene generation
# ---------------------------------------------------------------------------

from calibri.tools.synth_gen import (
    generate_landmarks, generate_corridor_landmarks,
    generate_scene_landmarks, generate_multicam_landmarks,
    generate_surround_landmarks,
)


class TestSceneGeneration:
    def test_random_landmarks_count(self):
        rng = np.random.default_rng(42)
        lm = generate_landmarks(100, rng=rng)
        assert len(lm) == 100

    def test_random_landmarks_range(self):
        rng = np.random.default_rng(42)
        lm = generate_landmarks(500, rng=rng)
        for l in lm:
            assert -8.0 <= l["X"] <= 8.0
            assert -4.0 <= l["Y"] <= 4.0
            assert 5.0 <= l["Z"] <= 30.0

    def test_corridor_landmarks_count(self):
        rng = np.random.default_rng(42)
        lm = generate_corridor_landmarks(200, rng=rng)
        assert len(lm) == 200

    def test_corridor_landmarks_structure(self):
        rng = np.random.default_rng(42)
        lm = generate_corridor_landmarks(1000, length_m=30, width_m=8, wall_height_m=4, rng=rng)
        xs = [l["X"] for l in lm]
        ys = [l["Y"] for l in lm]
        zs = [l["Z"] for l in lm]
        # Should have points near walls (x ~ +/-4) and near ground (y ~ 0)
        near_left_wall = sum(1 for x in xs if x < -3.0)
        near_right_wall = sum(1 for x in xs if x > 3.0)
        near_ground = sum(1 for y in ys if abs(y) < 0.5)
        assert near_left_wall > 200, "Expected substantial left wall points"
        assert near_right_wall > 200, "Expected substantial right wall points"
        assert near_ground > 200, "Expected substantial ground plane points"
        # All Z should be >= 3 (min depth)
        assert all(z >= 3.0 for z in zs)

    def test_corridor_deterministic(self):
        lm1 = generate_corridor_landmarks(100, rng=np.random.default_rng(99))
        lm2 = generate_corridor_landmarks(100, rng=np.random.default_rng(99))
        assert lm1 == lm2

    def test_scene_dispatch(self):
        rng = np.random.default_rng(42)
        lm_rand = generate_scene_landmarks("random", 100, 1, rng)
        rng = np.random.default_rng(42)
        lm_corr = generate_scene_landmarks("corridor", 100, 1, rng)
        # They should differ because corridor generates differently
        xs_rand = [l["X"] for l in lm_rand]
        xs_corr = [l["X"] for l in lm_corr]
        assert xs_rand != xs_corr  # different distributions

    def test_multicam_landmarks(self):
        lm = generate_multicam_landmarks(rng=np.random.default_rng(42))
        assert len(lm) == 750  # 200+150+150+150+50+50

    def test_surround_landmarks_count(self):
        rng = np.random.default_rng(42)
        lm = generate_surround_landmarks(600, rng=rng)
        assert len(lm) == 600

    def test_surround_landmarks_deterministic(self):
        lm1 = generate_surround_landmarks(200, rng=np.random.default_rng(77))
        lm2 = generate_surround_landmarks(200, rng=np.random.default_rng(77))
        assert lm1 == lm2

    def test_surround_landmarks_covers_all_directions(self):
        rng = np.random.default_rng(42)
        lm = generate_surround_landmarks(1000, radius_m=15.0, rng=rng)
        xs = [l["X"] for l in lm]
        zs = [l["Z"] for l in lm]
        # Points must exist in all four quadrants
        front = sum(1 for z in zs if z > 10.0)
        rear = sum(1 for z in zs if z < -10.0)
        left = sum(1 for x in xs if x < -10.0)
        right = sum(1 for x in xs if x > 10.0)
        assert front > 100, f"Front points: {front}, expected > 100"
        assert rear > 100, f"Rear points: {rear}, expected > 100"
        assert left > 100, f"Left points: {left}, expected > 100"
        assert right > 100, f"Right points: {right}, expected > 100"

    def test_surround_landmarks_has_ground(self):
        rng = np.random.default_rng(42)
        lm = generate_surround_landmarks(1000, rng=rng)
        ys = [l["Y"] for l in lm]
        near_ground = sum(1 for y in ys if abs(y) < 0.5)
        assert near_ground > 100, f"Ground points: {near_ground}, expected > 100"

    def test_surround_dispatch(self):
        rng = np.random.default_rng(42)
        lm = generate_scene_landmarks("surround", 500, 4, rng)
        assert len(lm) == 500

    def test_surround_config_valid(self):
        from calibri.framework.config import VALID_SCENE_TYPES
        assert "surround" in VALID_SCENE_TYPES

    def test_surround_multicam_observations(self):
        """Surround scene + 4-cam rig should produce observations on all cameras."""
        from calibri.tools.synth_gen import (
            generate_rig, generate_poses, generate_multicam_observations,
        )
        rng = np.random.default_rng(42)
        cameras = generate_rig(4)
        poses = generate_poses(10, trajectory_type="arc", multicam=True)
        lm = generate_surround_landmarks(500, rng=rng)
        obs = generate_multicam_observations(poses, lm, cameras, noise_sigma=0.5, rng=rng)
        cam_ids_seen = set(o["camera_id"] for o in obs)
        assert cam_ids_seen == {0, 1, 2, 3}, \
            f"Expected all 4 cameras to have observations, got camera ids: {cam_ids_seen}"
        # Each camera should have a meaningful number
        for cid in range(4):
            count = sum(1 for o in obs if o["camera_id"] == cid)
            assert count > 50, f"Camera {cid} has only {count} observations, expected > 50"


# ---------------------------------------------------------------------------
# A5: Trajectory generation
# ---------------------------------------------------------------------------

from calibri.tools.synth_gen import generate_poses


class TestTrajectoryGeneration:
    def test_straight(self):
        poses = generate_poses(10, trajectory_type="straight")
        assert len(poses) == 10
        for i, (ts, q, t) in enumerate(poses):
            assert t[0] == 0.0  # no lateral
            assert t[1] == 0.0  # no vertical
            assert abs(t[2] - i * 1.0) < 1e-6  # forward
            assert q == [1, 0, 0, 0] or abs(q[0] - 1.0) < 1e-6  # identity rotation

    def test_circular(self):
        poses = generate_poses(20, trajectory_type="circular")
        assert len(poses) == 20
        # Should have non-trivial lateral motion
        xs = [t[0] for _, _, t in poses]
        assert max(xs) > 5.0, "Circular should produce significant lateral motion"

    def test_slalom(self):
        poses = generate_poses(20, trajectory_type="slalom")
        assert len(poses) == 20
        xs = [t[0] for _, _, t in poses]
        # Slalom should oscillate laterally
        assert max(xs) > 1.0
        assert min(xs) < -1.0

    def test_arc_default(self):
        poses = generate_poses(15, trajectory_type="arc")
        assert len(poses) == 15

    def test_deterministic(self):
        p1 = generate_poses(10, trajectory_type="circular")
        p2 = generate_poses(10, trajectory_type="circular")
        for (_, q1, t1), (_, q2, t2) in zip(p1, p2):
            assert q1 == q2
            assert t1 == t2


# ---------------------------------------------------------------------------
# A6: Projection engine — distortion is actually applied
# ---------------------------------------------------------------------------

from calibri.tools.utils import project_pinhole_radtan, project_pinhole_fisheye
from calibri.tools.synth_gen import project_point, generate_camera


class TestProjection:
    def test_radtan_distortion_applied(self):
        """Verify that RadTan distortion produces different results from ideal pinhole."""
        X_cam = np.array([0.5, 0.3, 5.0])
        fx, fy, cx, cy = 950.0, 950.0, 640.0, 400.0

        # Ideal pinhole (zero distortion)
        u_ideal = fx * (X_cam[0] / X_cam[2]) + cx
        v_ideal = fy * (X_cam[1] / X_cam[2]) + cy

        # With distortion
        uv = project_pinhole_radtan(X_cam, fx, fy, cx, cy, -0.1, 0.01, 0.001, -0.001, 0.0)
        assert uv is not None
        u_dist, v_dist = uv

        # Should differ from ideal pinhole
        assert abs(u_dist - u_ideal) > 0.01, "Distortion must change projection"
        assert abs(v_dist - v_ideal) > 0.01, "Distortion must change projection"

    def test_zero_distortion_matches_pinhole(self):
        """Zero distortion should produce ideal pinhole result."""
        X_cam = np.array([0.5, 0.3, 5.0])
        fx, fy, cx, cy = 950.0, 950.0, 640.0, 400.0
        uv = project_pinhole_radtan(X_cam, fx, fy, cx, cy, 0, 0, 0, 0, 0)
        u_ideal = fx * (X_cam[0] / X_cam[2]) + cx
        v_ideal = fy * (X_cam[1] / X_cam[2]) + cy
        assert abs(uv[0] - u_ideal) < 1e-10
        assert abs(uv[1] - v_ideal) < 1e-10

    def test_behind_camera_radtan(self):
        X_cam = np.array([0.5, 0.3, -1.0])
        assert project_pinhole_radtan(X_cam, 950, 950, 640, 400, 0, 0, 0, 0, 0) is None

    def test_fisheye_projection(self):
        """Fisheye projection should produce valid results."""
        X_cam = np.array([0.5, 0.3, 5.0])
        uv = project_pinhole_fisheye(X_cam, 400, 400, 640, 400, 0.05, -0.02, 0.005, -0.001)
        assert uv is not None
        u, v = uv
        # Should be reasonable pixel coordinates
        assert 0 < u < 1280
        assert 0 < v < 800

    def test_fisheye_behind_camera(self):
        X_cam = np.array([0.5, 0.3, -1.0])
        assert project_pinhole_fisheye(X_cam, 400, 400, 640, 400, 0, 0, 0, 0) is None

    def test_fisheye_compresses_vs_pinhole(self):
        """Fisheye with positive k1 should move off-axis points toward center."""
        X_cam = np.array([2.0, 1.0, 3.0])  # wide angle
        fx, fy, cx, cy = 400.0, 400.0, 640.0, 400.0
        # Pure pinhole (no distortion)
        u_pin = fx * (X_cam[0] / X_cam[2]) + cx
        uv_fish = project_pinhole_fisheye(X_cam, fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0)
        # Fisheye with zero distortion uses theta/r scaling which compresses
        assert uv_fish is not None
        assert abs(uv_fish[0] - cx) < abs(u_pin - cx), "Fisheye should compress wide angles"

    def test_project_point_dispatches_pinhole(self):
        """project_point with Pinhole camera should apply RadTan distortion."""
        cam = generate_camera(model="Pinhole")
        cam["distortion"]["k1"] = -0.1  # nonzero distortion
        Xw = np.array([1.0, 0.5, 10.0])
        q = [1, 0, 0, 0]
        t = [0, 0, 0]
        uv = project_point(Xw, q, t, q, t, cam)
        assert uv is not None
        # Compare with ideal pinhole to verify distortion was applied
        fx, fy = cam["intrinsics"]["fx"], cam["intrinsics"]["fy"]
        cx, cy = cam["intrinsics"]["cx"], cam["intrinsics"]["cy"]
        u_ideal = fx * (Xw[0] / Xw[2]) + cx
        assert abs(uv[0] - u_ideal) > 0.01, "Distortion must be applied"

    def test_project_point_dispatches_fisheye(self):
        """project_point with Fisheye camera should use fisheye model."""
        cam = generate_camera(model="Fisheye")
        Xw = np.array([1.0, 0.5, 10.0])
        q = [1, 0, 0, 0]
        t = [0, 0, 0]
        uv = project_point(Xw, q, t, q, t, cam)
        assert uv is not None


# ---------------------------------------------------------------------------
# A9: Perturbation engine
# ---------------------------------------------------------------------------

from calibri.tools.synth_gen import (
    perturb_poses, perturb_landmarks, perturb_extrinsics,
    perturb_intrinsics, perturb_distortion, generate_rig,
)


class TestPerturbation:
    def test_gt_unchanged_after_pose_perturbation(self):
        poses = generate_poses(5, trajectory_type="arc")
        gt_copy = [(ts, list(q), list(t)) for ts, q, t in poses]
        _ = perturb_poses(poses, rng=np.random.default_rng(42))
        for (ts1, q1, t1), (ts2, q2, t2) in zip(poses, gt_copy):
            assert q1 == q2
            assert t1 == t2

    def test_first_pose_unchanged(self):
        poses = generate_poses(5)
        pert = perturb_poses(poses, rot_sigma_deg=5.0, trans_sigma_m=0.5,
                            rng=np.random.default_rng(42))
        assert pert[0][1] == poses[0][1]  # first pose q unchanged
        assert pert[0][2] == poses[0][2]  # first pose t unchanged
        # Other poses should be different
        assert pert[1][1] != poses[1][1]

    def test_gt_unchanged_after_landmark_perturbation(self):
        rng = np.random.default_rng(42)
        lm = generate_landmarks(50, rng=rng)
        gt_copy = [dict(l) for l in lm]
        _ = perturb_landmarks(lm, rng=np.random.default_rng(99))
        for l1, l2 in zip(lm, gt_copy):
            assert l1["X"] == l2["X"]

    def test_reference_camera_unchanged(self):
        cameras = generate_rig(4)
        pert = perturb_extrinsics(cameras, reference_id=0,
                                  rng=np.random.default_rng(42))
        assert pert[0]["extrinsic"]["q"] == cameras[0]["extrinsic"]["q"]
        assert pert[0]["extrinsic"]["t"] == cameras[0]["extrinsic"]["t"]

    def test_intrinsic_perturbation(self):
        cameras = generate_rig(4)
        pert = perturb_intrinsics(cameras, sigma_px=20.0,
                                  rng=np.random.default_rng(42))
        # Original camera should be unchanged
        assert cameras[0]["intrinsics"]["fx"] == 950.0
        # Perturbed should differ
        assert pert[0]["intrinsics"]["fx"] != 950.0

    def test_distortion_perturbation(self):
        cameras = generate_rig(4)
        pert = perturb_distortion(cameras, sigma=0.05,
                                  rng=np.random.default_rng(42))
        assert cameras[0]["distortion"]["k1"] == -0.1
        assert pert[0]["distortion"]["k1"] != -0.1


# ---------------------------------------------------------------------------
# A11: GT evaluation
# ---------------------------------------------------------------------------

from calibri.framework.evaluate import evaluate_against_gt, format_evaluation_text


class TestEvaluation:
    def _make_gt_and_result(self, tmp_path):
        """Create minimal GT and result files for testing."""
        gt = {
            "cameras": [{
                "id": 0,
                "model": "Pinhole",
                "width": 1280, "height": 800,
                "intrinsics": {"fx": 950.0, "fy": 950.0, "cx": 640.0, "cy": 400.0},
                "distortion": {"k1": -0.1, "k2": 0.01, "p1": 0.001, "p2": -0.001, "k3": 0.0},
                "extrinsic": {"q": [1, 0, 0, 0], "t": [0, 0, 0]},
            }],
            "poses": [
                {"timestamp_ns": 1000000000, "q": [1, 0, 0, 0], "t": [0, 0, 0]},
                {"timestamp_ns": 2000000000, "q": [1, 0, 0, 0], "t": [0, 0, 0.5]},
            ],
            "landmarks": [
                {"id": 1, "X": 1.0, "Y": 0.5, "Z": 10.0},
                {"id": 2, "X": -1.0, "Y": -0.5, "Z": 15.0},
            ],
        }

        result = {
            "final_params": {
                "cameras": [{
                    "id": 0,
                    "intr": [951.0, 949.5, 640.2, 399.8, -0.098, 0.011, 0.0012, -0.0008, 0.0001],
                    "extrinsic": {"q": [0.9999, 0.001, 0.002, 0.003], "t": [0.01, -0.005, 0.008]},
                }],
                "poses": [
                    {"timestamp_ns": 1000000000, "q": [1, 0, 0, 0], "t": [0, 0, 0]},
                    {"timestamp_ns": 2000000000, "q": [0.9999, 0.001, 0.0, 0.0], "t": [0.01, 0.005, 0.52]},
                ],
                "landmarks": [
                    {"id": 1, "X": 1.05, "Y": 0.48, "Z": 9.95},
                    {"id": 2, "X": -0.98, "Y": -0.52, "Z": 15.1},
                ],
            }
        }

        gt_path = tmp_path / "ground_truth.json"
        result_path = tmp_path / "result.json"
        with open(gt_path, "w") as f:
            json.dump(gt, f)
        with open(result_path, "w") as f:
            json.dump(result, f)

        return gt_path, result_path

    def test_evaluation_returns_all_sections(self, tmp_path):
        gt_path, result_path = self._make_gt_and_result(tmp_path)
        ev = evaluate_against_gt(result_path, gt_path)
        assert "extrinsics" in ev
        assert "intrinsics" in ev
        assert "distortion" in ev
        assert "poses" in ev
        assert "landmarks" in ev
        assert "summary" in ev

    def test_evaluation_pose_errors(self, tmp_path):
        gt_path, result_path = self._make_gt_and_result(tmp_path)
        ev = evaluate_against_gt(result_path, gt_path)
        assert len(ev["poses"]) == 2
        # First pose should have zero error (gauge-fixed)
        assert ev["poses"][0]["rot_err_deg"] == 0.0
        assert ev["poses"][0]["trans_err_m"] == 0.0
        # Second pose should have small errors
        assert ev["poses"][1]["rot_err_deg"] > 0
        assert ev["poses"][1]["trans_err_m"] > 0

    def test_evaluation_landmark_errors(self, tmp_path):
        gt_path, result_path = self._make_gt_and_result(tmp_path)
        ev = evaluate_against_gt(result_path, gt_path)
        assert ev["landmarks"]["count"] == 2
        assert ev["landmarks"]["mean_err_m"] > 0

    def test_evaluation_intrinsic_errors(self, tmp_path):
        gt_path, result_path = self._make_gt_and_result(tmp_path)
        ev = evaluate_against_gt(result_path, gt_path)
        assert len(ev["intrinsics"]) == 1
        assert ev["intrinsics"][0]["fx_err"] > 0

    def test_format_text(self, tmp_path):
        gt_path, result_path = self._make_gt_and_result(tmp_path)
        ev = evaluate_against_gt(result_path, gt_path)
        text = format_evaluation_text(ev)
        assert "Ground-Truth Evaluation" in text
        assert "Pose" in text


# ---------------------------------------------------------------------------
# Integration: generate observations with distortion
# ---------------------------------------------------------------------------

from calibri.tools.synth_gen import generate_observations


class TestObservationGeneration:
    def test_observations_with_distortion(self):
        """Observations should be generated using the full distortion model."""
        cam = generate_camera(model="Pinhole")
        cam["distortion"]["k1"] = -0.1
        rng = np.random.default_rng(42)
        poses = generate_poses(5, trajectory_type="arc")
        lm = generate_landmarks(50, rng=rng)
        obs = generate_observations(poses, lm, cam, noise_sigma=0.0, rng=rng)
        assert len(obs) > 0

        # Verify at least one observation differs from ideal pinhole
        fx = cam["intrinsics"]["fx"]
        fy = cam["intrinsics"]["fy"]
        cx = cam["intrinsics"]["cx"]
        cy = cam["intrinsics"]["cy"]
        from scipy.spatial.transform import Rotation
        from calibri.tools.utils import json_quat_to_scipy

        found_distortion_difference = False
        for ob in obs[:20]:  # check first 20
            lm_dict = next(l for l in lm if l["id"] == ob["landmark_id"])
            pose = next(p for p in poses if p[0] == ob["timestamp_ns"])
            Xw = np.array([lm_dict["X"], lm_dict["Y"], lm_dict["Z"]])
            R_wv = Rotation.from_quat(json_quat_to_scipy(pose[1]))
            X_v = R_wv.inv().apply(Xw - np.array(pose[2]))
            X_c = X_v  # identity extrinsic
            if X_c[2] <= 0.1:
                continue
            u_ideal = fx * (X_c[0] / X_c[2]) + cx
            if abs(ob["u"] - u_ideal) > 0.1:  # more than 0.1px difference
                found_distortion_difference = True
                break

        assert found_distortion_difference, \
            "Observations must differ from ideal pinhole when distortion is nonzero"

    def test_fisheye_observations(self):
        """Fisheye camera should generate valid observations."""
        cam = generate_camera(model="Fisheye")
        rng = np.random.default_rng(42)
        poses = generate_poses(5, trajectory_type="arc")
        lm = generate_landmarks(50, rng=rng)
        obs = generate_observations(poses, lm, cam, noise_sigma=0.5, rng=rng)
        assert len(obs) > 0
        for ob in obs:
            assert 0 <= ob["u"] < cam["width"]
            assert 0 <= ob["v"] < cam["height"]


# ---------------------------------------------------------------------------
# Manual runner — used when pytest is not available
# ---------------------------------------------------------------------------

def _run_without_pytest():
    """Run all tests without pytest, using a simple pass/fail harness."""
    import sys
    import os

    passed = 0
    failed = 0
    errors = []

    def run_test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            print(f"  PASS: {name}")
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  FAIL: {name} -- {e}")

    # --- A1: Config ---
    print("=== A1: Config ===")
    tc = TestConfigValidation()
    run_test("defaults_valid", tc.test_defaults_valid)
    run_test("scene_type_valid", tc.test_scene_type_valid)

    def _scene_invalid():
        try:
            tc._make_config(scene_type="forest")
            validate_config(tc._make_config(scene_type="forest"))
            assert False, "Should have raised"
        except ValueError:
            pass
    run_test("scene_type_invalid", _scene_invalid)

    run_test("trajectory_type_valid", tc.test_trajectory_type_valid)

    def _traj_invalid():
        try:
            validate_config(tc._make_config(trajectory_type="zigzag"))
            assert False, "Should have raised"
        except ValueError:
            pass
    run_test("trajectory_type_invalid", _traj_invalid)

    run_test("camera_model_valid", tc.test_camera_model_valid)

    def _outlier_range():
        validate_config(tc._make_config(outlier_ratio=0.0))
        validate_config(tc._make_config(outlier_ratio=0.5))
        try:
            validate_config(tc._make_config(outlier_ratio=-0.1))
            assert False
        except ValueError:
            pass
    run_test("outlier_ratio_range", _outlier_range)

    run_test("config_yaml_roundtrip", tc.test_config_yaml_roundtrip)

    # --- A3: Scene ---
    print("\n=== A3: Scene ===")
    ts = TestSceneGeneration()
    run_test("random_landmarks_count", ts.test_random_landmarks_count)
    run_test("random_landmarks_range", ts.test_random_landmarks_range)
    run_test("corridor_landmarks_count", ts.test_corridor_landmarks_count)
    run_test("corridor_landmarks_structure", ts.test_corridor_landmarks_structure)
    run_test("corridor_deterministic", ts.test_corridor_deterministic)
    run_test("scene_dispatch", ts.test_scene_dispatch)
    run_test("multicam_landmarks", ts.test_multicam_landmarks)
    run_test("surround_landmarks_count", ts.test_surround_landmarks_count)
    run_test("surround_landmarks_deterministic", ts.test_surround_landmarks_deterministic)
    run_test("surround_landmarks_covers_all_directions", ts.test_surround_landmarks_covers_all_directions)
    run_test("surround_landmarks_has_ground", ts.test_surround_landmarks_has_ground)
    run_test("surround_dispatch", ts.test_surround_dispatch)
    run_test("surround_config_valid", ts.test_surround_config_valid)
    run_test("surround_multicam_observations", ts.test_surround_multicam_observations)

    # --- A5: Trajectory ---
    print("\n=== A5: Trajectory ===")
    tt = TestTrajectoryGeneration()
    run_test("straight", tt.test_straight)
    run_test("circular", tt.test_circular)
    run_test("slalom", tt.test_slalom)
    run_test("arc_default", tt.test_arc_default)
    run_test("deterministic", tt.test_deterministic)

    # --- A6: Projection ---
    print("\n=== A6: Projection ===")
    tp = TestProjection()
    run_test("radtan_distortion_applied", tp.test_radtan_distortion_applied)
    run_test("zero_distortion_matches_pinhole", tp.test_zero_distortion_matches_pinhole)
    run_test("behind_camera_radtan", tp.test_behind_camera_radtan)
    run_test("fisheye_projection", tp.test_fisheye_projection)
    run_test("fisheye_behind_camera", tp.test_fisheye_behind_camera)
    run_test("fisheye_compresses_vs_pinhole", tp.test_fisheye_compresses_vs_pinhole)
    run_test("project_point_dispatches_pinhole", tp.test_project_point_dispatches_pinhole)
    run_test("project_point_dispatches_fisheye", tp.test_project_point_dispatches_fisheye)

    # --- A9: Perturbation ---
    print("\n=== A9: Perturbation ===")
    tpe = TestPerturbation()
    run_test("gt_unchanged_after_pose_perturbation", tpe.test_gt_unchanged_after_pose_perturbation)
    run_test("first_pose_unchanged", tpe.test_first_pose_unchanged)
    run_test("gt_unchanged_after_landmark_perturbation", tpe.test_gt_unchanged_after_landmark_perturbation)
    run_test("reference_camera_unchanged", tpe.test_reference_camera_unchanged)
    run_test("intrinsic_perturbation", tpe.test_intrinsic_perturbation)
    run_test("distortion_perturbation", tpe.test_distortion_perturbation)

    # --- A11: Evaluation ---
    print("\n=== A11: Evaluation ===")
    te = TestEvaluation()
    tmp = Path(tempfile.mkdtemp())
    run_test("evaluation_returns_all_sections", lambda: te.test_evaluation_returns_all_sections(tmp))
    run_test("evaluation_pose_errors", lambda: te.test_evaluation_pose_errors(tmp))
    run_test("evaluation_landmark_errors", lambda: te.test_evaluation_landmark_errors(tmp))
    run_test("evaluation_intrinsic_errors", lambda: te.test_evaluation_intrinsic_errors(tmp))
    run_test("format_text", lambda: te.test_format_text(tmp))

    # --- Integration ---
    print("\n=== Integration ===")
    ti = TestObservationGeneration()
    run_test("observations_with_distortion", ti.test_observations_with_distortion)
    run_test("fisheye_observations", ti.test_fisheye_observations)

    # --- Summary ---
    print(f"\n{'='*40}")
    print(f"  PASSED: {passed}")
    print(f"  FAILED: {failed}")
    print(f"{'='*40}")
    if errors:
        print()
        for name, msg in errors:
            print(f"  FAIL: {name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    if pytest is not None:
        pytest.main([__file__, "-v"])
    else:
        _run_without_pytest()
