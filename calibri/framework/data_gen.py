"""Wraps synth_gen.py via Python import for synthetic data generation."""

import json
from pathlib import Path
from typing import Tuple

import numpy as np

from .config import DataConfig, SolverFlags, RobustConfig
from calibri.tools import synth_gen


def generate_synthetic_data(
    output_dir: Path,
    data_config: DataConfig,
    solver_flags: SolverFlags,
    robust_config: RobustConfig,
) -> Tuple[Path, Path]:
    """Generate problem.json and ground_truth.json in output_dir.

    Uses synth_gen functions directly, then patches the flags and robust
    fields to match the experiment config.

    All generation parameters are driven by DataConfig (scene_type,
    trajectory_type, camera_model, perturbation sigmas, etc.).

    Returns:
        (problem_path, ground_truth_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(data_config.seed)
    multicam = data_config.num_cameras > 1
    pert = data_config.perturbation

    if multicam:
        cameras = synth_gen.generate_rig(
            data_config.num_cameras,
            model=data_config.camera_model,
            width=data_config.image_width,
            height=data_config.image_height,
        )
        gt_poses = synth_gen.generate_poses(
            data_config.num_poses,
            trajectory_type=data_config.trajectory_type,
            multicam=True,
        )
        gt_landmarks = synth_gen.generate_scene_landmarks(
            scene_type=data_config.scene_type,
            num_landmarks=data_config.num_landmarks,
            num_cameras=data_config.num_cameras,
            rng=rng,
            corridor_length_m=data_config.corridor_length_m,
            corridor_width_m=data_config.corridor_width_m,
            corridor_wall_height_m=data_config.corridor_wall_height_m,
        )
        observations = synth_gen.generate_multicam_observations(
            gt_poses, gt_landmarks, cameras,
            noise_sigma=data_config.noise_sigma,
            outlier_ratio=data_config.outlier_ratio,
            rng=rng,
        )

        # Ground truth (unperturbed)
        gt_json = synth_gen.build_problem_json(
            cameras, gt_poses, gt_landmarks, observations,
            opt_extrinsics=True, reference_camera_id=0,
        )
        gt_json["seed"] = data_config.seed

        # Perturbed problem
        pert_poses = synth_gen.perturb_poses(
            gt_poses, rot_sigma_deg=pert.pose_rot_sigma_deg,
            trans_sigma_m=pert.pose_trans_sigma_m, rng=rng,
        )
        pert_landmarks = synth_gen.perturb_landmarks(
            gt_landmarks, sigma_m=pert.landmark_sigma_m, rng=rng,
        )
        pert_cameras = synth_gen.perturb_extrinsics(
            cameras, rot_sigma_deg=pert.extrinsic_rot_sigma_deg,
            trans_sigma_m=pert.extrinsic_trans_sigma_m, rng=rng,
        )
        if pert.intrinsic_sigma_px > 0:
            pert_cameras = synth_gen.perturb_intrinsics(
                pert_cameras, sigma_px=pert.intrinsic_sigma_px, rng=rng,
            )
        if pert.distortion_sigma > 0:
            pert_cameras = synth_gen.perturb_distortion(
                pert_cameras, sigma=pert.distortion_sigma, rng=rng,
            )
        problem = synth_gen.build_problem_json(
            pert_cameras, pert_poses, pert_landmarks, observations,
            opt_extrinsics=True, reference_camera_id=0,
        )
    else:
        camera = synth_gen.generate_camera(
            model=data_config.camera_model,
            width=data_config.image_width,
            height=data_config.image_height,
        )
        cameras = [camera]
        gt_poses = synth_gen.generate_poses(
            data_config.num_poses,
            trajectory_type=data_config.trajectory_type,
        )
        gt_landmarks = synth_gen.generate_scene_landmarks(
            scene_type=data_config.scene_type,
            num_landmarks=data_config.num_landmarks,
            num_cameras=1,
            rng=rng,
            corridor_length_m=data_config.corridor_length_m,
            corridor_width_m=data_config.corridor_width_m,
            corridor_wall_height_m=data_config.corridor_wall_height_m,
        )
        observations = synth_gen.generate_observations(
            gt_poses, gt_landmarks, camera,
            noise_sigma=data_config.noise_sigma,
            outlier_ratio=data_config.outlier_ratio,
            rng=rng,
        )

        gt_json = synth_gen.build_problem_json(cameras, gt_poses, gt_landmarks, observations)
        gt_json["seed"] = data_config.seed

        pert_poses = synth_gen.perturb_poses(
            gt_poses, rot_sigma_deg=pert.pose_rot_sigma_deg,
            trans_sigma_m=pert.pose_trans_sigma_m, rng=rng,
        )
        pert_landmarks = synth_gen.perturb_landmarks(
            gt_landmarks, sigma_m=pert.landmark_sigma_m, rng=rng,
        )
        pert_cameras = cameras  # single camera: no extrinsic perturbation
        if pert.intrinsic_sigma_px > 0:
            pert_cameras = synth_gen.perturb_intrinsics(
                pert_cameras, sigma_px=pert.intrinsic_sigma_px, rng=rng,
            )
        if pert.distortion_sigma > 0:
            pert_cameras = synth_gen.perturb_distortion(
                pert_cameras, sigma=pert.distortion_sigma, rng=rng,
            )
        problem = synth_gen.build_problem_json(
            pert_cameras, pert_poses, pert_landmarks, observations,
        )

    # Patch flags to match experiment config
    problem["flags"] = {
        "opt_poses": solver_flags.opt_poses,
        "opt_landmarks": solver_flags.opt_landmarks,
        "opt_intrinsics": solver_flags.opt_intrinsics,
        "opt_distortion": solver_flags.opt_distortion,
        "opt_extrinsics": solver_flags.opt_extrinsics,
    }

    # Patch robust config
    problem["robust"] = {
        "type": robust_config.type,
        "scale": robust_config.scale,
    }

    # Write files
    problem_path = output_dir / "problem.json"
    gt_path = output_dir / "ground_truth.json"

    with open(problem_path, "w") as f:
        json.dump(problem, f, indent=2)

    with open(gt_path, "w") as f:
        json.dump(gt_json, f, indent=2)

    return problem_path, gt_path
