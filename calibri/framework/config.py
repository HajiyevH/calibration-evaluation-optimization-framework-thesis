"""YAML experiment config loading, validation, and serialization."""

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Union

import yaml


# Valid choices for enum-like config fields
VALID_SCENE_TYPES = ("random", "corridor", "surround")
VALID_TRAJECTORY_TYPES = ("arc", "straight", "circular", "slalom")
VALID_CAMERA_MODELS = ("Pinhole", "Fisheye")
VALID_ROBUST_TYPES = ("Huber", "Cauchy", "SoftL1", "None")


@dataclass
class PerturbationConfig:
    """Perturbation sigmas applied to initial guesses (GT remains untouched)."""
    pose_rot_sigma_deg: float = 1.0
    pose_trans_sigma_m: float = 0.02
    landmark_sigma_m: float = 0.05
    extrinsic_rot_sigma_deg: float = 2.0
    extrinsic_trans_sigma_m: float = 0.05
    intrinsic_sigma_px: float = 0.0
    distortion_sigma: float = 0.0


@dataclass
class DataConfig:
    # Scene
    scene_type: str = "random"
    num_landmarks: int = 500
    # Corridor scene geometry (used when scene_type == "corridor")
    corridor_length_m: float = 30.0
    corridor_width_m: float = 8.0
    corridor_wall_height_m: float = 4.0

    # Trajectory
    trajectory_type: str = "arc"
    num_poses: int = 15

    # Camera
    camera_model: str = "Pinhole"
    num_cameras: int = 1
    image_width: int = 1280
    image_height: int = 800

    # Observation noise and corruption
    noise_sigma: float = 0.5
    outlier_ratio: float = 0.0

    # Perturbation
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)

    # Reproducibility
    seed: int = 42


@dataclass
class SolverFlags:
    opt_poses: bool = True
    opt_landmarks: bool = True
    opt_intrinsics: bool = False
    opt_distortion: bool = False
    opt_extrinsics: bool = False


@dataclass
class RobustConfig:
    type: str = "Huber"
    scale: float = 1.0


@dataclass
class SolverConfig:
    flags: SolverFlags = field(default_factory=SolverFlags)
    robust: RobustConfig = field(default_factory=RobustConfig)
    timeout: int = 600


@dataclass
class AnalysisConfig:
    plot_errors: bool = True
    overlay_frames: List[Union[int, str]] = field(default_factory=lambda: [0, "mid", "last"])
    print_summary: bool = True
    evaluate_gt: bool = True


@dataclass
class ExperimentConfig:
    name: str = ""
    description: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


def _merge_dict(defaults: dict, overrides: dict) -> dict:
    """Recursively merge overrides into defaults."""
    result = dict(defaults)
    for k, v in overrides.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_dict(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    """Load an experiment config from a YAML file.

    Missing keys are filled with defaults. Unknown keys raise ValueError.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(raw).__name__}")

    top_level_known = {"name", "description", "data", "solver", "analysis"}
    top_unknown = set(raw) - top_level_known
    if top_unknown:
        raise ValueError(
            f"Unknown top-level key(s): {sorted(top_unknown)}. "
            f"Valid keys: {sorted(top_level_known)}"
        )

    solver_known = {"flags", "robust", "timeout"}
    solver_unknown = set(raw.get("solver", {})) - solver_known
    if solver_unknown:
        raise ValueError(
            f"Unknown key(s) in 'solver': {sorted(solver_unknown)}. "
            f"Valid keys: {sorted(solver_known)}"
        )

    data_raw = raw.get("data", {})
    flags_raw = raw.get("solver", {}).get("flags", {})
    robust_raw = raw.get("solver", {}).get("robust", {})
    analysis_raw = raw.get("analysis", {})

    def _check_unknown_keys(raw_dict: dict, dataclass_cls, section_name: str):
        known = set(dataclass_cls.__dataclass_fields__)
        unknown = set(raw_dict) - known
        if unknown:
            raise ValueError(
                f"Unknown key(s) in '{section_name}': {sorted(unknown)}. "
                f"Valid keys: {sorted(known)}"
            )

    # Extract nested perturbation before checking DataConfig keys
    perturbation_raw = data_raw.pop("perturbation", {})
    _check_unknown_keys(data_raw, DataConfig, "data")
    _check_unknown_keys(perturbation_raw, PerturbationConfig, "data.perturbation")
    _check_unknown_keys(flags_raw, SolverFlags, "solver.flags")
    _check_unknown_keys(robust_raw, RobustConfig, "solver.robust")
    _check_unknown_keys(analysis_raw, AnalysisConfig, "analysis")

    perturbation = PerturbationConfig(**perturbation_raw)
    data = DataConfig(**data_raw, perturbation=perturbation)
    flags = SolverFlags(**flags_raw)
    robust = RobustConfig(**robust_raw)
    analysis = AnalysisConfig(**analysis_raw)

    solver_raw = raw.get("solver", {})
    timeout = solver_raw.get("timeout", 600)

    config = ExperimentConfig(
        name=raw.get("name", ""),
        description=raw.get("description", ""),
        data=data,
        solver=SolverConfig(flags=flags, robust=robust, timeout=timeout),
        analysis=analysis,
    )

    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    """Validate an ExperimentConfig. Raises ValueError on problems."""
    errors = []

    # Name
    if not config.name:
        errors.append("'name' is required")
    elif not re.match(r"^[a-zA-Z0-9_-]+$", config.name):
        errors.append(f"'name' must be alphanumeric with _ or -, got '{config.name}'")

    # Data
    d = config.data
    if d.num_cameras not in (1, 4):
        errors.append(f"data.num_cameras must be 1 or 4, got {d.num_cameras}")
    if d.num_poses < 2:
        errors.append(f"data.num_poses must be >= 2, got {d.num_poses}")
    if d.num_landmarks < 10:
        errors.append(f"data.num_landmarks must be >= 10, got {d.num_landmarks}")
    if d.noise_sigma < 0:
        errors.append(f"data.noise_sigma must be >= 0, got {d.noise_sigma}")
    if d.seed < 0:
        errors.append(f"data.seed must be non-negative, got {d.seed}")
    if d.scene_type not in VALID_SCENE_TYPES:
        errors.append(
            f"data.scene_type must be one of {VALID_SCENE_TYPES}, got '{d.scene_type}'"
        )
    if d.trajectory_type not in VALID_TRAJECTORY_TYPES:
        errors.append(
            f"data.trajectory_type must be one of {VALID_TRAJECTORY_TYPES}, "
            f"got '{d.trajectory_type}'"
        )
    if d.camera_model not in VALID_CAMERA_MODELS:
        errors.append(
            f"data.camera_model must be one of {VALID_CAMERA_MODELS}, "
            f"got '{d.camera_model}'"
        )
    if not (0.0 <= d.outlier_ratio <= 1.0):
        errors.append(f"data.outlier_ratio must be between 0.0 and 1.0, got {d.outlier_ratio}")
    if d.image_width < 1 or d.image_height < 1:
        errors.append(f"data.image_width and image_height must be >= 1")

    # Perturbation sigmas must be non-negative
    p = d.perturbation
    for fname in ("pose_rot_sigma_deg", "pose_trans_sigma_m", "landmark_sigma_m",
                  "extrinsic_rot_sigma_deg", "extrinsic_trans_sigma_m",
                  "intrinsic_sigma_px", "distortion_sigma"):
        val = getattr(p, fname)
        if val < 0:
            errors.append(f"data.perturbation.{fname} must be >= 0, got {val}")

    # Solver
    r = config.solver.robust
    if r.type not in VALID_ROBUST_TYPES:
        errors.append(
            f"solver.robust.type must be one of {VALID_ROBUST_TYPES}, got '{r.type}'"
        )
    if r.scale <= 0:
        errors.append(f"solver.robust.scale must be > 0, got {r.scale}")

    # Solver timeout
    if config.solver.timeout < 1:
        errors.append(f"solver.timeout must be >= 1, got {config.solver.timeout}")

    # Analysis overlay_frames entries
    for entry in config.analysis.overlay_frames:
        if isinstance(entry, int):
            if entry < 0:
                errors.append(f"overlay_frames index must be >= 0, got {entry}")
        elif isinstance(entry, str):
            if entry not in ("mid", "last"):
                errors.append(f"overlay_frames string must be 'mid' or 'last', got '{entry}'")
        else:
            errors.append(f"overlay_frames entry must be int or 'mid'/'last', got {type(entry).__name__}")

    if errors:
        raise ValueError("Config validation failed:\n  " + "\n  ".join(errors))


def config_to_dict(config: ExperimentConfig) -> dict:
    """Convert an ExperimentConfig to a plain dict for YAML serialization."""
    return asdict(config)
