from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


BASE_SEED = 3594963098
SEEDS = list(range(7))
EVAL_EPISODES = int(os.getenv("PAPER71_EVAL_EPISODES", "12"))
ABLATION_EPISODES = int(os.getenv("PAPER71_ABLATION_EPISODES", "10"))
STRESS_EPISODES = int(os.getenv("PAPER71_STRESS_EPISODES", "8"))
TRAINING_EXAMPLES = int(os.getenv("PAPER71_TRAINING_EXAMPLES", "2200"))
STRESS_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]
STEPS = 90
SUCCESS_RADIUS = 0.080
WRONG_CONTACT_RADIUS = 0.105
BASE_POS = np.array([0.0, -0.58], dtype=float)
CAMERA_POS = np.array([0.0, -1.35], dtype=float)
WORKSPACE_LOW = np.array([-0.72, -0.48], dtype=float)
WORKSPACE_HIGH = np.array([0.72, 0.62], dtype=float)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


MODEL_XML = """
<mujoco model="object_permanence_self_occlusion">
  <compiler angle="radian"/>
  <option timestep="0.025" integrator="implicitfast" gravity="0 0 0"/>
  <default>
    <joint damping="0.35"/>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <geom name="table" type="plane" pos="0 0 -0.01" size="1.2 1.0 0.02"
          rgba="0.90 0.90 0.86 1"/>
    <geom name="base_marker" type="sphere" pos="0 -0.58 0.03" size="0.025"
          rgba="0.15 0.15 0.15 1"/>
    <body name="tool" pos="0 -0.58 0.05">
      <joint name="tool_x" type="slide" axis="1 0 0" range="-0.78 0.78" damping="0.40"/>
      <joint name="tool_y" type="slide" axis="0 1 0" range="-0.60 0.66" damping="0.40"/>
      <geom name="tool_tip" type="sphere" size="0.040" mass="0.15" rgba="0.10 0.10 0.12 1"/>
    </body>
    <body name="target" pos="0 0 0.035">
      <joint name="target_x" type="slide" axis="1 0 0" range="-0.74 0.74" damping="0.70"/>
      <joint name="target_y" type="slide" axis="0 1 0" range="-0.50 0.64" damping="0.70"/>
      <geom name="target_obj" type="sphere" size="0.045" mass="0.08" rgba="0.2 0.62 0.25 1"/>
    </body>
    <body name="distractor" pos="0 0 0.035">
      <joint name="distractor_x" type="slide" axis="1 0 0" range="-0.74 0.74" damping="0.70"/>
      <joint name="distractor_y" type="slide" axis="0 1 0" range="-0.50 0.64" damping="0.70"/>
      <geom name="distractor_obj" type="sphere" size="0.045" mass="0.08" rgba="0.76 0.25 0.20 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="tool_x_motor" joint="tool_x" gear="1" ctrllimited="true" ctrlrange="-4 4"/>
    <motor name="tool_y_motor" joint="tool_y" gear="1" ctrllimited="true" ctrlrange="-4 4"/>
  </actuator>
</mujoco>
"""


METHODS = [
    "last_seen_memory",
    "visibility_gated_kalman",
    "particle_belief_tracker",
    "learned_occlusion_regressor",
    "random_forest_occlusion_regressor",
    "hist_gradient_occlusion_regressor",
    "identity_consistency_tracker",
    "ensemble_uncertainty_planner",
    "risk_averse_particle_planner",
    "pomdp_style_belief_planner",
    "occlusion_aware_permanence_v4",
    "occlusion_aware_permanence_v5",
    "no_self_mask_ablation",
    "oracle_state",
]

ABLATION_METHODS = [
    "occlusion_full_v5",
    "ablate_no_self_mask",
    "ablate_no_branch_belief",
    "ablate_no_contact_update",
    "ablate_no_uncertainty_inflation",
    "ablate_no_identity_filter",
    "ablate_no_reacquisition_guard",
    "ablate_no_tail_risk_objective",
    "ablate_no_false_detection_rejection",
    "ablate_no_distractor_filter",
    "occlusion_aware_permanence_v4",
    "learned_only_branch",
]

STRESS_METHODS = [
    "visibility_gated_kalman",
    "particle_belief_tracker",
    "learned_occlusion_regressor",
    "random_forest_occlusion_regressor",
    "hist_gradient_occlusion_regressor",
    "identity_consistency_tracker",
    "ensemble_uncertainty_planner",
    "risk_averse_particle_planner",
    "pomdp_style_belief_planner",
    "occlusion_aware_permanence_v5",
    "no_self_mask_ablation",
    "oracle_state",
]


@dataclass(frozen=True)
class SplitSpec:
    name: str
    occlusion_width: float
    camera_noise: float
    dropout: float
    distractor_close_prob: float
    false_detection_prob: float
    object_displacement: float
    actuator_limit: float
    hidden_drift: float
    occlusion_bias: float


@dataclass(frozen=True)
class EpisodeConfig:
    split: SplitSpec
    seed: int
    episode: int
    tool_start: np.ndarray
    target_start: np.ndarray
    distractor_start: np.ndarray
    drift_vector: np.ndarray
    displacement_window: Tuple[int, int]
    camera_noise: float
    occlusion_width: float
    dropout: float
    false_detection_prob: float
    actuator_limit: float
    hidden_drift: float
    stress_level: float | None = None


@dataclass
class MethodState:
    method: str
    belief: np.ndarray
    velocity: np.ndarray
    covariance: float
    last_visible: np.ndarray
    last_tool: np.ndarray
    particles: np.ndarray
    weights: np.ndarray
    occlusion_age: int
    false_disappearance_steps: int
    wrong_contacts: int
    diagnostic_steps: int
    target_contacts: int
    belief_history: List[float]


@dataclass
class LearnedPack:
    ridge_model: object
    rf_model: object
    hgb_model: object
    training_rows: List[Dict[str, str]]
    ridge_train_error: float
    rf_train_error: float
    hgb_train_error: float


SPLITS = [
    SplitSpec("nominal", 0.028, 0.012, 0.00, 0.10, 0.02, 0.010, 3.25, 0.006, 0.55),
    SplitSpec("short_self_occlusion", 0.072, 0.020, 0.02, 0.16, 0.06, 0.016, 3.05, 0.012, 1.00),
    SplitSpec("long_self_occlusion", 0.118, 0.026, 0.05, 0.22, 0.10, 0.024, 2.88, 0.020, 1.35),
    SplitSpec("end_effector_occlusion", 0.098, 0.024, 0.04, 0.20, 0.12, 0.020, 2.95, 0.018, 1.25),
    SplitSpec("distractor_swap", 0.082, 0.026, 0.04, 0.82, 0.28, 0.020, 2.90, 0.016, 1.05),
    SplitSpec("near_identical_distractors", 0.088, 0.030, 0.06, 0.88, 0.32, 0.022, 2.82, 0.020, 1.10),
    SplitSpec("object_displacement", 0.078, 0.028, 0.05, 0.30, 0.14, 0.048, 2.75, 0.032, 1.00),
    SplitSpec("hidden_contact_drift", 0.092, 0.030, 0.07, 0.38, 0.20, 0.060, 2.68, 0.048, 1.12),
    SplitSpec("false_reappearance", 0.105, 0.034, 0.08, 0.78, 0.40, 0.038, 2.62, 0.034, 1.18),
    SplitSpec("camera_dropout_burst", 0.096, 0.038, 0.16, 0.48, 0.24, 0.034, 2.66, 0.036, 1.10),
    SplitSpec("stale_memory_trap", 0.122, 0.032, 0.10, 0.60, 0.26, 0.066, 2.54, 0.052, 1.30),
    SplitSpec("embodiment_control_shift", 0.094, 0.030, 0.08, 0.52, 0.24, 0.044, 2.20, 0.038, 1.08),
    SplitSpec("high_symmetry_layout", 0.100, 0.032, 0.07, 0.92, 0.32, 0.040, 2.58, 0.036, 1.20),
    SplitSpec("combined_stress", 0.124, 0.042, 0.12, 0.82, 0.38, 0.064, 2.36, 0.050, 1.36),
    SplitSpec("combined_extreme_stress", 0.140, 0.052, 0.18, 0.92, 0.46, 0.078, 2.04, 0.064, 1.52),
]


def unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return np.zeros_like(v, dtype=float)
    return v / norm


def ci95(values: Sequence[float]) -> float:
    vals = np.array(values, dtype=float)
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * np.std(vals, ddof=1) / math.sqrt(len(vals)))


def clamp_pos(pos: np.ndarray) -> np.ndarray:
    return np.clip(pos, WORKSPACE_LOW, WORKSPACE_HIGH)


def make_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(MODEL_XML)


def config_rng(seed: int, episode: int, split_name: str) -> np.random.Generator:
    offset = sum((i + 1) * ord(c) for i, c in enumerate(split_name))
    return np.random.default_rng(BASE_SEED + 6151 * seed + 131 * episode + offset)


def make_config(split: SplitSpec, seed: int, episode: int, stress_level: float | None = None) -> EpisodeConfig:
    rng = config_rng(seed, episode, split.name if stress_level is None else f"{split.name}_{stress_level:.2f}")
    x = rng.uniform(-0.42, 0.42)
    y = rng.uniform(0.10, 0.52)
    target = np.array([x, y], dtype=float)
    if rng.random() < split.distractor_close_prob:
        offset = rng.normal(0.0, 0.075, size=2)
        offset += 0.09 * unit(np.array([rng.choice([-1.0, 1.0]), rng.uniform(-0.2, 0.6)]))
    else:
        offset = rng.normal(0.0, 0.22, size=2)
    distractor = clamp_pos(target + offset)
    if float(np.linalg.norm(distractor - target)) < 0.075:
        distractor = clamp_pos(target + np.array([0.16, -0.08]))
    tool_start = np.array([rng.uniform(-0.12, 0.12), -0.56], dtype=float)
    drift_direction = unit(rng.normal(0.0, 1.0, size=2))
    start = int(rng.integers(24, 40))
    window = (start, min(STEPS - 18, start + int(rng.integers(14, 25))))

    if stress_level is None:
        camera_noise = split.camera_noise
        occlusion_width = split.occlusion_width
        dropout = split.dropout
        false_detection = split.false_detection_prob
        actuator_limit = split.actuator_limit
        hidden_drift = split.hidden_drift
        displacement = split.object_displacement
    else:
        camera_noise = 0.012 + 0.045 * stress_level
        occlusion_width = 0.030 + 0.105 * stress_level
        dropout = 0.02 + 0.16 * stress_level
        false_detection = 0.04 + 0.34 * stress_level
        actuator_limit = 3.25 - 0.92 * stress_level
        hidden_drift = 0.006 + 0.052 * stress_level
        displacement = 0.010 + 0.062 * stress_level

    return EpisodeConfig(
        split=split,
        seed=seed,
        episode=episode,
        tool_start=tool_start,
        target_start=target,
        distractor_start=distractor,
        drift_vector=drift_direction * displacement,
        displacement_window=window,
        camera_noise=camera_noise,
        occlusion_width=occlusion_width,
        dropout=dropout,
        false_detection_prob=false_detection,
        actuator_limit=actuator_limit,
        hidden_drift=hidden_drift,
        stress_level=stress_level,
    )


def point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        return float(np.linalg.norm(point - a)), 0.0
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(point - closest)), t


def self_occluded(tool: np.ndarray, obj: np.ndarray, width: float) -> bool:
    dist, t = point_segment_distance(obj, BASE_POS, tool)
    camera_line = unit(obj - CAMERA_POS)
    arm_line = unit(tool - BASE_POS)
    image_overlap = abs(float(np.cross(np.append(camera_line, 0.0), np.append(arm_line, 0.0))[2]))
    return bool(dist < width and 0.18 < t < 1.02 and obj[1] > -0.28 and image_overlap < 0.55)


def measurement(
    cfg: EpisodeConfig,
    rng: np.random.Generator,
    tool: np.ndarray,
    target: np.ndarray,
    distractor: np.ndarray,
) -> Tuple[np.ndarray | None, bool, bool, bool]:
    hidden = self_occluded(tool, target, cfg.occlusion_width)
    visible = (not hidden) and (rng.random() > cfg.dropout)
    if visible:
        return target + rng.normal(0.0, cfg.camera_noise, size=2), True, hidden, False
    false_detection = bool(rng.random() < cfg.false_detection_prob)
    if false_detection:
        return distractor + rng.normal(0.0, cfg.camera_noise * 1.2, size=2), False, hidden, True
    return None, False, hidden, False


def train_feature(
    last_visible: np.ndarray,
    tool: np.ndarray,
    tool_vel: np.ndarray,
    observed: np.ndarray | None,
    occluded: bool,
    false_detection: bool,
    age: int,
    camera_noise: float,
    hidden_drift: float,
) -> np.ndarray:
    obs = observed if observed is not None else np.array([0.0, 0.0], dtype=float)
    obs_flag = 0.0 if observed is None else 1.0
    return np.array(
        [
            last_visible[0],
            last_visible[1],
            tool[0],
            tool[1],
            tool_vel[0],
            tool_vel[1],
            obs[0],
            obs[1],
            obs_flag,
            float(occluded),
            float(false_detection),
            age / STEPS,
            camera_noise,
            hidden_drift,
            float(np.linalg.norm(tool - last_visible)),
        ],
        dtype=float,
    )


def generate_training_pack() -> LearnedPack:
    rng = np.random.default_rng(BASE_SEED + 707)
    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    csv_rows: List[Dict[str, str]] = []
    for idx in range(TRAINING_EXAMPLES):
        split = SPLITS[int(rng.integers(0, len(SPLITS)))]
        target = np.array([rng.uniform(-0.46, 0.46), rng.uniform(0.08, 0.54)], dtype=float)
        last_visible = target + rng.normal(0.0, split.camera_noise, size=2)
        tool = np.array([rng.uniform(-0.52, 0.52), rng.uniform(-0.50, 0.56)], dtype=float)
        tool_vel = rng.normal(0.0, 0.20, size=2)
        age = int(rng.integers(0, 42))
        occluded = bool(rng.random() < 0.62)
        false_detection = bool(occluded and rng.random() < split.false_detection_prob)
        drift = unit(rng.normal(0.0, 1.0, size=2)) * split.object_displacement * min(1.0, age / 18)
        true = clamp_pos(target + drift + rng.normal(0.0, split.hidden_drift * age / STEPS, size=2))
        observed: np.ndarray | None
        if not occluded or rng.random() < 0.35:
            observed = true + rng.normal(0.0, split.camera_noise, size=2)
        elif false_detection:
            observed = clamp_pos(true + rng.normal(0.12, 0.08, size=2))
        else:
            observed = None
        feat = train_feature(last_visible, tool, tool_vel, observed, occluded, false_detection, age, split.camera_noise, split.hidden_drift)
        x_rows.append(feat)
        y_rows.append(true)
        csv_rows.append(
            {
                "example": str(idx),
                "split": split.name,
                "occluded": str(int(occluded)),
                "false_detection": str(int(false_detection)),
                "age": str(age),
                "target_x": f"{true[0]:.5f}",
                "target_y": f"{true[1]:.5f}",
            }
        )
    x = np.vstack(x_rows)
    y = np.vstack(y_rows)
    ridge_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), Ridge(alpha=0.9))
    rf_model = RandomForestRegressor(
        n_estimators=72,
        max_depth=9,
        min_samples_leaf=3,
        random_state=BASE_SEED % 2**31,
        n_jobs=1,
    )
    hgb_model = MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=80, learning_rate=0.055, max_leaf_nodes=24, l2_regularization=0.02, random_state=17)
    )
    ridge_model.fit(x, y)
    rf_model.fit(x, y)
    hgb_model.fit(x, y)
    ridge_pred = ridge_model.predict(x)
    rf_pred = rf_model.predict(x)
    hgb_pred = hgb_model.predict(x)
    ridge_train_error = float(np.mean(np.linalg.norm(ridge_pred - y, axis=1)))
    rf_train_error = float(np.mean(np.linalg.norm(rf_pred - y, axis=1)))
    hgb_train_error = float(np.mean(np.linalg.norm(hgb_pred - y, axis=1)))
    return LearnedPack(
        ridge_model=ridge_model,
        rf_model=rf_model,
        hgb_model=hgb_model,
        training_rows=csv_rows,
        ridge_train_error=ridge_train_error,
        rf_train_error=rf_train_error,
        hgb_train_error=hgb_train_error,
    )


def init_state(method: str, start_belief: np.ndarray, rng: np.random.Generator) -> MethodState:
    particle_count = 96
    particles = start_belief + rng.normal(0.0, 0.035, size=(particle_count, 2))
    weights = np.full(particle_count, 1.0 / particle_count, dtype=float)
    return MethodState(
        method=method,
        belief=start_belief.copy(),
        velocity=np.zeros(2, dtype=float),
        covariance=0.025,
        last_visible=start_belief.copy(),
        last_tool=BASE_POS.copy(),
        particles=particles,
        weights=weights,
        occlusion_age=0,
        false_disappearance_steps=0,
        wrong_contacts=0,
        diagnostic_steps=0,
        target_contacts=0,
        belief_history=[],
    )


def update_particles(state: MethodState, observed: np.ndarray | None, visible_target: bool, cfg: EpisodeConfig, rng: np.random.Generator) -> None:
    state.particles += rng.normal(0.0, cfg.hidden_drift + 0.003, size=state.particles.shape)
    if observed is not None:
        sigma = cfg.camera_noise * (1.0 if visible_target else 2.8) + 0.010
        dist = np.linalg.norm(state.particles - observed[None, :], axis=1)
        likelihood = np.exp(-0.5 * (dist / sigma) ** 2) + 1e-8
        state.weights *= likelihood
        total = float(np.sum(state.weights))
        state.weights = state.weights / total if total > 0 else np.full_like(state.weights, 1.0 / len(state.weights))
        ess = 1.0 / float(np.sum(state.weights * state.weights))
        if ess < len(state.weights) * 0.45:
            idx = rng.choice(len(state.weights), size=len(state.weights), p=state.weights)
            state.particles = state.particles[idx] + rng.normal(0.0, 0.010, size=state.particles.shape)
            state.weights = np.full_like(state.weights, 1.0 / len(state.weights))
    state.belief = clamp_pos(np.average(state.particles, axis=0, weights=state.weights))
    state.covariance = float(np.average(np.linalg.norm(state.particles - state.belief[None, :], axis=1), weights=state.weights))


def update_method(
    state: MethodState,
    method: str,
    cfg: EpisodeConfig,
    pack: LearnedPack,
    tool: np.ndarray,
    tool_vel: np.ndarray,
    target: np.ndarray,
    distractor: np.ndarray,
    observed: np.ndarray | None,
    visible_target: bool,
    occluded: bool,
    false_detection: bool,
    rng: np.random.Generator,
) -> None:
    previous = state.belief.copy()
    if visible_target and observed is not None:
        state.last_visible = observed.copy()
        state.occlusion_age = 0
    else:
        state.occlusion_age += 1

    if method == "oracle_state":
        state.belief = target.copy()
        state.covariance = 0.0
    elif method == "last_seen_memory":
        if observed is not None:
            state.belief = observed.copy()
        state.covariance = min(0.35, state.covariance + (0.004 if occluded else 0.001))
    elif method == "visibility_gated_kalman":
        state.belief = state.belief + state.velocity
        state.covariance = min(0.45, state.covariance + (0.010 if occluded else 0.004))
        if observed is not None:
            gain = state.covariance / (state.covariance + cfg.camera_noise + (0.080 if not visible_target else 0.015))
            innovation = observed - state.belief
            state.belief += gain * innovation
            state.velocity = 0.55 * state.velocity + 0.45 * innovation
            state.covariance *= 1.0 - 0.55 * gain
    elif method == "particle_belief_tracker":
        update_particles(state, observed, visible_target, cfg, rng)
    elif method in {"learned_occlusion_regressor", "random_forest_occlusion_regressor", "hist_gradient_occlusion_regressor"}:
        feat = train_feature(state.last_visible, tool, tool_vel, observed, occluded, false_detection, state.occlusion_age, cfg.camera_noise, cfg.hidden_drift)
        if method == "random_forest_occlusion_regressor":
            pred = np.array(pack.rf_model.predict(feat.reshape(1, -1))[0], dtype=float)
            visible_weight, false_weight = 0.88, 0.30
            base_cov = 0.016
        elif method == "hist_gradient_occlusion_regressor":
            pred = np.array(pack.hgb_model.predict(feat.reshape(1, -1))[0], dtype=float)
            visible_weight, false_weight = 0.86, 0.32
            base_cov = 0.017
        else:
            pred = np.array(pack.ridge_model.predict(feat.reshape(1, -1))[0], dtype=float)
            visible_weight, false_weight = 0.85, 0.45
            base_cov = 0.018
        if visible_target and observed is not None:
            state.belief = visible_weight * observed + (1.0 - visible_weight) * pred
        elif observed is not None:
            state.belief = false_weight * observed + (1.0 - false_weight) * pred
        else:
            state.belief = pred
        state.covariance = min(0.32, base_cov + 0.003 * state.occlusion_age)
    elif method == "identity_consistency_tracker":
        predicted = state.belief + state.velocity
        if visible_target and observed is not None:
            state.belief = 0.90 * observed + 0.10 * predicted
            state.covariance = 0.014
        elif observed is not None and float(np.linalg.norm(observed - state.last_visible)) < 0.22 + 0.004 * state.occlusion_age:
            state.belief = 0.45 * observed + 0.55 * predicted
            state.covariance = min(0.26, state.covariance + 0.005)
        else:
            state.belief = predicted + 0.25 * cfg.drift_vector * int(occluded)
            state.covariance = min(0.34, state.covariance + 0.008 + 0.002 * int(false_detection))
    elif method == "risk_averse_particle_planner":
        update_particles(state, observed if not false_detection else None, visible_target, cfg, rng)
        state.covariance = min(0.42, state.covariance + 0.006 * int(occluded))
    elif method == "pomdp_style_belief_planner":
        predicted = state.belief + state.velocity + 0.25 * cfg.drift_vector * int(occluded)
        self_mask_prob = 0.80 if occluded else 0.12
        false_obs_prob = 0.75 if false_detection else 0.10
        if visible_target and observed is not None:
            state.belief = 0.88 * observed + 0.12 * predicted
            state.covariance = 0.014
        elif observed is not None and false_obs_prob < 0.50:
            state.belief = 0.40 * observed + 0.60 * predicted
            state.covariance = min(0.25, state.covariance + 0.006)
        else:
            state.belief = (0.65 + 0.20 * self_mask_prob) * predicted + (0.35 - 0.20 * self_mask_prob) * state.last_visible
            state.covariance = min(0.36, state.covariance + 0.008 + 0.003 * int(false_detection))
    elif method == "ensemble_uncertainty_planner":
        candidates = []
        for k in range(7):
            bias = np.array([math.sin(k + 0.7), math.cos(1.3 * k)], dtype=float) * cfg.hidden_drift * state.occlusion_age
            candidates.append(state.last_visible + 0.35 * state.velocity * state.occlusion_age + bias)
        if observed is not None:
            candidates.append(observed)
        stack = np.vstack(candidates)
        state.belief = clamp_pos(np.mean(stack, axis=0))
        state.covariance = float(np.mean(np.linalg.norm(stack - state.belief[None, :], axis=1))) + (0.04 if occluded else 0.01)
    elif method in {
        "occlusion_aware_permanence",
        "occlusion_aware_permanence_v4",
        "occlusion_aware_permanence_v5",
        "occlusion_full_v5",
        "no_self_mask_ablation",
        "ablate_no_self_mask",
        "ablate_no_branch_belief",
        "ablate_no_contact_update",
        "ablate_no_uncertainty_inflation",
        "ablate_no_identity_filter",
        "ablate_no_reacquisition_guard",
        "ablate_no_tail_risk_objective",
        "ablate_no_false_detection_rejection",
        "ablate_no_distractor_filter",
        "learned_only_branch",
    }:
        is_v4 = method in {"occlusion_aware_permanence", "occlusion_aware_permanence_v4"}
        uses_self_mask = method not in {"no_self_mask_ablation", "ablate_no_self_mask", "learned_only_branch"}
        uses_branch = method != "ablate_no_branch_belief"
        uses_contact = method != "ablate_no_contact_update"
        inflates_uncertainty = method != "ablate_no_uncertainty_inflation"
        filters_distractor = method not in {"ablate_no_distractor_filter", "ablate_no_false_detection_rejection"}
        uses_identity = method != "ablate_no_identity_filter"
        uses_reacquisition_guard = method != "ablate_no_reacquisition_guard"
        uses_tail_risk = method != "ablate_no_tail_risk_objective"

        predicted = state.belief + state.velocity
        feat = train_feature(state.last_visible, tool, tool_vel, observed, occluded, false_detection, state.occlusion_age, cfg.camera_noise, cfg.hidden_drift)
        learned_hidden = clamp_pos(np.array(pack.rf_model.predict(feat.reshape(1, -1))[0], dtype=float))
        if method == "learned_only_branch":
            state.belief = learned_hidden if not (visible_target and observed is not None) else 0.86 * observed + 0.14 * learned_hidden
            state.covariance = min(0.30, 0.017 + 0.003 * state.occlusion_age)
            state.belief = clamp_pos(state.belief)
            state.velocity = np.clip(state.belief - previous, -0.08, 0.08)
            if occluded and state.covariance > 0.18:
                state.false_disappearance_steps += 1
            state.belief_history.append(float(np.linalg.norm(state.belief - target)))
            return

        if is_v4:
            uses_identity = False
            uses_reacquisition_guard = False
            uses_tail_risk = False
            filters_distractor = method != "ablate_no_distractor_filter"

        if uses_self_mask and occluded and filters_distractor and false_detection:
            measurement_weight = 0.0
        elif uses_identity and observed is not None and (not visible_target) and float(np.linalg.norm(observed - state.last_visible)) > 0.20 + 0.003 * state.occlusion_age:
            measurement_weight = 0.0
        elif observed is not None:
            measurement_weight = 0.88 if visible_target else (0.22 if uses_self_mask else 0.38)
        else:
            measurement_weight = 0.0
        if observed is not None and measurement_weight > 0:
            new_belief = (1.0 - measurement_weight) * predicted + measurement_weight * observed
        else:
            drift_guess = cfg.drift_vector * (0.40 if uses_contact else 0.12) if occluded else 0.0
            learned_weight = 0.55 if (uses_self_mask and occluded and not is_v4) else 0.48
            new_belief = (1.0 - learned_weight) * (predicted + drift_guess) + learned_weight * learned_hidden if uses_self_mask and occluded else predicted + drift_guess
        if uses_branch and occluded:
            branch_a = new_belief
            branch_b = learned_hidden if uses_self_mask else state.last_visible + cfg.drift_vector
            self_mask_score = 0.75 if (uses_self_mask and not is_v4) else (0.70 if uses_self_mask else 0.35)
            if uses_tail_risk and state.covariance > 0.16:
                self_mask_score -= 0.08
            new_belief = self_mask_score * branch_a + (1.0 - self_mask_score) * branch_b
        if uses_reacquisition_guard and visible_target and observed is not None and state.occlusion_age <= 2:
            new_belief = 0.92 * observed + 0.08 * new_belief
        state.velocity = 0.55 * state.velocity + 0.45 * (new_belief - state.belief)
        state.belief = clamp_pos(new_belief)
        if visible_target and observed is not None:
            state.covariance = 0.012
        elif inflates_uncertainty:
            state.covariance = min(0.30, state.covariance + 0.005 + 0.002 * int(not uses_self_mask) + 0.001 * int(not uses_identity))
        else:
            state.covariance = min(0.14, state.covariance + 0.002)
    else:
        raise ValueError(f"unknown method {method}")

    state.belief = clamp_pos(state.belief)
    state.velocity = np.clip(state.belief - previous, -0.08, 0.08)
    if occluded and state.covariance > 0.18:
        state.false_disappearance_steps += 1
    state.belief_history.append(float(np.linalg.norm(state.belief - target)))


def plan_target(method: str, state: MethodState, tool: np.ndarray, occluded: bool) -> np.ndarray:
    diagnostic_methods = {
        "ensemble_uncertainty_planner",
        "risk_averse_particle_planner",
        "pomdp_style_belief_planner",
        "occlusion_aware_permanence_v5",
        "occlusion_full_v5",
    }
    if method in diagnostic_methods and occluded and state.covariance > 0.10 and state.occlusion_age < 18:
        state.diagnostic_steps += 1
        side = -1.0 if state.belief[0] > 0 else 1.0
        retreat = -0.18 if method in {"occlusion_aware_permanence_v5", "occlusion_full_v5"} else -0.14
        return clamp_pos(state.belief + np.array([0.20 * side, retreat]))
    if method in {"ablate_no_uncertainty_inflation", "no_self_mask_ablation", "ablate_no_self_mask"} and occluded and state.covariance > 0.18:
        return clamp_pos(state.belief)
    if method == "ablate_no_tail_risk_objective" and occluded:
        return clamp_pos(state.belief + 0.05 * unit(state.belief - tool))
    return state.belief


def simulate_episode(model: mujoco.MjModel, method: str, cfg: EpisodeConfig, pack: LearnedPack) -> Dict[str, str]:
    method_offset = sum((i + 1) * ord(c) for i, c in enumerate(method))
    rng = np.random.default_rng(BASE_SEED + 10007 * cfg.seed + 293 * cfg.episode + method_offset)
    data = mujoco.MjData(model)
    data.qpos[:2] = cfg.tool_start
    data.qpos[2:4] = cfg.target_start
    data.qpos[4:6] = cfg.distractor_start
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    initial_observed = cfg.target_start + rng.normal(0.0, cfg.camera_noise, size=2)
    state = init_state(method, initial_observed, rng)
    state.last_tool = cfg.tool_start.copy()
    samples: List[str] = []
    occluded_steps = 0
    visible_steps = 0
    wrong_contact_steps = 0
    target_contact_steps = 0
    false_detection_steps = 0
    identity_switch_steps = 0
    localization_during_occ: List[float] = []
    localization_after_occ: List[float] = []
    calibration_errors: List[float] = []
    reacquisition_latencies: List[int] = []
    waiting_for_reacquisition = False
    current_reacquisition_latency = 0
    path = 0.0
    success_step: int | None = None

    for step in range(STEPS):
        tool = np.array(data.qpos[:2], dtype=float)
        target = np.array(data.qpos[2:4], dtype=float)
        distractor = np.array(data.qpos[4:6], dtype=float)
        tool_vel = np.array(data.qvel[:2], dtype=float)
        observed, visible_target, occluded, false_detection = measurement(cfg, rng, tool, target, distractor)
        if occluded:
            occluded_steps += 1
            waiting_for_reacquisition = True
            current_reacquisition_latency = 0
        elif waiting_for_reacquisition and visible_target:
            reacquisition_latencies.append(current_reacquisition_latency)
            waiting_for_reacquisition = False
        elif waiting_for_reacquisition:
            current_reacquisition_latency += 1
        if visible_target:
            visible_steps += 1
        if false_detection:
            false_detection_steps += 1

        update_method(state, method, cfg, pack, tool, tool_vel, target, distractor, observed, visible_target, occluded, false_detection, rng)
        if false_detection and float(np.linalg.norm(state.belief - distractor)) + 0.025 < float(np.linalg.norm(state.belief - target)):
            identity_switch_steps += 1
        desired = plan_target(method, state, tool, occluded)
        data.qfrc_applied[:] = 0.0
        if occluded and cfg.displacement_window[0] <= step <= cfg.displacement_window[1]:
            data.qfrc_applied[2:4] += 0.58 * cfg.drift_vector + rng.normal(0.0, cfg.hidden_drift, size=2)
        if float(np.linalg.norm(tool - target)) < SUCCESS_RADIUS:
            target_contact_steps += 1
            state.target_contacts += 1
            if success_step is None:
                success_step = step
        if float(np.linalg.norm(tool - distractor)) < WRONG_CONTACT_RADIUS:
            wrong_contact_steps += 1
            state.wrong_contacts += 1

        ctrl = 8.2 * (desired - tool) - 1.15 * tool_vel
        data.ctrl[:2] = np.clip(ctrl, -cfg.actuator_limit, cfg.actuator_limit)
        prev_tool = tool.copy()
        mujoco.mj_step(model, data)
        data.qpos[:2] = clamp_pos(np.array(data.qpos[:2], dtype=float))
        data.qpos[2:4] = clamp_pos(np.array(data.qpos[2:4], dtype=float))
        data.qpos[4:6] = clamp_pos(np.array(data.qpos[4:6], dtype=float))
        data.qvel[:] = np.clip(data.qvel[:], -1.5, 1.5)
        mujoco.mj_forward(model, data)
        path += float(np.linalg.norm(np.array(data.qpos[:2], dtype=float) - prev_tool))

        err = float(np.linalg.norm(state.belief - np.array(data.qpos[2:4], dtype=float)))
        if occluded:
            localization_during_occ.append(err)
        elif occluded_steps > 0:
            localization_after_occ.append(err)
        calibration_errors.append(abs(state.covariance - err))
        if step % 12 == 0 or step == STEPS - 1:
            samples.append(
                f"{step}:{data.qpos[0]:.3f}:{data.qpos[1]:.3f}:b{state.belief[0]:.3f}:{state.belief[1]:.3f}:occ{int(occluded)}"
            )

    final_tool = np.array(data.qpos[:2], dtype=float)
    final_target = np.array(data.qpos[2:4], dtype=float)
    final_distractor = np.array(data.qpos[4:6], dtype=float)
    final_error = float(np.linalg.norm(state.belief - final_target))
    final_reach_dist = float(np.linalg.norm(final_tool - final_target))
    final_wrong_dist = float(np.linalg.norm(final_tool - final_distractor))
    success = int(final_reach_dist < SUCCESS_RADIUS or (success_step is not None and success_step >= STEPS - 22))
    wrong_object_contact = int(wrong_contact_steps > 1 and target_contact_steps == 0)
    false_disappearance_rate = state.false_disappearance_steps / max(1, occluded_steps)
    during_error = float(np.mean(localization_during_occ)) if localization_during_occ else final_error
    after_error = float(np.mean(localization_after_occ)) if localization_after_occ else final_error
    reacquisition_latency = float(np.mean(reacquisition_latencies)) if reacquisition_latencies else float(STEPS)
    calibration_error = float(np.mean(calibration_errors)) if calibration_errors else final_error
    return {
        "method": method,
        "split": cfg.split.name,
        "seed": str(cfg.seed),
        "episode": str(cfg.episode),
        "stress_level": "" if cfg.stress_level is None else f"{cfg.stress_level:.2f}",
        "success": str(success),
        "success_step": str(success_step if success_step is not None else -1),
        "final_reach_dist": f"{final_reach_dist:.5f}",
        "final_belief_error": f"{final_error:.5f}",
        "occluded_steps": str(occluded_steps),
        "visible_steps": str(visible_steps),
        "false_detection_steps": str(false_detection_steps),
        "mean_error_during_occlusion": f"{during_error:.5f}",
        "mean_error_after_occlusion": f"{after_error:.5f}",
        "false_disappearance_rate": f"{false_disappearance_rate:.5f}",
        "identity_switch_rate": f"{identity_switch_steps / max(1, false_detection_steps):.5f}",
        "wrong_object_contact": str(wrong_object_contact),
        "wrong_contact_rate": f"{wrong_contact_steps / STEPS:.5f}",
        "target_contact_rate": f"{target_contact_steps / STEPS:.5f}",
        "reacquisition_latency": f"{reacquisition_latency:.5f}",
        "diagnostic_steps": str(state.diagnostic_steps),
        "path_length": f"{path:.5f}",
        "calibration_error": f"{calibration_error:.5f}",
        "final_covariance": f"{state.covariance:.5f}",
        "final_wrong_dist": f"{final_wrong_dist:.5f}",
        "trajectory_samples": ";".join(samples),
    }


def group_rows(rows: Iterable[Dict[str, str]], fields: Sequence[str]) -> Dict[Tuple[str, ...], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in fields), []).append(row)
    return grouped


def mean_metric(rows: Sequence[Dict[str, str]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def build_seed_metrics(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for (method, split, seed), group in sorted(group_rows(rows, ["method", "split", "seed"]).items()):
        out.append(
            {
                "method": method,
                "split": split,
                "seed": seed,
                "episodes": str(len(group)),
                "success_rate": f"{mean_metric(group, 'success'):.5f}",
                "mean_final_belief_error": f"{mean_metric(group, 'final_belief_error'):.5f}",
                "mean_error_during_occlusion": f"{mean_metric(group, 'mean_error_during_occlusion'):.5f}",
                "mean_error_after_occlusion": f"{mean_metric(group, 'mean_error_after_occlusion'):.5f}",
                "false_disappearance_rate": f"{mean_metric(group, 'false_disappearance_rate'):.5f}",
                "identity_switch_rate": f"{mean_metric(group, 'identity_switch_rate'):.5f}",
                "wrong_object_contact_rate": f"{mean_metric(group, 'wrong_object_contact'):.5f}",
                "wrong_contact_rate": f"{mean_metric(group, 'wrong_contact_rate'):.5f}",
                "target_contact_rate": f"{mean_metric(group, 'target_contact_rate'):.5f}",
                "mean_reacquisition_latency": f"{mean_metric(group, 'reacquisition_latency'):.5f}",
                "mean_diagnostic_steps": f"{mean_metric(group, 'diagnostic_steps'):.5f}",
                "mean_path_length": f"{mean_metric(group, 'path_length'):.5f}",
                "mean_calibration_error": f"{mean_metric(group, 'calibration_error'):.5f}",
                "mean_occluded_steps": f"{mean_metric(group, 'occluded_steps'):.5f}",
            }
        )
    return out


def build_summary(seed_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    metrics = [
        "success_rate",
        "mean_final_belief_error",
        "mean_error_during_occlusion",
        "mean_error_after_occlusion",
        "false_disappearance_rate",
        "identity_switch_rate",
        "wrong_object_contact_rate",
        "wrong_contact_rate",
        "target_contact_rate",
        "mean_reacquisition_latency",
        "mean_diagnostic_steps",
        "mean_path_length",
        "mean_calibration_error",
        "mean_occluded_steps",
    ]
    rows: List[Dict[str, str]] = []
    for (method, split), group in sorted(group_rows(seed_rows, ["method", "split"]).items()):
        item: Dict[str, str] = {"method": method, "split": split, "seeds": str(len(group)), "episodes_per_seed": group[0]["episodes"]}
        for metric in metrics:
            vals = [float(row[metric]) for row in group]
            item[f"mean_{metric}"] = f"{float(np.mean(vals)):.5f}"
            item[f"ci95_{metric}"] = f"{ci95(vals):.5f}"
        rows.append(item)
    return rows


def build_pairwise(seed_rows: List[Dict[str, str]], reference: str = "occlusion_aware_permanence_v5") -> List[Dict[str, str]]:
    by_key = {(row["method"], row["split"], row["seed"]): row for row in seed_rows}
    rows: List[Dict[str, str]] = []
    methods = sorted({row["method"] for row in seed_rows if row["method"] != reference})
    splits = sorted({row["split"] for row in seed_rows})
    for split in splits:
        for method in methods:
            success_diffs = []
            err_diffs = []
            false_diffs = []
            for seed in [str(s) for s in SEEDS]:
                ref = by_key.get((reference, split, seed))
                other = by_key.get((method, split, seed))
                if ref is None or other is None:
                    continue
                success_diffs.append(float(ref["success_rate"]) - float(other["success_rate"]))
                err_diffs.append(float(other["mean_error_during_occlusion"]) - float(ref["mean_error_during_occlusion"]))
                false_diffs.append(float(other["false_disappearance_rate"]) - float(ref["false_disappearance_rate"]))
            if success_diffs:
                rows.append(
                    {
                        "split": split,
                        "reference": reference,
                        "comparison": method,
                        "paired_success_diff": f"{float(np.mean(success_diffs)):.5f}",
                        "ci95_success_diff": f"{ci95(success_diffs):.5f}",
                        "paired_occlusion_error_reduction": f"{float(np.mean(err_diffs)):.5f}",
                        "paired_false_disappearance_reduction": f"{float(np.mean(false_diffs)):.5f}",
                        "reference_better_seeds": str(sum(1 for d in success_diffs if d > 0)),
                        "seeds": str(len(success_diffs)),
                    }
                )
    return rows


def build_aggregate_metrics(seed_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    group_defs = {
        "all_splits": sorted({row["split"] for row in seed_rows}),
        "hard_splits": [
            "long_self_occlusion",
            "end_effector_occlusion",
            "near_identical_distractors",
            "object_displacement",
            "hidden_contact_drift",
            "false_reappearance",
            "camera_dropout_burst",
            "stale_memory_trap",
            "embodiment_control_shift",
            "high_symmetry_layout",
            "combined_stress",
            "combined_extreme_stress",
        ],
        "combined_and_extreme": ["combined_stress", "combined_extreme_stress"],
        "identity_hard": ["distractor_swap", "near_identical_distractors", "false_reappearance", "high_symmetry_layout"],
    }
    rows: List[Dict[str, str]] = []
    methods = sorted({row["method"] for row in seed_rows})
    for group_name, splits in group_defs.items():
        for method in methods:
            group = [row for row in seed_rows if row["method"] == method and row["split"] in splits]
            if not group:
                continue
            rows.append(
                {
                    "group": group_name,
                    "method": method,
                    "success": f"{float(np.mean([float(row['success_rate']) for row in group])):.5f}",
                    "false_disappearance": f"{float(np.mean([float(row['false_disappearance_rate']) for row in group])):.5f}",
                    "wrong_object_contact": f"{float(np.mean([float(row['wrong_object_contact_rate']) for row in group])):.5f}",
                    "identity_switch": f"{float(np.mean([float(row['identity_switch_rate']) for row in group])):.5f}",
                    "occlusion_error": f"{float(np.mean([float(row['mean_error_during_occlusion']) for row in group])):.5f}",
                    "reacquisition_latency": f"{float(np.mean([float(row['mean_reacquisition_latency']) for row in group])):.5f}",
                    "diagnostic_steps": f"{float(np.mean([float(row['mean_diagnostic_steps']) for row in group])):.5f}",
                    "calibration_error": f"{float(np.mean([float(row['mean_calibration_error']) for row in group])):.5f}",
                }
            )
    return rows


def build_ablation_aggregate(summary_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for method, group in sorted(group_rows(summary_rows, ["method"]).items()):
        rows.append(
            {
                "method": method[0],
                "success": f"{float(np.mean([float(row['mean_success_rate']) for row in group])):.5f}",
                "false_disappearance": f"{float(np.mean([float(row['mean_false_disappearance_rate']) for row in group])):.5f}",
                "wrong_object_contact": f"{float(np.mean([float(row['mean_wrong_object_contact_rate']) for row in group])):.5f}",
                "identity_switch": f"{float(np.mean([float(row['mean_identity_switch_rate']) for row in group])):.5f}",
                "occlusion_error": f"{float(np.mean([float(row['mean_mean_error_during_occlusion']) for row in group])):.5f}",
                "diagnostic_steps": f"{float(np.mean([float(row['mean_mean_diagnostic_steps']) for row in group])):.5f}",
            }
        )
    return rows


def build_fixed_risk_metrics(raw_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    hard_splits = {
        "long_self_occlusion",
        "end_effector_occlusion",
        "near_identical_distractors",
        "object_displacement",
        "hidden_contact_drift",
        "false_reappearance",
        "camera_dropout_burst",
        "stale_memory_trap",
        "embodiment_control_shift",
        "high_symmetry_layout",
        "combined_stress",
        "combined_extreme_stress",
    }
    budgets = [0.10, 0.20]
    rows: List[Dict[str, str]] = []
    for method, group in sorted(group_rows([row for row in raw_rows if row["split"] in hard_splits], ["method"]).items()):
        method_rows = group
        thresholds = sorted({float(row["final_covariance"]) for row in method_rows}) + [float("inf")]
        for budget in budgets:
            best: Dict[str, float] | None = None
            for threshold in thresholds:
                selected = [row for row in method_rows if float(row["final_covariance"]) <= threshold]
                if not selected:
                    continue
                false_rate = float(np.mean([float(row["false_disappearance_rate"]) for row in selected]))
                wrong_rate = float(np.mean([float(row["wrong_object_contact"]) for row in selected]))
                identity_rate = float(np.mean([float(row["identity_switch_rate"]) for row in selected]))
                risk = max(false_rate, wrong_rate, identity_rate)
                success = float(np.mean([float(row["success"]) for row in selected]))
                coverage = len(selected) / len(method_rows)
                if risk <= budget and (best is None or success > best["success"]):
                    best = {
                        "success": success,
                        "coverage": coverage,
                        "false_disappearance": false_rate,
                        "wrong_object_contact": wrong_rate,
                        "identity_switch": identity_rate,
                        "threshold": threshold,
                    }
            if best is None:
                best = {
                    "success": 0.0,
                    "coverage": 0.0,
                    "false_disappearance": 1.0,
                    "wrong_object_contact": 1.0,
                    "identity_switch": 1.0,
                    "threshold": float("inf"),
                }
            rows.append(
                {
                    "method": method[0],
                    "budget": f"{budget:.2f}",
                    "success_at_budget": f"{best['success']:.5f}",
                    "coverage": f"{best['coverage']:.5f}",
                    "false_disappearance": f"{best['false_disappearance']:.5f}",
                    "wrong_object_contact": f"{best['wrong_object_contact']:.5f}",
                    "identity_switch": f"{best['identity_switch']:.5f}",
                    "threshold": "inf" if math.isinf(best["threshold"]) else f"{best['threshold']:.5f}",
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(summary: List[Dict[str, str]], split_order: Sequence[str], methods: Sequence[str], metric: str, title: str, path: Path, ylim: Tuple[float, float] | None = None) -> None:
    width = min(0.095, 0.78 / max(1, len(methods)))
    x = np.arange(len(split_order))
    plt.figure(figsize=(15, 5.5))
    for idx, method in enumerate(methods):
        vals = []
        errs = []
        for split in split_order:
            row = [r for r in summary if r["method"] == method and r["split"] == split][0]
            vals.append(float(row[f"mean_{metric}"]))
            errs.append(float(row[f"ci95_{metric}"]))
        plt.bar(x + (idx - len(methods) / 2) * width, vals, width, yerr=errs, label=method)
    plt.xticks(x, split_order, rotation=20, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_ablation(ablation_summary: List[Dict[str, str]], path: Path) -> None:
    rows = [row for row in ablation_summary if row["split"] == "combined_stress"]
    plt.figure(figsize=(10, 4.8))
    plt.bar([row["method"] for row in rows], [float(row["mean_success_rate"]) for row in rows], yerr=[float(row["ci95_success_rate"]) for row in rows], color="#576f72")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("success rate")
    plt.title("Paper 71 occlusion-permanence ablations")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_stress(stress_summary: List[Dict[str, str]], path: Path) -> None:
    plt.figure(figsize=(9, 5))
    for method in sorted({row["method"] for row in stress_summary}):
        rows = sorted([row for row in stress_summary if row["method"] == method], key=lambda r: float(r["stress_level"]))
        x = [float(row["stress_level"]) for row in rows]
        y = [float(row["mean_success_rate"]) for row in rows]
        e = [float(row["ci95_success_rate"]) for row in rows]
        plt.errorbar(x, y, yerr=e, marker="o", label=method)
    plt.xlabel("stress level")
    plt.ylabel("success rate")
    plt.title("Paper 71 self-occlusion stress sweep")
    plt.ylim(0, 1.0)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def decide(
    aggregate: List[Dict[str, str]],
    fixed_risk: List[Dict[str, str]],
    pairwise: List[Dict[str, str]],
    ablation_aggregate: List[Dict[str, str]],
    stress_summary: List[Dict[str, str]],
) -> Tuple[str, str]:
    proposed = "occlusion_aware_permanence_v5"

    def agg(group: str, method: str) -> Dict[str, str]:
        return [row for row in aggregate if row["group"] == group and row["method"] == method][0]

    hard_v5 = agg("hard_splits", proposed)
    hard_best = max(
        [row for row in aggregate if row["group"] == "hard_splits" and row["method"] not in {proposed, "oracle_state"}],
        key=lambda row: float(row["success"]),
    )
    combined_v5 = agg("combined_and_extreme", proposed)
    combined_best = max(
        [row for row in aggregate if row["group"] == "combined_and_extreme" and row["method"] not in {proposed, "oracle_state"}],
        key=lambda row: float(row["success"]),
    )
    pair_rows = [row for row in pairwise if row["comparison"] == combined_best["method"] and row["split"] == "combined_extreme_stress"]
    if not pair_rows:
        pair_rows = [row for row in pairwise if row["comparison"] == combined_best["method"]]
    paired = float(pair_rows[0]["paired_success_diff"]) if pair_rows else 0.0
    paired_ci = float(pair_rows[0]["ci95_success_diff"]) if pair_rows else 1.0

    fixed_v5 = [row for row in fixed_risk if row["method"] == proposed and row["budget"] == "0.10"][0]
    fixed_best = max(
        [row for row in fixed_risk if row["budget"] == "0.10" and row["method"] not in {proposed, "oracle_state"}],
        key=lambda row: float(row["success_at_budget"]),
    )

    max_stress_rows = [
        row
        for row in stress_summary
        if row.get("split") == "combined_extreme_stress" and row["stress_level"] == "1.00" and row["method"] != "oracle_state"
    ]
    max_v5 = [row for row in max_stress_rows if row["method"] == proposed][0]
    max_best = max(max_stress_rows, key=lambda row: float(row["mean_success_rate"]))

    full_ablation = [row for row in ablation_aggregate if row["method"] == "occlusion_full_v5"][0]
    ablation_failures = [
        row["method"]
        for row in ablation_aggregate
        if row["method"] != "occlusion_full_v5"
        and float(row["success"]) >= float(full_ablation["success"]) - 0.020
        and float(row["false_disappearance"]) <= float(full_ablation["false_disappearance"]) + 0.020
    ]

    failures: List[str] = []
    if float(hard_v5["success"]) < float(hard_best["success"]) + 0.030:
        failures.append(
            f"v5 does not beat strongest hard-regime baseline {hard_best['method']} by 0.030 "
            f"(v5={float(hard_v5['success']):.3f}, best={float(hard_best['success']):.3f})"
        )
    if paired - paired_ci <= 0.0:
        failures.append(f"paired lower bound against {combined_best['method']} is not positive ({paired:.3f}+/-{paired_ci:.3f})")
    if float(combined_v5["success"]) < float(combined_best["success"]) + 0.030:
        failures.append(
            f"v5 does not beat strongest combined/extreme baseline {combined_best['method']} by 0.030 "
            f"(v5={float(combined_v5['success']):.3f}, best={float(combined_best['success']):.3f})"
        )
    if float(hard_v5["false_disappearance"]) > float(hard_best["false_disappearance"]) + 0.020:
        failures.append(f"false-disappearance safety gate fails versus {hard_best['method']}")
    if float(hard_v5["wrong_object_contact"]) > float(hard_best["wrong_object_contact"]) + 0.020:
        failures.append(f"wrong-object-contact safety gate fails versus {hard_best['method']}")
    if float(fixed_v5["success_at_budget"]) < float(fixed_best["success_at_budget"]) - 1e-9:
        failures.append(
            f"fixed-risk gate fails at budget 0.10 (v5={float(fixed_v5['success_at_budget']):.3f}, "
            f"best={fixed_best['method']} {float(fixed_best['success_at_budget']):.3f})"
        )
    if float(max_v5["mean_success_rate"]) < float(max_best["mean_success_rate"]) - 0.030:
        failures.append(
            f"maximum-stress gate fails to {max_best['method']} "
            f"(v5={float(max_v5['mean_success_rate']):.3f}, best={float(max_best['mean_success_rate']):.3f})"
        )
    if ablation_failures:
        failures.append("ablation gate fails because " + ", ".join(ablation_failures) + " matches or beats full v5")

    if failures:
        return "KILL_ARCHIVE", "; ".join(failures)
    return (
        "STRONG_REVISE",
        "v5 clears frozen simulated gates but remains below ICLR-main readiness without real-robot or public-benchmark validation",
    )


def negative_cases(raw_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    candidates = [
        r
        for r in raw_rows
        if r["method"] == "occlusion_aware_permanence_v5"
        and r["split"] in {"combined_stress", "combined_extreme_stress", "object_displacement", "false_reappearance", "stale_memory_trap"}
    ]
    worst = sorted(
        candidates,
        key=lambda r: (
            int(r["success"]),
            -float(r["mean_error_during_occlusion"]),
            -float(r["identity_switch_rate"]),
            -int(float(r["wrong_object_contact"])),
        ),
    )[:12]
    rows: List[Dict[str, str]] = []
    for i, row in enumerate(worst):
        lesson = "hidden displacement exceeded the branch belief update"
        if float(row["false_detection_steps"]) > 0:
            lesson = "distractor detections during self-occlusion competed with target permanence"
        if int(row["wrong_object_contact"]):
            lesson = "belief error caused wrong-object contact before recovery"
        rows.append(
            {
                "case": str(i),
                "split": row["split"],
                "seed": row["seed"],
                "episode": row["episode"],
                "success": row["success"],
                "wrong_object_contact": row["wrong_object_contact"],
                "mean_error_during_occlusion": row["mean_error_during_occlusion"],
                "false_detection_steps": row["false_detection_steps"],
                "identity_switch_rate": row["identity_switch_rate"],
                "lesson": lesson,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper 71 expanded MuJoCo self-occlusion benchmark")
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--ablation-episodes", type=int, default=ABLATION_EPISODES)
    parser.add_argument("--stress-episodes", type=int, default=STRESS_EPISODES)
    parser.add_argument("--train-scenes", type=int, default=TRAINING_EXAMPLES)
    parser.add_argument("--splits", nargs="*", default=[split.name for split in SPLITS])
    parser.add_argument(
        "--ablation-splits",
        nargs="*",
        default=["combined_stress", "combined_extreme_stress", "false_reappearance", "stale_memory_trap"],
    )
    parser.add_argument(
        "--stress-splits",
        nargs="*",
        default=["combined_stress", "combined_extreme_stress", "false_reappearance"],
    )
    parser.add_argument("--stress-levels", nargs="*", type=float, default=STRESS_LEVELS)
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--figures-dir", default=str(FIGURES))
    parser.add_argument("--workers", type=int, default=1, help="Accepted for protocol symmetry; Paper 71 runs serially to stay RAM-light.")
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> Tuple[List[SplitSpec], List[SplitSpec], List[SplitSpec]]:
    global SEEDS, EVAL_EPISODES, ABLATION_EPISODES, STRESS_EPISODES, TRAINING_EXAMPLES, RESULTS, FIGURES, STRESS_LEVELS
    SEEDS = list(range(args.seeds))
    EVAL_EPISODES = args.episodes
    ABLATION_EPISODES = args.ablation_episodes
    STRESS_EPISODES = args.stress_episodes
    TRAINING_EXAMPLES = args.train_scenes
    RESULTS = Path(args.results_dir)
    FIGURES = Path(args.figures_dir)
    STRESS_LEVELS = args.stress_levels
    split_by_name = {split.name: split for split in SPLITS}
    active_splits = [split_by_name[name] for name in args.splits if name in split_by_name]
    ablation_splits = [split_by_name[name] for name in args.ablation_splits if name in split_by_name]
    stress_splits = [split_by_name[name] for name in args.stress_splits if name in split_by_name]
    if not active_splits:
        raise ValueError("no active splits selected")
    if not ablation_splits:
        raise ValueError("no ablation splits selected")
    if not stress_splits:
        raise ValueError("no stress splits selected")
    return active_splits, ablation_splits, stress_splits


def main() -> None:
    args = parse_args()
    active_splits, ablation_splits, stress_splits = configure_from_args(args)
    start_time = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    pack = generate_training_pack()
    write_csv(RESULTS / "training_occlusion_examples.csv", pack.training_rows)
    write_csv(
        RESULTS / "training_summary.csv",
        [
            {
                "training_examples": str(TRAINING_EXAMPLES),
                "ridge_train_error": f"{pack.ridge_train_error:.5f}",
                "random_forest_train_error": f"{pack.rf_train_error:.5f}",
                "hist_gradient_train_error": f"{pack.hgb_train_error:.5f}",
                "feature_dim": "15",
            }
        ],
    )

    model = make_model()
    raw_rows: List[Dict[str, str]] = []
    for split in active_splits:
        for seed in SEEDS:
            for episode in range(EVAL_EPISODES):
                cfg = make_config(split, seed, episode)
                for method in METHODS:
                    raw_rows.append(simulate_episode(model, method, cfg, pack))
    write_csv(RESULTS / "self_occlusion_raw.csv", raw_rows)
    write_csv(RESULTS / "self_occlusion_rollouts.csv", raw_rows)
    seed_rows = build_seed_metrics(raw_rows)
    summary = build_summary(seed_rows)
    pairwise = build_pairwise(seed_rows)
    aggregate = build_aggregate_metrics(seed_rows)
    fixed_risk = build_fixed_risk_metrics(raw_rows)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "metrics.csv", summary)
    write_csv(RESULTS / "self_occlusion_metrics.csv", summary)
    write_csv(RESULTS / "pairwise_stats.csv", pairwise)
    write_csv(RESULTS / "self_occlusion_pairwise.csv", pairwise)
    write_csv(RESULTS / "aggregate_metrics.csv", aggregate)
    write_csv(RESULTS / "fixed_risk_metrics.csv", fixed_risk)

    ablation_raw: List[Dict[str, str]] = []
    for split in ablation_splits:
        for seed in SEEDS:
            for episode in range(ABLATION_EPISODES):
                cfg = make_config(split, seed, 1000 + episode)
                for method in ABLATION_METHODS:
                    row = simulate_episode(model, method, cfg, pack)
                    row["method"] = method
                    ablation_raw.append(row)
    write_csv(RESULTS / "self_occlusion_ablation_raw.csv", ablation_raw)
    ablation_summary = build_summary(build_seed_metrics(ablation_raw))
    ablation_aggregate = build_ablation_aggregate(ablation_summary)
    write_csv(RESULTS / "ablation_metrics.csv", ablation_summary)
    write_csv(RESULTS / "self_occlusion_ablation.csv", ablation_summary)
    write_csv(RESULTS / "ablation_aggregate_metrics.csv", ablation_aggregate)

    stress_raw: List[Dict[str, str]] = []
    for split in stress_splits:
        for stress_level in STRESS_LEVELS:
            for seed in SEEDS:
                for episode in range(STRESS_EPISODES):
                    cfg = make_config(split, seed, 2000 + episode, stress_level=float(stress_level))
                    for method in STRESS_METHODS:
                        row = simulate_episode(model, method, cfg, pack)
                        row["stress_level"] = f"{stress_level:.2f}"
                        stress_raw.append(row)
    write_csv(RESULTS / "stress_sweep_raw.csv", stress_raw)
    stress_summary: List[Dict[str, str]] = []
    for (method, split_name, stress_level), group in sorted(group_rows(stress_raw, ["method", "split", "stress_level"]).items()):
        seed_vals = []
        for seed in [str(s) for s in SEEDS]:
            rows = [r for r in group if r["seed"] == seed]
            if rows:
                seed_vals.append(float(np.mean([float(r["success"]) for r in rows])))
        stress_summary.append(
            {
                "method": method,
                "split": split_name,
                "stress_level": stress_level,
                "seeds": str(len(seed_vals)),
                "episodes_per_seed": str(STRESS_EPISODES),
                "mean_success_rate": f"{float(np.mean(seed_vals)):.5f}",
                "ci95_success_rate": f"{ci95(seed_vals):.5f}",
                "mean_false_disappearance_rate": f"{mean_metric(group, 'false_disappearance_rate'):.5f}",
                "mean_wrong_object_contact_rate": f"{mean_metric(group, 'wrong_object_contact'):.5f}",
                "mean_identity_switch_rate": f"{mean_metric(group, 'identity_switch_rate'):.5f}",
                "mean_mean_diagnostic_steps": f"{mean_metric(group, 'diagnostic_steps'):.5f}",
            }
        )
    write_csv(RESULTS / "stress_sweep.csv", stress_summary)
    write_csv(FIGURES / "stress_curve_data.csv", stress_summary)
    write_csv(RESULTS / "negative_cases.csv", negative_cases(raw_rows))

    split_order = [s.name for s in active_splits]
    plot_metric(summary, split_order, METHODS, "success_rate", "Paper 71 MuJoCo object permanence success", FIGURES / "self_occlusion_success_by_split.png", (0, 1.0))
    plot_metric(summary, split_order, METHODS, "mean_error_during_occlusion", "Paper 71 localization error during self-occlusion", FIGURES / "self_occlusion_error_by_split.png")
    plot_metric(summary, split_order, METHODS, "false_disappearance_rate", "Paper 71 false disappearance under self-occlusion", FIGURES / "self_occlusion_false_disappearance.png", (0, 1.0))
    plot_ablation(ablation_summary, FIGURES / "self_occlusion_ablation_success.png")
    plot_stress(stress_summary, FIGURES / "self_occlusion_stress_sweep.png")

    decision, reason = decide(aggregate, fixed_risk, pairwise, ablation_aggregate, stress_summary)
    elapsed = time.time() - start_time
    combined_rows = [r for r in summary if r["split"] == "combined_stress"]
    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Paper 71 object_permanence_under_self_occlusion expanded v5 MuJoCo rebuild\n")
        f.write(f"Terminal decision: {decision}\n")
        f.write(f"Terminal reason: {reason}\n")
        f.write(f"Main eval rows: {len(raw_rows)}\n")
        f.write(f"Ablation rows: {len(ablation_raw)}\n")
        f.write(f"Stress rows: {len(stress_raw)}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write(f"Eval episodes per seed/split: {EVAL_EPISODES}\n")
        f.write(f"Active splits: {[s.name for s in active_splits]}\n")
        f.write(f"Ablation splits: {[s.name for s in ablation_splits]}\n")
        f.write(f"Stress splits: {[s.name for s in stress_splits]}\n")
        f.write(f"Stress levels: {STRESS_LEVELS}\n")
        f.write(f"Runtime seconds: {elapsed:.2f}\n\n")
        f.write("Combined-stress summary:\n")
        for row in sorted(combined_rows, key=lambda r: -float(r["mean_success_rate"])):
            f.write(
                f"{row['method']} success={row['mean_success_rate']} ci95={row['ci95_success_rate']} "
                f"occ_error={row['mean_mean_error_during_occlusion']} false_disappearance={row['mean_false_disappearance_rate']}\n"
            )

    print(f"wrote Paper 71 expanded MuJoCo self-occlusion evidence to {RESULTS}")
    print(f"terminal decision: {decision}")
    print(reason)


if __name__ == "__main__":
    main()
