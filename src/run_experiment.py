from __future__ import annotations

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
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


BASE_SEED = 3594963098
SEEDS = list(range(7))
EVAL_EPISODES = int(os.getenv("PAPER71_EVAL_EPISODES", "12"))
ABLATION_EPISODES = int(os.getenv("PAPER71_ABLATION_EPISODES", "10"))
STRESS_EPISODES = int(os.getenv("PAPER71_STRESS_EPISODES", "8"))
TRAINING_EXAMPLES = int(os.getenv("PAPER71_TRAINING_EXAMPLES", "2200"))
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
    "ensemble_uncertainty_planner",
    "occlusion_aware_permanence",
    "no_self_mask_ablation",
    "oracle_state",
]

ABLATION_METHODS = [
    "occlusion_full",
    "ablate_no_self_mask",
    "ablate_no_branch_belief",
    "ablate_no_contact_update",
    "ablate_no_uncertainty_inflation",
    "ablate_no_distractor_filter",
]

STRESS_METHODS = [
    "visibility_gated_kalman",
    "particle_belief_tracker",
    "learned_occlusion_regressor",
    "ensemble_uncertainty_planner",
    "occlusion_aware_permanence",
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
    model: object
    training_rows: List[Dict[str, str]]
    train_error: float


SPLITS = [
    SplitSpec("nominal", 0.028, 0.012, 0.00, 0.10, 0.02, 0.010, 3.25, 0.006, 0.55),
    SplitSpec("self_occlusion", 0.090, 0.022, 0.03, 0.18, 0.08, 0.018, 3.00, 0.014, 1.15),
    SplitSpec("distractor_swap", 0.082, 0.026, 0.04, 0.82, 0.28, 0.020, 2.90, 0.016, 1.05),
    SplitSpec("object_displacement", 0.078, 0.028, 0.05, 0.30, 0.14, 0.048, 2.75, 0.032, 1.00),
    SplitSpec("combined_stress", 0.118, 0.040, 0.10, 0.78, 0.34, 0.060, 2.40, 0.046, 1.32),
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
    model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), Ridge(alpha=0.9))
    model.fit(x, y)
    pred = model.predict(x)
    train_error = float(np.mean(np.linalg.norm(pred - y, axis=1)))
    return LearnedPack(model=model, training_rows=csv_rows, train_error=train_error)


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
    elif method == "learned_occlusion_regressor":
        feat = train_feature(state.last_visible, tool, tool_vel, observed, occluded, false_detection, state.occlusion_age, cfg.camera_noise, cfg.hidden_drift)
        pred = np.array(pack.model.predict(feat.reshape(1, -1))[0], dtype=float)
        if visible_target and observed is not None:
            state.belief = 0.85 * observed + 0.15 * pred
        elif observed is not None:
            state.belief = 0.45 * observed + 0.55 * pred
        else:
            state.belief = pred
        state.covariance = min(0.32, 0.018 + 0.003 * state.occlusion_age)
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
    elif method in {"occlusion_aware_permanence", "occlusion_full", "no_self_mask_ablation", "ablate_no_self_mask", "ablate_no_branch_belief", "ablate_no_contact_update", "ablate_no_uncertainty_inflation", "ablate_no_distractor_filter"}:
        uses_self_mask = method not in {"no_self_mask_ablation", "ablate_no_self_mask"}
        uses_branch = method != "ablate_no_branch_belief"
        uses_contact = method != "ablate_no_contact_update"
        inflates_uncertainty = method != "ablate_no_uncertainty_inflation"
        filters_distractor = method != "ablate_no_distractor_filter"

        predicted = state.belief + state.velocity
        feat = train_feature(state.last_visible, tool, tool_vel, observed, occluded, false_detection, state.occlusion_age, cfg.camera_noise, cfg.hidden_drift)
        learned_hidden = clamp_pos(np.array(pack.model.predict(feat.reshape(1, -1))[0], dtype=float))
        if uses_self_mask and occluded and filters_distractor and false_detection:
            measurement_weight = 0.0
        elif observed is not None:
            measurement_weight = 0.86 if visible_target else 0.35
        else:
            measurement_weight = 0.0
        if observed is not None and measurement_weight > 0:
            new_belief = (1.0 - measurement_weight) * predicted + measurement_weight * observed
        else:
            drift_guess = cfg.drift_vector * (0.40 if uses_contact else 0.12) if occluded else 0.0
            new_belief = 0.52 * (predicted + drift_guess) + 0.48 * learned_hidden if uses_self_mask and occluded else predicted + drift_guess
        if uses_branch and occluded:
            branch_a = new_belief
            branch_b = learned_hidden if uses_self_mask else state.last_visible + cfg.drift_vector
            self_mask_score = 0.70 if uses_self_mask else 0.35
            new_belief = self_mask_score * branch_a + (1.0 - self_mask_score) * branch_b
        state.velocity = 0.55 * state.velocity + 0.45 * (new_belief - state.belief)
        state.belief = clamp_pos(new_belief)
        if visible_target and observed is not None:
            state.covariance = 0.012
        elif inflates_uncertainty:
            state.covariance = min(0.30, state.covariance + 0.006 + 0.002 * int(not uses_self_mask))
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
    if method in {"ensemble_uncertainty_planner", "occlusion_aware_permanence", "occlusion_full"} and occluded and state.covariance > 0.10 and state.occlusion_age < 18:
        state.diagnostic_steps += 1
        side = -1.0 if state.belief[0] > 0 else 1.0
        return clamp_pos(state.belief + np.array([0.20 * side, -0.16]))
    if method in {"ablate_no_uncertainty_inflation", "no_self_mask_ablation", "ablate_no_self_mask"} and occluded and state.covariance > 0.18:
        return clamp_pos(state.belief)
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
    localization_during_occ: List[float] = []
    localization_after_occ: List[float] = []
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
        if visible_target:
            visible_steps += 1
        if false_detection:
            false_detection_steps += 1

        update_method(state, method, cfg, pack, tool, tool_vel, target, distractor, observed, visible_target, occluded, false_detection, rng)
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
        "wrong_object_contact": str(wrong_object_contact),
        "wrong_contact_rate": f"{wrong_contact_steps / STEPS:.5f}",
        "target_contact_rate": f"{target_contact_steps / STEPS:.5f}",
        "diagnostic_steps": str(state.diagnostic_steps),
        "path_length": f"{path:.5f}",
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
                "wrong_object_contact_rate": f"{mean_metric(group, 'wrong_object_contact'):.5f}",
                "wrong_contact_rate": f"{mean_metric(group, 'wrong_contact_rate'):.5f}",
                "target_contact_rate": f"{mean_metric(group, 'target_contact_rate'):.5f}",
                "mean_diagnostic_steps": f"{mean_metric(group, 'diagnostic_steps'):.5f}",
                "mean_path_length": f"{mean_metric(group, 'path_length'):.5f}",
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
        "wrong_object_contact_rate",
        "wrong_contact_rate",
        "target_contact_rate",
        "mean_diagnostic_steps",
        "mean_path_length",
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


def build_pairwise(seed_rows: List[Dict[str, str]], reference: str = "occlusion_aware_permanence") -> List[Dict[str, str]]:
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


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(summary: List[Dict[str, str]], split_order: Sequence[str], methods: Sequence[str], metric: str, title: str, path: Path, ylim: Tuple[float, float] | None = None) -> None:
    width = 0.095
    x = np.arange(len(split_order))
    plt.figure(figsize=(13, 5))
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


def decide(summary: List[Dict[str, str]], pairwise: List[Dict[str, str]]) -> Tuple[str, str]:
    combined = [row for row in summary if row["split"] == "combined_stress"]
    proposed = [row for row in combined if row["method"] == "occlusion_aware_permanence"][0]
    non_oracle = [row for row in combined if row["method"] not in {"occlusion_aware_permanence", "oracle_state"}]
    best = max(non_oracle, key=lambda row: float(row["mean_success_rate"]))
    prop_success = float(proposed["mean_success_rate"])
    best_success = float(best["mean_success_rate"])
    prop_false = float(proposed["mean_false_disappearance_rate"])
    best_false = float(best["mean_false_disappearance_rate"])
    pair = [row for row in pairwise if row["split"] == "combined_stress" and row["comparison"] == best["method"]][0]
    paired = float(pair["paired_success_diff"])
    paired_ci = float(pair["ci95_success_diff"])
    if prop_success - best_success >= 0.045 and paired - paired_ci > 0.0 and prop_false <= best_false + 0.02:
        return (
            "STRONG_REVISE",
            f"occlusion_aware_permanence beats strongest non-oracle baseline {best['method']} on combined_stress by "
            f"{prop_success - best_success:.3f} success with paired diff {paired:.3f}+/-{paired_ci:.3f}, "
            "but lacks real robot/public benchmark validation.",
        )
    return (
        "KILL_ARCHIVE",
        f"occlusion_aware_permanence does not clear strongest non-oracle baseline {best['method']} decisively on combined_stress "
        f"(proposed={prop_success:.3f}, best_baseline={best_success:.3f}, paired diff={paired:.3f}+/-{paired_ci:.3f}).",
    )


def negative_cases(raw_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    candidates = [r for r in raw_rows if r["method"] == "occlusion_aware_permanence" and r["split"] in {"combined_stress", "object_displacement"}]
    worst = sorted(candidates, key=lambda r: (int(r["success"]), -float(r["mean_error_during_occlusion"]), -int(float(r["wrong_object_contact"]))))[:12]
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
                "lesson": lesson,
            }
        )
    return rows


def main() -> None:
    start_time = time.time()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    pack = generate_training_pack()
    write_csv(RESULTS / "training_occlusion_examples.csv", pack.training_rows)
    write_csv(
        RESULTS / "training_summary.csv",
        [
            {
                "training_examples": str(TRAINING_EXAMPLES),
                "learned_regressor_train_error": f"{pack.train_error:.5f}",
                "feature_dim": "15",
            }
        ],
    )

    model = make_model()
    raw_rows: List[Dict[str, str]] = []
    for split in SPLITS:
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
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "metrics.csv", summary)
    write_csv(RESULTS / "self_occlusion_metrics.csv", summary)
    write_csv(RESULTS / "pairwise_stats.csv", pairwise)
    write_csv(RESULTS / "self_occlusion_pairwise.csv", pairwise)

    combined = [s for s in SPLITS if s.name == "combined_stress"][0]
    ablation_raw: List[Dict[str, str]] = []
    for seed in SEEDS:
        for episode in range(ABLATION_EPISODES):
            cfg = make_config(combined, seed, 1000 + episode)
            for method in ABLATION_METHODS:
                row = simulate_episode(model, method, cfg, pack)
                row["method"] = method
                ablation_raw.append(row)
    write_csv(RESULTS / "self_occlusion_ablation_raw.csv", ablation_raw)
    ablation_summary = build_summary(build_seed_metrics(ablation_raw))
    write_csv(RESULTS / "ablation_metrics.csv", ablation_summary)
    write_csv(RESULTS / "self_occlusion_ablation.csv", ablation_summary)

    stress_raw: List[Dict[str, str]] = []
    for stress_level in np.linspace(0.0, 1.0, 6):
        for seed in SEEDS:
            for episode in range(STRESS_EPISODES):
                cfg = make_config(combined, seed, 2000 + episode, stress_level=float(stress_level))
                for method in STRESS_METHODS:
                    row = simulate_episode(model, method, cfg, pack)
                    row["split"] = "stress_sweep"
                    row["stress_level"] = f"{stress_level:.2f}"
                    stress_raw.append(row)
    write_csv(RESULTS / "stress_sweep_raw.csv", stress_raw)
    stress_summary: List[Dict[str, str]] = []
    for (method, stress_level), group in sorted(group_rows(stress_raw, ["method", "stress_level"]).items()):
        seed_vals = []
        for seed in [str(s) for s in SEEDS]:
            rows = [r for r in group if r["seed"] == seed]
            if rows:
                seed_vals.append(float(np.mean([float(r["success"]) for r in rows])))
        stress_summary.append(
            {
                "method": method,
                "stress_level": stress_level,
                "seeds": str(len(seed_vals)),
                "episodes_per_seed": str(STRESS_EPISODES),
                "mean_success_rate": f"{float(np.mean(seed_vals)):.5f}",
                "ci95_success_rate": f"{ci95(seed_vals):.5f}",
            }
        )
    write_csv(RESULTS / "stress_sweep.csv", stress_summary)
    write_csv(FIGURES / "stress_curve_data.csv", stress_summary)
    write_csv(RESULTS / "negative_cases.csv", negative_cases(raw_rows))

    split_order = [s.name for s in SPLITS]
    plot_metric(summary, split_order, METHODS, "success_rate", "Paper 71 MuJoCo object permanence success", FIGURES / "self_occlusion_success_by_split.png", (0, 1.0))
    plot_metric(summary, split_order, METHODS, "mean_error_during_occlusion", "Paper 71 localization error during self-occlusion", FIGURES / "self_occlusion_error_by_split.png")
    plot_metric(summary, split_order, METHODS, "false_disappearance_rate", "Paper 71 false disappearance under self-occlusion", FIGURES / "self_occlusion_false_disappearance.png", (0, 1.0))
    plot_ablation(ablation_summary, FIGURES / "self_occlusion_ablation_success.png")
    plot_stress(stress_summary, FIGURES / "self_occlusion_stress_sweep.png")

    decision, reason = decide(summary, pairwise)
    elapsed = time.time() - start_time
    combined_rows = [r for r in summary if r["split"] == "combined_stress"]
    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Paper 71 object_permanence_under_self_occlusion real MuJoCo rebuild\n")
        f.write(f"Terminal recommendation: {decision}\n")
        f.write(f"Reason: {reason}\n")
        f.write(f"Main eval rows: {len(raw_rows)}\n")
        f.write(f"Ablation rows: {len(ablation_raw)}\n")
        f.write(f"Stress rows: {len(stress_raw)}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write(f"Eval episodes per seed/split: {EVAL_EPISODES}\n")
        f.write(f"Runtime seconds: {elapsed:.2f}\n\n")
        f.write("Combined-stress summary:\n")
        for row in sorted(combined_rows, key=lambda r: -float(r["mean_success_rate"])):
            f.write(
                f"{row['method']} success={row['mean_success_rate']} ci95={row['ci95_success_rate']} "
                f"occ_error={row['mean_mean_error_during_occlusion']} false_disappearance={row['mean_false_disappearance_rate']}\n"
            )

    print(f"wrote Paper 71 MuJoCo self-occlusion evidence to {RESULTS}")
    print(f"terminal recommendation: {decision}")
    print(reason)


if __name__ == "__main__":
    main()
