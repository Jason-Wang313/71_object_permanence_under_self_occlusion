# Paper 71 Rebuild Plan: Object Permanence Under Self-Occlusion

Date: 2026-06-13

## Goal

Rebuild Paper 71 into a real ICLR-main-target robotics submission candidate, or terminate it honestly as `STRONG_REVISE` / `KILL_ARCHIVE` if real evidence does not justify submission. The central question is whether explicitly modeling robot self-occlusion improves object permanence and closed-loop manipulation compared with strong tracking, filtering, memory, and learned baselines.

## Core Claim To Test

Robot self-occlusion should be treated as a distinct world-model failure mode. When the robot's own arm or tool hides an object, the policy should maintain an action-relevant object belief instead of collapsing to the last visible frame, generic uncertainty, or hallucinated disappearance.

## High-Fidelity Benchmark

Build a MuJoCo tabletop scene with a controllable planar or 3D manipulator, a target object, distractors, and a fixed camera. The robot must reach or push the target after periods where its own end-effector/link occludes the object from the camera view.

Required task families:

- Nominal visible reaching: object remains visible most of the episode.
- Self-occlusion: robot link/tool passes between camera and target.
- Distractor swap: a distractor appears near the target while the true target is occluded.
- Object displacement under occlusion: target may move slightly due to contact or perturbation while hidden.
- Combined stress: self-occlusion plus distractors, observation noise, actuator limits, and object displacement.

The benchmark must log MuJoCo states, camera-derived visibility, occlusion masks or visibility flags, estimated object belief, planned action, contact events, final task success, localization error, false disappearance, and unsafe contacts.

## Methods To Implement

- `last_seen_memory`: holds the last visible object position during occlusion.
- `visibility_gated_kalman`: Kalman-style filter that increases uncertainty during occlusion.
- `particle_belief_tracker`: particle filter over object position with contact/dynamics noise.
- `learned_occlusion_regressor`: supervised model trained from simulated occlusion traces to estimate hidden object state.
- `ensemble_uncertainty_planner`: ensemble of predictors with uncertainty-aware action selection.
- `occlusion_aware_permanence`: proposed method; explicitly uses robot-link visibility geometry to distinguish self-occlusion from object disappearance and maintains branch beliefs over hidden object hypotheses.
- `no_self_mask_ablation`: proposed method without robot self-occlusion geometry.
- `oracle_state`: upper bound with access to the true object state.

## Metrics

- Closed-loop manipulation success.
- Object localization error during and after occlusion.
- Recovery success after the object becomes visible again.
- False disappearance rate.
- Wrong-object contact rate.
- Unsafe robot-object/distractor contact rate.
- Calibration of belief uncertainty under occlusion.
- Tail risk on combined stress.

## Experimental Rigor

- Use at least 5 random seeds; target 7 if runtime stays manageable.
- Evaluate multiple task families and held-out perturbation settings.
- Report mean, 95 percent confidence intervals, and paired comparisons against the strongest non-oracle baseline.
- Include ablations: no self-occlusion mask, no branch belief, no contact update, no uncertainty inflation, no distractor filtering.
- Include stress sweeps over occlusion duration, camera noise, distractor proximity, object displacement, and control limits.
- Save raw per-episode rollouts and per-seed summaries for auditability.

## Submission Gate

The paper can only move above archive if `occlusion_aware_permanence` beats the best non-oracle baseline on combined stress with a meaningful paired effect, lower false disappearance, and no safety regression. If a learned tracker, Kalman/particle filter, or ensemble baseline matches or beats it, the paper remains `KILL_ARCHIVE` or at best `STRONG_REVISE`.

## Deliverables

- Replace the synthetic script with a reproducible MuJoCo self-occlusion benchmark runner.
- Generate raw rollout CSVs, metrics, pairwise statistics, ablations, stress sweep tables, negative cases, and figures.
- Rewrite README, claims, reviewer attacks, novelty boundary, final audit, ICLR gate, and submission decision around actual evidence.
- Rewrite `paper/main.tex` as either a real negative-result paper or a submission-candidate manuscript.
- Compile `paper/main.pdf`, copy exactly to `C:/Users/wangz/Downloads/71.pdf`, and do not copy any PDF to Desktop.
- Commit and push the final Paper 71 repo, then update shared root reports before moving to Paper 72.
