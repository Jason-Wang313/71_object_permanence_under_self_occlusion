# Hostile Reviewer Response

Paper: 71 Object Permanence Under Self-Occlusion

## Strongest Technical Threats

- Failure-Aware RL: Reliable Offline-to-Online Reinforcement Learning with Self-Recovery for Real-World Manipulation (2026)
- Vision-language model-driven scene understanding and robotic object manipulation (2024)
- Self-Supervised Learning of Multi-Object Keypoints for Robotic Manipulation (2022)
- Scaling Cross-Environment Failure Reasoning Data for Vision-Language Robotic Manipulation (2025)
- Related object-tracking, scene-memory, uncertainty-planning, and manipulation-world-model work.

## ICLR Main Response

A hostile ICLR reviewer would still be correct to reject this as a main-conference submission. The paper now has real MuJoCo evidence and implemented baselines, but the result is not decisive. The proposed method has the best mean combined-stress success among non-oracles, yet its paired advantage over the closest baseline is 0.119 +/- 0.123, and ablations are close.

## Honest Action

The paper is marked `KILL_ARCHIVE`. This avoids overstating a promising but statistically non-decisive MuJoCo result.

## What Would Be Needed To Revive

- Decisive paired win over learned/tracking baselines.
- Stronger ablations isolating robot self-occlusion geometry.
- Lower false disappearance under combined stress.
- Real robot or public benchmark validation.
- Manual full-paper related-work audit.
