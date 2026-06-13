# Novelty Boundary Map

## Crowded Territory

- Generic object tracking under occlusion.
- Kalman, particle, and learned state estimation.
- Scene memory for robot manipulation.
- Uncertainty-aware planning.
- Benchmark-only contributions.

## Claimed Boundary Tested

Robot self-occlusion is treated as a special failure mode: the system uses robot-link visibility geometry to know when the robot itself, not object disappearance, hides the target.

## v4 Evidence Outcome

The boundary is promising but not decisive. On combined stress, occlusion-aware permanence reaches 0.905 success, while the closest non-oracle baseline reaches 0.786. However, the paired difference is 0.119 +/- 0.123, and the ablation grid keeps no-self-mask and other variants close to the full method.

## What Falsified Submission Readiness

The method does not prove a decisive, statistically robust advantage over the closest baseline, and the ablations do not isolate the self-occlusion mechanism strongly enough for an ICLR-main claim.
