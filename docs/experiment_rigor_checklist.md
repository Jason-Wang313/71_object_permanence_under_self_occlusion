# Experiment Rigor Checklist

## v4 Real-Evidence Rigor

- [x] Real MuJoCo dynamics benchmark.
- [x] Camera/self-occlusion visibility geometry.
- [x] Task families: nominal, self-occlusion, distractor swap, object displacement, combined stress.
- [x] Seven random seeds.
- [x] 3,360 main eval rollouts.
- [x] 420 ablation rollouts.
- [x] 2,016 stress-sweep rollouts.
- [x] Learned occlusion regressor baseline.
- [x] Tracking baselines: last-seen memory, visibility-gated Kalman, particle tracker.
- [x] Ensemble uncertainty planner.
- [x] Oracle state upper bound.
- [x] Raw rollout CSVs.
- [x] Per-seed metrics with confidence intervals.
- [x] Pairwise statistics against the proposed method.
- [x] Paper-specific figures.
- [x] Negative cases.

## Remaining ICLR Main Blockers

- [ ] Real-robot validation.
- [ ] External public benchmark comparison.
- [ ] Manual exhaustive full-paper related-work synthesis.
- [ ] Decisive paired win over closest non-oracle baseline.
- [ ] Ablations prove the self-mask/branch mechanism is necessary.

Decision: fail ICLR main empirical-rigor gate; archive as a real MuJoCo negative result.
