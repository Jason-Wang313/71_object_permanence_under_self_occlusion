# Submission Version Log

## v1 - Generated Draft

- Original continuation-batch generated paper and toy single-seed experiment.

## v2 - Submission Hardening

- Added hostile reviewer attack log and response docs.
- Replaced the toy experiment with seven-seed synthetic metrics, stronger synthetic baselines, ablations, stress tests, and negative cases.
- Narrowed claims to synthetic diagnostic evidence.
- Terminal decision: WORKSHOP_ONLY.

## v3 - ICLR Main Gate Archive

- Applied the stricter ICLR-main-conference standard.
- Determined that missing real-robot/high-fidelity evidence, template-generated experiments, and unresolved novelty threats were fatal.
- Terminal decision: KILL_ARCHIVE.

## v4 - Real MuJoCo Negative Result

- Replaced the synthetic probability scaffold with a MuJoCo self-occlusion object-permanence benchmark.
- Added learned/tracking/planning baselines, occlusion-aware permanence, ablations, stress sweeps, raw rollouts, and figures.
- Ran 3,360 main evaluation rollouts, 420 ablation rollouts, and 2,016 stress rollouts across seven seeds.
- Found a promising but non-decisive mean gain over the closest baseline.
- Rewrote the manuscript as a real negative-result paper.
- Terminal decision: KILL_ARCHIVE.
