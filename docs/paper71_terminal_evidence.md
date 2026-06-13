# Paper 71 Terminal Evidence

Date: 2026-06-13

Terminal decision: KILL_ARCHIVE

## Experiment Scale

- Main eval rows: 3,360
- Ablation rows: 420
- Stress rows: 2,016
- Seeds: 0, 1, 2, 3, 4, 5, 6
- Eval episodes per seed/split: 12
- Training examples: 2,200

## Combined-Stress Result

| Method | Success | 95 CI | Occlusion Error | False Disappearance |
|---|---:|---:|---:|---:|
| oracle_state | 1.000 | 0.000 | 0.000 | 0.000 |
| occlusion_aware_permanence | 0.905 | 0.083 | 0.054 | 0.575 |
| no_self_mask_ablation | 0.786 | 0.079 | 0.069 | 0.699 |
| learned_occlusion_regressor | 0.714 | 0.100 | 0.077 | 0.202 |
| visibility_gated_kalman | 0.250 | 0.118 | 0.202 | 0.002 |
| particle_belief_tracker | 0.238 | 0.083 | 0.131 | 0.049 |
| last_seen_memory | 0.190 | 0.099 | 0.168 | 0.422 |
| ensemble_uncertainty_planner | 0.107 | 0.047 | 0.216 | 0.668 |

Paired proposed versus closest non-oracle baseline (`no_self_mask_ablation`): 0.119 success difference with 0.123 95 CI.

## Stress-Level 1.00 Result

| Method | Success | 95 CI |
|---|---:|---:|
| oracle_state | 1.000 | 0.000 |
| occlusion_aware_permanence | 0.768 | 0.083 |
| learned_occlusion_regressor | 0.661 | 0.128 |
| ensemble_uncertainty_planner | 0.161 | 0.103 |
| particle_belief_tracker | 0.089 | 0.045 |
| visibility_gated_kalman | 0.089 | 0.088 |

## Ablation Outcome

The ablation suite does not prove a decisive component story. `occlusion_full` reaches 0.914 success, while `ablate_no_self_mask` reaches 0.886 and `ablate_no_uncertainty_inflation` reaches 0.886.

## Terminal Rationale

The paper now has real MuJoCo evidence and strong implemented baselines. The evidence is promising but not decisive enough for ICLR main: the closest ablation/baseline remains within the paired confidence interval, and false disappearance remains high under combined stress. The paper is archived as a real negative result.
