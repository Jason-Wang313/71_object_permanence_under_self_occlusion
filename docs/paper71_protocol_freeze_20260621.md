# Paper 71 Protocol Freeze - 2026-06-21

This protocol was frozen after the development probe and before the full run. The probe result did not relax any gate.

## Frozen Command

```powershell
python src\run_experiment.py --seeds 8 --episodes 6 --ablation-episodes 4 --stress-episodes 3 --train-scenes 2400 --splits nominal short_self_occlusion long_self_occlusion end_effector_occlusion distractor_swap near_identical_distractors object_displacement hidden_contact_drift false_reappearance camera_dropout_burst stale_memory_trap embodiment_control_shift high_symmetry_layout combined_stress combined_extreme_stress --ablation-splits combined_stress combined_extreme_stress false_reappearance stale_memory_trap --stress-splits combined_stress combined_extreme_stress false_reappearance --stress-levels 0.0 0.25 0.5 0.75 1.0 --results-dir results --figures-dir figures --workers 1
```

## Frozen Scale

- Seeds: 8
- Main episodes per seed/split: 6
- Main splits: 15
- Main methods: 14
- Expected main rows: 10,080
- Ablation episodes per seed/split: 4
- Ablation splits: 4
- Ablation methods: 12
- Expected ablation rows: 1,536
- Stress episodes per seed/level/split: 3
- Stress splits: 3
- Stress levels: 5
- Stress methods: 12
- Expected stress rows: 4,320

## Frozen Gates

- Hard-regime gate: `occlusion_aware_permanence_v5` must beat the strongest non-oracle hard-regime baseline by at least 0.030 success.
- Paired gate: paired success lower bound against the strongest combined/extreme baseline must be positive.
- Combined/extreme gate: v5 must beat the strongest non-oracle combined/extreme baseline by at least 0.030 success.
- Safety gate: v5 must not exceed the strongest hard-regime baseline by more than 0.020 in false disappearance or wrong-object contact.
- Fixed-risk gate: v5 must not be worse than the best non-oracle method at the 0.10 fixed-risk budget.
- Maximum-stress gate: v5 must be within 0.030 of the best non-oracle method at stress level 1.00 on `combined_extreme_stress`.
- Ablation gate: every removed-component variant must be at least 0.020 success below full v5 or clearly worse on safety/calibration.

## Frozen Terminal Policy

If any gate fails, the paper remains `KILL_ARCHIVE` unless all simulated gates pass but external validation is missing, in which case `STRONG_REVISE` is allowed. No result can be upgraded to ICLR-main ready without real-robot or public-benchmark validation and a manual related-work audit.
