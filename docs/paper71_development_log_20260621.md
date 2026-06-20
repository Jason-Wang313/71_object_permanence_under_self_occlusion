# Paper 71 Development Log - 2026-06-21

## Starting Point

The v4 repository was clean on `main`. Existing evidence was a real MuJoCo near miss: 3,360 main rows, 420 ablation rows, 2,016 stress rows, and terminal `KILL_ARCHIVE` because the combined-stress paired gain over `no_self_mask_ablation` was non-decisive.

## Planned v5 Changes

- Expanded the benchmark from 5 to 15 task splits.
- Added stronger learned baselines: random forest and histogram-gradient hidden-state regressors.
- Added identity-consistency, risk-averse particle, and lightweight POMDP-style belief baselines.
- Added v4 replay, v5 proposed method, and 12 ablation variants.
- Added identity-switch, reacquisition-latency, calibration-error, fixed-risk, aggregate, and ablation-aggregate outputs.
- Replaced the old single combined-stress gate with frozen hostile-review gates.

## Development Probe

Command:

```powershell
python src\run_experiment.py --seeds 2 --episodes 1 --ablation-episodes 1 --stress-episodes 1 --train-scenes 200 --splits combined_stress combined_extreme_stress --ablation-splits combined_extreme_stress --stress-splits combined_extreme_stress --results-dir results\dev_probe --figures-dir figures\dev_probe
```

Result: completed successfully. Terminal decision was `KILL_ARCHIVE`; the probe correctly killed v5 when it tied `occlusion_aware_permanence_v4` and multiple ablations matched or beat full v5. No code changes were made to relax gates after seeing this result.

Runtime note: the probe took enough time that the frozen full run should be launched as a background process and polled, rather than executed as a single fragile foreground call.
