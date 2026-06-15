# Paper 71 Terminal Audit - 2026-06-15

Paper: `object_permanence_under_self_occlusion`
Decision: `KILL_ARCHIVE`
ICLR-main ready: no

## Verification Performed

1. Source compile gate passed with `python -m py_compile src/run_experiment.py`.
2. CSV integrity gate passed for all result CSVs: files are present, nonempty, finite, and schema-readable. Blank `stress_level` values are expected only in non-stress rollout tables.
3. Evidence scale matched the reported claims:
   - Main rollouts: 3,360
   - Ablation rollouts: 420
   - Stress rollouts: 2,016
   - Seeds: 0, 1, 2, 3, 4, 5, 6
4. Baselines were present in the main evidence: `last_seen_memory`, `visibility_gated_kalman`, `particle_belief_tracker`, `learned_occlusion_regressor`, `ensemble_uncertainty_planner`, `no_self_mask_ablation`, and `oracle_state`.
5. PDF rebuild completed and `C:/Users/wangz/Downloads/71.pdf` was refreshed.
6. BibTeX sort warnings were repaired by adding stable `key` fields to the local reference entries.
7. No visible Desktop copy of `71.pdf` was present after the audit.

## Fatal Evidence

The proposed method is a strong near miss, but it fails the requested ICLR-main decision rule. On combined stress, `occlusion_aware_permanence` reaches 0.905 success versus 0.786 for `no_self_mask_ablation`; however, the paired success difference is 0.119 +/- 0.123. The lower bound does not establish decisive separation from the closest baseline.

At stress level 1.00, `occlusion_aware_permanence` reaches 0.768 success while `learned_occlusion_regressor` reaches 0.661. This supports continued investigation but does not repair the non-decisive combined-stress gate or the lack of external validation.

The ablation evidence is also too close for a mechanism claim. On the ablation combined-stress grid, `occlusion_full` reaches 0.914 success, while `ablate_no_self_mask` and `ablate_no_uncertainty_inflation` each reach 0.886. The full mechanism is not isolated strongly enough for an ICLR-main submission.

## Decision

Paper 71 remains `KILL_ARCHIVE`. It is a reproducible MuJoCo near miss with useful evidence, but it is not an honest ICLR-main submission candidate under the requested bar.

## Revival Requirements

To revive this paper, a future version would need a decisive paired win over the closest self-mask and learned/tracking baselines, materially lower false disappearance, ablations that clearly isolate the self-occlusion mechanism, and either real-robot or public-benchmark validation.

