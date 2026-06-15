# Paper 71 ICLR-Main Submission-Readiness Execution Plan

Date: 2026-06-15
Paper: 71 - `object_permanence_under_self_occlusion`
Target venue posture: ICLR main only if supported by decisive evidence
Current terminal label entering audit: `KILL_ARCHIVE`

## Goal

Rebuild and audit Paper 71 as a real submission candidate rather than a cosmetic manuscript. The audit must decide whether the MuJoCo self-occlusion evidence can honestly support an ICLR-main submission, or whether the paper remains a terminal negative result.

## Decision Rule

Upgrade from `KILL_ARCHIVE` only if all of the following are true:

1. `occlusion_aware_permanence` decisively beats the strongest non-oracle learned/tracking baseline on combined stress.
2. The paired confidence interval supports a real effect rather than a close or ambiguous mean gain.
3. Stress-level results remain favorable at the hardest occlusion settings.
4. Ablations isolate the claimed self-occlusion mechanism; removing the self mask, uncertainty inflation, or update logic should clearly hurt performance.
5. False disappearance and occlusion-error behavior are compatible with a robotics submission claim.
6. The evidence is reproducible from checked-in code, raw CSVs, and a clean PDF build.

If any of these gates fail, preserve `KILL_ARCHIVE` and document the exact failure mode.

## Evidence Gates

Run these checks before changing the decision:

1. Code integrity: compile the experiment source with `python -m py_compile src/run_experiment.py`.
2. Result integrity: verify all required CSVs exist, are nonempty, finite, and schema-valid.
3. Scale check: confirm the recorded evidence includes 7 seeds, 3,360 main rollouts, 420 ablation rollouts, and 2,016 stress rollouts.
4. Baseline check: verify `last_seen_memory`, `visibility_gated_kalman`, `particle_belief_tracker`, `learned_occlusion_regressor`, `ensemble_uncertainty_planner`, `no_self_mask_ablation`, and `oracle_state` are present.
5. Stress check: confirm stress-level 1.00 results are represented and compare the proposed method against the strongest non-oracle stress baseline.
6. Ablation check: confirm whether `occlusion_full` decisively beats removed-component variants.
7. Paper build: run LaTeX/BibTeX to produce a clean PDF and copy only the numbered PDF to `C:/Users/wangz/Downloads/71.pdf`.
8. Artifact hygiene: confirm no numbered PDF is copied to the visible Desktop.
9. GitHub hygiene: confirm the matching public GitHub repository exists and the local commit is pushed.
10. Root-report hygiene: update `GLOBAL_POOL_STATUS.md`, `BATCH_STATUS.md`, `SUBMISSION_STATUS.md`, `MASTER_REPORT.md`, and `MASTER_SUBMISSION_REPORT.md`.

## Expected Risk

The existing evidence summary reports that `occlusion_aware_permanence` has the best non-oracle mean on combined stress, but its paired advantage over `no_self_mask_ablation` is 0.119 +/- 0.123. That interval is not decisive under the requested ICLR-main bar. Unless verification contradicts the summary, Paper 71 cannot honestly become submission-ready in this pass.

## Execution Order

1. Re-check repository cleanliness and result inventory.
2. Run code and CSV integrity gates.
3. Rebuild the paper PDF and repair recoverable build warnings.
4. Write a terminal audit with exact evidence and rejection rationale.
5. Update child status, local audit docs, and root reports.
6. Commit and push the Paper 71 repository.
7. Verify `Downloads/71.pdf`, no Desktop copy, public GitHub visibility, clean git state, and root report consistency.

