# Final Audit

1. Chosen thesis: object permanence under self-occlusion treats robot-induced occlusion as a distinct world-model failure mode.
2. ICLR-main decision: KILL_ARCHIVE.
3. Submission-hardening version: v4.
4. Evidence: real MuJoCo benchmark with 3,360 main eval rollouts, 420 ablation rollouts, 2,016 stress rollouts, learned/tracking baselines, raw CSVs, and figures.
5. Terminal result: on combined stress, `occlusion_aware_permanence` reaches 0.905 success versus 0.786 for `no_self_mask_ablation`, but the paired success difference is 0.119 +/- 0.123, so the result does not clear the strict gate.
6. Stress result: at stress level 1.00, occlusion-aware permanence reaches 0.768 success versus 0.661 for the learned occlusion regressor.
7. Ablation result: `occlusion_full` reaches 0.914 while `ablate_no_self_mask` reaches 0.886; the component story is too close for an ICLR-main mechanism claim.
8. Closest hostile prior work: see `docs/hostile_prior_work.md`, `docs/hostile_prior_work_100_cards.csv`, and `docs/hostile_reviewer_response.md`.
9. Reproducibility: `python src\run_experiment.py` regenerates metrics and figures.
10. Claim-validity status: main-conference claims killed; real negative evidence retained.
11. Exact Downloads PDF path: `C:/Users/wangz/Downloads/71.pdf`
12. GitHub URL: https://github.com/Jason-Wang313/71_object_permanence_under_self_occlusion
13. Confirmation: no visible Desktop copy was requested or made.
