# Claims

- Mechanism claim tested: robot self-occlusion should be treated as a distinct object-permanence failure mode, using robot-link visibility geometry to maintain target belief while the arm/tool hides the object.
- Evidence claim: the v4 runner evaluates this in MuJoCo with camera-derived visibility, target/distractor objects, hidden displacement, false distractor detections, learned/tracking baselines, ablations, and stress sweeps.
- Main result: `occlusion_aware_permanence` reaches 0.905 success on combined stress, ahead of `no_self_mask_ablation` at 0.786, but the paired success difference is 0.119 +/- 0.123 and therefore not decisive.
- Stress result: at stress level 1.00, `occlusion_aware_permanence` reaches 0.768 success versus 0.661 for `learned_occlusion_regressor`.
- Ablation result: full occlusion permanence reaches 0.914 on the ablation grid, but `ablate_no_self_mask` reaches 0.886 and other ablations remain close.
- Scope claim: this is a real MuJoCo negative/kill-archive package, not a real-robot or public-benchmark submission.
- Unsupported claim explicitly avoided: no ICLR-main-ready or state-of-the-art robotics claim is made.
