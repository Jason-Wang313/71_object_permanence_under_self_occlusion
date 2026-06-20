# 71 Object Permanence Under Self-Occlusion

Submission-hardening version: v5 expanded frozen archive

Terminal decision: KILL_ARCHIVE for ICLR main conference.

Canonical PDF: `C:/Users/wangz/Downloads/71.pdf`

Final PDF pages: 55

Final PDF SHA256: `780086C93CDEFB9962E13851043BD302C58512AB6A2ACDF5403247A09D0A18A2`

GitHub: `https://github.com/Jason-Wang313/71_object_permanence_under_self_occlusion`

Paper 71 was rebuilt from a short near-miss report into a frozen MuJoCo negative-result archive for robot object permanence under self-occlusion. The expanded benchmark simulates a robot tool, target object, distractor object, camera visibility, robot self-occlusion geometry, hidden target displacement, false reappearance, hidden contact drift, stale-memory traps, camera dropout bursts, embodiment/control shift, high-symmetry layouts, ablations, fixed-risk analysis, and stress sweeps.

The result is honest and negative. On hard regimes, `occlusion_aware_permanence_v5` reaches 0.929 success, while the strongest non-oracle hard-regime baseline, `occlusion_aware_permanence_v4`, reaches 0.908. That margin does not clear the frozen +0.030 gate, and the paired lower bound against v4 is not positive. On combined/extreme regimes, v5 reaches 0.719 while v4 reaches 0.740. At a fixed 10 percent failure budget, v5 reaches 0.000 while `random_forest_occlusion_regressor` reaches 0.933. The ablation gate also fails because multiple ablations match or beat the full v5 mechanism.

## Frozen Evidence Scale

- Training occlusion examples: 2,400
- Main method-evaluation rows: 10,080
- Seed summary rows: 1,680
- Split-method metric rows: 210
- Paired comparison rows: 195
- Ablation rollout rows: 1,536
- Ablation metric rows: 48
- Stress rollout rows: 4,320
- Stress metric rows: 180
- Negative cases: 12

## Reproduce Frozen MuJoCo Evidence

```powershell
python src\run_experiment.py --seeds 8 --episodes 6 --ablation-episodes 4 --stress-episodes 3 --train-scenes 2400 --splits nominal short_self_occlusion long_self_occlusion end_effector_occlusion distractor_swap near_identical_distractors object_displacement hidden_contact_drift false_reappearance camera_dropout_burst stale_memory_trap embodiment_control_shift high_symmetry_layout combined_stress combined_extreme_stress --ablation-splits combined_stress combined_extreme_stress false_reappearance stale_memory_trap --stress-splits combined_stress combined_extreme_stress false_reappearance --stress-levels 0.0 0.25 0.5 0.75 1.0 --results-dir results --figures-dir figures --workers 1
```

The frozen run completed in 6,890.40 seconds on CPU with one worker.

## Rebuild PDF

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_submission_pdf.ps1
```

The build regenerates submission tables from CSV artifacts, compiles the ICLR-style PDF, and writes the numbered PDF to `C:/Users/wangz/Downloads/71.pdf`.

## Validation

```powershell
python scripts\validate_submission_artifacts.py
```

Latest validation passed: counts, figures, TeX links, Downloads PDF, repo URL, and Desktop hygiene are OK.

No visible Desktop PDF copy should be made.
