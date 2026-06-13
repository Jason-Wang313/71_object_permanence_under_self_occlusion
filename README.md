# 71 Object Permanence Under Self-Occlusion

Submission-hardening version: v4

Terminal decision: KILL_ARCHIVE for ICLR main conference.

Paper 71 was rebuilt from a synthetic scaffold into a real MuJoCo benchmark for robot object permanence under self-occlusion. The benchmark simulates a robot tool, target object, distractor object, camera-derived visibility, self-occlusion geometry, hidden target displacement, false distractor detections, ablations, and stress sweeps.

The result is a real but non-decisive negative result. On combined stress, `occlusion_aware_permanence` reaches 0.905 success, ahead of the closest non-oracle baseline `no_self_mask_ablation` at 0.786, but the paired success difference is 0.119 +/- 0.123, so the gain is not statistically decisive under the strict gate. Ablations are also too close: `occlusion_full` reaches 0.914 while `ablate_no_self_mask` reaches 0.886.

## Reproduce Real MuJoCo Evidence

```powershell
python src\run_experiment.py
```

The default full run uses seven seeds, 12 eval episodes per seed/split, 10 ablation episodes per seed, 8 stress episodes per seed/level, and 2,200 training examples.

Main outputs:

- `results/self_occlusion_raw.csv`
- `results/metrics.csv`
- `results/pairwise_stats.csv`
- `results/ablation_metrics.csv`
- `results/stress_sweep.csv`
- `figures/self_occlusion_success_by_split.png`
- `figures/self_occlusion_error_by_split.png`
- `figures/self_occlusion_stress_sweep.png`

## Rebuild PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/71.pdf`

No visible Desktop PDF copy should be made.
