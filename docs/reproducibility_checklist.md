# Reproducibility Checklist

## What Reproduces

- [x] `python src/run_experiment.py`
- [x] MuJoCo self-occlusion object-permanence benchmark.
- [x] Learned occlusion-regressor baseline.
- [x] Last-seen, Kalman, particle, ensemble, proposed, ablation, and oracle methods.
- [x] `results/self_occlusion_raw.csv`
- [x] `results/metrics.csv`
- [x] `results/raw_seed_metrics.csv`
- [x] `results/pairwise_stats.csv`
- [x] `results/ablation_metrics.csv`
- [x] `results/stress_sweep.csv`
- [x] `results/negative_cases.csv`
- [x] Paper-specific figures in `figures/`.
- [x] `paper/main.tex`
- [x] Canonical PDF: `C:/Users/wangz/Downloads/71.pdf`

## What Does Not Reproduce

- [ ] Real robot results.
- [ ] External public benchmark results.
- [ ] Real camera calibration or hardware occlusion masks.
- [ ] Manual exhaustive related-work synthesis.

This is reproducible as a real MuJoCo negative-result package, not as an ICLR-main-ready robotics submission.
