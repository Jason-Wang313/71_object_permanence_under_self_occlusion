# Paper 71 Expanded Submission Plan - 2026-06-21

Paper: `object_permanence_under_self_occlusion`

Target venue style: ICLR main, but only if frozen evidence survives hostile review.

Current state: v4 is a real MuJoCo near miss, not a submission. The proposed method beats the closest no-self-mask baseline in mean combined-stress success, but the paired interval crosses zero and the ablations do not isolate the self-occlusion mechanism. The v5 pass must therefore expand evidence, add stronger baselines, and be prepared to kill the paper.

## Goal

Rebuild Paper 71 into a 25+ page ICLR-style submission archive with real CPU-only MuJoCo evidence, theory, strong baselines, hostile stress tests, negative cases, and a frozen terminal decision. Do not optimize for pretty results. Optimize for a result that survives hostile review.

## Central Claim To Test

Explicit robot self-occlusion geometry and object permanence should improve closed-loop manipulation when the robot hides the target from its own camera. The claim is only valid if a branch-aware permanence mechanism beats strong learned, filtering, particle, uncertainty, identity-tracking, and geometry-ablated baselines under hard occlusion regimes without increasing false disappearance or wrong-object contacts.

## Development Rules

- Keep the run CPU-only and RAM-light. Prefer streaming rows, compact NumPy state, and scikit-learn models over deep checkpoints.
- Use small development probes to expose weaknesses before the frozen run.
- Freeze the protocol before the final run: splits, methods, seeds, metrics, gates, and row counts.
- After freeze, do not alter gates, cherry-pick splits, or remove failures.
- Report all predefined results honestly, including negative and embarrassing cases.
- Produce the final numbered PDF only at `C:/Users/wangz/Downloads/71.pdf`; do not copy a PDF to Desktop.

## v5 Benchmark Expansion

Keep the existing planar MuJoCo self-occlusion scene, but expand it into a richer partial-observability benchmark:

- Nominal visible reaching.
- Short robot-link self-occlusion.
- Long robot-link self-occlusion.
- End-effector self-occlusion during approach.
- Distractor swap while target is hidden.
- Near-identical target/distractor appearances.
- Hidden target displacement.
- Hidden target drift after incidental contact.
- False reappearance from distractor detections.
- Camera dropout bursts.
- Stale-memory trap after prolonged occlusion.
- Embodiment/control-limit shift.
- High-symmetry object layout.
- Combined stress.
- Combined extreme stress.

Each rollout must log true target state, distractor state, tool state, visibility flags, self-occlusion geometry features, false detections, belief estimate, uncertainty/calibration, action target, final success, localization error during occlusion, reacquisition latency, false disappearance, identity switch, wrong-object contact, and diagnostic/replanning burden.

## Methods And Baselines

Main methods:

- `last_seen_memory`: last visible target position.
- `visibility_gated_kalman`: constant-velocity Kalman-style tracker.
- `particle_belief_tracker`: particle filter with occlusion dynamics.
- `learned_occlusion_regressor`: supervised hidden-state regressor.
- `random_forest_occlusion_regressor`: stronger nonlinear learned baseline.
- `hist_gradient_occlusion_regressor`: stronger tabular learned baseline when available.
- `identity_consistency_tracker`: object identity tracker using appearance, motion, and proximity.
- `ensemble_uncertainty_planner`: uncertainty-aware ensemble with viewing moves.
- `risk_averse_particle_planner`: particle planner with tail-risk action choice.
- `pomdp_style_belief_planner`: lightweight POMDP-style belief update baseline.
- `occlusion_aware_permanence_v4`: frozen replay of the old method.
- `occlusion_aware_permanence_v5`: proposed branch-aware permanence method.
- `no_self_mask_ablation`: v5 without explicit self-occlusion geometry.
- `oracle_state`: true-state upper bound.

Ablations:

- Full v5.
- No self-mask geometry.
- No branch belief.
- No contact update.
- No uncertainty inflation.
- No identity filter.
- No reacquisition guard.
- No tail-risk action objective.
- No false-detection rejection.
- V4 replay.
- Learned-only branch.

## Metrics

Primary:

- Closed-loop success.
- Paired success difference versus strongest non-oracle baseline.
- False disappearance rate.
- Wrong-object contact rate.
- Identity-switch rate.
- Mean localization error during occlusion.
- Reacquisition latency after visibility returns.

Secondary:

- Belief calibration/ECE-like binned error.
- Diagnostic/replanning burden.
- Stress-sweep success at fixed stress levels.
- Fixed false-disappearance operating point.
- Fixed wrong-contact operating point.
- Negative-case categories.

## Frozen Gates

The paper cannot be called ICLR-main ready unless all gates pass:

- Hard-regime gate: v5 beats the strongest non-oracle hard-regime baseline by at least 0.030 success.
- Paired gate: the paired success lower bound against the strongest hard-regime baseline is positive.
- Combined/extreme gate: v5 beats the strongest non-oracle combined/extreme baseline by at least 0.030 success.
- Safety gate: v5 does not exceed the strongest baseline by more than 0.020 in false disappearance or wrong-object contact.
- Fixed-risk gate: at fixed false-disappearance and wrong-contact budgets, v5 is not worse than the best baseline.
- Maximum-stress gate: v5 is within 0.030 of the best non-oracle method at stress level 1.00.
- Ablation gate: every removed-component variant must be at least 0.020 success below full v5 or clearly worse on safety/calibration.

If any gate fails, the terminal decision is `KILL_ARCHIVE` unless the evidence is genuinely promising but externally incomplete, in which case `STRONG_REVISE` is allowed. Do not upgrade the decision for writing quality alone.

## Frozen Scale Target

Use a CPU-light but review-serious run:

- Seeds: 8.
- Main episodes: 6 per seed/split.
- Main methods: at least 14.
- Main splits: 15.
- Main rows target: at least 10,000.
- Ablation episodes: 4 per seed/split.
- Ablation splits: at least 4 hard splits.
- Ablation rows target: at least 1,400.
- Stress episodes: 3 per seed/level/split.
- Stress levels: 0.00, 0.25, 0.50, 0.75, 1.00.
- Stress rows target: at least 2,800.

If runtime becomes excessive, reduce episodes before reducing baselines or stress diversity.

## Paper And Artifact Expansion

The final manuscript must be at least 25 pages without padding. It should contain:

- Clear question and hostile-review framing.
- Related work anchored in object permanence, occlusion-aware tracking, object-centric world models, manipulation under partial observability, and robot failure/stress benchmarks.
- Formal problem setup with visibility, self-occlusion, belief, and safety definitions.
- Propositions explaining when self-occlusion geometry is identifiable and when learned/particle baselines can match it.
- Frozen protocol and decision gates.
- Main results, fixed-risk analysis, stress sweep, ablations, negative cases, limitations, and archival decision.
- Generated appendix tables for full metrics, seed-level metrics, pairwise comparisons, ablations, stress sweeps, and negative cases.
- Bright boxed clickable citation links routed to the references.

## Deliverables

- `docs/paper71_expanded_submission_plan_20260621.md`
- `docs/paper71_development_log_20260621.md`
- `docs/paper71_protocol_freeze_20260621.md`
- Expanded `src/run_experiment.py`
- Generated CSVs and figures under `results/` and `figures/`
- `scripts/render_submission_assets.py`
- `scripts/build_submission_pdf.ps1`
- `scripts/validate_submission_artifacts.py`
- Expanded `paper/main.tex` and `paper/references.bib`
- `C:/Users/wangz/Downloads/71.pdf`
- Updated `README.md`, `child_status.md`, and `docs/submission_readiness_decision.md`
- Public GitHub push to `https://github.com/Jason-Wang313/71_object_permanence_under_self_occlusion`
- Updated root ledgers before moving to Paper 72
