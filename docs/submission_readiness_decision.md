# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

Reason: the v5 expanded rebuild replaces the short near-miss report with a frozen MuJoCo archive, stronger baselines, stress tests, fixed-risk analysis, ablations, negative cases, generated tables, and a 55-page ICLR-style PDF. The evidence is real, but it does not support an ICLR-main mechanism claim. `occlusion_aware_permanence_v5` reaches 0.929 hard-regime success versus 0.908 for `occlusion_aware_permanence_v4`, which does not clear the frozen +0.030 gate; the paired lower bound against v4 is not positive. On combined/extreme regimes, v5 reaches 0.719 while v4 reaches 0.740. At a fixed 10 percent failure budget, v5 reaches 0.000 while `random_forest_occlusion_regressor` reaches 0.933. Multiple ablations also match or beat full v5.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper to ICLR main in its current form.

Revival condition: the project would need a decisive paired win over strong learned/filter/planning baselines, fixed-risk safety that survives failure-budget constraints, ablations proving that the self-occlusion mechanism is necessary, and either real-robot or public-benchmark validation.
