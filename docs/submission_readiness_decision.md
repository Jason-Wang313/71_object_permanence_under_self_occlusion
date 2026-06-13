# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

Reason: the v4 rebuild replaces the synthetic scaffold with real MuJoCo evidence, and the proposed method is promising in mean success. However, it does not clear the closest non-oracle baseline decisively: 0.905 success versus 0.786 on combined stress, with paired success difference 0.119 +/- 0.123. The ablation story is also too close, with `occlusion_full` at 0.914 and `ablate_no_self_mask` at 0.886.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper to ICLR main in its current form.

Revival condition: the project would need a decisive paired win over strong tracking/learned baselines, stronger ablations proving the self-occlusion mechanism, real-robot or public-benchmark validation, and a manual full-paper related-work audit.
