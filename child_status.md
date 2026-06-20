# Child Status 71

Current stage: ICLR main expanded frozen-evidence terminal
Last update: 2026-06-21 03:58 Asia/Shanghai
PDF: C:/Users/wangz/Downloads/71.pdf
PDF pages: 55
PDF SHA256: 780086C93CDEFB9962E13851043BD302C58512AB6A2ACDF5403247A09D0A18A2
GitHub: https://github.com/Jason-Wang313/71_object_permanence_under_self_occlusion
Submission-hardening version: v5 expanded
Terminal decision: KILL_ARCHIVE
ICLR main ready: no

v5 expanded evidence: frozen MuJoCo self-occlusion benchmark with 10,080 main method-evaluation rows, 1,680 seed summaries, 1,536 ablation rollouts, 4,320 stress rollouts, 195 paired comparisons, 12 negative cases, strengthened learned/filter/planning baselines, fixed-risk gates, ablation gates, raw CSVs, figures, a 55-page ICLR-style PDF, and bright boxed clickable citations.

Decision reason: v5 does not clear the +0.030 hard-regime gate over the strongest non-oracle baseline, the paired lower bound against v4 is not positive, v5 trails v4 on combined/extreme regimes, the fixed-risk gate fails at budget 0.10, and multiple ablations match or beat the full v5 mechanism.

Validation: `python scripts\validate_submission_artifacts.py` passed. Final PDF visual QA passed after rendering to PNGs. No `C:/Users/wangz/Desktop/71.pdf` copy exists.
