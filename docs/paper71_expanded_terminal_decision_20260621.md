# Paper 71 Expanded Terminal Decision

Decision: KILL_ARCHIVE

Reason: v5 does not beat strongest hard-regime baseline occlusion_aware_permanence_v4 by 0.030 (v5=0.929, best=0.908); paired lower bound against occlusion_aware_permanence_v4 is not positive (-0.042+/-0.135); v5 does not beat strongest combined/extreme baseline occlusion_aware_permanence_v4 by 0.030 (v5=0.719, best=0.740); fixed-risk gate fails at budget 0.10 (v5=0.000, best=random_forest_occlusion_regressor 0.933); ablation gate fails because ablate_no_branch_belief, ablate_no_distractor_filter, ablate_no_false_detection_rejection, ablate_no_reacquisition_guard, ablate_no_tail_risk_objective, ablate_no_uncertainty_inflation matches or beats full v5

Training rows: 2400
Main method-evaluation rows: 10080
Seed summary rows: 1680
Split-method metric rows: 210
Ablation rows: 1536
Stress rows: 4320
Negative cases: 12

This decision is generated from frozen CSV artifacts, not hand-transcribed table values.

## Final Artifact Check

Canonical PDF: `C:/Users/wangz/Downloads/71.pdf`

Pages: 55

SHA256: `780086C93CDEFB9962E13851043BD302C58512AB6A2ACDF5403247A09D0A18A2`

Build command: `powershell -ExecutionPolicy Bypass -File scripts\build_submission_pdf.ps1`

Validation command: `python scripts\validate_submission_artifacts.py`

Validation result: passed counts, figures, TeX links, Downloads PDF, repo URL, and Desktop hygiene.

Visual QA: passed after rendering the final PDF to PNGs and inspecting title/citation boxes, main result pages, dense appendix tables, late seed-level tables, and references.

Desktop hygiene: no `C:/Users/wangz/Desktop/71.pdf` copy exists.
