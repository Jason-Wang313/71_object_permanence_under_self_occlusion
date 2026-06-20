# Plan

Build paper 71 `object_permanence_under_self_occlusion` from the shared pool, compile PDF to Downloads only, and publish the exact-name public repo.

## 2026-06-15 Continuation Plan

1. Re-audit the real MuJoCo self-occlusion evidence before making any submission-readiness claim.
2. Confirm the experiment source compiles and all raw CSVs are present, finite, and at the claimed scale.
3. Rebuild the PDF, repair recoverable LaTeX/BibTeX issues, and copy only `71.pdf` to Downloads.
4. Preserve `KILL_ARCHIVE` unless `occlusion_aware_permanence` decisively beats the closest baseline and ablations clearly support the self-occlusion mechanism.
5. Update child docs, root reports, and GitHub state before moving to Paper 72.

## 2026-06-21 Expanded-Standard Plan

Detailed plan: `docs/paper71_expanded_submission_plan_20260621.md`

1. Expand the MuJoCo self-occlusion benchmark from the v4 near miss into a hostile partial-observability suite with 15 splits, stronger learned/filtering/particle/POMDP-style baselines, v4 replay, v5 mechanism, and component ablations.
2. Run small development probes first, document recoverable failures, then freeze seeds, splits, methods, metrics, and gates before the full run.
3. Execute the frozen CPU-only full run with at least 10,000 main rows, 1,400 ablation rows, and 2,800 stress rows unless runtime forces episode reduction.
4. Build generated tables, full appendix evidence, bright boxed clickable citations, and a 25+ page ICLR-style PDF.
5. Validate CSV counts, TeX links, Downloads-only `71.pdf`, Desktop hygiene, visual PDF rendering, public GitHub push, and root ledger updates before starting Paper 72.
