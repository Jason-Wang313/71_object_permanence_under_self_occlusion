from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
GENERATED = PAPER / "generated"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def label(value: str) -> str:
    return tex_escape(value.replace("_", " "))


def fmt(value: object, digits: int = 3) -> str:
    text = str(value)
    if text.lower() == "inf":
        return r"$\infty$"
    try:
        return f"{float(text):.{digits}f}"
    except ValueError:
        return tex_escape(text)


def table_cell(name: str, value: str) -> str:
    if name in {"method", "split", "comparison", "reference", "group"}:
        return label(value)
    if name in {"seed", "episode", "reference_better_seeds", "seeds"}:
        try:
            return str(int(float(value)))
        except ValueError:
            return tex_escape(value)
    return fmt(value)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_summary() -> tuple[str, str]:
    text = (RESULTS / "summary.txt").read_text(encoding="utf-8")
    decision = re.search(r"Terminal decision: (.+)", text)
    reason = re.search(r"Terminal reason: (.+)", text)
    return (
        decision.group(1).strip() if decision else "UNKNOWN",
        reason.group(1).strip() if reason else "No terminal reason found.",
    )


def count_rows(name: str) -> int:
    return len(read_csv(RESULTS / name))


def row_lookup(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def aggregate_table(rows: list[dict[str, str]], group: str) -> str:
    rows = [row for row in rows if row["group"] == group]
    rows = sorted(rows, key=lambda row: float(row["success"]), reverse=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{label(group)} aggregate results. Higher success is better; lower failure metrics are better.}}",
        rf"\label{{tab:{group}}}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Success & False disp. & Wrong obj. & Id switch & Occ. err. \\",
        r"\midrule",
    ]
    for row in rows[:10]:
        lines.append(
            f"{label(row['method'])} & {fmt(row['success'])} & {fmt(row['false_disappearance'])} & "
            f"{fmt(row['wrong_object_contact'])} & {fmt(row['identity_switch'])} & {fmt(row['occlusion_error'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def selected_split_table(metrics: list[dict[str, str]], split: str) -> str:
    wanted = {
        "occlusion_aware_permanence_v5",
        "occlusion_aware_permanence_v4",
        "random_forest_occlusion_regressor",
        "hist_gradient_occlusion_regressor",
        "learned_occlusion_regressor",
        "no_self_mask_ablation",
        "pomdp_style_belief_planner",
        "oracle_state",
    }
    rows = [row for row in metrics if row["split"] == split and row["method"] in wanted]
    rows = sorted(rows, key=lambda row: float(row["mean_success_rate"]), reverse=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Selected methods on {label(split)}.}}",
        r"\label{tab:selected-split}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Success & False disp. & Wrong obj. & Id switch & Occ. err. \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{label(row['method'])} & {fmt(row['mean_success_rate'])} & {fmt(row['mean_false_disappearance_rate'])} & "
            f"{fmt(row['mean_wrong_object_contact_rate'])} & {fmt(row['mean_identity_switch_rate'])} & "
            f"{fmt(row['mean_mean_error_during_occlusion'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def fixed_risk_table(rows: list[dict[str, str]]) -> str:
    rows = [row for row in rows if row["budget"] == "0.10"]
    rows = sorted(rows, key=lambda row: float(row["success_at_budget"]), reverse=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Fixed-risk operating points at a 10 percent maximum failure budget over hard splits.}",
        r"\label{tab:fixed-risk}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Success & Coverage & False disp. & Wrong obj. & Id switch \\",
        r"\midrule",
    ]
    for row in rows[:10]:
        lines.append(
            f"{label(row['method'])} & {fmt(row['success_at_budget'])} & {fmt(row['coverage'])} & "
            f"{fmt(row['false_disappearance'])} & {fmt(row['wrong_object_contact'])} & {fmt(row['identity_switch'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def ablation_table(rows: list[dict[str, str]]) -> str:
    rows = sorted(rows, key=lambda row: float(row["success"]), reverse=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Occlusion-aware permanence v5 ablations aggregated over frozen ablation splits.}",
        r"\label{tab:ablation}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Ablation & Success & False disp. & Wrong obj. & Id switch & Occ. err. \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{label(row['method'])} & {fmt(row['success'])} & {fmt(row['false_disappearance'])} & "
            f"{fmt(row['wrong_object_contact'])} & {fmt(row['identity_switch'])} & {fmt(row['occlusion_error'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def stress_table(rows: list[dict[str, str]]) -> str:
    chosen = {
        "occlusion_aware_permanence_v5",
        "occlusion_aware_permanence_v4",
        "random_forest_occlusion_regressor",
        "hist_gradient_occlusion_regressor",
        "learned_occlusion_regressor",
        "oracle_state",
    }
    rows = [row for row in rows if row["method"] in chosen and row["split"] == "combined_extreme_stress"]
    rows = sorted(rows, key=lambda row: (float(row["stress_level"]), row["method"]))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Combined-extreme stress sweep.}",
        r"\label{tab:stress}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Level & Success & False disp. & Wrong obj. \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{label(row['method'])} & {fmt(row['stress_level'], 2)} & {fmt(row['mean_success_rate'])} & "
            f"{fmt(row['mean_false_disappearance_rate'])} & {fmt(row['mean_wrong_object_contact_rate'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def negative_cases_table(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Representative frozen negative cases for occlusion-aware permanence v5.}",
        r"\label{tab:negative-cases}",
        r"\scriptsize",
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Case & Split & Seed & Ep. & Success & Wrong & Occ. err. & Id switch \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['case']} & {label(row['split'])} & {row['seed']} & {row['episode']} & {row['success']} & "
            f"{row['wrong_object_contact']} & {fmt(row['mean_error_during_occlusion'])} & {fmt(row['identity_switch_rate'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def longtable(rows: list[dict[str, str]], columns: list[tuple], caption: str, label_name: str) -> str:
    # Long appendix tables use compact wrapped headers so the exhaustive archive stays legible.
    colspec = "@{}" + " ".join(column[1] for column in columns) + "@{}"
    header = " & ".join(tex_escape(column[2] if len(column) > 2 else column[0]) for column in columns) + r" \\"
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{0.8pt}",
        r"\renewcommand{\arraystretch}{0.88}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        rf"\begin{{longtable}}{{{colspec}}}",
        rf"\caption{{{tex_escape(caption)}}}\label{{{label_name}}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        vals = []
        for column in columns:
            name = column[0]
            vals.append(table_cell(name, row.get(name, "")))
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{longtable}", r"\endgroup", ""]
    return "\n".join(lines)


def render() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    decision, reason = parse_summary()
    metrics = read_csv(RESULTS / "metrics.csv")
    aggregate = read_csv(RESULTS / "aggregate_metrics.csv")
    fixed = read_csv(RESULTS / "fixed_risk_metrics.csv")
    ablation = read_csv(RESULTS / "ablation_aggregate_metrics.csv")
    stress = read_csv(RESULTS / "stress_sweep.csv")
    negative = read_csv(RESULTS / "negative_cases.csv")
    seed = read_csv(RESULTS / "raw_seed_metrics.csv")
    pairwise = read_csv(RESULTS / "pairwise_stats.csv")

    v5 = "occlusion_aware_permanence_v5"
    hard_v5 = row_lookup(aggregate, group="hard_splits", method=v5)
    hard_best = max(
        [row for row in aggregate if row["group"] == "hard_splits" and row["method"] not in {v5, "oracle_state"}],
        key=lambda row: float(row["success"]),
    )
    combined_v5 = row_lookup(aggregate, group="combined_and_extreme", method=v5)
    combined_best = max(
        [row for row in aggregate if row["group"] == "combined_and_extreme" and row["method"] not in {v5, "oracle_state"}],
        key=lambda row: float(row["success"]),
    )
    fixed_v5 = row_lookup(fixed, budget="0.10", method=v5)
    fixed_best = max(
        [row for row in fixed if row["budget"] == "0.10" and row["method"] not in {v5, "oracle_state"}],
        key=lambda row: float(row["success_at_budget"]),
    )
    text_col = r">{\raggedright\arraybackslash}p{%s\linewidth}"
    num_col = r">{\raggedleft\arraybackslash}p{%s\linewidth}"

    macros = rf"""
\newcommand{{\PaperDecision}}{{{tex_escape(decision)}}}
\newcommand{{\PaperReason}}{{{tex_escape(reason)}}}
\newcommand{{\MainRows}}{{{count_rows('self_occlusion_raw.csv'):,}}}
\newcommand{{\SeedRows}}{{{count_rows('raw_seed_metrics.csv'):,}}}
\newcommand{{\MetricRows}}{{{count_rows('metrics.csv'):,}}}
\newcommand{{\PairwiseRows}}{{{count_rows('pairwise_stats.csv'):,}}}
\newcommand{{\AblationRows}}{{{count_rows('self_occlusion_ablation_raw.csv'):,}}}
\newcommand{{\StressRows}}{{{count_rows('stress_sweep_raw.csv'):,}}}
\newcommand{{\NegativeRows}}{{{count_rows('negative_cases.csv'):,}}}
\newcommand{{\VFiveHardSuccess}}{{{fmt(hard_v5['success'])}}}
\newcommand{{\BestHardMethod}}{{{label(hard_best['method'])}}}
\newcommand{{\BestHardSuccess}}{{{fmt(hard_best['success'])}}}
\newcommand{{\VFiveCombinedSuccess}}{{{fmt(combined_v5['success'])}}}
\newcommand{{\BestCombinedMethod}}{{{label(combined_best['method'])}}}
\newcommand{{\BestCombinedSuccess}}{{{fmt(combined_best['success'])}}}
\newcommand{{\VFiveFixedRiskSuccess}}{{{fmt(fixed_v5['success_at_budget'])}}}
\newcommand{{\BestFixedRiskMethod}}{{{label(fixed_best['method'])}}}
\newcommand{{\BestFixedRiskSuccess}}{{{fmt(fixed_best['success_at_budget'])}}}
""".strip()
    write(GENERATED / "result_macros.tex", macros + "\n")
    write(GENERATED / "hard_aggregate_table.tex", aggregate_table(aggregate, "hard_splits"))
    write(GENERATED / "combined_aggregate_table.tex", aggregate_table(aggregate, "combined_and_extreme"))
    write(GENERATED / "selected_split_table.tex", selected_split_table(metrics, "combined_extreme_stress"))
    write(GENERATED / "fixed_risk_table.tex", fixed_risk_table(fixed))
    write(GENERATED / "ablation_table.tex", ablation_table(ablation))
    write(GENERATED / "stress_table.tex", stress_table(stress))
    write(GENERATED / "negative_cases_table.tex", negative_cases_table(negative))

    write(
        GENERATED / "full_metrics_longtable.tex",
        longtable(
            metrics,
            [
                ("method", text_col % "0.20", "Method"),
                ("split", text_col % "0.18", "Split"),
                ("mean_success_rate", num_col % "0.055", "Succ."),
                ("mean_false_disappearance_rate", num_col % "0.060", "False"),
                ("mean_wrong_object_contact_rate", num_col % "0.055", "Wrong"),
                ("mean_identity_switch_rate", num_col % "0.045", "ID"),
                ("mean_mean_error_during_occlusion", num_col % "0.055", "Err."),
            ],
            "Full split-method metrics.",
            "tab:full-metrics",
        ),
    )
    write(
        GENERATED / "full_aggregate_longtable.tex",
        longtable(
            aggregate,
            [
                ("group", text_col % "0.18", "Group"),
                ("method", text_col % "0.20", "Method"),
                ("success", num_col % "0.055", "Succ."),
                ("false_disappearance", num_col % "0.060", "False"),
                ("wrong_object_contact", num_col % "0.055", "Wrong"),
                ("identity_switch", num_col % "0.045", "ID"),
                ("occlusion_error", num_col % "0.055", "Err."),
            ],
            "Full aggregate metrics.",
            "tab:full-aggregate",
        ),
    )
    write(
        GENERATED / "all_seed_metrics_longtable.tex",
        longtable(
            seed,
            [
                ("method", text_col % "0.17", "Method"),
                ("split", text_col % "0.15", "Split"),
                ("seed", num_col % "0.045", "Seed"),
                ("success_rate", num_col % "0.055", "Succ."),
                ("false_disappearance_rate", num_col % "0.060", "False"),
                ("wrong_object_contact_rate", num_col % "0.055", "Wrong"),
                ("identity_switch_rate", num_col % "0.045", "ID"),
                ("mean_error_during_occlusion", num_col % "0.055", "Err."),
            ],
            "All seed-level metrics.",
            "tab:all-seeds",
        ),
    )
    write(
        GENERATED / "full_pairwise_longtable.tex",
        longtable(
            pairwise,
            [
                ("split", text_col % "0.19", "Split"),
                ("comparison", text_col % "0.25", "Comparison"),
                ("paired_success_diff", num_col % "0.060", "dSucc."),
                ("ci95_success_diff", num_col % "0.060", "CI95"),
                ("paired_occlusion_error_reduction", num_col % "0.060", "dErr."),
                ("paired_false_disappearance_reduction", num_col % "0.060", "dFalse"),
                ("reference_better_seeds", num_col % "0.050", "Wins"),
            ],
            "Full paired seed differences versus occlusion-aware permanence v5.",
            "tab:full-pairwise",
        ),
    )
    write(
        GENERATED / "full_ablation_longtable.tex",
        longtable(
            read_csv(RESULTS / "ablation_metrics.csv"),
            [
                ("method", text_col % "0.20", "Method"),
                ("split", text_col % "0.18", "Split"),
                ("mean_success_rate", num_col % "0.055", "Succ."),
                ("mean_false_disappearance_rate", num_col % "0.060", "False"),
                ("mean_wrong_object_contact_rate", num_col % "0.055", "Wrong"),
                ("mean_identity_switch_rate", num_col % "0.045", "ID"),
                ("mean_mean_error_during_occlusion", num_col % "0.055", "Err."),
            ],
            "Full ablation metrics.",
            "tab:full-ablation",
        ),
    )
    write(
        GENERATED / "full_stress_longtable.tex",
        longtable(
            stress,
            [
                ("method", text_col % "0.20", "Method"),
                ("split", text_col % "0.18", "Split"),
                ("stress_level", num_col % "0.045", "Lvl"),
                ("mean_success_rate", num_col % "0.055", "Succ."),
                ("mean_false_disappearance_rate", num_col % "0.060", "False"),
                ("mean_wrong_object_contact_rate", num_col % "0.055", "Wrong"),
                ("mean_identity_switch_rate", num_col % "0.045", "ID"),
            ],
            "Full stress sweep metrics.",
            "tab:full-stress",
        ),
    )

    terminal = f"""# Paper 71 Expanded Terminal Decision

Decision: {decision}

Reason: {reason}

Training rows: {count_rows('training_occlusion_examples.csv')}
Main method-evaluation rows: {count_rows('self_occlusion_raw.csv')}
Seed summary rows: {count_rows('raw_seed_metrics.csv')}
Split-method metric rows: {count_rows('metrics.csv')}
Ablation rows: {count_rows('self_occlusion_ablation_raw.csv')}
Stress rows: {count_rows('stress_sweep_raw.csv')}
Negative cases: {count_rows('negative_cases.csv')}

This decision is generated from frozen CSV artifacts, not hand-transcribed table values.
"""
    write(ROOT / "docs" / "paper71_expanded_terminal_decision_20260621.md", terminal)


if __name__ == "__main__":
    render()
