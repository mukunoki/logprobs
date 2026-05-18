#!/usr/bin/env python3
"""Build VerilogEval formal-result tables and figures for the paper."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PAPER_DIR = Path(__file__).resolve().parent
RESULTS_DIR = (
    PAPER_DIR.parent
    / "results"
    / "verilog_eval_formal_qwen35_9b_awq4_20260509_100trials_k10_w8_mt16384_s12345"
)
GROUP_CSV = PAPER_DIR.parent / "results" / "verilog_eval_problem_groups.csv"
DATASET_DIR = (
    PAPER_DIR.parent
    / "external"
    / "verilog-eval"
    / "dataset_code-complete-iccad2023"
)

METHODS = ["Gen", "MeanLogP", "TailLogP", "MinLogP"]
METHOD_LABELS = {
    "Gen": "Gen",
    "MeanLogP": "Mean",
    "TailLogP": "Tail",
    "MinLogP": "MinLogP",
}
COLORS = {
    "Gen": "#4D4D4D",
    "MeanLogP": "#0072B2",
    "TailLogP": "#E69F00",
    "MinLogP": "#009E73",
}

FEATURE_ORDER = [
    "simple logic",
    "procedural comb.",
    "arithmetic datapath",
    "sequential datapath",
    "FSM/control",
]
FEATURE_COLORS = {
    "simple logic": "#4E79A7",
    "procedural comb.": "#F28E2B",
    "arithmetic datapath": "#59A14F",
    "sequential datapath": "#B07AA1",
    "FSM/control": "#E15759",
}
FEATURE_LABELS = {
    "simple logic": "Simple logic",
    "procedural comb.": "Procedural comb.",
    "arithmetic datapath": "Arithmetic",
    "sequential datapath": "Sequential",
    "FSM/control": "FSM/control",
}
GAIN_METHODS = ["MeanLogP", "TailLogP", "MinLogP"]
CORRELATION_ROWS = [
    ("生成結果", "Any-success rate", "any_success"),
    ("生成結果", "成功候補数", "success_count"),
    ("生成結果", "Gen検証回数", "raw_Gen"),
    ("生成結果", "長大出力率", "length_rate"),
    ("コード特徴", "FSM/control flag", "fsm_word"),
    ("コード特徴", "参照実装LOC", "ref_loc"),
    ("コード特徴", "posedge/negedge flag", "posedge"),
    ("コード特徴", "continuous assign flag", "continuous_assign_only"),
]


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def short_label(problem: str) -> str:
    return problem.replace("Prob", "P").replace("_", "\n")


def reduction(gen: float, value: float) -> float:
    return 100.0 * (gen - value) / gen if gen else 0.0


def format_pct(value: float) -> str:
    return f"{value:+.1f}\\%"


def load_groups() -> dict[str, str]:
    with GROUP_CSV.open(encoding="utf-8") as handle:
        return {row["problem"]: row["group"] for row in csv.DictReader(handle)}


def load_records() -> list[dict]:
    return [json.loads(line) for line in (RESULTS_DIR / "candidates.jsonl").open()]


def load_trials() -> list[dict[str, str]]:
    return list(csv.DictReader((RESULTS_DIR / "trial_summary.csv").open()))


def strip_comments(text: str) -> str:
    return re.sub(r"//.*", "", text)


def static_features(problem: str) -> dict[str, float | int | str]:
    ref = (DATASET_DIR / f"{problem}_ref.sv").read_text(
        encoding="utf-8", errors="ignore"
    )
    prompt = (DATASET_DIR / f"{problem}_prompt.txt").read_text(
        encoding="utf-8", errors="ignore"
    )
    code = strip_comments(ref)
    lower = "\n".join([problem, prompt, ref]).lower()

    def count(pattern: str) -> int:
        return len(re.findall(pattern, code))

    posedge = int(bool(re.search(r"\b(posedge|negedge)\b", code)))
    always_count = count(r"\balways\b")
    assign_count = count(r"\bassign\b")
    procedural_comb = int(
        bool(re.search(r"\balways\s*@\s*\(\s*\*|always_comb", code))
        and not posedge
    )
    continuous_assign_only = int(always_count == 0 and assign_count > 0)
    fsm_word = int(
        bool(
            re.search(
                r"\b(fsm|state|next_state|lemmings|timer|gshare|serial|ps2|hdlc)\b",
                lower,
            )
        )
    )
    arithmetic_ops = count(r"(\+|-|\*|/|%|<<|>>)")

    return {
        "ref_loc": sum(1 for line in code.splitlines() if line.strip()),
        "ref_chars": len(ref),
        "prompt_chars": len(prompt),
        "assign_count": assign_count,
        "always_count": always_count,
        "case_count": count(r"\bcase[zx]?\b"),
        "if_count": count(r"\bif\b"),
        "posedge": posedge,
        "procedural_comb_static": procedural_comb,
        "continuous_assign_only": continuous_assign_only,
        "fsm_word": fsm_word,
        "arithmetic_ops": arithmetic_ops,
    }


def feature_group(row: dict) -> str:
    if row["fsm_word"] or row["group"] == "fsm_control":
        return "FSM/control"
    if row["procedural_comb_static"]:
        return "procedural comb."
    if row["posedge"]:
        return "sequential datapath"
    if row["arithmetic_ops"] >= 2 or row["group"] == "arithmetic_datapath":
        return "arithmetic datapath"
    return "simple logic"


def count_until(order: list[dict]) -> int:
    count = 0
    for record in order:
        count += 1
        if record.get("success"):
            return count
    return count


def validation_after_compile(order: list[dict]) -> int:
    count = 0
    for record in [item for item in order if item.get("compile_ok")]:
        count += 1
        if record.get("success"):
            return count
    return count


def build_trial_rows(records: list[dict], groups: dict[str, str]) -> list[dict]:
    by_trial: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        by_trial[(record["problem"], int(record["trial"]))].append(record)
    feature_by_problem = {}
    for problem, _trial in by_trial:
        if problem not in feature_by_problem:
            features = static_features(problem)
            features["feature_group"] = feature_group(
                {"group": groups.get(problem, "unknown"), **features}
            )
            feature_by_problem[problem] = features

    rows: list[dict] = []
    for (problem, trial), candidates in sorted(by_trial.items()):
        if len(candidates) != 10:
            continue
        ordered = sorted(candidates, key=lambda item: int(item["candidate"]))
        mean = sorted(ordered, key=lambda item: item.get("mean_logprob") or -1e99, reverse=True)
        tail = sorted(ordered, key=lambda item: item.get("tail_logprob") or -1e99, reverse=True)
        minp = sorted(ordered, key=lambda item: item.get("min_logprob") or -1e99, reverse=True)
        row = {
                "problem": problem,
                "trial": trial,
                "group": groups.get(problem, "unknown"),
                "any": any(item.get("success") for item in ordered),
                "compile": sum(bool(item.get("compile_ok")) for item in ordered),
                "success_count": sum(bool(item.get("success")) for item in ordered),
                "length": sum(item.get("finish_reason") == "length" for item in ordered),
                "raw_Gen": count_until(ordered),
                "raw_MeanLogP": count_until(mean),
                "raw_TailLogP": count_until(tail),
                "raw_MinLogP": count_until(minp),
                "cf_Gen": validation_after_compile(ordered),
                "cf_MeanLogP": validation_after_compile(mean),
                "cf_TailLogP": validation_after_compile(tail),
                "cf_MinLogP": validation_after_compile(minp),
            }
        row.update(feature_by_problem[problem])
        rows.append(row)
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def best_gain(values: dict[str, float]) -> tuple[str, float]:
    best = min(["MeanLogP", "TailLogP", "MinLogP"], key=lambda method: values[method])
    return best, reduction(values["Gen"], values[best])


def build_category_table(rows: list[dict]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{VerilogEval正式評価の問題種類別結果．各種類は分析用の分類であり，手法はこの分類情報を用いない．成功率はany-success rate，通過数は平均コンパイル通過候補数，打切は\texttt{max\_tokens}到達候補数である．Genは平均検証回数，Mean，Tail，MinLogPはGen比削減率（\%）を示す．CF最良はcompile-filter後のvalidation実行回数における最良削減率である．}",
        r"\label{tab:verilog_eval_category}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrrrrrrr@{}}",
        r"\hline",
        r"種類 & 問題数 & 成功率 & 通過数 & 打切 & Gen & Mean & Tail & MinLogP & CF最良 \\",
        r"\hline",
    ]
    for group in sorted({row["group"] for row in rows}):
        selected = [row for row in rows if row["group"] == group]
        raw = {
            method: mean([row[f"raw_{method}"] for row in selected])
            for method in METHODS
        }
        cf = {
            method: mean([row[f"cf_{method}"] for row in selected])
            for method in METHODS
        }
        cf_best, cf_gain = best_gain(cf)
        lines.append(
            " & ".join(
                [
                    tex_escape(group),
                    str(len({row["problem"] for row in selected})),
                    f"{mean([row['any'] for row in selected]):.3f}",
                    f"{mean([row['compile'] for row in selected]):.2f}",
                    f"{mean([row['length'] for row in selected]):.2f}",
                    f"{raw['Gen']:.2f}",
                    format_pct(reduction(raw["Gen"], raw["MeanLogP"])),
                    format_pct(reduction(raw["Gen"], raw["TailLogP"])),
                    format_pct(reduction(raw["Gen"], raw["MinLogP"])),
                    f"{METHOD_LABELS[cf_best]} {cf_gain:+.1f}\\%",
                ]
            )
            + r" \\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    (PAPER_DIR / "verilog_eval_category_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def problem_summaries(rows: list[dict]) -> list[dict]:
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_problem[row["problem"]].append(row)

    summaries: list[dict] = []
    for problem, selected in sorted(by_problem.items()):
        raw = {method: mean([row[f"raw_{method}"] for row in selected]) for method in METHODS}
        cf = {method: mean([row[f"cf_{method}"] for row in selected]) for method in METHODS}
        summary = {
            "problem": problem,
            "group": selected[0]["group"],
            "feature_group": selected[0]["feature_group"],
            "any_success": mean([row["any"] for row in selected]),
            "compile": mean([row["compile"] for row in selected]),
            "success_count": mean([row["success_count"] for row in selected]),
            "length": mean([row["length"] for row in selected]),
            "length_rate": mean([row["length"] for row in selected]) / 10.0,
        }
        for key in [
            "ref_loc",
            "ref_chars",
            "prompt_chars",
            "assign_count",
            "always_count",
            "case_count",
            "if_count",
            "posedge",
            "procedural_comb_static",
            "continuous_assign_only",
            "fsm_word",
            "arithmetic_ops",
        ]:
            summary[key] = selected[0][key]
        for method in METHODS:
            summary[f"raw_{method}"] = raw[method]
            summary[f"cf_{method}"] = cf[method]
        for method in GAIN_METHODS:
            summary[f"raw_gain_{method}"] = reduction(raw["Gen"], raw[method])
            summary[f"cf_gain_{method}"] = reduction(cf["Gen"], cf[method])
        summary["raw_best_gain"] = max(
            summary[f"raw_gain_{method}"] for method in GAIN_METHODS
        )
        summary["cf_best_gain"] = max(
            summary[f"cf_gain_{method}"] for method in GAIN_METHODS
        )
        summaries.append(summary)
    return summaries


def build_feature_table(rows: list[dict]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{VerilogEval正式評価のコード特徴別結果．分類は公開されている問題文・問題名・参照実装の構文特徴に基づく分析用分類であり，順位付け手法はこの分類情報を用いない．成功率はany-success rate，通過数は平均コンパイル通過候補数，打切は\texttt{max\_tokens}到達候補数である．Genは平均検証試行回数，Mean，Tail，MinLogPはGen比削減率（\%），CF最良はcompile-filter後のsimulation実行回数における最良削減率を示す．}",
        r"\label{tab:verilog_eval_feature}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrrrrrrr@{}}",
        r"\hline",
        r"コード特徴 & 問題数 & 成功率 & 通過数 & 打切 & Gen & Mean & Tail & MinLogP & CF最良 \\",
        r"\hline",
    ]
    for group in FEATURE_ORDER:
        selected = [row for row in rows if row["feature_group"] == group]
        if not selected:
            continue
        raw = {
            method: mean([row[f"raw_{method}"] for row in selected])
            for method in METHODS
        }
        cf = {
            method: mean([row[f"cf_{method}"] for row in selected])
            for method in METHODS
        }
        cf_best, cf_gain = best_gain(cf)
        lines.append(
            " & ".join(
                [
                    tex_escape(FEATURE_LABELS[group]),
                    str(len({row["problem"] for row in selected})),
                    f"{mean([row['any'] for row in selected]):.3f}",
                    f"{mean([row['compile'] for row in selected]):.2f}",
                    f"{mean([row['length'] for row in selected]):.2f}",
                    f"{raw['Gen']:.2f}",
                    format_pct(reduction(raw["Gen"], raw["MeanLogP"])),
                    format_pct(reduction(raw["Gen"], raw["TailLogP"])),
                    format_pct(reduction(raw["Gen"], raw["MinLogP"])),
                    f"{METHOD_LABELS[cf_best]} {cf_gain:+.1f}\\%",
                ]
            )
            + r" \\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    (PAPER_DIR / "verilog_eval_feature_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        rank = (index + end) / 2.0 + 1.0
        for pos in range(index, end + 1):
            ranks[order[pos]] = rank
        index = end + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    xmean = mean(xs)
    ymean = mean(ys)
    xden = sum((value - xmean) ** 2 for value in xs) ** 0.5
    yden = sum((value - ymean) ** 2 for value in ys) ** 0.5
    if xden == 0 or yden == 0:
        return 0.0
    return sum((x - xmean) * (y - ymean) for x, y in zip(xs, ys)) / (xden * yden)


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rankdata(xs), rankdata(ys))


def build_correlation_table(summaries: list[dict]) -> None:
    target = [row["raw_best_gain"] for row in summaries]
    lines = [
        r"\begin{table}[t]",
        r"\caption{VerilogEval 156問における効果量と特徴量のSpearman相関．効果量はMean，Tail，MinLogPのうち各問題で最も大きいGen比削減率であり，提案手法の選択規則ではなく分析用の指標である．}",
        r"\label{tab:verilog_eval_correlation}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llr@{}}",
        r"\hline",
        r"種別 & 特徴量 & $\rho$ \\",
        r"\hline",
    ]
    for kind, label, key in CORRELATION_ROWS:
        values = [float(row[key]) for row in summaries]
        lines.append(
            f"{tex_escape(kind)} & {tex_escape(label)} & {spearman(values, target):+.3f} \\\\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    (PAPER_DIR / "verilog_eval_correlation_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def build_feature_analysis_figure(summaries: list[dict]) -> None:
    fig, (ax_box, ax_scatter) = plt.subplots(
        1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )

    positions = np.arange(len(FEATURE_ORDER))
    offsets = np.linspace(-0.22, 0.22, len(GAIN_METHODS))
    for method, offset in zip(GAIN_METHODS, offsets):
        data = [
            [row[f"raw_gain_{method}"] for row in summaries if row["feature_group"] == group]
            for group in FEATURE_ORDER
        ]
        bp = ax_box.boxplot(
            data,
            positions=positions + offset,
            widths=0.16,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            whis=(0, 100),
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markeredgewidth": 0.6,
                "markersize": 2.5,
            },
            medianprops={"color": "black", "linewidth": 0.7},
            whiskerprops={"color": "0.25", "linewidth": 0.55},
            capprops={"color": "0.25", "linewidth": 0.55},
        )
        for box in bp["boxes"]:
            box.set_facecolor(COLORS[method])
            box.set_edgecolor("black")
            box.set_linewidth(0.5)
    ax_box.axhline(0, color="0.25", linewidth=0.7)
    ax_box.set_ylabel("Reduction vs. Gen (%)", fontsize=8)
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels([FEATURE_LABELS[group] for group in FEATURE_ORDER], rotation=25, ha="right", fontsize=7)
    ax_box.tick_params(axis="y", labelsize=7)
    ax_box.grid(axis="y", linestyle=":", linewidth=0.5, color="0.78")
    ax_box.set_title("(a) Per-problem reduction", fontsize=8, pad=2)

    for group in FEATURE_ORDER:
        selected = [row for row in summaries if row["feature_group"] == group]
        ax_scatter.scatter(
            [row["raw_Gen"] for row in selected],
            [row["raw_best_gain"] for row in selected],
            s=13,
            color=FEATURE_COLORS[group],
            marker="o",
            edgecolors="none",
            label=FEATURE_LABELS[group],
        )
    ax_scatter.axhline(0, color="0.25", linewidth=0.7)
    ax_scatter.set_xlabel("Gen verifications", fontsize=8)
    ax_scatter.set_ylabel("Best reduction (%)", fontsize=8)
    ax_scatter.set_xlim(0.7, 10.3)
    ax_scatter.set_xticks(range(1, 11))
    ax_scatter.tick_params(axis="both", labelsize=7)
    ax_scatter.grid(True, linestyle=":", linewidth=0.5, color="0.78")
    ax_scatter.set_title("(b) Effect vs. Gen cost", fontsize=8, pad=2)

    method_handles = [
        Patch(facecolor=COLORS[method], edgecolor="black", label=METHOD_LABELS[method])
        for method in GAIN_METHODS
    ]
    group_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=4,
            color=FEATURE_COLORS[group],
            label=FEATURE_LABELS[group],
        )
        for group in FEATURE_ORDER
    ]
    fig.legend(
        handles=method_handles + group_handles,
        loc="lower center",
        ncol=4,
        fontsize=7,
        frameon=True,
        framealpha=1.0,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1), pad=0.35)
    fig.savefig(PAPER_DIR / "verilog_eval_feature_analysis.eps", format="eps")
    fig.savefig(PAPER_DIR / "verilog_eval_feature_analysis.png", dpi=250)
    plt.close(fig)


def main() -> None:
    groups = load_groups()
    rows = build_trial_rows(load_records(), groups)
    summaries = problem_summaries(rows)
    build_category_table(rows)
    build_feature_table(rows)
    build_correlation_table(summaries)
    build_feature_analysis_figure(summaries)


if __name__ == "__main__":
    main()
