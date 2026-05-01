#!/usr/bin/env python3
"""Create 20-problem case figures for the 9B k=10 evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np


PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent
RESULT_DIR = (
    REPO_ROOT
    / "results/selected20_9b_k10_by_problem_qwen35_9b_awq4_20260430_222029_100trials_w5_mt4096"
)
BY_PROBLEM_DIR = RESULT_DIR / "by_problem"
HIGH_CONF_THRESHOLD = 0.75


CASES = [
    (
        "LogP-Separable",
        "logp_separable",
        [
            "pareval_sort_ignore_zero_i32",
            "utf8_validate_strict",
            "pareval_largest_component_i32",
            "csr_spmv_axpy_dot",
            "transpose_strided_f64",
        ],
    ),
    (
        "Tail-Sensitive",
        "tail_sensitive",
        [
            "pareval_convex_hull_perimeter_f64",
            "topk_ignore_nan_f32",
            "stencil3d_mixed_7pt",
            "stencil2d_halo5",
            "cholesky_spd_f64",
        ],
    ),
    (
        "Weak-Signal",
        "weak_signal",
        [
            "crop2d_strided_u8",
            "matrix_transpose_cache",
            "rmsnorm_mixed",
            "heap_sort_implementation",
            "quadratic_roots_stable_f64",
        ],
    ),
    (
        "Unreliable-Signal",
        "unreliable_signal",
        [
            "conv2d_3x3_multi_channel",
            "radix_sort_u32_pairs",
            "floyd_warshall_blocked",
            "lower_tri_solve_strided_f64",
            "banded_edit_distance_i32",
        ],
    ),
]


PROBLEM_LABELS = {
    "banded_edit_distance_i32": "Banded\nEdit",
    "cholesky_spd_f64": "Cholesky\nSPD",
    "conv2d_3x3_multi_channel": "Conv2D",
    "crop2d_strided_u8": "Crop2D",
    "csr_spmv_axpy_dot": "SpMV",
    "floyd_warshall_blocked": "Floyd",
    "heap_sort_implementation": "HeapSort",
    "lower_tri_solve_strided_f64": "Triangular\nSolve",
    "matrix_transpose_cache": "Transpose",
    "pareval_convex_hull_perimeter_f64": "Convex\nHull",
    "pareval_largest_component_i32": "Largest\nComponent",
    "pareval_sort_ignore_zero_i32": "Sort\nNonzero",
    "quadratic_roots_stable_f64": "Quadratic\nRoots",
    "radix_sort_u32_pairs": "Radix",
    "rmsnorm_mixed": "RMSNorm",
    "stencil2d_halo5": "Stencil2D",
    "stencil3d_mixed_7pt": "Stencil3D",
    "topk_ignore_nan_f32": "Top-k\nNaN",
    "transpose_strided_f64": "Strided\nTranspose",
    "utf8_validate_strict": "UTF-8\nValidate",
}


METHODS = ["Gen", "Mean", "Tail", "MinLogP"]
COLORS = ["#f39c12", "#27ae60", "#8e44ad", "#2c7fb8"]


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "legend.fontsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "ps.useafm": True,
    "pdf.use14corefonts": True,
    "lines.linewidth": 1.15,
})


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def load_problem(problem: str) -> list[dict]:
    with open(BY_PROBLEM_DIR / f"{safe_name(problem)}.json", encoding="utf-8") as f:
        return json.load(f)["candidate_sets"]


def valid_logprobs(candidate: dict) -> np.ndarray:
    return np.array([float(x) for x in candidate.get("token_logprobs", []) if x is not None], dtype=float)


def tail_score(candidate: dict) -> float:
    vals = valid_logprobs(candidate)
    if len(vals) == 0:
        return float("-inf")
    return float(np.quantile(vals, 0.25))


def min_logp_score(candidate: dict) -> float:
    vals = valid_logprobs(candidate)
    if len(vals) == 0:
        return float("-inf")
    return float(np.min(vals))


def method_order(candidates: list[dict], method: str) -> list[int]:
    if method == "Gen":
        return list(range(len(candidates)))
    if method == "Mean":
        return sorted(range(len(candidates)), key=lambda i: (-float(candidates[i]["confidence"]), i))
    if method == "Tail":
        return sorted(range(len(candidates)), key=lambda i: (-tail_score(candidates[i]), i))
    if method == "MinLogP":
        return sorted(range(len(candidates)), key=lambda i: (-min_logp_score(candidates[i]), i))
    raise ValueError(method)


def collect_tests(trials: list[dict]) -> dict[str, list[float]]:
    out = {method: [] for method in METHODS}
    for trial in trials:
        candidates = trial["candidates"]
        for method in METHODS:
            tests = float(len(candidates))
            for rank, idx in enumerate(method_order(candidates, method), start=1):
                if candidates[idx]["success"]:
                    tests = float(rank)
                    break
            out[method].append(tests)
    return out


def draw_boxplot(ax, series_list: list[list[float]]) -> None:
    bp = ax.boxplot(
        series_list,
        positions=np.arange(len(series_list)),
        widths=0.56,
        patch_artist=True,
        whis=1.5,
        showfliers=False,
        showmeans=True,
        meanprops=dict(marker="D", markeredgecolor="white", markeredgewidth=0.4, markersize=3.0),
        medianprops=dict(color="#222222", linewidth=0.9),
        whiskerprops=dict(color="#555555", linewidth=0.8),
        capprops=dict(color="#555555", linewidth=0.8),
        boxprops=dict(edgecolor="#555555", linewidth=0.8),
    )
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
    for mean_marker, color in zip(bp["means"], COLORS):
        mean_marker.set_markerfacecolor(color)
        mean_marker.set_markeredgecolor("white")


def visible_box_ylim(series_list: list[list[float]]) -> tuple[float, float]:
    visible_values: list[float] = []
    for series in series_list:
        values = np.array(series, dtype=float)
        if len(values) == 0:
            continue
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        whisker_values = values[(values >= lower_bound) & (values <= upper_bound)]
        if len(whisker_values) == 0:
            whisker_values = values
        visible_values.extend([
            float(np.min(whisker_values)),
            float(np.max(whisker_values)),
            float(np.mean(values)),
        ])
    if not visible_values:
        return 0.5, 1.5
    ymin = max(0.5, math.floor((min(visible_values) - 0.2) * 2) / 2)
    ymax = math.ceil((max(visible_values) + 0.25) * 2) / 2
    if ymax <= ymin:
        ymax = ymin + 0.5
    return ymin, ymax


def write_xbb(output_path: Path, fig) -> None:
    width = fig.get_figwidth() * 72
    height = fig.get_figheight() * 72
    output_path.with_suffix(".xbb").write_text(
        "\n".join([
            f"%%Title: {output_path.name}",
            "%%Creator: paper/create_case_figures_20problems.py",
            f"%%BoundingBox: 0 0 {int(round(width))} {int(round(height))}",
            f"%%HiResBoundingBox: 0.000000 0.000000 {width:.6f} {height:.6f}",
            "",
        ]),
        encoding="utf-8",
    )


def plot_case(case_name: str, slug: str, problems: list[str]) -> None:
    datasets = {problem: load_problem(problem) for problem in problems}
    output_path = PAPER_DIR / f"cbba_case_{slug}.png"
    vertical_gap = 0.52 if slug == "logp_separable" else 0.25

    fig = plt.figure(figsize=(7.15, 3.55))
    gs_main = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.0, 1.12],
        hspace=vertical_gap,
        left=0.065,
        right=0.992,
        bottom=0.16,
        top=0.88,
    )
    gs_top = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_main[0], wspace=0.32)
    gs_bottom = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_main[1], wspace=0.32)

    for col, problem in enumerate(problems):
        ax = fig.add_subplot(gs_top[0, col])
        success_conf = []
        failure_conf = []
        for trial in datasets[problem]:
            for cand in trial["candidates"]:
                (success_conf if cand["success"] else failure_conf).append(float(cand["confidence"]))

        hc_total = sum(1 for trial in datasets[problem] for cand in trial["candidates"] if cand["confidence"] >= HIGH_CONF_THRESHOLD)
        hc_success = sum(
            1
            for trial in datasets[problem]
            for cand in trial["candidates"]
            if cand["confidence"] >= HIGH_CONF_THRESHOLD and cand["success"]
        )
        hc_rate = 100.0 * hc_success / hc_total if hc_total else 0.0
        hc_fail = [conf for conf in failure_conf if conf >= HIGH_CONF_THRESHOLD]

        bins = np.arange(0.50, 1.01, 0.03)
        ax.axvspan(HIGH_CONF_THRESHOLD, 1.00, color="#f0f0f0", zorder=0)
        n_succ, _, _ = ax.hist(
            success_conf,
            bins=bins,
            histtype="step",
            color="#238b45",
            linewidth=1.05,
            zorder=3,
        )
        n_fail, _, _ = ax.hist(
            failure_conf,
            bins=bins,
            histtype="step",
            color="#cb181d",
            linestyle="--",
            linewidth=1.05,
            zorder=4,
        )
        ax.axvline(HIGH_CONF_THRESHOLD, color="#555555", linestyle=":", linewidth=0.85)
        max_count = max(max(n_fail) if len(n_fail) else 0, max(n_succ) if len(n_succ) else 0, 1)
        if hc_fail:
            ax.scatter(
                hc_fail,
                np.full(len(hc_fail), -max_count * 0.07),
                marker="|",
                s=24,
                linewidths=0.75,
                color="#9e1b1b",
                zorder=5,
            )
        ax.set_ylim(-max_count * 0.14, max_count * 1.25)
        ax.set_xlim(0.50, 1.00)
        ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_xticklabels(["0.5", "0.75", "1.0"])
        ax.set_xlabel("Conf.", fontsize=6, labelpad=1)
        ax.set_title(
            f"{PROBLEM_LABELS.get(problem, problem[:8])}\n(HC {hc_rate:.1f}%)",
            fontsize=6.2,
            pad=3,
            linespacing=0.95,
        )
        if col == 0:
            ax.set_ylabel("Candidates", fontsize=7)
        ax.grid(axis="y", linestyle=":", linewidth=0.5)
        ax.tick_params(labelsize=5.7, pad=1)

    for col, problem in enumerate(problems):
        ax = fig.add_subplot(gs_bottom[0, col])
        distributions = collect_tests(datasets[problem])
        series = [distributions[method] for method in METHODS]
        draw_boxplot(ax, series)
        ax.set_ylim(*visible_box_ylim(series))
        ax.set_xticks(np.arange(len(METHODS)))
        ax.set_xticklabels(METHODS, fontsize=5.7, rotation=32, ha="right")
        if col == 0:
            ax.set_ylabel("Tests", fontsize=7)
        ax.grid(axis="y", linestyle=":", linewidth=0.5)
        ax.tick_params(labelsize=5.8, pad=1)

    fig.suptitle(case_name, fontsize=8.5, weight="bold", y=0.985)
    confidence_handles = [
        plt.Line2D([0], [0], color="#238b45", linewidth=1.1),
        plt.Line2D([0], [0], color="#cb181d", linestyle="--", linewidth=1.1),
        plt.Line2D([0], [0], marker="|", linestyle="none", color="#9e1b1b", markersize=6),
    ]
    method_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none") for color in COLORS]
    fig.legend(
        confidence_handles + method_handles,
        ["Pass", "Fail", "HC fail"] + METHODS,
        loc="lower center",
        fontsize=6.2,
        frameon=False,
        ncol=7,
        bbox_to_anchor=(0.5, 0.015),
        columnspacing=0.9,
        handletextpad=0.35,
    )

    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".eps"), format="eps")
    write_xbb(output_path, fig)
    print(f"saved {output_path}")


def main() -> None:
    for case_name, slug, problems in CASES:
        plot_case(case_name, slug, problems)


if __name__ == "__main__":
    main()
