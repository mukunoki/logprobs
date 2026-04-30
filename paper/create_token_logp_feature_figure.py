#!/usr/bin/env python3
"""Create a 20-panel feature-space figure for MeanLogP and TailLogP."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent
RESULT_DIR = (
    REPO_ROOT
    / "results/selected20_9b_k10_qwen35_9b_awq4_20260430_172159_50trials_w5_mt4096"
)
BY_PROBLEM_DIR = RESULT_DIR / "by_problem"
ANALYSIS_PATH = RESULT_DIR / "selected20_9b_k10_method_analysis.json"
OUT_PATH = PAPER_DIR / "token_logp_feature_space.png"

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


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 7,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "ps.useafm": True,
    "pdf.use14corefonts": True,
})


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def load_analysis_order() -> list[tuple[str, float]]:
    with open(ANALYSIS_PATH, encoding="utf-8") as f:
        analysis = json.load(f)
    rows = []
    for problem, item in analysis.items():
        gen = float(item["methods"]["Gen"]["avg_tests"])
        tail = float(item["methods"]["Tail"]["avg_tests"])
        rows.append((problem, gen - tail))
    return sorted(rows, key=lambda row: row[1], reverse=True)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("-inf")
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def tail_logp(candidate: dict) -> float:
    return percentile([float(x) for x in candidate.get("token_logprobs", []) if x is not None], 0.25)


def auc(scores: list[float], labels: list[bool]) -> float | None:
    pos = [score for score, label in zip(scores, labels) if label]
    neg = [score for score, label in zip(scores, labels) if not label]
    if not pos or not neg:
        return None
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def write_xbb(output_path: Path, fig) -> None:
    width = fig.get_figwidth() * 72
    height = fig.get_figheight() * 72
    output_path.with_suffix(".xbb").write_text(
        "\n".join([
            f"%%Title: {output_path.name}",
            "%%Creator: paper/create_token_logp_feature_figure.py",
            f"%%BoundingBox: 0 0 {int(round(width))} {int(round(height))}",
            f"%%HiResBoundingBox: 0.000000 0.000000 {width:.6f} {height:.6f}",
            "",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    ordered = load_analysis_order()
    fig, axes = plt.subplots(5, 4, figsize=(7.15, 8.25), sharex=True, sharey=True)
    methods_xlim = (0.50, 1.005)
    methods_ylim = (0.40, 1.005)

    for ax, (problem, tail_gain) in zip(axes.ravel(), ordered):
        mean_scores: list[float] = []
        tail_scores: list[float] = []
        labels: list[bool] = []
        with open(BY_PROBLEM_DIR / f"{safe_name(problem)}.json", encoding="utf-8") as f:
            data = json.load(f)
        for trial in data["candidate_sets"]:
            for cand in trial["candidates"]:
                mean_scores.append(float(cand["confidence"]))
                tail_scores.append(math.exp(tail_logp(cand)))
                labels.append(bool(cand["success"]))

        pass_idx = np.array(labels, dtype=bool)
        x = np.array(mean_scores)
        y = np.array(tail_scores)
        ax.scatter(
            x[~pass_idx],
            y[~pass_idx],
            s=10,
            marker="x",
            color="#cb181d",
            linewidths=0.65,
            label="Fail",
        )
        ax.scatter(
            x[pass_idx],
            y[pass_idx],
            s=10,
            marker="o",
            facecolors="none",
            edgecolors="#238b45",
            linewidths=0.65,
            label="Pass",
        )
        ax.set_xlim(*methods_xlim)
        ax.set_ylim(*methods_ylim)
        ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_yticks([0.5, 0.75, 1.0])
        ax.grid(linestyle=":", linewidth=0.5, color="#c7c7c7")

        mean_auc = auc(mean_scores, labels)
        tail_auc = auc(tail_scores, labels)
        mean_text = "nan" if mean_auc is None else f"{mean_auc:.2f}"
        tail_text = "nan" if tail_auc is None else f"{tail_auc:.2f}"
        ax.set_title(
            f"{PROBLEM_LABELS.get(problem, problem[:8])}\nAUC {mean_text}/{tail_text}, $\\Delta_T$={tail_gain:+.2f}",
            fontsize=5.7,
            pad=3,
        )

    for row in range(5):
        axes[row, 0].set_ylabel("TailLogP")
    for ax in axes[-1, :]:
        ax.set_xlabel("MeanLogP")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                   markeredgecolor="#238b45", markersize=5, label="Pass"),
        plt.Line2D([0], [0], marker="x", linestyle="none", color="#cb181d",
                   markersize=5, label="Fail"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.text(
        0.995,
        0.018,
        "20 benchmarks, panels sorted by Tail gain",
        ha="right",
        va="bottom",
        fontsize=6.5,
    )
    fig.tight_layout(rect=(0.04, 0.055, 0.995, 0.995), h_pad=1.15, w_pad=0.85)

    fig.savefig(OUT_PATH, dpi=300)
    fig.savefig(OUT_PATH.with_suffix(".eps"), format="eps")
    write_xbb(OUT_PATH, fig)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
