#!/usr/bin/env python3
"""Create representative token loss distribution plots for selected benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent
RESULT_DIR = (
    REPO_ROOT
    / "results/selected20_9b_k10_by_problem_qwen35_9b_awq4_20260430_222029_100trials_w5_mt4096"
)
BY_PROBLEM_DIR = RESULT_DIR / "by_problem"
ANALYSIS_PATH = RESULT_DIR / "selected20_9b_k10_by_problem_method_analysis.json"
OUT_PATH = PAPER_DIR / "token_loss_distribution.png"

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

SELECTED_PROBLEMS = [
    "pareval_sort_ignore_zero_i32",
    "stencil3d_mixed_7pt",
    "crop2d_strided_u8",
    "floyd_warshall_blocked",
]


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


def survival_curve(losses: list[float], x: np.ndarray) -> np.ndarray:
    if not losses:
        return np.zeros(len(x))
    arr = np.sort(np.asarray(losses, dtype=float))
    idx = np.searchsorted(arr, x, side="left")
    return (len(arr) - idx).astype(float) / float(len(arr))


def load_tail_gains() -> dict[str, float]:
    with open(ANALYSIS_PATH, encoding="utf-8") as f:
        analysis = json.load(f)
    gains = {}
    for problem, item in analysis.items():
        gen = float(item["methods"]["Gen"]["avg_tests"])
        tail = float(item["methods"]["Tail"]["avg_tests"])
        gains[problem] = gen - tail
    return gains


def write_xbb(output_path: Path, fig) -> None:
    width = fig.get_figwidth() * 72
    height = fig.get_figheight() * 72
    output_path.with_suffix(".xbb").write_text(
        "\n".join([
            f"%%Title: {output_path.name}",
            "%%Creator: paper/create_token_loss_distribution_figure.py",
            f"%%BoundingBox: 0 0 {int(round(width))} {int(round(height))}",
            f"%%HiResBoundingBox: 0.000000 0.000000 {width:.6f} {height:.6f}",
            "",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    tail_gains = load_tail_gains()
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.25), sharex=True, sharey=True)
    x = np.linspace(0.0, 16.0, 161)
    total_tokens = 0

    for ax, problem in zip(axes.ravel(), SELECTED_PROBLEMS):
        tail_gain = tail_gains[problem]
        pass_losses: list[float] = []
        fail_losses: list[float] = []
        with open(BY_PROBLEM_DIR / f"{safe_name(problem)}.json", encoding="utf-8") as f:
            data = json.load(f)
        for trial in data["candidate_sets"]:
            for cand in trial["candidates"]:
                target = pass_losses if cand["success"] else fail_losses
                for logp in cand.get("token_logprobs", []):
                    if logp is not None:
                        target.append(-float(logp))

        total_tokens += len(pass_losses) + len(fail_losses)
        pass_survival = survival_curve(pass_losses, x)
        fail_survival = survival_curve(fail_losses, x)

        ax.plot(
            x,
            pass_survival,
            color="#238b45",
            linewidth=1.05,
            marker="o",
            markersize=1.6,
            markevery=20,
            label="Pass tokens",
        )
        ax.plot(
            x,
            fail_survival,
            color="#cb181d",
            linestyle="--",
            linewidth=1.05,
            marker="x",
            markersize=1.9,
            markevery=20,
            label="Fail tokens",
        )
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1.05)
        ax.set_xlim(0.0, 16.0)
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.grid(linestyle=":", linewidth=0.5, color="#c7c7c7")
        ax.set_title(
            f"{PROBLEM_LABELS.get(problem, problem[:8])}\n$\\Delta_T$={tail_gain:+.2f}",
            fontsize=5.6,
            pad=3,
        )

    for row in range(2):
        axes[row, 0].set_ylabel("Frac. tokens\nloss >= x")
    for ax in axes[-1, :]:
        ax.set_xlabel("Token loss (-logP)")

    handles = [
        plt.Line2D([0], [0], color="#238b45", linewidth=1.2, marker="o", markersize=2.8, label="Pass"),
        plt.Line2D([0], [0], color="#cb181d", linestyle="--", linewidth=1.2, marker="x", markersize=3.2, label="Fail"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0.02, 0.07, 0.995, 0.995), h_pad=1.05, w_pad=0.45)

    fig.savefig(OUT_PATH, dpi=300)
    fig.savefig(OUT_PATH.with_suffix(".eps"), format="eps")
    write_xbb(OUT_PATH, fig)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
