#!/usr/bin/env python3
"""
論文用の図を見やすく作り直す（選抜12問・3グループ版）

既存結果と新規ベンチを組み合わせ，コード自体の傾向で3グループに分類:
- Indexing and Robust Numerics
- Ordered Algorithm Kernels
- Numeric and Validation Kernels

各ベンチマークごとに：
1. 確信度分布（4問のヒストグラム、confidence=0.75 の閾値線付き）
2. テスト数（Gen, Mean, Tail の3手法比較）

横軸ラベルは最下段のみ表示

出力:
- 図1: Indexing and Robust Numerics
- 図2: Ordered Algorithm Kernels
- 図3: Numeric and Validation Kernels
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

HIGH_CONF_THRESHOLD = 0.75

# フォント設定
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "ps.useafm": True,  # EPS出力用
    "pdf.use14corefonts": True,  # PDF/EPS出力用
    "lines.linewidth": 1.3,
})

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent
FIGURE_OUT_DIR = Path(os.environ.get("CBBA_FIGURE_OUT_DIR", PAPER_DIR))


RESULT_PATHS = {
    "paper_easy4": REPO_ROOT / "results/paper12_sameprompt_20260429_092647_100trials_w4_mt2048/same_prompt_eval_paper_easy4_b5.json",
    "paper_medium4": REPO_ROOT / "results/paper12_sameprompt_20260429_092647_100trials_w4_mt2048/same_prompt_eval_paper_medium4_b5.json",
    "paper_hard4": REPO_ROOT / "results/paper12_sameprompt_20260429_092647_100trials_w4_mt2048/same_prompt_eval_paper_hard4_b5.json",
    "paper_stride4": REPO_ROOT / "results/stride4_sameprompt_20260429_182351_50trials_w4_mt2048/same_prompt_eval_paper_stride4_b5.json",
    "paper_numeric2": REPO_ROOT / "results/numeric2_sameprompt_20260429_193832_50trials_w4_mt2048/same_prompt_eval_paper_numeric2_b5.json",
    "paper_tailcase4": REPO_ROOT / "results/tailcase4_sameprompt_20260429_173608_50trials_w4_mt2048/same_prompt_eval_paper_tailcase4_b5.json",
    "paper_general_selected2": REPO_ROOT / "results/general_selected2_sameprompt_20260429_222827_50trials_w4_mt4096/same_prompt_eval_paper_general_selected2_b5.json",
    "paper_general4": REPO_ROOT / "results/general4_sameprompt_20260429_205621_50trials_w4_mt4096/same_prompt_eval_paper_general4_b5.json",
    "paper_general_numeric4": REPO_ROOT / "results/general_numeric4_sameprompt_20260429_232617_50trials_w4_mt4096/same_prompt_eval_paper_general_numeric4_b5.json",
    "paper_pareval4": REPO_ROOT / "results/pareval4_sameprompt_20260430_111507_50trials_w4_mt4096/same_prompt_eval_paper_pareval4_b5.json",
}

PROBLEM_SOURCES = {
    "crop2d_strided_u8": "paper_stride4",
    "transpose_strided_f64": "paper_stride4",
    "cholesky_spd_f64": "paper_general_selected2",
    "pareval_convex_hull_perimeter_f64": "paper_pareval4",
    "radix_sort_u32_pairs": "paper_hard4",
    "floyd_warshall_blocked": "paper_medium4",
    "heap_sort_implementation": "paper_easy4",
    "pareval_sort_ignore_zero_i32": "paper_pareval4",
    "lower_tri_solve_strided_f64": "paper_numeric2",
    "stencil3d_mixed_7pt": "paper_hard4",
    "rmsnorm_mixed": "paper_medium4",
    "utf8_validate_strict": "paper_general_selected2",
}

GROUPS = [
    ("Indexing and Robust Numerics", [
        "crop2d_strided_u8",
        "transpose_strided_f64",
        "cholesky_spd_f64",
        "pareval_convex_hull_perimeter_f64",
    ]),
    ("Ordered Algorithm Kernels", [
        "radix_sort_u32_pairs",
        "floyd_warshall_blocked",
        "heap_sort_implementation",
        "pareval_sort_ignore_zero_i32",
    ]),
    ("Numeric and Validation Kernels", [
        "lower_tri_solve_strided_f64",
        "stencil3d_mixed_7pt",
        "rmsnorm_mixed",
        "utf8_validate_strict",
    ]),
]

PROBLEM_LABELS = {
    "array_sum_unroll": "ArraySum",
    "dot_product_unroll": "DotProd",
    "max_value_branchless": "MaxVal",
    "vector_add_simd": "VecAdd",
    "matrix_transpose_cache": "Transpose",
    "heap_sort_implementation": "HeapSort",
    "matmul_mixed_blocked": "MatMulMix",
    "rmsnorm_mixed": "RMSNorm",
    "stencil3d_mixed_7pt": "Stencil",
    "conv2d_3x3_multi_channel": "Conv2D",
    "csr_spmv_axpy_dot": "SpMV",
    "radix_sort_u32_pairs": "Radix",
    "floyd_warshall_blocked": "Floyd",
    "nbody_tiled_step": "NBody",
    "crop2d_strided_u8": "Crop2D",
    "transpose_strided_f64": "Strided\nTranspose",
    "gather_rows_strided_f64": "Gather\nRows",
    "gemm_strided_alpha_beta_f64": "Strided\nGEMM",
    "lower_tri_solve_strided_f64": "Triangular\nSolve",
    "segmented_prefix_sum_i32": "Segmented\nPrefix",
    "utf8_validate_strict": "UTF-8\nValidate",
    "cholesky_spd_f64": "Cholesky\nSPD",
    "pareval_sort_ignore_zero_i32": "Sort\nNonzero",
    "pareval_convex_hull_perimeter_f64": "Convex\nHull",
}


def load_data(problems):
    """問題ごとに対応する保存済み候補集合をロードする。"""
    result = {}
    cache = {}
    for problem in problems:
        source = PROBLEM_SOURCES[problem]
        if source not in cache:
            path = RESULT_PATHS[source]
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            by_problem = {}
            for item in data["candidate_sets"]:
                by_problem.setdefault(item["problem_name"], []).append(item)
            cache[source] = by_problem
        result[problem] = cache[source][problem]
    return result


def write_xbb(output_path, fig, left=0.02, right=0.99, bottom=0.06, top=0.97):
    """LaTeX用.xbbファイル生成（図全体のサイズを記録）"""
    width = fig.get_figwidth() * 72
    height = fig.get_figheight() * 72
    xbb_path = output_path.with_suffix(".xbb")
    xbb_path.write_text(
        "\n".join([
            f"%%Title: {output_path.name}",
            "%%Creator: analysis/create_clean_figures_12problems.py",
            f"%%BoundingBox: 0 0 {int(round(width))} {int(round(height))}",
            f"%%HiResBoundingBox: 0.000000 0.000000 {width:.6f} {height:.6f}",
            "",
        ])
    )


def collect_trial_distributions(trial_list, methods):
    """各手法について trial 単位の tests を集める。"""
    k = len(trial_list[0]["candidates"])
    distributions = {method: {"tests": []} for method in set(methods) | {"Gen", "Mean", "Tail"}}

    for trial in trial_list:
        candidates = trial["candidates"]

        # Gen
        gen_tests = float(k)
        for rank, cand in enumerate(candidates, start=1):
            if cand["success"]:
                gen_tests = float(rank)
                break
        distributions["Gen"]["tests"].append(gen_tests)

        # Mean-CBBA
        sorted_cands = sorted(enumerate(candidates), key=lambda item: (-item[1]["confidence"], item[0]))
        mean_tests = float(k)
        for rank, (_, cand) in enumerate(sorted_cands, start=1):
            if cand["success"]:
                mean_tests = float(rank)
                break
        distributions["Mean"]["tests"].append(mean_tests)

        # Tail: token logprob の下位四分位で順位付け
        def tail_score(candidate):
            values = candidate.get("token_logprobs") or []
            if not values:
                return float("-inf")
            return float(np.quantile(values, 0.25))

        sorted_tail_cands = sorted(enumerate(candidates), key=lambda item: (-tail_score(item[1]), item[0]))
        tail_tests = float(k)
        for rank, (_, cand) in enumerate(sorted_tail_cands, start=1):
            if cand["success"]:
                tail_tests = float(rank)
                break
        distributions["Tail"]["tests"].append(tail_tests)

    return distributions


def draw_boxplot(ax, series_list, colors):
    """色付き箱ひげ図を描画する。平均値と外れ値は小さな点で示す。"""
    bp = ax.boxplot(
        series_list,
        positions=np.arange(len(series_list)),
        widths=0.56,
        patch_artist=True,
        whis=1.5,
        showfliers=True,
        showmeans=True,
        meanprops=dict(
            marker="D",
            markerfacecolor="#222222",
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=3.2,
        ),
        medianprops=dict(color="#222222", linewidth=1.0),
        whiskerprops=dict(color="#555555", linewidth=0.9),
        capprops=dict(color="#555555", linewidth=0.9),
        boxprops=dict(edgecolor="#555555", linewidth=0.9),
        flierprops=dict(
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#444444",
            markeredgewidth=0.7,
            markersize=2.3,
            alpha=0.65,
        ),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    for mean, color in zip(bp["means"], colors):
        mean.set_markerfacecolor(color)
        mean.set_markeredgecolor("white")
        mean.set_markeredgewidth(0.45)
    for flier, color in zip(bp["fliers"], colors):
        flier.set_markerfacecolor("white")
        flier.set_markeredgecolor(color)
        flier.set_markeredgewidth(0.75)
        flier.set_alpha(0.7)
    return bp


def overlay_trial_points(ax, series_list, colors, jitter=0.10, size=12):
    """箱ひげ図の上に trial 単位の点を重ねる。"""
    for idx, (series, color) in enumerate(zip(series_list, colors)):
        if not series:
            continue
        if len(series) == 1:
            offsets = np.array([0.0])
        else:
            offsets = np.linspace(-jitter, jitter, len(series))
        ax.scatter(
            np.full(len(series), idx, dtype=float) + offsets,
            series,
            s=size,
            marker="o",
            facecolors=color,
            edgecolors="white",
            linewidths=0.35,
            alpha=0.75,
            zorder=3,
        )


def compute_row_axis(values, floor_min=0.0, step=0.5, pad_low=0.1, pad_high=0.3):
    """問題群内で共有する縦軸レンジと目盛りを決める。"""
    vmin = min(values)
    vmax = max(values)
    lower = max(floor_min, np.floor((vmin - pad_low) / step) * step)
    upper = np.ceil((vmax + pad_high) / step) * step
    if upper <= lower:
        upper = lower + step
    ticks = np.arange(lower, upper + 0.5 * step, step)
    return float(lower), float(upper), ticks


def thin_ticks(ticks, max_labels=3):
    """目盛り数を間引いて狭い subplot でも読めるようにする。"""
    if len(ticks) <= max_labels:
        return ticks
    mid = ticks[len(ticks) // 2]
    return np.array([ticks[0], mid, ticks[-1]])


def plot_group_figure(group_name, problems, output_path, id_offset):
    """問題群別の図を作成（12問版: 4列×2段）"""
    # データロード
    dataset = load_data(problems)

    # 図作成（2行4列、1カラム幅用にコンパクト化）
    fig = plt.figure(figsize=(3.35, 3.85))
    gs_main = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.18], hspace=0.62,
                       left=0.13, right=0.985, bottom=0.18, top=0.84)
    # 1段目用のGridSpec（4列）
    gs_top = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[0], wspace=0.35)
    # 2段目用のGridSpec（4列）
    gs_bottom = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[1], wspace=0.35)

    # 手法定義: Gen、Mean-CBBA、Tail-CBBA
    methods = ["Gen", "Mean", "Tail"]
    colors = ["#f39c12", "#27ae60", "#8e44ad"]

    # ===== 1段目: 確信度分布（縦軸レンジ個別調整） =====
    top_axes = []
    for col, problem in enumerate(problems):
        ax = fig.add_subplot(gs_top[0, col])
        top_axes.append(ax)

        success_conf = []
        failure_conf = []
        for trial_data in dataset[problem]:
            for cand in trial_data["candidates"]:
                if cand["success"]:
                    success_conf.append(cand["confidence"])
                else:
                    failure_conf.append(cand["confidence"])

        high_conf_total = sum(1 for trial_data in dataset[problem] for cand in trial_data["candidates"] if cand["confidence"] >= HIGH_CONF_THRESHOLD)
        high_conf_success = sum(1 for trial_data in dataset[problem] for cand in trial_data["candidates"] if cand["confidence"] >= HIGH_CONF_THRESHOLD and cand["success"])
        high_conf_rate = (100.0 * high_conf_success / high_conf_total) if high_conf_total else 0.0
        high_conf_failures = [conf for conf in failure_conf if conf >= HIGH_CONF_THRESHOLD]

        bins = np.arange(0.50, 1.01, 0.03)
        ax.axvspan(HIGH_CONF_THRESHOLD, 1.00, color="#eeeeee", alpha=0.45, zorder=0)
        n_succ, _, _ = ax.hist(
            success_conf, bins=bins, label="Pass", color="#2ca25f", edgecolor="none", alpha=0.58, zorder=2
        )
        n_fail, _, _ = ax.hist(
            failure_conf, bins=bins, label="Fail", color="#de2d26", edgecolor="#9e1b1b",
            linewidth=0.45, alpha=0.48, zorder=3
        )
        ax.axvline(HIGH_CONF_THRESHOLD, color="#555555", linestyle="--", linewidth=0.8, alpha=0.8)

        ax.set_xlim(0.50, 1.00)
        ax.set_xticks([0.5, HIGH_CONF_THRESHOLD, 1.0])
        ax.set_xticklabels(["0.5", "0.75", "1.0"])
        ax.set_xlabel("Conf.", fontsize=6, labelpad=1)
        # 縦軸レンジを各問題のデータに合わせて調整（目盛りは自動）
        max_count = max(max(n_fail) if len(n_fail) > 0 else 0, max(n_succ) if len(n_succ) > 0 else 0, 1)
        ax.set_ylim(-max_count * 0.14, max_count * 1.25)
        ax.set_yticks([tick for tick in ax.get_yticks() if tick >= 0])
        if high_conf_failures:
            rug_y = -max_count * 0.07
            ax.scatter(
                high_conf_failures,
                np.full(len(high_conf_failures), rug_y),
                marker="|",
                s=28,
                linewidths=0.8,
                color="#9e1b1b",
                alpha=0.85,
                zorder=5,
                label="HC fail",
            )

        ax.set_title(
            f"{PROBLEM_LABELS.get(problem, problem[:8])}\n(HC {high_conf_rate:.1f}%)",
            fontsize=6.6,
            pad=3,
            linespacing=0.95,
        )
        if col == 0:
            ax.set_ylabel("Count", fontsize=7)
        ax.grid(axis="y", linestyle=":")
        ax.tick_params(labelsize=5.8, pad=1)
    problem_distributions = [collect_trial_distributions(dataset[problem], methods) for problem in problems]

    # ===== 2段目: trial 単位のテスト数（箱ひげ図） =====
    for col, distributions in enumerate(problem_distributions):
        ax = fig.add_subplot(gs_bottom[0, col])

        series = [distributions[method]["tests"] for method in methods]
        draw_boxplot(ax, series, colors)
        local_test_values = [value for values in series for value in values]
        test_ymin, test_ymax, test_ticks = compute_row_axis(
            local_test_values, floor_min=0.5, step=0.5, pad_low=0.15, pad_high=0.25
        )
        test_ticks = thin_ticks(test_ticks, max_labels=3)

        x = np.arange(len(methods))
        ax.set_ylim(test_ymin, test_ymax)
        ax.set_yticks(test_ticks)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=6, rotation=30, ha="right")
        if col == 0:
            ax.set_ylabel("Tests", fontsize=7)
            ax.tick_params(labelsize=6)
        else:
            ax.tick_params(labelsize=5, pad=1)
        ax.grid(axis="y", linestyle=":")

    confidence_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#2ca25f", edgecolor="none", alpha=0.58),
        plt.Rectangle((0, 0), 1, 1, facecolor="#de2d26", edgecolor="#9e1b1b", alpha=0.48),
        plt.Line2D([0], [0], marker="|", linestyle="none", color="#9e1b1b", markersize=6),
    ]
    confidence_legend_y = min(ax.get_position().y0 for ax in top_axes) - 0.105
    fig.legend(
        confidence_handles,
        ["Pass", "Fail", "HC fail"],
        loc="center",
        fontsize=6.2,
        frameon=False,
        ncol=3,
        bbox_to_anchor=(0.5, confidence_legend_y),
        columnspacing=1.0,
        handletextpad=0.45,
    )

    method_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor="none")
        for i in range(len(methods))
    ]
    fig.legend(
        method_handles,
        methods,
        loc="lower center",
        fontsize=6.5,
        frameon=False,
        ncol=3,
        bbox_to_anchor=(0.5, 0.035),
        columnspacing=1.1,
        handletextpad=0.5,
    )

    # タイトルなし（キャプションで記載するため）
    # tight_layoutは使わない（マージンを保持するため）
    # PNG形式で保存（300dpi）
    fig.savefig(output_path, dpi=300)
    write_xbb(output_path, fig, left=0.13, right=0.985, bottom=0.18, top=0.84)
    # EPS形式でも保存
    eps_path = output_path.with_suffix(".eps")
    fig.savefig(eps_path, format='eps')
    print(f"✓ {output_path.name} + {eps_path.name}")


def main():
    """メイン処理"""
    FIGURE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("論文用の図を作成（選抜12問・3グループ版）")
    print("手法: Gen, Mean, Tail")
    print("="*70)
    print()

    for idx, (group_name, problems) in enumerate(GROUPS):
        fig_num = idx + 1
        output_file = f"cbba_group_{group_name.lower().replace(' kernels', '').replace(' ', '_')}.png"
        output_path = FIGURE_OUT_DIR / output_file

        plot_group_figure(group_name, problems, output_path, id_offset=0)

    print()
    print("="*70)
    print("全ての図を生成しました！")
    print("図1: Indexing and Robust Numerics, 図2: Ordered Algorithm, 図3: Numeric and Validation")
    print("="*70)


if __name__ == "__main__":
    main()
