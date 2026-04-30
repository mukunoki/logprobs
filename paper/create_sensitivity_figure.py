#!/usr/bin/env python3
"""Generate sensitivity analysis figure from summarized experiment JSON."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 論文用のフォント設定（図3と同じ）
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.3,
    "axes.linewidth": 1.0,
})

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent

def default_summary_json() -> Path:
    candidates = sorted(
        REPO_ROOT.glob("results/paper_sensitivity_*/sensitivity_summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]
    return REPO_ROOT / "results" / "paper_sensitivity_latest" / "sensitivity_summary.json"


SUMMARY_JSON = Path(os.environ.get("CBBA_SENSITIVITY_JSON", default_summary_json()))


def write_xbb(output_path: Path, fig: plt.Figure):
    """Write .xbb file for LaTeX."""
    width = fig.get_figwidth() * 72
    height = fig.get_figheight() * 72
    xbb_path = output_path.with_suffix(".xbb")
    xbb_path.write_text(
        "\n".join([
            f"%%Title: {output_path.name}",
            "%%Creator: analysis/create_sensitivity_figure.py",
            f"%%BoundingBox: 0 0 {int(round(width))} {int(round(height))}",
            f"%%HiResBoundingBox: 0.000000 0.000000 {width:.6f} {height:.6f}",
            "",
        ])
    )


def create_sensitivity_figure():
    """Generate side-by-side sensitivity analysis figure."""
    if not SUMMARY_JSON.exists():
        raise FileNotFoundError(f"Sensitivity summary JSON not found: {SUMMARY_JSON}")

    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    k_rows = sorted(summary["k_variation"], key=lambda row: row["k"])
    temp_rows = sorted(summary["temperature_variation"], key=lambda row: row["temperature"])

    k_values = [row["k"] for row in k_rows]
    hyb_c_success_k = [row["cbba_c"]["success_rate"] * 100.0 for row in k_rows]
    hyb_a_success_k = [row["cbba_a"]["success_rate"] * 100.0 for row in k_rows]
    hyb_c_avg_test_k = [row["cbba_c"]["avg_tests"] for row in k_rows]
    hyb_a_avg_test_k = [row["cbba_a"]["avg_tests"] for row in k_rows]

    temp_values = [row["temperature"] for row in temp_rows]
    hyb_c_success_temp = [row["cbba_c"]["success_rate"] * 100.0 for row in temp_rows]
    hyb_a_success_temp = [row["cbba_a"]["success_rate"] * 100.0 for row in temp_rows]
    hyb_c_avg_test_temp = [row["cbba_c"]["avg_tests"] for row in temp_rows]
    hyb_a_avg_test_temp = [row["cbba_a"]["avg_tests"] for row in temp_rows]

    # 図4は1カラム幅（figure）、図3は2カラム幅（figure*）
    # 1カラム = 2カラムの約半分なので、横幅を半分にして縦横比を調整
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.0, 2.5))

    # 色設定
    color_hybc = "#2ecc71"  # CBBA-C: 緑
    color_hyba = "#e74c3c"  # CBBA-A: 赤

    # ========== 左図: k変化 (temperature=0.8) ==========
    ax1_twin = ax1.twinx()

    # 成功率（左軸）
    line1 = ax1.plot(k_values, hyb_c_success_k, 'o-', color=color_hybc,
                     linewidth=1.5, markersize=4, label='CBBA-C success rate')
    line2 = ax1.plot(k_values, hyb_a_success_k, 's-', color=color_hyba,
                     linewidth=1.5, markersize=4, label='CBBA-A success rate')

    # 平均テスト数（右軸）
    line3 = ax1_twin.plot(k_values, hyb_c_avg_test_k, 'o--', color=color_hybc,
                          linewidth=1.3, markersize=3.5, alpha=0.7, label='CBBA-C avg tests')
    line4 = ax1_twin.plot(k_values, hyb_a_avg_test_k, 's--', color=color_hyba,
                          linewidth=1.3, markersize=3.5, alpha=0.7, label='CBBA-A avg tests')

    ax1.set_xlabel('Candidate budget $k$\n(temperature=0.8)')
    ax1.set_ylabel('Success rate (%)')
    ax1_twin.set_ylabel('')  # 右軸ラベルなし
    ax1_twin.tick_params(labelright=False)  # 右軸の目盛りラベルを非表示
    ax1.set_xticks(k_values)
    ax1.set_ylim(85, 105)
    ax1_twin.set_ylim(1.8, 3.0)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax1.set_title('(a) $k$ variation', pad=8)

    # ========== 右図: temperature変化 (k=3) ==========
    ax2_twin = ax2.twinx()

    # 成功率（左軸）
    line5 = ax2.plot(temp_values, hyb_c_success_temp, 'o-', color=color_hybc,
                     linewidth=1.5, markersize=4, label='CBBA-C success rate')
    line6 = ax2.plot(temp_values, hyb_a_success_temp, 's-', color=color_hyba,
                     linewidth=1.5, markersize=4, label='CBBA-A success rate')

    # 平均テスト数（右軸）
    line7 = ax2_twin.plot(temp_values, hyb_c_avg_test_temp, 'o--', color=color_hybc,
                          linewidth=1.3, markersize=3.5, alpha=0.7, label='CBBA-C avg tests')
    line8 = ax2_twin.plot(temp_values, hyb_a_avg_test_temp, 's--', color=color_hyba,
                          linewidth=1.3, markersize=3.5, alpha=0.7, label='CBBA-A avg tests')

    ax2.set_xlabel('Temperature (k=3)')
    ax2.set_ylabel('')  # 左軸ラベルなし
    ax2.tick_params(labelleft=False)  # 左軸の目盛りラベルを非表示
    ax2_twin.set_ylabel('Avg tests per trial')
    ax2.set_xticks(temp_values)
    ax2.set_ylim(85, 105)
    ax2_twin.set_ylim(1.8, 3.0)
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax2.set_title('(b) Temperature variation', pad=8)

    # 共通凡例を図の下に配置（枠なし、1カラム幅なので縦2段に配置）
    lines_all = line1 + line2 + line3 + line4
    labels_all = [l.get_label() for l in lines_all]
    fig.legend(lines_all, labels_all, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.08), fontsize=6, frameon=False,
               columnspacing=1.0, handlelength=1.5)

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.28)  # 凡例の上余白を十分確保

    # 保存
    output = PAPER_DIR / "cbba_sensitivity.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    write_xbb(output, fig)
    print(f"✓ Generated: {output.name} (300 dpi) from {SUMMARY_JSON}")

    return fig


def main():
    """Generate sensitivity analysis figure."""
    print("="*60)
    print("感度分析図の生成（横2列レイアウト）")
    print("="*60)
    print()

    create_sensitivity_figure()

    print()
    print("="*60)
    print("図の生成が完了しました")
    print("="*60)


if __name__ == "__main__":
    main()
