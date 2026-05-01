#!/usr/bin/env python3
"""Create the sensitivity-analysis figure used in the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PAPER_DIR = Path(__file__).resolve().parent
OUT_PATH = PAPER_DIR / "cbba_sensitivity.png"

ROWS = [
    {"temperature": 0.4, "k": 3, "pool": 95.0, "gen": 1.45, "mean": 1.23, "tail": 1.18, "gain": 0.27},
    {"temperature": 0.4, "k": 5, "pool": 97.5, "gen": 1.60, "mean": 1.30, "tail": 1.40, "gain": 0.20},
    {"temperature": 0.4, "k": 10, "pool": 100.0, "gen": 1.62, "mean": 1.07, "tail": 1.10, "gain": 0.52},
    {"temperature": 0.8, "k": 3, "pool": 97.5, "gen": 1.43, "mean": 1.23, "tail": 1.15, "gain": 0.28},
    {"temperature": 0.8, "k": 5, "pool": 97.5, "gen": 1.70, "mean": 1.32, "tail": 1.30, "gain": 0.40},
    {"temperature": 0.8, "k": 10, "pool": 100.0, "gen": 1.80, "mean": 1.15, "tail": 1.25, "gain": 0.55},
    {"temperature": 1.0, "k": 3, "pool": 95.0, "gen": 1.52, "mean": 1.38, "tail": 1.45, "gain": 0.07},
    {"temperature": 1.0, "k": 5, "pool": 97.5, "gen": 1.65, "mean": 1.57, "tail": 1.55, "gain": 0.10},
    {"temperature": 1.0, "k": 10, "pool": 100.0, "gen": 1.82, "mean": 1.32, "tail": 1.30, "gain": 0.52},
]

COLORS = {0.4: "#1f77b4", 0.8: "#238b45", 1.0: "#cb181d"}
MARKERS = {0.4: "o", 0.8: "s", 1.0: "^"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8.8,
    "axes.titlesize": 8.8,
    "axes.labelsize": 8.8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "lines.linewidth": 1.2,
    "ps.useafm": True,
    "pdf.use14corefonts": True,
})


def write_xbb(output_path: Path, fig) -> None:
    width = fig.get_figwidth() * 72
    height = fig.get_figheight() * 72
    output_path.with_suffix(".xbb").write_text(
        "\n".join([
            f"%%Title: {output_path.name}",
            "%%Creator: paper/create_sensitivity_figure.py",
            f"%%BoundingBox: 0 0 {int(round(width))} {int(round(height))}",
            f"%%HiResBoundingBox: 0.000000 0.000000 {width:.6f} {height:.6f}",
            "",
        ]),
        encoding="utf-8",
    )


def rows_for_temp(temp: float) -> list[dict]:
    return sorted([row for row in ROWS if row["temperature"] == temp], key=lambda row: row["k"])


def main() -> None:
    fig, (ax_tests, ax_gain) = plt.subplots(1, 2, figsize=(3.45, 2.35), sharex=True)

    for temp in [0.4, 0.8, 1.0]:
        rows = rows_for_temp(temp)
        ks = [row["k"] for row in rows]
        color = COLORS[temp]
        marker = MARKERS[temp]
        ax_tests.plot(
            ks,
            [row["gen"] for row in rows],
            color=color,
            marker=marker,
            markersize=3.0,
            linestyle="-",
            label=f"Gen, T={temp}",
        )
        ax_tests.plot(
            ks,
            [row["tail"] for row in rows],
            color=color,
            marker=marker,
            markersize=3.0,
            linestyle="--",
            label=f"Tail, T={temp}",
        )
        ax_gain.plot(
            ks,
            [row["gain"] for row in rows],
            color=color,
            marker=marker,
            markersize=3.0,
            linestyle="-",
            label=f"T={temp}",
        )

    ax_tests.set_title("(a) Avg. tests")
    ax_tests.set_xlabel("$k$")
    ax_tests.set_ylabel("Avg. tests")
    ax_tests.set_xticks([3, 5, 10])
    ax_tests.set_ylim(1.0, 1.9)
    ax_tests.grid(linestyle=":", linewidth=0.5, color="#c7c7c7")

    ax_gain.axhline(0.0, color="#555555", linewidth=0.8)
    ax_gain.set_title("(b) Tail gain over Gen")
    ax_gain.set_xlabel("$k$")
    ax_gain.set_ylabel("Gain")
    ax_gain.set_xticks([3, 5, 10])
    ax_gain.set_ylim(0.0, 0.62)
    ax_gain.grid(linestyle=":", linewidth=0.5, color="#c7c7c7")

    handles = [
        Line2D([0], [0], color=COLORS[0.4], marker=MARKERS[0.4], markersize=3.0, linestyle="-", label="T=0.4"),
        Line2D([0], [0], color=COLORS[0.8], marker=MARKERS[0.8], markersize=3.0, linestyle="-", label="T=0.8"),
        Line2D([0], [0], color=COLORS[1.0], marker=MARKERS[1.0], markersize=3.0, linestyle="-", label="T=1.0"),
        Line2D([0], [0], color="#333333", linestyle="-", label="Gen"),
        Line2D([0], [0], color="#333333", linestyle="--", label="Tail"),
    ]
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.018),
        columnspacing=0.45,
        handlelength=1.05,
        handletextpad=0.25,
        fontsize=6.2,
    )
    fig.tight_layout(rect=(0.02, 0.085, 0.995, 0.995), w_pad=0.8)

    fig.savefig(OUT_PATH, dpi=300)
    fig.savefig(OUT_PATH.with_suffix(".eps"), format="eps")
    write_xbb(OUT_PATH, fig)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
