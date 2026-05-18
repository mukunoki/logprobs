#!/usr/bin/env python3
"""Per-problem Hybrid vs Random scatter and structural bar chart."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAPER = Path(__file__).resolve().parent
INPUT_CSV = PAPER / "verilog_hybrid_problem_dependence.csv"
STRUCT_CSV = PAPER / "verilog_hybrid_struct_summary.csv"
OUT_SCATTER = PAPER / "verilog_hybrid_per_problem"
OUT_STRUCT = PAPER / "verilog_hybrid_struct_bar"


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "ps.useafm": True,
    "pdf.use14corefonts": True,
})


def load_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def write_xbb(path: Path, fig) -> None:
    width = fig.get_figwidth() * 72
    height = fig.get_figheight() * 72
    path.with_suffix(".xbb").write_text(
        "\n".join([
            f"%%Title: {path.name}",
            "%%Creator: paper/build_hybrid_per_problem_figure.py",
            f"%%BoundingBox: 0 0 {int(round(width))} {int(round(height))}",
            f"%%HiResBoundingBox: 0.000000 0.000000 {width:.6f} {height:.6f}",
            "",
        ]),
        encoding="utf-8",
    )


def make_scatter():
    rows = load_rows(INPUT_CSV)
    r_det = np.array([float(r["random_det@16"]) for r in rows])
    h_det = np.array([float(r["hybrid_det@16"]) for r in rows])
    n = np.array([int(r["n"]) for r in rows])
    delta = h_det - r_det

    # color by primary tag (counter, shift, or other)
    colors = []
    for r in rows:
        tags = r["tags"].split(";")
        if "counter" in tags or "timer" in tags or "bcd_display" in tags:
            colors.append("tab:green")
        elif "shift" in tags or "shift_op" in tags:
            colors.append("tab:red")
        elif "fsm_state" in tags:
            colors.append("tab:blue")
        else:
            colors.append("tab:gray")

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sizes = np.clip(np.log1p(n) * 8, 8, 80)
    ax.scatter(r_det, h_det, s=sizes, c=colors, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Random det@16 (per problem)")
    ax.set_ylabel("Hybrid-UGT det@16 (per problem)")
    ax.set_title("Hybrid-UGT vs Random per problem (FULL eval, 126 problems)")
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)

    # legend
    handles = [
        plt.Line2D([0], [0], marker='o', linestyle='', color='tab:green', label='counter / timer / BCD'),
        plt.Line2D([0], [0], marker='o', linestyle='', color='tab:red', label='shift / rotate'),
        plt.Line2D([0], [0], marker='o', linestyle='', color='tab:blue', label='FSM state'),
        plt.Line2D([0], [0], marker='o', linestyle='', color='tab:gray', label='other'),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)

    # annotate top winners/losers
    rows_sorted = sorted(zip(rows, delta), key=lambda x: -x[1])
    for r, d in rows_sorted[:3] + rows_sorted[-3:]:
        label = r["problem"].replace("Prob", "P")
        # truncate label
        if len(label) > 18:
            label = label[:18]
        ax.annotate(
            label,
            xy=(float(r["random_det@16"]), float(r["hybrid_det@16"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
            alpha=0.8,
        )
    fig.tight_layout()
    fig.savefig(OUT_SCATTER.with_suffix(".png"), dpi=300)
    fig.savefig(OUT_SCATTER.with_suffix(".eps"), format="eps")
    write_xbb(OUT_SCATTER.with_suffix(".png"), fig)
    print(f"wrote {OUT_SCATTER}.png/eps")
    plt.close(fig)


def make_struct_bar():
    rows = load_rows(STRUCT_CSV)
    rows = [r for r in rows if int(r["n_candidates"]) >= 200]
    rows.sort(key=lambda r: -float(r["delta_det@16"]))
    tags = [r["tag"] for r in rows]
    delta = np.array([float(r["delta_det@16"]) for r in rows])
    n_cand = np.array([int(r["n_candidates"]) for r in rows])

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    colors = ["tab:green" if d > 0.01 else "tab:red" if d < -0.01 else "tab:gray" for d in delta]
    bars = ax.barh(range(len(tags)), delta, color=colors, edgecolor="white")
    ax.set_yticks(range(len(tags)))
    labels = [f"{t} (n={c:,})" for t, c in zip(tags, n_cand)]
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Hybrid-UGT − Random  (det@16 difference)")
    ax.set_title("Hybrid-UGT advantage by structural tag (FULL eval)")
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    for i, (bar, d) in enumerate(zip(bars, delta)):
        x = d + (0.005 if d >= 0 else -0.005)
        ha = "left" if d >= 0 else "right"
        ax.text(x, i, f"{d:+.3f}", va="center", ha=ha, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_STRUCT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT_STRUCT.with_suffix(".eps"), format="eps")
    write_xbb(OUT_STRUCT.with_suffix(".png"), fig)
    print(f"wrote {OUT_STRUCT}.png/eps")
    plt.close(fig)


def main():
    make_scatter()
    make_struct_bar()


if __name__ == "__main__":
    main()
