"""Build LaTeX tables and EPS figures from the v2 evaluation results.

Reads ``eval_v2.jsonl`` produced by ``run_full_eval.py`` and writes:

* ``verilog_ugt_overall_full_table.tex`` (overall det@k per method),
* ``verilog_ugt_struct_full_table.tex``  (per-tag Hybrid - Random delta),
* ``verilog_hybrid_struct_bar.eps/.png/.xbb`` (bar chart of per-tag delta),
* ``verilog_hybrid_per_problem.eps/.png/.xbb`` (scatter).

Problem tags are derived from the problem name and the reference module
source (counter, shift, fsm_state, case_construct, posedge, arith_op).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MAX_K = 16
KS = (1, 2, 4, 8, 16)


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _problem_tags(problem: str, dataset_dir: Path) -> list[str]:
    tags: list[str] = []
    lower = problem.lower()
    if re.search(r"count|counter", lower): tags.append("counter")
    if re.search(r"timer", lower): tags.append("timer")
    if re.search(r"shift|rotate", lower): tags.append("shift")
    if re.search(r"fsm|state|ps2|hdlc|lemmings|serial|always_case", lower):
        tags.append("fsm_state")
    if re.search(r"case|mux", lower): tags.append("case_mux")
    if re.search(r"add|sub|alu|adder", lower): tags.append("arith")
    if re.search(r"bcd|seven", lower): tags.append("bcd_display")
    ref_path = dataset_dir / f"{problem}_ref.sv"
    if ref_path.exists():
        ref = ref_path.read_text(encoding="utf-8", errors="ignore")
        ref = re.sub(r"//.*|/\*.*?\*/", "", ref, flags=re.DOTALL)
        if re.search(r"\bcase[zx]?\b", ref) and "case_construct" not in tags:
            tags.append("case_construct")
        if re.search(r"<<|>>", ref) and "shift_op" not in tags:
            tags.append("shift_op")
        if re.search(r"\+|-|\*|/|%", ref) and "arith_op" not in tags:
            tags.append("arith_op")
        if re.search(r"always\s*@\s*\(\s*posedge", ref) and "posedge" not in tags:
            tags.append("posedge")
    if not tags:
        tags.append("other")
    return tags


def _load_eval(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            out.append(d)
    return out


def _det_stats(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {f"det@{k}": 0.0 for k in KS} | {"mean": 0.0}
    n = len(rows)
    out = {}
    for k in KS:
        c = sum(1 for r in rows
                if r.get("status") == "ok"
                and r.get("tests_to_detect") is not None
                and r["tests_to_detect"] <= k)
        out[f"det@{k}"] = c / n
    ttd_sum = 0
    for r in rows:
        if r.get("status") == "ok" and r.get("tests_to_detect") is not None:
            ttd_sum += r["tests_to_detect"]
        else:
            ttd_sum += MAX_K + 1
    out["mean"] = ttd_sum / n
    out["n"] = n
    return out


def build_overall_table(rows: list[dict], n_candidates: int, out_path: Path,
                        is_full_eval: bool = False) -> None:
    method_labels = {
        "random": "Random",
        "generic": "Generic",
        "category": "Category-UGT",
        "span": "Span-UGT",
        "hybrid": "Hybrid-UGT",
    }
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    set_descr = "生成token数$\\le 512$の全件評価，caps無し" if is_full_eval \
        else "per-problem stratified sub-sample，seed 54321"
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{Compile-pass official-fail候補{n_candidates:,}件（{set_descr}）に対する不確実性誘導テストの全体性能．det@$B$は先頭$B$個以内のvalidation input vectorで検出できた割合，mean detection vectorsは検出できなかった候補を$B_{{\max}}+1=17$として平均した値．}}",
        r"\label{tab:ugt_overall_full}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\hline",
        r"手法 & det@1 & det@2 & det@4 & det@8 & det@16 & mean \\",
        r"\hline",
    ]
    for m in ("random", "generic", "category", "span", "hybrid"):
        s = _det_stats(by_method.get(m, []))
        lines.append(
            f"{method_labels[m]} & "
            f"{s['det@1']:.3f} & {s['det@2']:.3f} & {s['det@4']:.3f} & "
            f"{s['det@8']:.3f} & {s['det@16']:.3f} & {s['mean']:.2f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


def build_struct_table(
    rows: list[dict],
    problem_tags: dict[str, list[str]],
    out_path: Path,
    n_candidates: int,
    min_n: int = 100,
) -> None:
    # Aggregate per (tag, method)
    tag_rows: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    tag_problems: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        for tag in problem_tags.get(r["problem"], ["other"]):
            tag_rows[tag][r["method"]].append(r)
            tag_problems[tag].add(r["problem"])

    tag_label = {
        "counter": "Counter (count, countbcd等)",
        "timer": "Timer",
        "arith_op": "Arith op (+, -, *, /)",
        "case_construct": "case文",
        "case_mux": "Case/mux",
        "fsm_state": "FSM/state遷移",
        "arith": "Arithmetic datapath",
        "bcd_display": "BCD display",
        "shift": "Shift / rotate",
        "shift_op": "Shift op (<<, >>)",
        "posedge": "posedge clock",
        "other": "Other",
    }
    table_rows = []
    for tag, by_m in tag_rows.items():
        r = by_m.get("random", [])
        h = by_m.get("hybrid", [])
        n = min(len(r), len(h))
        if n < min_n:
            continue
        rs = _det_stats(r); hs = _det_stats(h)
        table_rows.append({
            "tag": tag,
            "n_problems": len(tag_problems[tag]),
            "n_candidates": n,
            "random_det1": rs["det@1"],
            "hybrid_det1": hs["det@1"],
            "delta_det1": hs["det@1"] - rs["det@1"],
            "random_det16": rs["det@16"],
            "hybrid_det16": hs["det@16"],
            "delta_det16": hs["det@16"] - rs["det@16"],
        })
    table_rows.sort(key=lambda r: -r["delta_det1"])

    all_problems = set()
    for r in rows:
        all_problems.add(r["problem"])
    lines = [
        r"\begin{table*}[t]",
        rf"\caption{{構文タグ別Hybrid-UGT対Randomの差分（CPOF候補 {n_candidates:,}件・{len(all_problems)}問対象）．候補数{min_n}以上のタグを$\Delta$det@1降順で表示．各候補は複数タグに属し得るため候補数は重複を含む．det@1はfail-fast screening指標，det@16は16ベクトル予算での累積検出率．}}",
        r"\label{tab:ugt_struct_full}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\hline",
        r"構文タグ & 問題数 & 候補数 & R det@1 & H det@1 & $\Delta$@1 & R det@16 & H det@16 & $\Delta$@16 \\",
        r"\hline",
    ]
    for tr in table_rows:
        label = _tex_escape(tag_label.get(tr["tag"], tr["tag"]))
        lines.append(
            f"{label} & {tr['n_problems']} & {tr['n_candidates']:,} & "
            f"{tr['random_det1']:.3f} & {tr['hybrid_det1']:.3f} & {tr['delta_det1']:+.3f} & "
            f"{tr['random_det16']:.3f} & {tr['hybrid_det16']:.3f} & {tr['delta_det16']:+.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


def _write_xbb(path: Path, fig) -> None:
    w = fig.get_figwidth() * 72
    h = fig.get_figheight() * 72
    path.with_suffix(".xbb").write_text(
        f"%%Title: {path.name}\n"
        f"%%Creator: build_v2_assets.py\n"
        f"%%BoundingBox: 0 0 {int(round(w))} {int(round(h))}\n"
        f"%%HiResBoundingBox: 0.000000 0.000000 {w:.6f} {h:.6f}\n",
        encoding="utf-8",
    )


def build_struct_bar(
    rows: list[dict],
    problem_tags: dict[str, list[str]],
    out_stem: Path,
    min_n: int = 100,
) -> None:
    tag_rows: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for tag in problem_tags.get(r["problem"], ["other"]):
            tag_rows[tag][r["method"]].append(r)

    entries = []
    for tag, by_m in tag_rows.items():
        r = by_m.get("random", []); h = by_m.get("hybrid", [])
        n = min(len(r), len(h))
        if n < min_n: continue
        rs = _det_stats(r); hs = _det_stats(h)
        entries.append((tag, n,
                        hs["det@1"] - rs["det@1"],
                        hs["det@16"] - rs["det@16"]))
    entries.sort(key=lambda e: -e[2])  # sort by Δdet@1

    tags = [e[0] for e in entries]
    ns = [e[1] for e in entries]
    d1 = np.array([e[2] for e in entries])
    d16 = np.array([e[3] for e in entries])

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    y = np.arange(len(tags))
    bar_h = 0.4
    bars1 = ax.barh(y - bar_h/2, d1, height=bar_h,
                    color=["tab:green" if v > 0.005 else "tab:red" if v < -0.005 else "tab:gray" for v in d1],
                    edgecolor="white", label=r"$\Delta$det@1")
    bars16 = ax.barh(y + bar_h/2, d16, height=bar_h,
                     color=["#2ca02c" if v > 0.005 else "#d62728" if v < -0.005 else "#7f7f7f" for v in d16],
                     edgecolor="white", alpha=0.6, label=r"$\Delta$det@16")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t} (n={c:,})" for t, c in zip(tags, ns)])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Hybrid-UGT − Random (det@k difference)")
    ax.set_title("Hybrid-UGT advantage by structural tag (B-plan, 6,377)")
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    for i, (v1, v16) in enumerate(zip(d1, d16)):
        for d, off in ((v1, -bar_h/2), (v16, bar_h/2)):
            x = d + (0.004 if d >= 0 else -0.004)
            ha = "left" if d >= 0 else "right"
            ax.text(x, i + off, f"{d:+.3f}", va="center", ha=ha, fontsize=6)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(str(out_stem) + ".png", dpi=300)
    fig.savefig(str(out_stem) + ".eps", format="eps")
    _write_xbb(out_stem.with_suffix(".png"), fig)
    print(f"wrote {out_stem}.png/eps")
    plt.close(fig)


def build_per_problem_scatter(
    rows: list[dict],
    problem_tags: dict[str, list[str]],
    out_stem: Path,
) -> None:
    by_prob: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_prob[r["problem"]][r["method"]].append(r)

    data = []
    for prob, by_m in by_prob.items():
        r = by_m.get("random", []); h = by_m.get("hybrid", [])
        n = min(len(r), len(h))
        if n < 5: continue
        rs = _det_stats(r); hs = _det_stats(h)
        tags = problem_tags.get(prob, [])
        # Color by primary tag class.
        if "counter" in tags or "timer" in tags or "bcd_display" in tags:
            c = "tab:green"
        elif "shift" in tags:
            c = "tab:red"
        elif "fsm_state" in tags:
            c = "tab:blue"
        else:
            c = "tab:gray"
        data.append((prob, n, rs["det@16"], hs["det@16"], c, hs["det@16"] - rs["det@16"]))

    if not data:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ns = np.array([d[1] for d in data])
    rd = np.array([d[2] for d in data])
    hd = np.array([d[3] for d in data])
    sizes = np.clip(np.log1p(ns) * 8, 8, 80)
    colors = [d[4] for d in data]
    ax.scatter(rd, hd, s=sizes, c=colors, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Random det@16 (per problem)")
    ax.set_ylabel("Hybrid-UGT det@16 (per problem)")
    ax.set_title(f"Hybrid-UGT vs Random per problem (B-plan, {len(data)} problems)")
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    handles = [
        plt.Line2D([0], [0], marker='o', linestyle='', color='tab:green', label='counter / timer / BCD'),
        plt.Line2D([0], [0], marker='o', linestyle='', color='tab:red', label='shift / rotate'),
        plt.Line2D([0], [0], marker='o', linestyle='', color='tab:blue', label='FSM state'),
        plt.Line2D([0], [0], marker='o', linestyle='', color='tab:gray', label='other'),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    sorted_data = sorted(data, key=lambda d: -d[5])
    for d in sorted_data[:3] + sorted_data[-3:]:
        label = d[0].replace("Prob", "P")[:18]
        ax.annotate(label, xy=(d[2], d[3]), xytext=(4, 4),
                    textcoords="offset points", fontsize=6, alpha=0.8)
    fig.tight_layout()
    fig.savefig(str(out_stem) + ".png", dpi=300)
    fig.savefig(str(out_stem) + ".eps", format="eps")
    _write_xbb(out_stem.with_suffix(".png"), fig)
    print(f"wrote {out_stem}.png/eps")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path(
        "/home/mukunoki/bot/pocketnika2/work/topk/external/verilog-eval/"
        "dataset_code-complete-iccad2023"))
    parser.add_argument("--paper-dir", type=Path, default=Path(
        "/home/mukunoki/bot/pocketnika2/work/topk/paper"))
    args = parser.parse_args()

    rows = _load_eval(args.eval_jsonl)
    print(f"loaded {len(rows)} eval rows")

    n_candidates = len({(r["problem"], r["trial"], r["candidate"]) for r in rows})
    problems = {r["problem"] for r in rows}
    problem_tags = {p: _problem_tags(p, args.dataset_dir) for p in problems}

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "ps.useafm": True, "pdf.use14corefonts": True,
    })

    build_overall_table(rows, n_candidates,
                        args.paper_dir / "verilog_ugt_overall_full_table.tex",
                        is_full_eval=(n_candidates >= 30000))
    build_struct_table(rows, problem_tags,
                       args.paper_dir / "verilog_ugt_struct_full_table.tex",
                       n_candidates=n_candidates)
    build_struct_bar(rows, problem_tags,
                     args.paper_dir / "verilog_hybrid_struct_bar")
    build_per_problem_scatter(rows, problem_tags,
                              args.paper_dir / "verilog_hybrid_per_problem")


if __name__ == "__main__":
    main()
