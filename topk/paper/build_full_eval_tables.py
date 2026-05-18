#!/usr/bin/env python3
"""Build LaTeX tables for the full-scale eval results."""
from __future__ import annotations

import csv
import json
from pathlib import Path


PAPER = Path(__file__).resolve().parent
FULL_JSON = PAPER / "verilog_span_guided_full_512_test_eval.json"
STRUCT_CSV = PAPER / "verilog_hybrid_struct_summary.csv"
OUT_OVERALL_TEX = PAPER / "verilog_ugt_overall_full_table.tex"
OUT_STRUCT_TEX = PAPER / "verilog_ugt_struct_full_table.tex"


def load_full():
    return json.loads(FULL_JSON.read_text())


def load_struct():
    rows = []
    with open(STRUCT_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def build_overall_table():
    data = load_full()
    methods_order = ["random", "generic", "category", "span", "hybrid"]
    name = {"random": "Random", "generic": "Generic", "category": "Category-UGT",
            "span": "Span-UGT", "hybrid": "Hybrid-UGT"}

    lines = [
        "\\begin{table}[t]",
        "\\caption{Compile-pass official-fail候補31,842件全件に対する不確実性誘導テストの全体性能．det@$B$は先頭$B$個以内のvalidation input vectorで検出できた割合，mean detection vectorsは検出できなかった候補を$B_{\\max}+1=17$として平均した値．}",
        "\\label{tab:ugt_overall_full}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lcccccc@{}}",
        "\\hline",
        "手法 & det@1 & det@2 & det@4 & det@8 & det@16 & mean \\\\",
        "\\hline",
    ]
    for m in methods_order:
        item = data["methods"][m]
        det = item["detection_at"]
        lines.append(
            f"{name[m]} & {det['1']:.3f} & {det['2']:.3f} & {det['4']:.3f} & "
            f"{det['8']:.3f} & {det['16']:.3f} & {item['mean_tests_to_detect']:.2f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    OUT_OVERALL_TEX.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_OVERALL_TEX}")


def build_struct_table():
    rows = load_struct()
    # Keep tags with reasonable sample size
    rows = [r for r in rows if int(r["n_candidates"]) >= 200]
    rows.sort(key=lambda r: -float(r["delta_det@16"]))

    tag_label = {
        "counter": "Counter (count, countbcd等)",
        "arith_op": "Arith op (+, -, *, /)",
        "case_mux": "Case/mux",
        "fsm_state": "FSM/state遷移",
        "always_comb": "always\\_comb",
        "arith": "Arithmetic datapath",
        "comb_logic": "Combinational logic",
        "other": "Other",
        "posedge": "posedge clock",
        "case_construct": "case文",
        "shift": "Shift / rotate",
        "shift_op": "Shift op (<<, >>)",
        "bcd_display": "BCD display",
        "timer": "Timer",
    }

    lines = [
        "\\begin{table*}[t]",
        "\\caption{構文タグ別Hybrid-UGT対Randomの差分（FULL評価，31,842候補）．候補数200以上のタグを表示．各候補は複数タグに属し得るため候補数は重複を含む．Δdet@16はHybrid-UGTからRandomを引いた値．}",
        "\\label{tab:ugt_struct_full}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\hline",
        "構文タグ & 問題数 & 候補数 & Random det@16 & Hybrid det@16 & $\\Delta$det@16 & $\\Delta$mean \\\\",
        "\\hline",
    ]
    for r in rows:
        tag = tag_label.get(r["tag"], r["tag"])
        lines.append(
            f"{tag} & {r['n_problems']} & {int(r['n_candidates']):,} & "
            f"{float(r['random_det@16']):.3f} & {float(r['hybrid_det@16']):.3f} & "
            f"{float(r['delta_det@16']):+.3f} & {float(r['delta_mean']):+.2f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}"])
    OUT_STRUCT_TEX.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_STRUCT_TEX}")


def main():
    build_overall_table()
    build_struct_table()


if __name__ == "__main__":
    main()
