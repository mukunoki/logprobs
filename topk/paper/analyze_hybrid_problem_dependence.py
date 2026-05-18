#!/usr/bin/env python3
"""Problem-dependent analysis of Hybrid-UGT vs Random.

Outputs:
1. Per-problem stats CSV: problem, n, R@1..16, H@1..16, delta_det@16, delta_mean
2. Structural categorization based on participation in name/reference patterns
3. Aggregated win/loss by structural category
4. Bootstrap CI for top win/loss problems
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PAPER = Path("/home/mukunoki/bot/pocketnika2/work/topk/paper")
DATASET = Path("/home/mukunoki/bot/pocketnika2/work/topk/external/verilog-eval/dataset_code-complete-iccad2023")
FULL_CSV = PAPER / "verilog_span_guided_full_512_test_eval.csv"
OUT_PROB_CSV = PAPER / "verilog_hybrid_problem_dependence.csv"
OUT_STRUCT_CSV = PAPER / "verilog_hybrid_struct_summary.csv"
OUT_JSON = PAPER / "verilog_hybrid_problem_dependence.json"

MAX_TESTS = 16


def load_eval():
    out = defaultdict(dict)
    with FULL_CSV.open() as f:
        for r in csv.DictReader(f):
            key = (r["problem"], int(r["trial"]), int(r["candidate"]))
            tt = r["tests_to_detect"]
            tt = int(tt) if tt and tt != "" else None
            out[key][r["method"]] = {
                "detected": r["detected"].lower() == "true",
                "tests_to_detect": tt,
                "feature_group": r["feature_group"],
            }
    return out


def per_subset(eval_map, keys, methods=("random", "generic", "category", "span", "hybrid")):
    s = {}
    for m in methods:
        det_at = {b: 0 for b in (1, 2, 4, 8, 16)}
        sum_t = 0
        n = 0
        for k in keys:
            if k not in eval_map or m not in eval_map[k]:
                continue
            row = eval_map[k][m]
            n += 1
            tt = row["tests_to_detect"]
            if tt is not None:
                for b in (1, 2, 4, 8, 16):
                    if tt <= b:
                        det_at[b] += 1
                sum_t += tt
            else:
                sum_t += MAX_TESTS + 1
        if n == 0:
            continue
        s[m] = {
            "n": n,
            "det@1": det_at[1] / n,
            "det@4": det_at[4] / n,
            "det@8": det_at[8] / n,
            "det@16": det_at[16] / n,
            "mean": sum_t / n,
        }
    return s


def classify_problem_struct(problem: str) -> list[str]:
    """Return all structural tags for the problem.
    A problem can belong to multiple tags. Tags are derived from the problem
    name and the reference implementation file."""
    tags = []
    lower = problem.lower()

    # Name-based tags
    if re.search(r"count|counter", lower):
        tags.append("counter")
    if re.search(r"timer", lower):
        tags.append("timer")
    if re.search(r"shift|rotate", lower):
        tags.append("shift")
    if re.search(r"fsm|state|ps2|hdlc|gshare|lemmings|serial|always_case", lower):
        tags.append("fsm_state")
    if re.search(r"case|mux", lower):
        tags.append("case_mux")
    if re.search(r"add|sub|mul|alu|adder", lower):
        tags.append("arith")
    if re.search(r"bcd|seven|sevenseg", lower):
        tags.append("bcd_display")
    if re.search(r"vector|gate|logic|wire|conditional|reduction|popcount", lower):
        tags.append("comb_logic")

    # Reference-based tags
    ref_path = DATASET / f"{problem}_ref.sv"
    if ref_path.exists():
        ref = ref_path.read_text(errors="ignore")
        ref_no_c = re.sub(r"//.*|/\*.*?\*/", "", ref, flags=re.DOTALL)
        if re.search(r"\bcase[zx]?\b", ref_no_c):
            if "case_construct" not in tags:
                tags.append("case_construct")
        if re.search(r"\balways\s*@\s*\(\s*posedge", ref_no_c):
            if "posedge" not in tags:
                tags.append("posedge")
        if re.search(r"<<|>>", ref_no_c):
            if "shift_op" not in tags:
                tags.append("shift_op")
        if "always_comb" in ref_no_c or re.search(r"always\s*@\s*\(\s*\*\s*\)", ref_no_c):
            if "always_comb" not in tags:
                tags.append("always_comb")
        if re.search(r"\+|-|\*|/|%", ref_no_c):
            if "arith_op" not in tags:
                tags.append("arith_op")

    if not tags:
        tags.append("other")
    return tags


def main():
    print("loading full eval...")
    eval_map = load_eval()
    print(f"loaded {len(eval_map)} candidate keys")

    # Group keys by problem
    by_problem = defaultdict(list)
    for key in eval_map:
        by_problem[key[0]].append(key)

    # Per-problem stats
    rows = []
    for problem, keys in sorted(by_problem.items()):
        s = per_subset(eval_map, keys, methods=("random", "hybrid", "span", "category", "generic"))
        if "random" not in s or "hybrid" not in s:
            continue
        r = s["random"]
        h = s["hybrid"]
        tags = classify_problem_struct(problem)
        feat = eval_map[keys[0]]["random"]["feature_group"]
        rows.append({
            "problem": problem,
            "feature_group": feat,
            "tags": ";".join(tags),
            "n": r["n"],
            "random_det@1": r["det@1"],
            "random_det@4": r["det@4"],
            "random_det@16": r["det@16"],
            "random_mean": r["mean"],
            "hybrid_det@1": h["det@1"],
            "hybrid_det@4": h["det@4"],
            "hybrid_det@16": h["det@16"],
            "hybrid_mean": h["mean"],
            "delta_det@16": h["det@16"] - r["det@16"],
            "delta_mean": h["mean"] - r["mean"],
            "span_det@16": s.get("span", {}).get("det@16"),
            "span_mean": s.get("span", {}).get("mean"),
        })

    # Sort by delta_det@16 desc
    rows.sort(key=lambda r: -r["delta_det@16"])

    with OUT_PROB_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"wrote {OUT_PROB_CSV} ({len(rows)} problems)")

    # Structural category summary (each problem may belong to multiple tags)
    by_tag = defaultdict(list)
    for r in rows:
        for tag in r["tags"].split(";"):
            by_tag[tag].append(r)

    struct_rows = []
    for tag in sorted(by_tag, key=lambda t: (-len(by_tag[t]), t)):
        problems = by_tag[tag]
        # Aggregate per-candidate (sum across problems)
        all_keys = []
        for r in problems:
            all_keys.extend([k for k in eval_map if k[0] == r["problem"]])
        if not all_keys:
            continue
        s = per_subset(eval_map, all_keys, methods=("random", "hybrid", "span"))
        if "random" not in s or "hybrid" not in s:
            continue
        struct_rows.append({
            "tag": tag,
            "n_problems": len(problems),
            "n_candidates": s["random"]["n"],
            "random_det@16": s["random"]["det@16"],
            "hybrid_det@16": s["hybrid"]["det@16"],
            "span_det@16": s.get("span", {}).get("det@16"),
            "delta_det@16": s["hybrid"]["det@16"] - s["random"]["det@16"],
            "random_mean": s["random"]["mean"],
            "hybrid_mean": s["hybrid"]["mean"],
            "delta_mean": s["hybrid"]["mean"] - s["random"]["mean"],
        })

    struct_rows.sort(key=lambda r: -r["delta_det@16"])
    with OUT_STRUCT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(struct_rows[0].keys()))
        writer.writeheader()
        for r in struct_rows:
            writer.writerow(r)
    print(f"wrote {OUT_STRUCT_CSV}")

    # Print structural summary
    print()
    print("=== Hybrid vs Random by structural tag (FULL data) ===")
    print(f"{'tag':22s} {'n_prob':>6s} {'n_cand':>7s} {'R@16':>7s} {'H@16':>7s} {'Δdet':>7s} {'R_mean':>7s} {'H_mean':>7s} {'Δmean':>7s}")
    for r in struct_rows:
        print(
            f"{r['tag']:22s} {r['n_problems']:>6d} {r['n_candidates']:>7d} "
            f"{r['random_det@16']:>7.3f} {r['hybrid_det@16']:>7.3f} {r['delta_det@16']:>+7.3f} "
            f"{r['random_mean']:>7.2f} {r['hybrid_mean']:>7.2f} {r['delta_mean']:>+7.2f}"
        )

    # Bootstrap CI for top winner / loser / overall
    rng = np.random.default_rng(20260516)

    def bootstrap_diff(keys, n_boot=10000):
        diffs_det = []
        diffs_mean = []
        per_cand_r_det = []
        per_cand_h_det = []
        per_cand_r_mean = []
        per_cand_h_mean = []
        for k in keys:
            r = eval_map[k]["random"]
            h = eval_map[k]["hybrid"]
            r_det = 1 if r["tests_to_detect"] is not None and r["tests_to_detect"] <= 16 else 0
            h_det = 1 if h["tests_to_detect"] is not None and h["tests_to_detect"] <= 16 else 0
            r_mean = r["tests_to_detect"] if r["tests_to_detect"] is not None else MAX_TESTS + 1
            h_mean = h["tests_to_detect"] if h["tests_to_detect"] is not None else MAX_TESTS + 1
            per_cand_r_det.append(r_det)
            per_cand_h_det.append(h_det)
            per_cand_r_mean.append(r_mean)
            per_cand_h_mean.append(h_mean)
        r_det_arr = np.asarray(per_cand_r_det)
        h_det_arr = np.asarray(per_cand_h_det)
        r_mean_arr = np.asarray(per_cand_r_mean, dtype=float)
        h_mean_arr = np.asarray(per_cand_h_mean, dtype=float)
        n = len(keys)
        idx = rng.integers(0, n, size=(n_boot, n))
        det_diff = (h_det_arr[idx].mean(axis=1) - r_det_arr[idx].mean(axis=1))
        mean_diff = (h_mean_arr[idx].mean(axis=1) - r_mean_arr[idx].mean(axis=1))
        return (
            (float(det_diff.mean()), float(np.quantile(det_diff, 0.025)), float(np.quantile(det_diff, 0.975))),
            (float(mean_diff.mean()), float(np.quantile(mean_diff, 0.025)), float(np.quantile(mean_diff, 0.975))),
        )

    print()
    print("=== Bootstrap CI for Hybrid - Random (FULL data) ===")
    all_keys = list(eval_map.keys())
    det_ci, mean_ci = bootstrap_diff(all_keys)
    print(f"Overall (n={len(all_keys)}): det@16 diff = {det_ci[0]:+.4f} [{det_ci[1]:+.4f}, {det_ci[2]:+.4f}], mean diff = {mean_ci[0]:+.3f} [{mean_ci[1]:+.3f}, {mean_ci[2]:+.3f}]")

    # Per top structural tags
    print()
    print("=== Bootstrap CI for Hybrid - Random per tag ===")
    for tag in ["counter", "timer", "fsm_state", "always_comb", "case_construct", "case_mux",
                "arith", "shift", "shift_op", "posedge", "comb_logic"]:
        if tag not in by_tag:
            continue
        keys = []
        for r in by_tag[tag]:
            keys.extend([k for k in eval_map if k[0] == r["problem"]])
        if len(keys) < 100:
            print(f"  {tag:22s} (n={len(keys)} skipped, too small)")
            continue
        det_ci, mean_ci = bootstrap_diff(keys, n_boot=5000)
        print(f"  {tag:22s} (n={len(keys):>6d}): det@16 = {det_ci[0]:+.4f} [{det_ci[1]:+.4f}, {det_ci[2]:+.4f}], "
              f"mean = {mean_ci[0]:+.3f} [{mean_ci[1]:+.3f}, {mean_ci[2]:+.3f}]")

    # Save JSON summary
    summary_json = {
        "overall": {
            "n": len(all_keys),
            "random_det@16": per_subset(eval_map, all_keys, ("random",))["random"]["det@16"],
            "hybrid_det@16": per_subset(eval_map, all_keys, ("hybrid",))["hybrid"]["det@16"],
        },
        "by_tag": struct_rows,
        "top_winners": rows[:15],
        "top_losers": rows[-15:],
    }
    OUT_JSON.write_text(json.dumps(summary_json, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
