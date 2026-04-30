#!/usr/bin/env python3
"""Offline evaluation for hybrid threshold + confidence-order CBBA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


PAPER_GROUPS = {
    "Easy-4": [
        ("paper_easy4", "matrix_transpose_cache"),
        ("paper_easy4", "array_sum_unroll"),
        ("paper_easy4", "csr_spmv_axpy_dot"),
        ("paper_medium4", "max_value_branchless"),
    ],
    "Medium-4": [
        ("paper_easy4", "heap_sort_implementation"),
        ("paper_medium4", "rmsnorm_mixed"),
        ("paper_medium4", "vector_add_simd"),
        ("paper_hard4", "conv2d_3x3_multi_channel"),
    ],
    "Hard-4": [
        ("paper_medium4", "floyd_warshall_blocked"),
        ("paper_hard4", "stencil3d_mixed_7pt"),
        ("paper_hard4", "radix_sort_u32_pairs"),
        ("paper_hard4", "nbody_tiled_step"),
    ],
}


def aggressive_budget(confidence: float, max_budget: int) -> int:
    if confidence >= 0.90:
        return 1
    if confidence >= 0.85:
        return min(max_budget, max(2, math.ceil((max_budget + 1) / 2)))
    return max_budget


def conservative_budget(confidence: float, max_budget: int) -> int:
    if confidence >= 0.90:
        return min(max_budget, 2)
    if confidence >= 0.85:
        return min(max_budget, 3)
    return max_budget


POLICIES = {
    "hybrid_aggressive": aggressive_budget,
    "hybrid_conservative": conservative_budget,
}


def evaluate_hybrid(candidates: list[dict[str, Any]], max_budget: int, policy_name: str) -> dict[str, Any]:
    if not candidates:
        return {
            "method": policy_name,
            "success": False,
            "tests_executed": 0,
            "generated_candidates_counted": 0,
            "llm_calls_counted": 0,
            "selected_idx": None,
            "selected_confidence": None,
            "selected_time_ms": None,
            "budget_miss": False,
        }

    initial = candidates[0]
    budget = POLICIES[policy_name](float(initial["confidence"]), max_budget)
    budget = max(1, min(budget, max_budget, len(candidates)))
    selected_pool = list(range(budget))
    order = sorted(selected_pool, key=lambda idx: (-candidates[idx]["confidence"], candidates[idx]["idx"]))

    tests = 0
    for idx in order:
        tests += 1
        cand = candidates[idx]
        if cand["success"]:
            return {
                "method": policy_name,
                "success": True,
                "tests_executed": tests,
                "generated_candidates_counted": budget,
                "llm_calls_counted": 1 if budget == 1 else 2,
                "selected_idx": idx,
                "selected_confidence": cand["confidence"],
                "selected_time_ms": cand["time_ms"],
                "initial_confidence": initial["confidence"],
                "adaptive_budget": budget,
                "confidence_order": order,
                "budget_miss": False,
            }

    full_has_success = any(cand["success"] for cand in candidates[:max_budget])
    return {
        "method": policy_name,
        "success": False,
        "tests_executed": len(order),
        "generated_candidates_counted": budget,
        "llm_calls_counted": 1 if budget == 1 else 2,
        "selected_idx": None,
        "selected_confidence": None,
        "selected_time_ms": None,
        "initial_confidence": initial["confidence"],
        "adaptive_budget": budget,
        "confidence_order": order,
        "budget_miss": full_has_success,
    }


def aggregate(rows: list[dict[str, Any]], policy_name: str) -> dict[str, Any]:
    evals = [row["offline_evaluation"][policy_name] for row in rows]
    total = len(evals)
    selected_times = [e["selected_time_ms"] for e in evals if e["selected_time_ms"] is not None]
    budget_counts: dict[str, int] = {}
    for e in evals:
        key = str(e["adaptive_budget"])
        budget_counts[key] = budget_counts.get(key, 0) + 1
    return {
        "success_count": sum(1 for e in evals if e["success"]),
        "success_rate": sum(1 for e in evals if e["success"]) / total if total else 0.0,
        "avg_tests_executed": mean(e["tests_executed"] for e in evals) if evals else 0.0,
        "avg_generated_candidates_counted": mean(e["generated_candidates_counted"] for e in evals) if evals else 0.0,
        "avg_llm_calls_counted": mean(e["llm_calls_counted"] for e in evals) if evals else 0.0,
        "budget_miss_count": sum(1 for e in evals if e["budget_miss"]),
        "budget_counts": budget_counts,
        "avg_selected_time_ms": mean(selected_times) if selected_times else None,
        "median_selected_time_ms": median(selected_times) if selected_times else None,
    }


def process_dataset(name: str, path: Path, max_budget: int) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out_rows = []
    for row in data["candidate_sets"]:
        candidates = row["candidates"][:max_budget]
        offline = {}
        for policy_name in POLICIES:
            offline[policy_name] = evaluate_hybrid(candidates, max_budget, policy_name)
        copied = {k: v for k, v in row.items() if k != "offline_evaluation"}
        copied["offline_evaluation"] = offline
        out_rows.append(copied)
    return {
        "method": "hybrid_cbba_offline_eval",
        "source": str(path),
        "benchmark": name,
        "max_budget": max_budget,
        "policies": {
            "hybrid_aggressive": "c>=0.90 -> 1, c>=0.85 -> mid, otherwise k; confidence-order tests",
            "hybrid_conservative": "c>=0.90 -> 2, c>=0.85 -> 3, otherwise k; confidence-order tests",
        },
        "summary": {policy: aggregate(out_rows, policy) for policy in POLICIES},
        "candidate_sets": out_rows,
    }


def paper_group_rows(
    datasets: dict[str, dict[str, Any]],
    members: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for dataset_name, problem_name in members:
        for row in datasets[dataset_name]["candidate_sets"]:
            if row["problem_name"] == problem_name:
                rows.append(row)
    return rows


def write_summary(output: Path, datasets: dict[str, dict[str, Any]]) -> None:
    lines = ["# Hybrid CBBA Summary", ""]
    lines.append("Policy definitions:")
    lines.append("- `hybrid_aggressive`: initial confidence >=0.90 uses budget 1; >=0.85 uses a middle budget; otherwise max budget; tests generated candidates by confidence order.")
    lines.append("- `hybrid_conservative`: initial confidence >=0.90 uses budget 2; >=0.85 uses budget 3; otherwise max budget; tests generated candidates by confidence order.")
    lines.append("")
    lines.append("## By Dataset")
    lines.append("")
    lines.append("| Dataset | Policy | Success | Success rate | Avg generated | Avg tests | Avg LLM calls | Budget misses | Budget counts |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for name, data in datasets.items():
        total = len(data["candidate_sets"])
        for policy, row in data["summary"].items():
            lines.append(
                f"| {name} | {policy} | {row['success_count']}/{total} | {row['success_rate']:.3f} | "
                f"{row['avg_generated_candidates_counted']:.3f} | {row['avg_tests_executed']:.3f} | "
                f"{row['avg_llm_calls_counted']:.3f} | {row['budget_miss_count']} | {row['budget_counts']} |"
            )
    lines.append("")
    lines.append("## Paper Groups")
    lines.append("")
    lines.append("| Group | Policy | Success | Success rate | Avg generated | Avg tests | Avg LLM calls | Budget misses |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for group, members in PAPER_GROUPS.items():
        rows = paper_group_rows(datasets, members)
        total = len(rows)
        for policy in POLICIES:
            row = aggregate(rows, policy)
            lines.append(
                f"| {group} | {policy} | {row['success_count']}/{total} | {row['success_rate']:.3f} | "
                f"{row['avg_generated_candidates_counted']:.3f} | {row['avg_tests_executed']:.3f} | "
                f"{row['avg_llm_calls_counted']:.3f} | {row['budget_miss_count']} |"
            )
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-easy4", type=Path, required=True)
    parser.add_argument("--paper-medium4", type=Path, required=True)
    parser.add_argument("--paper-hard4", type=Path, required=True)
    parser.add_argument("--budget-paper-easy4", type=int, default=5)
    parser.add_argument("--budget-paper-medium4", type=int, default=5)
    parser.add_argument("--budget-paper-hard4", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "paper_easy4": process_dataset("paper_easy4", args.paper_easy4, args.budget_paper_easy4),
        "paper_medium4": process_dataset("paper_medium4", args.paper_medium4, args.budget_paper_medium4),
        "paper_hard4": process_dataset("paper_hard4", args.paper_hard4, args.budget_paper_hard4),
    }
    for name, data in datasets.items():
        output_path = args.output_dir / f"hybrid_cbba_{name}.json"
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(args.output_dir / "hybrid_cbba_summary.md", datasets)
    print(args.output_dir / "hybrid_cbba_summary.md")


if __name__ == "__main__":
    main()
