#!/usr/bin/env python3
"""Summarize paper CBBA results from hybrid candidate pools."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def default_hybrid_dir() -> Path:
    candidates = sorted(
        REPO_ROOT.glob("results/paper12_k5_*/hybrid"),
        key=lambda path: path.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]
    return REPO_ROOT / "results" / "paper12_k5_latest" / "hybrid"


DEFAULT_HYBRID_DIR = Path(os.environ.get("CBBA_HYBRID_DIR", default_hybrid_dir()))

DATASET_FILES = {
    "paper_easy4": "hybrid_cbba_paper_easy4.json",
    "paper_medium4": "hybrid_cbba_paper_medium4.json",
    "paper_hard4": "hybrid_cbba_paper_hard4.json",
}

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

PROBLEM_LABELS = {
    "array_sum_unroll": "ArraySum",
    "matrix_transpose_cache": "Transpose",
    "csr_spmv_axpy_dot": "SpMV",
    "rmsnorm_mixed": "RMSNorm",
    "vector_add_simd": "VecAdd",
    "floyd_warshall_blocked": "Floyd",
    "max_value_branchless": "MaxVal",
    "conv2d_3x3_multi_channel": "Conv2D",
    "heap_sort_implementation": "HeapSort",
    "stencil3d_mixed_7pt": "Stencil",
    "radix_sort_u32_pairs": "Radix",
    "nbody_tiled_step": "NBody",
}

METHOD_DISPLAY = ["First", "Full", "Gen", "Rand", "Mean", "Tail", "TG-Tail"]


def load_datasets(hybrid_dir: Path) -> dict[str, dict[str, Any]]:
    datasets = {}
    for dataset_name, filename in DATASET_FILES.items():
        path = hybrid_dir / filename
        datasets[dataset_name] = json.loads(path.read_text(encoding="utf-8"))
    return datasets


def candidate_order_by_confidence(candidates: list[dict[str, Any]]) -> list[int]:
    return sorted(
        range(len(candidates)),
        key=lambda idx: (-candidates[idx]["confidence"], candidates[idx]["idx"]),
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("-inf")
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def tail_confidence(candidate: dict[str, Any], q: float = 0.25) -> float:
    return percentile([float(x) for x in candidate.get("token_logprobs", [])], q)


def candidate_order_by_tail_confidence(candidates: list[dict[str, Any]]) -> list[int]:
    return sorted(
        range(len(candidates)),
        key=lambda idx: (-tail_confidence(candidates[idx]), candidates[idx]["idx"]),
    )


def evaluate_order(candidates: list[dict[str, Any]], order: list[int]) -> dict[str, Any]:
    k = len(order)
    for tests_executed, idx in enumerate(order, start=1):
        if candidates[idx]["success"]:
            return {"success": True, "tests_executed": float(tests_executed)}
    return {"success": False, "tests_executed": float(k)}


def evaluate_full(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    k = len(candidates)
    return {"success": any(cand["success"] for cand in candidates), "tests_executed": float(k)}


def evaluate_random_expected(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    k = len(candidates)
    success_count = sum(1 for cand in candidates if cand["success"])
    if success_count == 0:
        return {"success": False, "tests_executed": float(k)}
    return {"success": True, "tests_executed": float((k + 1) / (success_count + 1))}


def evaluate_test_gated_tail(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {"success": False, "tests_executed": 0.0, "generated": 0.0}
    if candidates[0]["success"]:
        return {"success": True, "tests_executed": 1.0, "generated": 1.0}

    generated = float(len(candidates))
    rest_order = sorted(
        range(1, len(candidates)),
        key=lambda idx: (-tail_confidence(candidates[idx]), candidates[idx]["idx"]),
    )
    for tests_executed, idx in enumerate(rest_order, start=2):
        if candidates[idx]["success"]:
            return {"success": True, "tests_executed": float(tests_executed), "generated": generated}
    return {"success": False, "tests_executed": generated, "generated": generated}


def per_row_methods(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = row["candidates"]
    k = float(len(candidates))
    gen = evaluate_order(candidates, list(range(len(candidates))))
    rand = evaluate_random_expected(candidates)
    cbba = evaluate_order(candidates, candidate_order_by_confidence(candidates))
    tail_cbba = evaluate_order(candidates, candidate_order_by_tail_confidence(candidates))
    test_gated_tail = evaluate_test_gated_tail(candidates)
    return {
        "First": {
            "success": bool(candidates and candidates[0]["success"]),
            "tests_executed": 1.0 if candidates else 0.0,
            "generated": 1.0 if candidates else 0.0,
        },
        "Full": {"success": any(c["success"] for c in candidates), "tests_executed": k, "generated": k},
        "Gen": {"success": gen["success"], "tests_executed": gen["tests_executed"], "generated": k},
        "Rand": {"success": rand["success"], "tests_executed": rand["tests_executed"], "generated": k},
        "Mean": {"success": cbba["success"], "tests_executed": cbba["tests_executed"], "generated": k},
        "Tail": {
            "success": tail_cbba["success"],
            "tests_executed": tail_cbba["tests_executed"],
            "generated": k,
        },
        "TG-Tail": {
            "success": test_gated_tail["success"],
            "tests_executed": test_gated_tail["tests_executed"],
            "generated": test_gated_tail["generated"],
        },
    }


def classify_candidate_pool(row: dict[str, Any]) -> str:
    candidates = row["candidates"]
    if any(cand["success"] for cand in candidates):
        return "pass"
    if all(cand.get("error_type") == "compile_error" for cand in candidates):
        return "compile_error"
    if any(cand.get("error_type") == "test_failed" for cand in candidates):
        return "test_failed"
    return "compile_error"


def ranking_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reciprocal_ranks: list[float] = []
    recall_at_1 = 0
    recall_at_2 = 0
    success_sets = 0

    for row in rows:
        candidates = row["candidates"]
        ranked = candidate_order_by_confidence(candidates)
        success_positions = [rank for rank, idx in enumerate(ranked, start=1) if candidates[idx]["success"]]
        if not success_positions:
            continue
        success_sets += 1
        first_success_rank = min(success_positions)
        reciprocal_ranks.append(1.0 / first_success_rank)
        if first_success_rank <= 1:
            recall_at_1 += 1
        if first_success_rank <= 2:
            recall_at_2 += 1

    return {
        "sets_with_any_success": success_sets,
        "recall_at_1": recall_at_1 / success_sets if success_sets else 0.0,
        "recall_at_2": recall_at_2 / success_sets if success_sets else 0.0,
        "mrr": mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = {name: [] for name in METHOD_DISPLAY}
    pool_counts = {"pass": 0, "compile_error": 0, "test_failed": 0}
    diagnostic_counts = {
        "no_success_rows_with_any_test_failed_candidate": 0,
        "no_success_rows_all_compile_error": 0,
    }

    for row in rows:
        evaluated = per_row_methods(row)
        for method_name in METHOD_DISPLAY:
            methods[method_name].append(evaluated[method_name])
        pool_counts[classify_candidate_pool(row)] += 1
        if not any(cand["success"] for cand in row["candidates"]):
            if any(cand.get("error_type") == "test_failed" for cand in row["candidates"]):
                diagnostic_counts["no_success_rows_with_any_test_failed_candidate"] += 1
            else:
                diagnostic_counts["no_success_rows_all_compile_error"] += 1

    method_summary = {}
    total = len(rows)
    for method_name, metrics in methods.items():
        method_summary[method_name] = {
            "success_count": sum(1 for item in metrics if item["success"]),
            "success_rate": sum(1 for item in metrics if item["success"]) / total if total else 0.0,
            "avg_tests": mean(item["tests_executed"] for item in metrics) if metrics else 0.0,
            "avg_generated": mean(item["generated"] for item in metrics) if metrics else 0.0,
        }

    return {
        "instances": total,
        "methods": method_summary,
        "candidate_pool_breakdown": pool_counts,
        "candidate_pool_breakdown_diagnostics": diagnostic_counts,
        "ranking_metrics": ranking_metrics(rows),
    }


def select_rows(datasets: dict[str, dict[str, Any]], pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    row_map = {}
    for dataset_name, data in datasets.items():
        trial_map = {}
        for row in data["candidate_sets"]:
            trial_map.setdefault(row["problem_name"], []).append(row)
        row_map[dataset_name] = trial_map

    selected = []
    for dataset_name, problem_name in pairs:
        rows = row_map[dataset_name][problem_name]
        selected.extend(rows)
    return sorted(selected, key=lambda row: (row["trial"], row["problem_name"]))


def build_summary(datasets: dict[str, dict[str, Any]], hybrid_dir: Path) -> dict[str, Any]:
    group_summaries = {}
    all_rows: list[dict[str, Any]] = []

    for group_name, pairs in PAPER_GROUPS.items():
        rows = select_rows(datasets, pairs)
        group_summaries[group_name] = {
            "problems": [PROBLEM_LABELS[problem_name] for _, problem_name in pairs],
            "problem_ids": [problem_name for _, problem_name in pairs],
            "summary": summarize_rows(rows),
        }
        all_rows.extend(rows)

    overall_rows = sorted(all_rows, key=lambda row: (row["trial"], row["problem_name"]))
    return {
        "hybrid_dir": str(hybrid_dir),
        "paper_groups": group_summaries,
        "overall": summarize_rows(overall_rows),
    }


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    overall = summary["overall"]
    lines = ["# Paper Summary", ""]
    lines.append(f"- Source: `{summary['hybrid_dir']}`")
    lines.append(f"- Total instances: {overall['instances']}")
    lines.append("")
    lines.append("## Overall Methods")
    lines.append("")
    lines.append("| Method | Success | Success rate | Avg tests | Avg generated |")
    lines.append("|---|---:|---:|---:|---:|")
    for method_name in METHOD_DISPLAY:
        row = overall["methods"][method_name]
        lines.append(
            f"| {method_name} | {row['success_count']}/{overall['instances']} | "
            f"{row['success_rate'] * 100:.1f}% | {row['avg_tests']:.2f} | {row['avg_generated']:.2f} |"
        )
    lines.append("")
    lines.append("## Group Methods")
    lines.append("")
    lines.append("| Group | Problems | Method | Success | Success rate | Avg tests | Avg generated |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for group_name, group_data in summary["paper_groups"].items():
        problems = ", ".join(group_data["problems"])
        instances = group_data["summary"]["instances"]
        for method_name in METHOD_DISPLAY:
            row = group_data["summary"]["methods"][method_name]
            lines.append(
                f"| {group_name} | {problems} | {method_name} | {row['success_count']}/{instances} | "
                f"{row['success_rate'] * 100:.1f}% | {row['avg_tests']:.2f} | {row['avg_generated']:.2f} |"
            )
    lines.append("")
    lines.append("## Candidate Pool Breakdown")
    lines.append("")
    lines.append("| Group | PASS | Compile error | Test failed |")
    lines.append("|---|---:|---:|---:|")
    for group_name, group_data in summary["paper_groups"].items():
        breakdown = group_data["summary"]["candidate_pool_breakdown"]
        lines.append(
            f"| {group_name} | {breakdown['pass']} | {breakdown['compile_error']} | {breakdown['test_failed']} |"
        )
    lines.append("")
    lines.append("## Ranking Metrics")
    lines.append("")
    lines.append("| Group | Recall@1 | Recall@2 | MRR |")
    lines.append("|---|---:|---:|---:|")
    for group_name, group_data in summary["paper_groups"].items():
        metrics = group_data["summary"]["ranking_metrics"]
        lines.append(
            f"| {group_name} | {metrics['recall_at_1'] * 100:.1f}% | {metrics['recall_at_2'] * 100:.1f}% | {metrics['mrr']:.3f} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-dir", type=Path, default=DEFAULT_HYBRID_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hybrid_dir = args.hybrid_dir
    output_dir = args.output_dir or hybrid_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(hybrid_dir)
    summary = build_summary(datasets, hybrid_dir)

    json_path = output_dir / "paper_summary.json"
    md_path = output_dir / "paper_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, md_path)

    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
