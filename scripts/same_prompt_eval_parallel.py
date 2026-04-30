#!/usr/bin/env python3
"""Parallel same-prompt candidate evaluation for CBBA.

Each problem/trial generates all candidates from the same problem prompt.  This
removes the previous initial/refinement mixture and makes confidence-based
ordering compare candidates conditioned on the same prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

from paper_benchmark_sets import BENCHMARKS
from threshold_refinement_eval import (
    RESULTS_DIR,
    build_output,
    canonical_benchmark,
    make_candidate_record,
    save_output,
)
from ugir import generate_with_logprobs


def sorted_candidate_sets(candidate_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidate_sets, key=lambda item: (item["trial"], item["problem_index"]))


def aggressive_budget(confidence: float, max_budget: int) -> int:
    if confidence >= 0.90:
        return 1
    if confidence >= 0.85:
        return min(max_budget, max(2, (max_budget + 2) // 2))
    return max_budget


def conservative_budget(confidence: float, max_budget: int) -> int:
    if confidence >= 0.90:
        return min(max_budget, 2)
    if confidence >= 0.85:
        return min(max_budget, 3)
    return max_budget


def confidence_order(candidates: list[dict[str, Any]], pool_size: int) -> list[int]:
    selected_pool = list(range(min(pool_size, len(candidates))))
    return sorted(selected_pool, key=lambda idx: (-candidates[idx]["confidence"], candidates[idx]["idx"]))


def evaluate_adaptive(candidates: list[dict[str, Any]], max_budget: int, policy_name: str) -> dict[str, Any]:
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
    policy = aggressive_budget if policy_name == "hybrid_aggressive" else conservative_budget
    budget = max(1, min(policy(float(initial["confidence"]), max_budget), max_budget, len(candidates)))
    order = confidence_order(candidates, budget)

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


def evaluate_candidate_set(candidates: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    return {
        "hybrid_conservative": evaluate_adaptive(candidates, budget, "hybrid_conservative"),
        "hybrid_aggressive": evaluate_adaptive(candidates, budget, "hybrid_aggressive"),
    }


def run_one(args: argparse.Namespace, trial: int, problem_index: int, problem: dict[str, Any]) -> dict[str, Any]:
    print(
        f"[start] trial={trial}/{args.num_trials} "
        f"problem={problem_index + 1}/{args.num_problems} {problem['name']}",
        flush=True,
    )
    problem_start = time.time()

    raw_candidates = generate_with_logprobs(
        problem["optimization_prompt"],
        n=args.budget,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    candidates = [
        make_candidate_record(raw_candidate, idx, "same_prompt", problem["test_code"])
        for idx, raw_candidate in enumerate(raw_candidates)
    ]

    offline = evaluate_candidate_set(candidates, args.budget)
    row = {
        "trial": trial,
        "problem_index": problem_index,
        "problem_name": problem["name"],
        "category": problem.get("category", "unknown"),
        "candidate_count": len(candidates),
        "problem_elapsed_sec": time.time() - problem_start,
        "candidates": candidates,
        "offline_evaluation": offline,
    }
    print(
        f"[done] trial={trial}/{args.num_trials} "
        f"problem={problem_index + 1}/{args.num_problems} {problem['name']} "
        f"elapsed={row['problem_elapsed_sec']:.1f}s",
        flush=True,
    )
    return row


def load_existing(args: argparse.Namespace, benchmark: str) -> tuple[list[dict[str, Any]], set[tuple[int, int]]]:
    if not args.resume or not os.path.exists(args.output):
        return [], set()

    with open(args.output, "r", encoding="utf-8") as f:
        existing = json.load(f)
    if canonical_benchmark(existing.get("benchmark", "")) != benchmark:
        raise ValueError(
            f"Cannot resume: output benchmark={existing.get('benchmark')} "
            f"does not match requested benchmark={benchmark}"
        )
    candidate_sets = existing.get("candidate_sets", [])
    completed_keys = {
        (item["trial"], item["problem_index"])
        for item in candidate_sets
        if "trial" in item and "problem_index" in item
    }
    print(f"Resuming from {args.output}: {len(candidate_sets)} completed instances", flush=True)
    return candidate_sets, completed_keys


def method_success_rate(candidate_sets: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidate_sets:
        return {}
    full_success = [any(c["success"] for c in row["candidates"]) for row in candidate_sets]
    first_success = [row["candidates"][0]["success"] if row["candidates"] else False for row in candidate_sets]
    hybc = [row["offline_evaluation"]["hybrid_conservative"] for row in candidate_sets]
    hyba = [row["offline_evaluation"]["hybrid_aggressive"] for row in candidate_sets]
    total = len(candidate_sets)
    return {
        "full_success_count": sum(full_success),
        "full_success_rate": sum(full_success) / total,
        "first_success_count": sum(first_success),
        "first_success_rate": sum(first_success) / total,
        "cbba_c_success_count": sum(item["success"] for item in hybc),
        "cbba_c_success_rate": sum(item["success"] for item in hybc) / total,
        "cbba_c_avg_generated": mean(item["generated_candidates_counted"] for item in hybc),
        "cbba_c_avg_tests": mean(item["tests_executed"] for item in hybc),
        "cbba_a_success_count": sum(item["success"] for item in hyba),
        "cbba_a_success_rate": sum(item["success"] for item in hyba) / total,
        "cbba_a_avg_generated": mean(item["generated_candidates_counted"] for item in hyba),
        "cbba_a_avg_tests": mean(item["tests_executed"] for item in hyba),
    }


def build_same_prompt_output(args: argparse.Namespace, candidate_sets: list[dict[str, Any]], elapsed_sec: float) -> dict[str, Any]:
    output = build_output(args, sorted_candidate_sets(candidate_sets), elapsed_sec)
    output["workers"] = args.workers
    output["method"] = "same_prompt_eval_parallel"
    output["generation_protocol"] = "same_prompt_n"
    output["summary"] = method_success_rate(output["candidate_sets"])
    return output


def save_same_prompt_output(path: str, output: dict[str, Any], compact: bool) -> None:
    if not compact:
        save_output(path, output)
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp_path, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    benchmark = canonical_benchmark(args.benchmark)
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    problems = list(BENCHMARKS[benchmark])
    if args.problem_name:
        wanted = set(args.problem_name)
        selected = [problem for problem in problems if problem["name"] in wanted]
        found = {problem["name"] for problem in selected}
        missing = [name for name in args.problem_name if name not in found]
        if missing:
            raise ValueError(f"Unknown problem name(s) for benchmark {benchmark}: {missing}")
        order = {name: idx for idx, name in enumerate(args.problem_name)}
        problems = sorted(selected, key=lambda problem: order[problem["name"]])
    if args.limit is not None:
        problems = problems[: args.limit]
    args.num_problems = len(problems)
    args.thresholds = []

    candidate_sets, completed_keys = load_existing(args, benchmark)
    tasks = [
        (trial, problem_index, problem)
        for trial in range(1, args.num_trials + 1)
        for problem_index, problem in enumerate(problems)
        if (trial, problem_index) not in completed_keys
    ]
    print(
        f"Same-prompt parallel run: benchmark={benchmark} trials={args.num_trials} "
        f"problems={len(problems)} tasks={len(tasks)} workers={args.workers}",
        flush=True,
    )

    started_at = time.time()
    completed_since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, args, trial, problem_index, problem) for trial, problem_index, problem in tasks]
        for future in as_completed(futures):
            candidate_sets.append(future.result())
            completed_since_checkpoint += 1
            if args.checkpoint_every and completed_since_checkpoint >= args.checkpoint_every:
                output = build_same_prompt_output(args, candidate_sets, time.time() - started_at)
                save_same_prompt_output(args.output, output, args.compact_output)
                completed_since_checkpoint = 0
                print(f"checkpoint saved to {args.output}: {len(candidate_sets)} instances", flush=True)

    output = build_same_prompt_output(args, candidate_sets, time.time() - started_at)
    save_same_prompt_output(args.output, output, args.compact_output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="paper_easy4")
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--problem-name",
        action="append",
        default=[],
        help="Restrict the benchmark to the named problem. Can be repeated.",
    )
    parser.add_argument("--output", default=str(RESULTS_DIR / "same_prompt_eval_paper_easy4_b5.json"))
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compact-output", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = run(parse_args())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr, flush=True)
        raise
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
