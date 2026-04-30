#!/usr/bin/env python3
"""Parallel shared-candidate evaluation for threshold-refinement CBBA."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from paper_benchmark_sets import BENCHMARKS
from threshold_refinement_eval import (
    RESULTS_DIR,
    build_output,
    canonical_benchmark,
    evaluate_candidate_set,
    make_candidate_record,
    refinement_prompt,
    save_output,
)
from ugir import generate_with_logprobs


def sorted_candidate_sets(candidate_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidate_sets, key=lambda item: (item["trial"], item["problem_index"]))


def run_one(args: argparse.Namespace, trial: int, problem_index: int, problem: dict[str, Any]) -> dict[str, Any]:
    print(
        f"[start] trial={trial}/{args.num_trials} "
        f"problem={problem_index + 1}/{args.num_problems} {problem['name']}",
        flush=True,
    )
    problem_start = time.time()

    initial_raw = generate_with_logprobs(
        problem["optimization_prompt"],
        n=1,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    refine_raw = generate_with_logprobs(
        refinement_prompt(problem["optimization_prompt"]),
        n=max(args.budget - 1, 0),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    candidates = []
    for idx, raw_candidate in enumerate(initial_raw):
        candidates.append(make_candidate_record(raw_candidate, idx, "initial", problem["test_code"]))
    for offset, raw_candidate in enumerate(refine_raw, start=len(candidates)):
        candidates.append(make_candidate_record(raw_candidate, offset, "refinement", problem["test_code"]))

    offline = evaluate_candidate_set(candidates, args.thresholds, args.budget)
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


def build_parallel_output(args: argparse.Namespace, candidate_sets: list[dict[str, Any]], elapsed_sec: float) -> dict[str, Any]:
    output = build_output(args, sorted_candidate_sets(candidate_sets), elapsed_sec)
    output["workers"] = args.workers
    output["method"] = "threshold_refinement_eval_parallel"
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    benchmark = canonical_benchmark(args.benchmark)
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    problems = list(BENCHMARKS[benchmark])
    if args.limit is not None:
        problems = problems[: args.limit]
    args.num_problems = len(problems)

    candidate_sets, completed_keys = load_existing(args, benchmark)
    tasks = [
        (trial, problem_index, problem)
        for trial in range(1, args.num_trials + 1)
        for problem_index, problem in enumerate(problems)
        if (trial, problem_index) not in completed_keys
    ]
    print(
        f"Parallel run: benchmark={benchmark} trials={args.num_trials} "
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
                output = build_parallel_output(args, candidate_sets, time.time() - started_at)
                save_output(args.output, output)
                completed_since_checkpoint = 0
                print(f"checkpoint saved to {args.output}: {len(candidate_sets)} instances", flush=True)

    output = build_parallel_output(args, candidate_sets, time.time() - started_at)
    save_output(args.output, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="paper_competitive4")
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.80, 0.85, 0.90])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=str(RESULTS_DIR / "threshold_refinement_eval_paper_competitive4_b5.json"))
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = run(parse_args())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr, flush=True)
        raise
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
