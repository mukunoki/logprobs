#!/usr/bin/env python3
"""
Shared-candidate offline evaluation for CBBA.

This script generates one candidate set per problem/trial, tests every candidate once,
and then evaluates multiple selection orders offline on the exact same candidates.
It is intended to separate the effect of early stopping from the effect of confidence
ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.append(os.path.dirname(__file__))

from cbba_method import compute_average_probability, extract_c_code, test_c_code
from paper_benchmark_sets import BENCHMARKS, LEGACY_BENCHMARK_ALIASES
from ugir import MODEL_NAME, VLLM_API_URL, generate_with_logprobs


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"


def canonical_benchmark(name: str) -> str:
    return LEGACY_BENCHMARK_ALIASES.get(name, name)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def get_gcc_version() -> str:
    try:
        result = subprocess.run(
            ["gcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.splitlines()[0] if result.stdout else "unknown"
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"unavailable: {exc}"


def get_cpu_model() -> str:
    try:
        result = subprocess.run(
            ["lscpu"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
        return "unknown"
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"unavailable: {exc}"


def classify_test_message(success: bool, message: str) -> str:
    if success:
        return "pass"

    lowered = message.lower()
    if "compile" in lowered:
        return "compile_error"
    if "timeout" in lowered:
        return "timeout"
    if "failed" in lowered:
        return "test_failed"
    if "extract" in lowered:
        return "code_extraction_failure"
    return "unknown_failure"


def make_candidate_record(raw_candidate: Dict[str, Any], idx: int, test_code: str) -> Dict[str, Any]:
    generated_text = raw_candidate.get("text", "")
    code = extract_c_code(generated_text)
    confidence = compute_average_probability(raw_candidate)
    token_logprobs = raw_candidate.get("token_logprobs", [])
    valid_logprobs = [lp for lp in token_logprobs if lp is not None]

    start = time.time()
    success, time_ms, message = test_c_code(code, test_code)
    test_elapsed = time.time() - start

    return {
        "idx": idx,
        "confidence": confidence,
        "success": success,
        "time_ms": time_ms if success else None,
        "message": message,
        "error_type": classify_test_message(success, message),
        "test_elapsed_sec": test_elapsed,
        "code_hash": sha256_text(code),
        "generated_text_hash": sha256_text(generated_text),
        "code": code,
        "generated_text": generated_text,
        "token_count": len(token_logprobs),
        "valid_logprob_count": len(valid_logprobs),
        "avg_logprob": (sum(valid_logprobs) / len(valid_logprobs)) if valid_logprobs else None,
        "token_logprobs": token_logprobs,
    }


def evaluate_order(candidates: List[Dict[str, Any]], order: Iterable[int], method: str) -> Dict[str, Any]:
    ordered_indices = list(order)
    tests_executed = 0

    for idx in ordered_indices:
        tests_executed += 1
        cand = candidates[idx]
        if cand["success"]:
            return {
                "method": method,
                "success": True,
                "tests_executed": tests_executed,
                "selected_idx": idx,
                "selected_confidence": cand["confidence"],
                "selected_time_ms": cand["time_ms"],
                "order": ordered_indices,
            }

    return {
        "method": method,
        "success": False,
        "tests_executed": len(ordered_indices),
        "selected_idx": None,
        "selected_confidence": None,
        "selected_time_ms": None,
        "order": ordered_indices,
    }


def evaluate_full_topk(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    passing = [cand for cand in candidates if cand["success"]]
    if not passing:
        return {
            "method": "full_topk_best_time",
            "success": False,
            "tests_executed": len(candidates),
            "selected_idx": None,
            "selected_confidence": None,
            "selected_time_ms": None,
        }

    selected = min(passing, key=lambda cand: cand["time_ms"] if cand["time_ms"] is not None else float("inf"))
    return {
        "method": "full_topk_best_time",
        "success": True,
        "tests_executed": len(candidates),
        "selected_idx": selected["idx"],
        "selected_confidence": selected["confidence"],
        "selected_time_ms": selected["time_ms"],
    }


def evaluate_oracle(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    passing = [cand for cand in candidates if cand["success"]]
    if not passing:
        return {
            "method": "oracle_early_stop",
            "success": False,
            "tests_executed": len(candidates),
            "selected_idx": None,
            "selected_confidence": None,
            "selected_time_ms": None,
        }

    selected = min(passing, key=lambda cand: cand["time_ms"] if cand["time_ms"] is not None else float("inf"))
    return {
        "method": "oracle_early_stop",
        "success": True,
        "tests_executed": 1,
        "selected_idx": selected["idx"],
        "selected_confidence": selected["confidence"],
        "selected_time_ms": selected["time_ms"],
    }


def evaluate_random_orders(
    candidates: List[Dict[str, Any]],
    repeats: int,
    base_seed: int,
    trial: int,
    problem_index: int,
) -> Dict[str, Any]:
    per_repeat = []
    for repeat in range(repeats):
        rng = random.Random(base_seed + trial * 100000 + problem_index * 1000 + repeat)
        order = list(range(len(candidates)))
        rng.shuffle(order)
        per_repeat.append(evaluate_order(candidates, order, "random_order_early_stop"))

    tests = [item["tests_executed"] for item in per_repeat]
    selected_times = [item["selected_time_ms"] for item in per_repeat if item["selected_time_ms"] is not None]
    success_rate = sum(1 for item in per_repeat if item["success"]) / repeats if repeats else 0.0

    return {
        "method": "random_order_early_stop",
        "repeats": repeats,
        "success_rate_over_repeats": success_rate,
        "avg_tests_executed": statistics.mean(tests) if tests else 0.0,
        "min_tests_executed": min(tests) if tests else 0,
        "max_tests_executed": max(tests) if tests else 0,
        "avg_selected_time_ms": statistics.mean(selected_times) if selected_times else None,
        "per_repeat": per_repeat,
    }


def evaluate_candidate_set(
    candidates: List[Dict[str, Any]],
    random_repeats: int,
    random_seed: int,
    trial: int,
    problem_index: int,
) -> Dict[str, Any]:
    generation_order = list(range(len(candidates)))
    confidence_order = sorted(
        generation_order,
        key=lambda idx: (-candidates[idx]["confidence"], candidates[idx]["idx"]),
    )

    return {
        "full_topk_best_time": evaluate_full_topk(candidates),
        "generation_order_early_stop": evaluate_order(
            candidates, generation_order, "generation_order_early_stop"
        ),
        "confidence_order_early_stop": evaluate_order(
            candidates, confidence_order, "confidence_order_early_stop"
        ),
        "random_order_early_stop": evaluate_random_orders(
            candidates, random_repeats, random_seed, trial, problem_index
        ),
        "oracle_early_stop": evaluate_oracle(candidates),
    }


def confidence_ranking_metrics(candidate_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
    reciprocal_ranks = []
    recall_at_1 = 0
    recall_at_2 = 0
    success_sets = 0

    for item in candidate_sets:
        candidates = item["candidates"]
        ranked = sorted(
            range(len(candidates)),
            key=lambda idx: (-candidates[idx]["confidence"], candidates[idx]["idx"]),
        )
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
        "mrr": statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "recall_at_1": recall_at_1 / success_sets if success_sets else 0.0,
        "recall_at_2": recall_at_2 / success_sets if success_sets else 0.0,
    }


def aggregate_results(candidate_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
    methods = [
        "full_topk_best_time",
        "generation_order_early_stop",
        "confidence_order_early_stop",
        "oracle_early_stop",
    ]
    total_instances = len(candidate_sets)
    summary: Dict[str, Any] = {}

    for method in methods:
        rows = [item["offline_evaluation"][method] for item in candidate_sets]
        successes = sum(1 for row in rows if row["success"])
        tests = [row["tests_executed"] for row in rows]
        selected_times = [row["selected_time_ms"] for row in rows if row["selected_time_ms"] is not None]
        summary[method] = {
            "success_count": successes,
            "success_rate": successes / total_instances if total_instances else 0.0,
            "total_tests_executed": sum(tests),
            "avg_tests_executed": statistics.mean(tests) if tests else 0.0,
            "avg_selected_time_ms": statistics.mean(selected_times) if selected_times else None,
            "median_selected_time_ms": statistics.median(selected_times) if selected_times else None,
        }

    random_rows = [item["offline_evaluation"]["random_order_early_stop"] for item in candidate_sets]
    if random_rows:
        summary["random_order_early_stop"] = {
            "avg_success_rate_over_repeats": statistics.mean(
                row["success_rate_over_repeats"] for row in random_rows
            ),
            "avg_tests_executed": statistics.mean(row["avg_tests_executed"] for row in random_rows),
            "total_tests_executed_mean": sum(row["avg_tests_executed"] for row in random_rows),
            "avg_selected_time_ms": statistics.mean(
                row["avg_selected_time_ms"] for row in random_rows if row["avg_selected_time_ms"] is not None
            )
            if any(row["avg_selected_time_ms"] is not None for row in random_rows)
            else None,
        }

    summary["confidence_ranking"] = confidence_ranking_metrics(candidate_sets)
    return summary


def build_output(args: argparse.Namespace, candidate_sets: List[Dict[str, Any]], elapsed_sec: float) -> Dict[str, Any]:
    problems = list(BENCHMARKS[args.benchmark])
    if args.limit is not None:
        problems = problems[: args.limit]

    return {
        "method": "shared_candidate_offline_eval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": canonical_benchmark(args.benchmark),
        "num_trials": args.num_trials,
        "num_problems": len(problems),
        "k": args.k,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "random_repeats": args.random_repeats,
        "random_seed": args.random_seed,
        "llm": {
            "model": MODEL_NAME,
            "api_url": VLLM_API_URL,
            "generation_seed_controlled": False,
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "gcc": get_gcc_version(),
            "cpu_model": get_cpu_model(),
        },
        "elapsed_sec": elapsed_sec,
        "is_partial": len(candidate_sets) < len(problems) * args.num_trials,
        "summary": aggregate_results(candidate_sets) if candidate_sets else {},
        "candidate_sets": candidate_sets,
    }


def save_output(path: str, output: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    benchmark = canonical_benchmark(args.benchmark)
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    problems = list(BENCHMARKS[benchmark])
    if args.limit is not None:
        problems = problems[: args.limit]

    candidate_sets = []
    completed_keys = set()
    if args.resume and os.path.exists(args.output):
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
        print(
            f"Resuming from {args.output}: {len(candidate_sets)} completed problem instances",
            flush=True,
        )

    started_at = time.time()

    for trial in range(1, args.num_trials + 1):
        print(f"\n=== Trial {trial}/{args.num_trials} ===", flush=True)
        for problem_index, problem in enumerate(problems):
            if (trial, problem_index) in completed_keys:
                print(
                    f"[{problem_index + 1}/{len(problems)}] {problem['name']} "
                    f"({problem.get('category', 'unknown')}) - skipped (already completed)",
                    flush=True,
                )
                continue

            print(
                f"[{problem_index + 1}/{len(problems)}] {problem['name']} "
                f"({problem.get('category', 'unknown')})",
                flush=True,
            )
            problem_start = time.time()
            raw_candidates = generate_with_logprobs(
                problem["optimization_prompt"],
                n=args.k,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            candidates = []
            for idx, raw_candidate in enumerate(raw_candidates):
                print(f"  testing candidate {idx + 1}/{len(raw_candidates)}", flush=True)
                candidates.append(make_candidate_record(raw_candidate, idx, problem["test_code"]))

            offline = evaluate_candidate_set(
                candidates,
                random_repeats=args.random_repeats,
                random_seed=args.random_seed,
                trial=trial,
                problem_index=problem_index,
            )

            candidate_sets.append(
                {
                    "trial": trial,
                    "problem_index": problem_index,
                    "problem_name": problem["name"],
                    "category": problem.get("category", "unknown"),
                    "candidate_count": len(candidates),
                    "problem_elapsed_sec": time.time() - problem_start,
                    "candidates": candidates,
                    "offline_evaluation": offline,
                }
            )

            if args.checkpoint_every and len(candidate_sets) % args.checkpoint_every == 0:
                output = build_output(args, candidate_sets, time.time() - started_at)
                save_output(args.output, output)
                print(f"  checkpoint saved to {args.output}", flush=True)

    output = build_output(args, candidate_sets, time.time() - started_at)
    save_output(args.output, output)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="paper12")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=20260421)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N problems for smoke tests.")
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "shared_candidate_offline_eval_paper12.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save a partial JSON after this many problem instances. Set 0 to disable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output JSON and skip completed trial/problem pairs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
