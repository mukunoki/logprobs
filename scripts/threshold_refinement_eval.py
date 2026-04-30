#!/usr/bin/env python3
"""
Shared-candidate evaluation for threshold-refinement CBBA.

For each problem/trial, this script generates one initial candidate and a fixed
set of refinement candidates. Thresholds are then evaluated offline on the same
candidate set so that threshold comparisons are not affected by generation
randomness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    except Exception as exc:
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
    except Exception as exc:
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


def refinement_prompt(problem_prompt: str) -> str:
    return f"""{problem_prompt}

IMPORTANT: Generate an IMPROVED version with:
1. Better optimization techniques
2. Correct algorithm implementation
3. Proper edge case handling
4. Return only C code
5. Required #include directives and helper definitions are allowed
6. Do not include markdown fences or explanatory text."""


def make_candidate_record(
    raw_candidate: Dict[str, Any],
    idx: int,
    source: str,
    test_code: str,
) -> Dict[str, Any]:
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
        "source": source,
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
        "finish_reason": raw_candidate.get("finish_reason"),
        "stop_reason": raw_candidate.get("stop_reason"),
    }


def evaluate_threshold(candidates: List[Dict[str, Any]], threshold: float, budget: int) -> Dict[str, Any]:
    if not candidates:
        return {
            "threshold": threshold,
            "success": False,
            "tests_executed": 0,
            "generated_candidates_counted": 0,
            "llm_calls_counted": 0,
            "refinement_triggered": False,
            "selected_idx": None,
            "selected_source": None,
            "selected_confidence": None,
            "selected_time_ms": None,
        }

    initial = candidates[0]
    tested = [initial]
    refinement_triggered = initial["confidence"] < threshold

    if refinement_triggered:
        tested.extend(candidates[1:budget])

    passing = [cand for cand in tested if cand["success"]]
    if passing:
        selected = min(passing, key=lambda cand: cand["time_ms"] if cand["time_ms"] is not None else float("inf"))
        success = True
    else:
        selected = None
        success = False

    return {
        "threshold": threshold,
        "success": success,
        "tests_executed": len(tested),
        "generated_candidates_counted": min(budget, len(candidates)) if refinement_triggered else 1,
        "llm_calls_counted": 2 if refinement_triggered else 1,
        "refinement_triggered": refinement_triggered,
        "initial_confidence": initial["confidence"],
        "initial_success": initial["success"],
        "selected_idx": selected["idx"] if selected else None,
        "selected_source": selected["source"] if selected else None,
        "selected_confidence": selected["confidence"] if selected else None,
        "selected_time_ms": selected["time_ms"] if selected else None,
    }


def evaluate_candidate_set(
    candidates: List[Dict[str, Any]],
    thresholds: Iterable[float],
    budget: int,
) -> Dict[str, Any]:
    return {
        f"threshold_{threshold:.2f}": evaluate_threshold(candidates, threshold, budget)
        for threshold in thresholds
    }


def aggregate_results(candidate_sets: List[Dict[str, Any]], thresholds: Iterable[float]) -> Dict[str, Any]:
    total_instances = len(candidate_sets)
    summary: Dict[str, Any] = {}

    for threshold in thresholds:
        key = f"threshold_{threshold:.2f}"
        rows = [item["offline_evaluation"][key] for item in candidate_sets]
        successes = sum(1 for row in rows if row["success"])
        tests = [row["tests_executed"] for row in rows]
        generated = [row["generated_candidates_counted"] for row in rows]
        llm_calls = [row["llm_calls_counted"] for row in rows]
        selected_times = [row["selected_time_ms"] for row in rows if row["selected_time_ms"] is not None]
        refinement_count = sum(1 for row in rows if row["refinement_triggered"])
        high_conf_failures = sum(
            1 for row in rows if (not row["refinement_triggered"]) and (not row["initial_success"])
        )

        summary[key] = {
            "threshold": threshold,
            "success_count": successes,
            "success_rate": successes / total_instances if total_instances else 0.0,
            "total_tests_executed": sum(tests),
            "avg_tests_executed": statistics.mean(tests) if tests else 0.0,
            "total_generated_candidates_counted": sum(generated),
            "avg_generated_candidates_counted": statistics.mean(generated) if generated else 0.0,
            "total_llm_calls_counted": sum(llm_calls),
            "avg_llm_calls_counted": statistics.mean(llm_calls) if llm_calls else 0.0,
            "refinement_count": refinement_count,
            "refinement_rate": refinement_count / total_instances if total_instances else 0.0,
            "high_confidence_failure_count": high_conf_failures,
            "avg_selected_time_ms": statistics.mean(selected_times) if selected_times else None,
            "median_selected_time_ms": statistics.median(selected_times) if selected_times else None,
        }

    return summary


def build_output(args: argparse.Namespace, candidate_sets: List[Dict[str, Any]], elapsed_sec: float) -> Dict[str, Any]:
    problems = list(BENCHMARKS[args.benchmark])
    if args.limit is not None:
        problems = problems[: args.limit]

    return {
        "method": "threshold_refinement_eval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": canonical_benchmark(args.benchmark),
        "num_trials": args.num_trials,
        "num_problems": len(problems),
        "budget": args.budget,
        "thresholds": args.thresholds,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
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
        "summary": aggregate_results(candidate_sets, args.thresholds) if candidate_sets else {},
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
        print(f"Resuming from {args.output}: {len(candidate_sets)} completed instances", flush=True)

    started_at = time.time()

    for trial in range(1, args.num_trials + 1):
        print(f"\n=== Trial {trial}/{args.num_trials} ===", flush=True)
        for problem_index, problem in enumerate(problems):
            if (trial, problem_index) in completed_keys:
                print(f"[{problem_index + 1}/{len(problems)}] {problem['name']} - skipped", flush=True)
                continue

            print(
                f"[{problem_index + 1}/{len(problems)}] {problem['name']} "
                f"({problem.get('category', 'unknown')})",
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
                print("  testing initial candidate", flush=True)
                candidates.append(make_candidate_record(raw_candidate, idx, "initial", problem["test_code"]))
            for offset, raw_candidate in enumerate(refine_raw, start=len(candidates)):
                print(f"  testing refinement candidate {offset}/{args.budget - 1}", flush=True)
                candidates.append(make_candidate_record(raw_candidate, offset, "refinement", problem["test_code"]))

            offline = evaluate_candidate_set(candidates, args.thresholds, args.budget)
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
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.80, 0.85, 0.90])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=str(RESULTS_DIR / "threshold_refinement_eval_paper12.json"))
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
