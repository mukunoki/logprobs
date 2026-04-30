#!/usr/bin/env python3
"""
Adaptive CBBA Evaluation
問題カテゴリに応じて閾値を自動調整するCBBAの評価
"""

import json
import subprocess
import tempfile
import os
import sys
import time
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

from ugir import generate_with_logprobs

sys.path.append(os.path.dirname(__file__))
from extended_optimization_problems import EXTENDED_OPTIMIZATION_PROBLEMS


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# カテゴリ別閾値マッピング（代表5問評価結果に基づく）
CATEGORY_THRESHOLDS = {
    'Algorithm': 0.90,  # 33.3% → 66.7% (+33.3%)
    'Branch': 0.85,     # 100% → 66.7% (-33.3%)  元の0.85が良い
    'Loop': 0.85,       # 83.3% → 83.3% (同等)   元の0.85が良い
    'SIMD': 0.85        # 100% → 66.7% (-33.3%)  元の0.85が良い
}


def extract_c_code(response_text: str) -> str:
    """Extract C code from response"""
    if "```c" in response_text:
        parts = response_text.split("```c")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return code.strip()
    elif "```" in response_text:
        parts = response_text.split("```")
        if len(parts) >= 3:
            return parts[1].strip()
    return response_text.strip()


def test_c_code(code: str, test_code: str) -> Tuple[bool, float, str]:
    """Test C code and measure performance"""
    with tempfile.TemporaryDirectory() as tmpdir:
        code_file = os.path.join(tmpdir, "code.c")
        test_file = os.path.join(tmpdir, "test.c")
        executable = os.path.join(tmpdir, "test")

        with open(code_file, "w") as f:
            f.write(code)

        with open(test_file, "w") as f:
            f.write(test_code)

        compile_cmd = f"gcc -o {executable} {code_file} {test_file} -lm -O3 2>&1"
        result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            return False, 0.0, f"Compile error: {result.stderr[:200]}"

        try:
            result = subprocess.run([executable], capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return False, 0.0, "Timeout"

        if result.returncode == 0 and "PASS" in result.stdout:
            match = re.search(r'PASS\s+([\d.]+)', result.stdout)
            if match:
                time_ms = float(match.group(1))
                return True, time_ms, "Test passed"
            else:
                return True, 0.0, "Test passed (no time)"
        else:
            return False, 0.0, f"Test failed: {result.stdout[:200]}"


def compute_average_probability(candidate: Dict) -> float:
    """Compute average probability from token logprobs"""
    token_logprobs = candidate.get('token_logprobs', [])
    valid_logprobs = [lp for lp in token_logprobs if lp is not None]

    if not valid_logprobs:
        return 0.0

    avg_logprob = sum(valid_logprobs) / len(valid_logprobs)
    avg_prob = math.exp(avg_logprob)

    return avg_prob


def adaptive_cbba_method(problem: Dict, budget: int = 3) -> Dict:
    """
    Adaptive CBBA: 問題カテゴリに応じて閾値を自動調整
    """
    category = problem.get('category', 'unknown')
    confidence_threshold = CATEGORY_THRESHOLDS.get(category, 0.85)  # デフォルト0.85

    print(f"\n[Adaptive CBBA] Category: {category}, Threshold: {confidence_threshold}")
    print(f"  Generating initial candidate...")
    start_time = time.time()

    initial_candidates = generate_with_logprobs(problem['optimization_prompt'], n=1, temperature=0.8, max_tokens=512)

    if not initial_candidates:
        return {
            "success": False,
            "time_ms": 0.0,
            "confidence": 0.0,
            "total_time": 0.0,
            "tests_run": 0,
            "threshold_used": confidence_threshold,
            "category": category
        }

    initial_candidate = initial_candidates[0]
    initial_conf = compute_average_probability(initial_candidate)
    initial_code = extract_c_code(initial_candidate['text'])

    print(f"  Initial candidate confidence: {initial_conf:.4f} (threshold: {confidence_threshold:.4f})")

    tests_run = 1
    print(f"  Testing initial candidate...", end=" ", flush=True)
    try:
        success, time_ms, message = test_c_code(initial_code, problem['test_code'])
        if success:
            print(f"✓ PASS ({time_ms:.2f}ms)")
            best_result = {
                "success": True,
                "time_ms": time_ms,
                "confidence": initial_conf,
                "candidate_idx": 0,
                "method": "initial",
                "threshold_used": confidence_threshold,
                "category": category
            }
        else:
            print(f"✗ FAIL")
            best_result = None
    except Exception as e:
        print(f"✗ ERROR: {e}")
        best_result = None

    # 適応的閾値による判定
    if initial_conf >= confidence_threshold:
        print(f"  ✓ High confidence ({initial_conf:.4f} ≥ {confidence_threshold}), accepting without refinement")
        elapsed = time.time() - start_time
        if best_result:
            best_result['total_time'] = elapsed
            best_result['tests_run'] = tests_run
            best_result['refinement_triggered'] = False
            return best_result
        else:
            return {
                "success": False,
                "time_ms": 0.0,
                "confidence": initial_conf,
                "total_time": elapsed,
                "tests_run": tests_run,
                "refinement_triggered": False,
                "threshold_used": confidence_threshold,
                "category": category
            }

    print(f"  ⚠ Low confidence ({initial_conf:.4f} < {confidence_threshold}), generating {budget-1} refinements...")

    refinement_prompt = f"""{problem['optimization_prompt']}

IMPORTANT: Generate an IMPROVED version with:
1. Better optimization techniques
2. Correct algorithm implementation
3. Proper edge case handling

Focus on creating a high-quality, correct implementation."""

    refined_candidates = generate_with_logprobs(refinement_prompt, n=budget-1, temperature=0.8, max_tokens=512)

    best_time = best_result['time_ms'] if best_result else float('inf')

    for idx, candidate in enumerate(refined_candidates):
        tests_run += 1
        code = extract_c_code(candidate['text'])
        conf = compute_average_probability(candidate)

        print(f"  Testing refined candidate {idx+1}/{budget-1} (conf={conf:.4f})...", end=" ", flush=True)

        try:
            success, time_ms, message = test_c_code(code, problem['test_code'])

            if success:
                print(f"✓ PASS ({time_ms:.2f}ms)")
                if time_ms < best_time:
                    best_time = time_ms
                    best_result = {
                        "success": True,
                        "time_ms": time_ms,
                        "confidence": conf,
                        "candidate_idx": idx + 1,
                        "method": "refined",
                        "threshold_used": confidence_threshold,
                        "category": category
                    }
            else:
                print(f"✗ FAIL")
        except Exception as e:
            print(f"✗ ERROR: {e}")

    elapsed = time.time() - start_time

    if best_result:
        best_result['total_time'] = elapsed
        best_result['tests_run'] = tests_run
        best_result['refinement_triggered'] = True
        return best_result
    else:
        return {
            "success": False,
            "time_ms": 0.0,
            "confidence": 0.0,
            "total_time": elapsed,
            "tests_run": tests_run,
            "refinement_triggered": True,
            "threshold_used": confidence_threshold,
            "category": category
        }


def run_evaluation(problem: Dict, num_trials: int = 3) -> Dict:
    """Evaluate Adaptive CBBA on a single problem"""
    print(f"\n{'='*70}")
    print(f"Problem: {problem['name']} [{problem.get('category', 'unknown')}]")
    print(f"Adaptive Threshold: {CATEGORY_THRESHOLDS.get(problem.get('category', 'unknown'), 0.85)}")
    print(f"{'='*70}")

    results = []

    for trial in range(1, num_trials + 1):
        print(f"\n--- Trial {trial}/{num_trials} ---")

        result = adaptive_cbba_method(problem, budget=3)
        results.append(result)

        if result['success']:
            print(f"\n  ✓ Success: {result['tests_run']} tests, {result['time_ms']:.2f}ms")
        else:
            print(f"\n  ✗ Failed: {result['tests_run']} tests")

    return {
        "problem": problem['name'],
        "category": problem.get('category', 'unknown'),
        "threshold": CATEGORY_THRESHOLDS.get(problem.get('category', 'unknown'), 0.85),
        "results": results
    }


def main():
    print("="*70)
    print("Adaptive CBBA Evaluation (Extended 10 Problems)")
    print("="*70)
    print(f"Problems: {len(EXTENDED_OPTIMIZATION_PROBLEMS)}")
    print(f"Trials per problem: 3")
    print(f"Budget: 3 candidates per trial")
    print("\nCategory Thresholds:")
    for cat, thresh in CATEGORY_THRESHOLDS.items():
        print(f"  {cat}: {thresh}")
    print("="*70)

    all_results = []

    for problem in EXTENDED_OPTIMIZATION_PROBLEMS:
        result = run_evaluation(problem, num_trials=3)
        all_results.append(result)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = str(RESULTS_DIR / 'adaptive_cbba_evaluation.json')

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

    # Summary statistics
    total_trials = sum(len(r['results']) for r in all_results)
    total_success = sum(sum(1 for trial in r['results'] if trial['success']) for r in all_results)
    success_rate = (total_success / total_trials * 100) if total_trials > 0 else 0

    total_tests = sum(sum(trial['tests_run'] for trial in r['results']) for r in all_results)
    avg_tests = total_tests / total_trials if total_trials > 0 else 0

    print(f"\nAdaptive CBBA Summary:")
    print(f"  Success Rate: {success_rate:.1f}% ({total_success}/{total_trials})")
    print(f"  Average Tests: {avg_tests:.1f}")

    # Category breakdown
    print(f"\nCategory Breakdown:")
    categories = {}
    for result in all_results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = {'total': 0, 'success': 0}
        categories[cat]['total'] += len(result['results'])
        categories[cat]['success'] += sum(1 for trial in result['results'] if trial['success'])

    for cat in sorted(categories.keys()):
        total = categories[cat]['total']
        success = categories[cat]['success']
        rate = (success / total * 100) if total > 0 else 0
        thresh = CATEGORY_THRESHOLDS.get(cat, 0.85)
        print(f"  {cat:12s} (threshold={thresh:.2f}): {rate:5.1f}% ({success}/{total})")


if __name__ == "__main__":
    main()
