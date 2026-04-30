#!/usr/bin/env python3
"""
CBBA: Confidence-Based Budget Allocation

確信度に基づく予算配分手法：
- 高確信度（≥0.85）: そのまま採用（追加生成なし）
- 低確信度（<0.85）: refinement実行（残り予算を使用）

UTERの「refinementが逆効果」問題を回避するため、
高確信度候補を保護する戦略。
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
from typing import Dict, List, Tuple, Optional

from ugir import generate_with_logprobs

sys.path.append(os.path.dirname(__file__))
from simple_optimization_problems import SIMPLE_OPTIMIZATION_PROBLEMS


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


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

        # Compile
        compile_cmd = f"gcc -o {executable} {code_file} {test_file} -lm -O3 2>&1"
        result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            return False, 0.0, f"Compile error: {result.stderr[:200]}"

        # Run test
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


def baseline_method(problem: Dict, k: int = 3) -> Dict:
    """Baseline: Generate k candidates, test all, return best"""
    print(f"\n[Baseline] Generating {k} candidates...")
    start_time = time.time()

    candidates = generate_with_logprobs(problem['optimization_prompt'], n=k, temperature=0.8, max_tokens=512)

    best_result = None
    best_time = float('inf')
    tests_run = 0

    for idx, candidate in enumerate(candidates):
        tests_run += 1
        code = extract_c_code(candidate['text'])
        conf = compute_average_probability(candidate)

        print(f"  Testing candidate {idx+1}/{k} (conf={conf:.4f})...", end=" ", flush=True)

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
                        "candidate_idx": idx
                    }
            else:
                print(f"✗ FAIL")
        except Exception as e:
            print(f"✗ ERROR: {e}")

    elapsed = time.time() - start_time

    if best_result:
        best_result['total_time'] = elapsed
        best_result['tests_run'] = tests_run
        return best_result
    else:
        return {
            "success": False,
            "time_ms": 0.0,
            "confidence": 0.0,
            "total_time": elapsed,
            "tests_run": tests_run
        }


def cbba_method(problem: Dict, budget: int = 3, confidence_threshold: float = 0.85) -> Dict:
    """
    CBBA: Confidence-Based Budget Allocation

    1. Generate initial candidate
    2. Check confidence
    3. If confidence >= threshold: accept and stop
    4. If confidence < threshold: generate refinements
    5. Return best candidate
    """
    print(f"\n[CBBA] Generating initial candidate...")
    start_time = time.time()

    # Step 1: Generate initial candidate
    initial_candidates = generate_with_logprobs(problem['optimization_prompt'], n=1, temperature=0.8, max_tokens=512)

    if not initial_candidates:
        return {"success": False, "time_ms": 0.0, "confidence": 0.0, "total_time": 0.0, "tests_run": 0}

    initial_candidate = initial_candidates[0]
    initial_conf = compute_average_probability(initial_candidate)
    initial_code = extract_c_code(initial_candidate['text'])

    print(f"  Initial candidate confidence: {initial_conf:.4f}")

    # Test initial
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
                "method": "initial"
            }
        else:
            print(f"✗ FAIL")
            best_result = None
    except Exception as e:
        print(f"✗ ERROR: {e}")
        best_result = None

    # Step 2: Check confidence threshold
    if initial_conf >= confidence_threshold:
        print(f"  ✓ High confidence ({initial_conf:.4f} ≥ {confidence_threshold}), accepting without refinement")
        elapsed = time.time() - start_time
        if best_result:
            best_result['total_time'] = elapsed
            best_result['tests_run'] = tests_run
            best_result['refinement_triggered'] = False
            return best_result
        else:
            # High confidence but test failed - still no refinement in CBBA
            return {
                "success": False,
                "time_ms": 0.0,
                "confidence": initial_conf,
                "total_time": elapsed,
                "tests_run": tests_run,
                "refinement_triggered": False
            }

    # Step 3: Low confidence - allocate remaining budget to refinement
    print(f"  ⚠ Low confidence ({initial_conf:.4f} < {confidence_threshold}), generating {budget-1} refinements...")

    refinement_prompt = f"""{problem['optimization_prompt']}

IMPORTANT: Generate an IMPROVED version with:
1. Better optimization techniques
2. Correct algorithm implementation
3. Proper edge case handling

Focus on creating a high-quality, correct implementation."""

    refined_candidates = generate_with_logprobs(refinement_prompt, n=budget-1, temperature=0.8, max_tokens=512)

    # Step 4: Test refined candidates
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
                        "method": "refined"
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
            "refinement_triggered": True
        }


def run_comparison(problem: Dict, num_trials: int = 3) -> Dict:
    """Compare Baseline vs CBBA on a single problem"""
    print(f"\n{'='*70}")
    print(f"Problem: {problem['name']}")
    print(f"{'='*70}")

    baseline_results = []
    cbba_results = []

    for trial in range(1, num_trials + 1):
        print(f"\n--- Trial {trial}/{num_trials} ---")

        # Baseline
        baseline_result = baseline_method(problem, k=3)
        baseline_results.append(baseline_result)

        # CBBA
        cbba_result = cbba_method(problem, budget=3, confidence_threshold=0.85)
        cbba_results.append(cbba_result)

        # Comparison
        if baseline_result['success'] and cbba_result['success']:
            speedup = baseline_result['time_ms'] / cbba_result['time_ms']
            tests_saved = baseline_result['tests_run'] - cbba_result['tests_run']
            print(f"\n  📊 CBBA: {cbba_result['tests_run']} tests (saved {tests_saved}), speedup: {speedup:.2f}x")

    return {
        "problem": problem['name'],
        "baseline": baseline_results,
        "cbba": cbba_results
    }


def main():
    print("="*70)
    print("CBBA Method Evaluation (Simple Optimization Problems)")
    print("="*70)
    print(f"Problems: {len(SIMPLE_OPTIMIZATION_PROBLEMS)}")
    print(f"Trials per problem: 3")
    print(f"Budget: 3 candidates per trial")
    print(f"Confidence threshold: 0.85")
    print("="*70)

    all_results = []

    for problem in SIMPLE_OPTIMIZATION_PROBLEMS:
        result = run_comparison(problem, num_trials=3)
        all_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    baseline_total_success = 0
    cbba_total_success = 0
    baseline_total_tests = 0
    cbba_total_tests = 0
    cbba_refinement_count = 0
    total_trials = 0

    for result in all_results:
        problem_name = result['problem']
        baseline_successes = sum(1 for r in result['baseline'] if r['success'])
        cbba_successes = sum(1 for r in result['cbba'] if r['success'])

        baseline_tests = sum(r['tests_run'] for r in result['baseline'])
        cbba_tests = sum(r['tests_run'] for r in result['cbba'])
        refinement_triggered = sum(1 for r in result['cbba'] if r.get('refinement_triggered', False))

        baseline_times = [r['time_ms'] for r in result['baseline'] if r['success']]
        cbba_times = [r['time_ms'] for r in result['cbba'] if r['success']]

        print(f"\n{problem_name}:")
        print(f"  Baseline:  {baseline_successes}/3 success, {baseline_tests} tests, avg time: {sum(baseline_times)/len(baseline_times) if baseline_times else 0:.2f}ms")
        print(f"  CBBA:      {cbba_successes}/3 success, {cbba_tests} tests, avg time: {sum(cbba_times)/len(cbba_times) if cbba_times else 0:.2f}ms")
        print(f"  Refinement triggered: {refinement_triggered}/3 trials")

        if baseline_times and cbba_times:
            speedup = (sum(baseline_times)/len(baseline_times)) / (sum(cbba_times)/len(cbba_times))
            print(f"  Speedup: {speedup:.2f}x")

        baseline_total_success += baseline_successes
        cbba_total_success += cbba_successes
        baseline_total_tests += baseline_tests
        cbba_total_tests += cbba_tests
        cbba_refinement_count += refinement_triggered
        total_trials += 3

    print(f"\n{'='*70}")
    print(f"Overall Results:")
    print(f"  Baseline:  {baseline_total_success}/{total_trials} = {baseline_total_success/total_trials*100:.1f}% success, {baseline_total_tests} total tests")
    print(f"  CBBA:      {cbba_total_success}/{total_trials} = {cbba_total_success/total_trials*100:.1f}% success, {cbba_total_tests} total tests")
    print(f"  Tests saved: {baseline_total_tests - cbba_total_tests} ({(baseline_total_tests - cbba_total_tests)/baseline_total_tests*100:.1f}%)")
    print(f"  Refinement triggered: {cbba_refinement_count}/{total_trials} = {cbba_refinement_count/total_trials*100:.1f}%")
    print(f"{'='*70}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = str(RESULTS_DIR / "cbba_simple_evaluation.json")

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
