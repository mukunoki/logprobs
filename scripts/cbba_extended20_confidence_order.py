#!/usr/bin/env python3
"""
CBBA: Extended 20評価

確信度ベース候補選択をExtended 20問題セットで評価
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

# Extended 20問題セット
sys.path.append(os.path.dirname(__file__))
from extended_20_optimization_problems import EXTENDED_20_OPTIMIZATION_PROBLEMS


OUTPUT_FILE = Path(__file__).resolve().parents[1] / "results" / "cbba_extended20_confidence_order.json"


def extract_c_code(response_text: str) -> str:
    """Extract C code from markdown response"""
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
    """Compile and test C code"""
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
            return False, 0.0, f"Compile error"

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
            return False, 0.0, f"Test failed"


def compute_average_probability(candidate: Dict) -> float:
    """Compute average token probability from logprobs"""
    token_logprobs = candidate.get('token_logprobs', [])
    valid_logprobs = [lp for lp in token_logprobs if lp is not None]

    if not valid_logprobs:
        return 0.0

    avg_logprob = sum(valid_logprobs) / len(valid_logprobs)
    return math.exp(avg_logprob)


def cbba_confidence_order_method(problem: Dict, k: int = 3) -> Dict:
    """
    CBBA: Generate k candidates, sort by confidence, test in order
    """
    start_time = time.time()

    # Generate k candidates with logprobs
    print(f"  Generating {k} candidates...", end=" ", flush=True)
    candidates = generate_with_logprobs(
        problem['optimization_prompt'],
        n=k,
        temperature=0.8,
        max_tokens=512
    )

    if not candidates:
        return {
            'success': False,
            'tests_run': 0,
            'total_time': time.time() - start_time,
            'error': 'Failed to generate candidates'
        }

    print(f"✓")

    # Compute confidence for each candidate
    candidate_data = []
    for i, cand in enumerate(candidates):
        code = extract_c_code(cand['text'])
        conf = compute_average_probability(cand)
        candidate_data.append({
            'idx': i,
            'code': code,
            'confidence': conf
        })

    # Sort by confidence (descending)
    candidate_data.sort(key=lambda x: x['confidence'], reverse=True)

    conf_str = ", ".join([f"{c['confidence']:.4f}" for c in candidate_data])
    print(f"  Confidence ranking: [{conf_str}]")

    # Test in confidence order (stop on first success)
    tests_run = 0
    for cand in candidate_data:
        tests_run += 1
        idx = cand['idx']
        conf = cand['confidence']
        code = cand['code']

        print(f"  Testing candidate {idx+1}/{k} (conf={conf:.4f})...", end=" ", flush=True)

        try:
            success, time_ms, message = test_c_code(code, problem['test_code'])

            if success:
                print(f"✓ PASS ({time_ms:.2f}ms)")
                total_time = time.time() - start_time
                return {
                    'success': True,
                    'time_ms': time_ms,
                    'confidence': conf,
                    'candidate_idx': idx,
                    'tests_run': tests_run,
                    'total_time': total_time
                }
            else:
                print(f"✗ FAIL")

        except Exception as e:
            print(f"✗ ERROR: {e}")

    # All candidates failed
    total_time = time.time() - start_time
    return {
        'success': False,
        'tests_run': tests_run,
        'total_time': total_time
    }


def run_cbba_extended20_confidence_order_evaluation(num_trials: int = 5) -> Dict:
    """Run CBBA evaluation on Extended 20 problems"""
    print(f"\n{'='*80}")
    print(f"CBBA Extended 20 Evaluation ({num_trials} trial(s))")
    print(f"Method: Confidence-sorted testing (k=3, stop on first success)")
    print(f"{'='*80}")

    all_trials = []

    for trial in range(num_trials):
        print(f"\n{'='*80}")
        print(f"Trial {trial + 1}/{num_trials}")
        print(f"{'='*80}")

        trial_results = []
        total_success = 0
        total_tests = 0

        category_stats = {}

        for i, problem in enumerate(EXTENDED_20_OPTIMIZATION_PROBLEMS):
            problem_name = problem['name']
            category = problem.get('category', 'unknown')

            print(f"\n[{i+1}/{len(EXTENDED_20_OPTIMIZATION_PROBLEMS)}] {problem_name} ({category})")

            result = cbba_confidence_order_method(problem, k=3)
            result['problem_name'] = problem_name
            result['category'] = category

            trial_results.append(result)

            if result['success']:
                total_success += 1

            total_tests += result['tests_run']

            # Category stats
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'success': 0, 'tests': []}
            category_stats[category]['total'] += 1
            if result['success']:
                category_stats[category]['success'] += 1
            category_stats[category]['tests'].append(result['tests_run'])

        # Trial summary
        success_rate = total_success / len(EXTENDED_20_OPTIMIZATION_PROBLEMS)
        avg_tests = total_tests / len(EXTENDED_20_OPTIMIZATION_PROBLEMS)
        total_time = sum(r['total_time'] for r in trial_results)

        print(f"\nTrial {trial + 1} Summary:")
        print(f"  Success: {total_success}/{len(EXTENDED_20_OPTIMIZATION_PROBLEMS)} = {success_rate*100:.1f}%")
        print(f"  Avg tests/problem: {avg_tests:.2f}")
        print(f"  Total time: {total_time:.2f}s")

        all_trials.append({
            'trial': trial + 1,
            'total_problems': len(EXTENDED_20_OPTIMIZATION_PROBLEMS),
            'success_count': total_success,
            'success_rate': success_rate,
            'avg_tests_per_problem': avg_tests,
            'total_time': total_time,
            'category_stats': category_stats,
            'results': trial_results
        })

    # Overall statistics
    print(f"\n{'='*80}")
    print("Category Summary")
    print(f"{'='*80}")

    all_category_stats = {}
    for trial in all_trials:
        for cat, stats in trial['category_stats'].items():
            if cat not in all_category_stats:
                all_category_stats[cat] = []
            all_category_stats[cat].append(stats['success'] / stats['total'])

    for cat, rates in sorted(all_category_stats.items()):
        mean_rate = sum(rates) / len(rates)
        print(f"{cat}: {mean_rate*100:.1f}%")

    # Save results
    output = {
        'method': 'cbba_extended20_confidence_order',
        'num_trials': num_trials,
        'total_problems': len(EXTENDED_20_OPTIMIZATION_PROBLEMS),
        'trials': all_trials
    }

    output_file = str(OUTPUT_FILE)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    print(f"\n{'='*80}")
    print("Evaluation Complete")
    print(f"{'='*80}")

    return output


if __name__ == "__main__":
    run_cbba_extended20_confidence_order_evaluation(num_trials=5)
