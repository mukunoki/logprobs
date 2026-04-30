#!/usr/bin/env python3
"""
Baseline Extended 20 Full Top-k Evaluation

Extended 20問題でベースライン手法（k=3全候補テスト）を評価する。
全候補を生成し、全てをテストして最良の候補を選択する。
"""

import json
import sys
import os
import time
from pathlib import Path
from typing import Dict, List

from ugir import generate_with_logprobs

sys.path.append(os.path.dirname(__file__))
from cbba_method import extract_c_code, test_c_code
from extended_20_optimization_problems import EXTENDED_20_OPTIMIZATION_PROBLEMS


OUTPUT_FILE = Path(__file__).resolve().parents[1] / "results" / "baseline_extended20_full_topk.json"


def baseline_method(problem: Dict, k: int = 3) -> Dict:
    """
    ベースライン手法: k個の候補を全て生成・テストし、最良を選択

    Args:
        problem: 問題定義
        k: 生成する候補数（デフォルト3）

    Returns:
        result: 実行結果の辞書
    """
    start_time = time.time()

    # k個の候補を生成
    candidates = generate_with_logprobs(problem['optimization_prompt'], n=k, temperature=0.8, max_tokens=512)

    if not candidates:
        return {
            'success': False,
            'tests_run': 0,
            'total_time': time.time() - start_time,
            'error': 'Failed to generate candidates'
        }

    # 全候補をテスト
    best_result = None
    best_time_ms = float('inf')
    tests_run = 0

    for i, cand in enumerate(candidates):
        tests_run += 1
        code = extract_c_code(cand['text'])

        print(f"  Testing candidate {i+1}/{k}...", end=" ", flush=True)

        try:
            success, time_ms, message = test_c_code(code, problem['test_code'])

            if success:
                print(f"✓ PASS ({time_ms:.2f}ms)")
                if time_ms < best_time_ms:
                    best_time_ms = time_ms
                    best_result = {
                        'success': True,
                        'time_ms': time_ms,
                        'candidate_idx': i,
                        'tests_run': tests_run
                    }
            else:
                print(f"✗ FAIL")

        except Exception as e:
            print(f"✗ ERROR: {e}")

    total_time = time.time() - start_time

    if best_result:
        best_result['total_time'] = total_time
        return best_result
    else:
        # 全失敗
        return {
            'success': False,
            'tests_run': tests_run,
            'total_time': total_time
        }


def run_baseline_extended20_full_topk(num_trials: int = 1) -> Dict:
    """Extended 20問題でのFull Top-kベースライン評価を実行"""
    print(f"\n{'='*80}")
    print(f"Baseline Extended 20 Full Top-k Evaluation ({num_trials} trial(s))")
    print(f"Method: Top-k (k=3, test all candidates)")
    print(f"{'='*80}")

    all_results = []
    trial_stats = []

    for trial in range(num_trials):
        if num_trials > 1:
            print(f"\n--- Trial {trial + 1}/{num_trials} ---")

        trial_results = []
        total_success = 0
        total_tests = 0
        total_time = 0.0

        category_stats = {}

        for i, problem in enumerate(EXTENDED_20_OPTIMIZATION_PROBLEMS):
            problem_name = problem['name']
            category = problem.get('category', 'unknown')

            print(f"\n[{i+1}/20] {problem_name} ({category})")

            result = baseline_method(problem, k=3)
            result['problem_name'] = problem_name
            result['category'] = category
            trial_results.append(result)

            # 統計更新
            if result['success']:
                total_success += 1
                print(f"  ✓ Success (time={result['time_ms']:.2f}ms, tests={result['tests_run']})")
            else:
                print(f"  ✗ Failed (tests={result['tests_run']})")

            total_tests += result['tests_run']
            total_time += result['total_time']

            # カテゴリ別統計
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'success': 0, 'tests': []}
            category_stats[category]['total'] += 1
            if result['success']:
                category_stats[category]['success'] += 1
            category_stats[category]['tests'].append(result['tests_run'])

        # 試行統計
        success_rate = total_success / len(EXTENDED_20_OPTIMIZATION_PROBLEMS)
        avg_tests = total_tests / len(EXTENDED_20_OPTIMIZATION_PROBLEMS)

        trial_stat = {
            'trial': trial + 1,
            'total_problems': len(EXTENDED_20_OPTIMIZATION_PROBLEMS),
            'success_count': total_success,
            'success_rate': success_rate,
            'avg_tests_per_problem': avg_tests,
            'total_time': total_time,
            'category_stats': category_stats,
            'results': trial_results
        }

        trial_stats.append(trial_stat)
        all_results.extend(trial_results)

        print(f"\nTrial {trial + 1} Summary:")
        print(f"  Success: {total_success}/{len(EXTENDED_20_OPTIMIZATION_PROBLEMS)} = {success_rate*100:.1f}%")
        print(f"  Avg tests/problem: {avg_tests:.2f}")
        print(f"  Total time: {total_time:.2f}s")

    # カテゴリ別サマリー
    print(f"\n{'='*80}")
    print(f"Category Summary")
    print(f"{'='*80}")

    # 全試行のカテゴリ統計を集約
    aggregated_categories = {}
    for trial_stat in trial_stats:
        for cat, stats in trial_stat['category_stats'].items():
            if cat not in aggregated_categories:
                aggregated_categories[cat] = {'total': 0, 'success': 0}
            aggregated_categories[cat]['total'] += stats['total']
            aggregated_categories[cat]['success'] += stats['success']

    for cat in sorted(aggregated_categories.keys()):
        stats = aggregated_categories[cat]
        success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"{cat}: {success_rate*100:.1f}%")

    # 結果を保存
    output = {
        'method': 'baseline_extended20_full_topk',
        'num_trials': num_trials,
        'total_problems': len(EXTENDED_20_OPTIMIZATION_PROBLEMS),
        'trials': trial_stats
    }

    output_file = str(OUTPUT_FILE)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    print(f"\n{'='*80}")
    print(f"Evaluation Complete")
    print(f"{'='*80}")

    return output


if __name__ == '__main__':
    num_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_baseline_extended20_full_topk(num_trials)
