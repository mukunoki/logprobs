#!/usr/bin/env python3
"""
CBBA Adaptive Threshold - Reproducibility Verification

Extended 10問題で適応閾値CBBAを5回試行し、再現性を検証する。
各試行で全10問題を評価し、成功率の安定性を確認する。
"""

import json
import sys
import os
import time
from pathlib import Path
from typing import Dict, List

sys.path.append(os.path.dirname(__file__))
from cbba_adaptive_threshold import cbba_adaptive_method
from extended_optimization_problems import EXTENDED_OPTIMIZATION_PROBLEMS


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "adaptive_cbba_reproducibility.json"


def run_single_trial(trial_num: int) -> Dict:
    """単一試行を実行"""
    print(f"\n{'='*80}")
    print(f"Trial {trial_num}/5: Adaptive Threshold CBBA on Extended 10 Problems")
    print(f"{'='*80}")

    results = []
    total_success = 0
    total_tests = 0
    total_time = 0.0

    category_stats = {
        'algorithm': {'total': 0, 'success': 0, 'tests': []},
        'performance': {'total': 0, 'success': 0, 'tests': []}
    }

    for i, problem in enumerate(EXTENDED_OPTIMIZATION_PROBLEMS):
        problem_name = problem['name']
        print(f"\n--- Problem {i+1}/10: {problem_name} ---")

        result = cbba_adaptive_method(problem, k=3)
        result['problem_name'] = problem_name
        results.append(result)

        # 統計更新
        if result['success']:
            total_success += 1
        total_tests += result['tests_run']
        total_time += result['total_time']

        # カテゴリ別統計
        category = result['category']
        category_stats[category]['total'] += 1
        if result['success']:
            category_stats[category]['success'] += 1
        category_stats[category]['tests'].append(result['tests_run'])

    # 試行結果サマリー
    success_rate = (total_success / len(results)) * 100
    avg_tests = total_tests / len(results)

    print(f"\n{'='*80}")
    print(f"Trial {trial_num} Results")
    print(f"{'='*80}")
    print(f"Success Rate: {success_rate:.1f}% ({total_success}/{len(results)})")
    print(f"Average Tests: {avg_tests:.2f}")
    print(f"Total Time: {total_time:.2f}s")

    # カテゴリ別サマリー
    for category in ['algorithm', 'performance']:
        stats = category_stats[category]
        if stats['total'] > 0:
            cat_success_rate = (stats['success'] / stats['total']) * 100
            cat_avg_tests = sum(stats['tests']) / stats['total'] if stats['tests'] else 0.0
            print(f"  {category.capitalize()}: {cat_success_rate:.1f}% ({stats['success']}/{stats['total']}), Avg Tests: {cat_avg_tests:.2f}")

    return {
        'trial': trial_num,
        'total_problems': len(results),
        'success_count': total_success,
        'success_rate': success_rate / 100,
        'avg_tests_per_problem': avg_tests,
        'total_time': total_time,
        'category_stats': category_stats,
        'results': results
    }


def compute_reproducibility_statistics(trials: List[Dict]) -> Dict:
    """再現性統計を計算"""
    import statistics

    # 全体成功率
    success_rates = [trial['success_rate'] for trial in trials]
    mean_sr = statistics.mean(success_rates)
    stdev_sr = statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0

    # 95%信頼区間 (t分布、自由度4)
    # t値 (df=4, 95%): 2.776
    t_value = 2.776
    margin_of_error = t_value * (stdev_sr / (len(success_rates) ** 0.5)) if stdev_sr > 0 else 0.0
    ci_lower = mean_sr - margin_of_error
    ci_upper = mean_sr + margin_of_error

    # カテゴリ別成功率
    algorithm_rates = []
    performance_rates = []

    for trial in trials:
        cat_stats = trial['category_stats']

        # Algorithm
        if cat_stats['algorithm']['total'] > 0:
            alg_rate = cat_stats['algorithm']['success'] / cat_stats['algorithm']['total']
            algorithm_rates.append(alg_rate)

        # Performance
        if cat_stats['performance']['total'] > 0:
            perf_rate = cat_stats['performance']['success'] / cat_stats['performance']['total']
            performance_rates.append(perf_rate)

    # 平均テスト数
    avg_tests_list = [trial['avg_tests_per_problem'] for trial in trials]
    mean_tests = statistics.mean(avg_tests_list)
    stdev_tests = statistics.stdev(avg_tests_list) if len(avg_tests_list) > 1 else 0.0

    stats = {
        'num_trials': len(trials),
        'overall': {
            'mean_success_rate': mean_sr,
            'stdev_success_rate': stdev_sr,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'min_success_rate': min(success_rates),
            'max_success_rate': max(success_rates),
            'all_success_rates': success_rates
        },
        'category': {
            'algorithm': {
                'mean_success_rate': statistics.mean(algorithm_rates) if algorithm_rates else 0.0,
                'stdev_success_rate': statistics.stdev(algorithm_rates) if len(algorithm_rates) > 1 else 0.0,
                'all_success_rates': algorithm_rates
            },
            'performance': {
                'mean_success_rate': statistics.mean(performance_rates) if performance_rates else 0.0,
                'stdev_success_rate': statistics.stdev(performance_rates) if len(performance_rates) > 1 else 0.0,
                'all_success_rates': performance_rates
            }
        },
        'tests': {
            'mean_tests_per_problem': mean_tests,
            'stdev_tests_per_problem': stdev_tests,
            'all_avg_tests': avg_tests_list
        }
    }

    return stats


def main():
    print("\n" + "="*80)
    print("CBBA Adaptive Threshold - Reproducibility Verification")
    print("5 Trials on Extended 10 Problems")
    print("="*80)

    trials = []

    # 5回試行を実行
    for trial_num in range(1, 6):
        trial_result = run_single_trial(trial_num)
        trials.append(trial_result)

        # 試行間に短い待機（API負荷軽減）
        if trial_num < 5:
            print(f"\nWaiting 5 seconds before next trial...")
            time.sleep(5)

    # 再現性統計を計算
    print(f"\n{'='*80}")
    print("Computing Reproducibility Statistics...")
    print(f"{'='*80}")

    stats = compute_reproducibility_statistics(trials)

    # 結果表示
    print(f"\n{'='*80}")
    print("Reproducibility Results Summary")
    print(f"{'='*80}")
    print(f"\nOverall Success Rate:")
    print(f"  Mean: {stats['overall']['mean_success_rate']*100:.1f}%")
    print(f"  Stdev: {stats['overall']['stdev_success_rate']*100:.2f}%")
    print(f"  95% CI: [{stats['overall']['ci_95_lower']*100:.1f}%, {stats['overall']['ci_95_upper']*100:.1f}%]")
    print(f"  Range: [{stats['overall']['min_success_rate']*100:.1f}%, {stats['overall']['max_success_rate']*100:.1f}%]")

    print(f"\nCategory-wise Success Rate:")
    print(f"  Algorithm: {stats['category']['algorithm']['mean_success_rate']*100:.1f}% ± {stats['category']['algorithm']['stdev_success_rate']*100:.2f}%")
    print(f"  Performance: {stats['category']['performance']['mean_success_rate']*100:.1f}% ± {stats['category']['performance']['stdev_success_rate']*100:.2f}%")

    print(f"\nAverage Tests per Problem:")
    print(f"  Mean: {stats['tests']['mean_tests_per_problem']:.2f}")
    print(f"  Stdev: {stats['tests']['stdev_tests_per_problem']:.2f}")

    # 結果を保存
    output = {
        'method': 'cbba_adaptive_reproducibility',
        'num_trials': len(trials),
        'statistics': stats,
        'trials': trials
    }

    output_path = str(OUTPUT_PATH)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")

    # 成功判定: 平均成功率が85%以上かつ標準偏差が10%以下
    success = (stats['overall']['mean_success_rate'] >= 0.85 and
               stats['overall']['stdev_success_rate'] <= 0.10)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
