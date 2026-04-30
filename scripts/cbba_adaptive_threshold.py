#!/usr/bin/env python3
"""
CBBA with Adaptive Threshold Strategy

問題カテゴリに応じて確信度閾値を動的に選択：
- Algorithm問題: 0.90（高精度が必要）
- Performance問題: 0.85（標準設定）

Extended 10問題での評価を実施し、固定閾値との比較を行う。
"""

import json
import sys
import os
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple

from ugir import generate_with_logprobs

sys.path.append(os.path.dirname(__file__))
from cbba_method import extract_c_code, test_c_code, compute_average_probability
from extended_optimization_problems import EXTENDED_OPTIMIZATION_PROBLEMS


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "adaptive_cbba_extended_evaluation.json"


def detect_problem_category(problem: Dict) -> str:
    """
    問題のカテゴリを判定

    Args:
        problem: 問題辞書（name, description, category等を含む）

    Returns:
        'algorithm' または 'performance'
    """
    # カテゴリフィールドが存在する場合はそれを使用
    if 'category' in problem:
        category = problem['category']
        # 'algorithm'カテゴリはそのまま
        if category == 'algorithm':
            return 'algorithm'
        # それ以外（loop, simd等）はperformance扱い
        else:
            return 'performance'

    # カテゴリフィールドがない場合は名前・説明から判定
    name = problem.get('name', '').lower()
    description = problem.get('description', '').lower()
    prompt = problem.get('optimization_prompt', '').lower()

    # Algorithm関連キーワード
    algorithm_keywords = [
        'binary_search', 'search', 'sort', 'sorting',
        'algorithm', 'data structure', 'tree', 'graph',
        'hash', 'recursion', 'dynamic programming'
    ]

    text = f"{name} {description} {prompt}"

    for keyword in algorithm_keywords:
        if keyword in text:
            return 'algorithm'

    # デフォルトはperformance
    return 'performance'


def select_threshold(category: str) -> float:
    """
    カテゴリに応じた閾値を選択

    Args:
        category: 'algorithm' または 'performance'

    Returns:
        閾値（0.90 または 0.85）
    """
    if category == 'algorithm':
        return 0.90
    else:
        return 0.85


def cbba_adaptive_method(problem: Dict, k: int = 3) -> Dict:
    """
    CBBA with Adaptive Threshold

    問題カテゴリを判定し、適切な閾値を選択してCBBAを実行

    Args:
        problem: 問題辞書
        k: 生成候補数

    Returns:
        結果辞書（success, tests_run, confidence, category, threshold等）
    """
    # カテゴリ判定と閾値選択
    category = detect_problem_category(problem)
    threshold = select_threshold(category)

    print(f"\n[CBBA-Adaptive] Category: {category}, Threshold: {threshold}")
    print(f"[CBBA-Adaptive] Generating {k} candidates...")

    start_time = time.time()

    # 初期生成
    candidates = generate_with_logprobs(problem['optimization_prompt'], n=k, temperature=0.8, max_tokens=512)

    if not candidates:
        return {
            "success": False,
            "category": category,
            "threshold": threshold,
            "tests_run": 0,
            "total_time": time.time() - start_time,
            "error": "Failed to generate candidates"
        }

    # 確信度計算
    for i, cand in enumerate(candidates):
        confidence = compute_average_probability(cand)
        cand['confidence'] = confidence
        print(f"  Candidate {i+1}: confidence={confidence:.4f}")

    # 閾値でフィルタリング
    high_confidence = [c for c in candidates if c['confidence'] >= threshold]
    low_confidence = [c for c in candidates if c['confidence'] < threshold]

    print(f"[CBBA-Adaptive] High confidence: {len(high_confidence)}, Low confidence: {len(low_confidence)}")

    # 高確信度候補をテスト
    tests_run = 0
    best_result = None
    best_time_ms = float('inf')

    for idx, cand in enumerate(high_confidence):
        tests_run += 1
        code = extract_c_code(cand['text'])

        print(f"  Testing high-conf candidate {idx+1}/{len(high_confidence)}...", end=" ", flush=True)

        try:
            success, time_ms, message = test_c_code(code, problem['test_code'])

            if success:
                print(f"✓ PASS ({time_ms:.2f}ms)")
                if time_ms < best_time_ms:
                    best_time_ms = time_ms
                    best_result = {
                        "success": True,
                        "time_ms": time_ms,
                        "confidence": cand['confidence'],
                        "candidate_idx": idx,
                        "source": "high_confidence"
                    }
                # 高確信度で成功したので即座に返す
                break
            else:
                print(f"✗ FAIL")
        except Exception as e:
            print(f"✗ ERROR: {e}")

    # 高確信度で成功した場合は結果を返す
    if best_result:
        elapsed = time.time() - start_time
        best_result.update({
            "category": category,
            "threshold": threshold,
            "tests_run": tests_run,
            "total_time": elapsed
        })
        return best_result

    # 低確信度候補をテスト（残り予算の範囲内）
    remaining_budget = k - tests_run

    if low_confidence and remaining_budget > 0:
        print(f"[CBBA-Adaptive] Testing low-confidence candidates (budget={remaining_budget})...")

        for idx, cand in enumerate(low_confidence[:remaining_budget]):
            tests_run += 1
            code = extract_c_code(cand['text'])

            print(f"  Testing low-conf candidate {idx+1}/{min(len(low_confidence), remaining_budget)}...", end=" ", flush=True)

            try:
                success, time_ms, message = test_c_code(code, problem['test_code'])

                if success:
                    print(f"✓ PASS ({time_ms:.2f}ms)")
                    if time_ms < best_time_ms:
                        best_time_ms = time_ms
                        best_result = {
                            "success": True,
                            "time_ms": time_ms,
                            "confidence": cand['confidence'],
                            "candidate_idx": idx,
                            "source": "low_confidence"
                        }
                else:
                    print(f"✗ FAIL")
            except Exception as e:
                print(f"✗ ERROR: {e}")

    # 最終結果
    elapsed = time.time() - start_time

    if best_result:
        best_result.update({
            "category": category,
            "threshold": threshold,
            "tests_run": tests_run,
            "total_time": elapsed
        })
        return best_result
    else:
        return {
            "success": False,
            "category": category,
            "threshold": threshold,
            "tests_run": tests_run,
            "total_time": elapsed
        }


def evaluate_adaptive_cbba_on_extended():
    """Extended 10問題で適応閾値CBBAを評価"""
    print("\n" + "="*80)
    print("CBBA Adaptive Threshold Evaluation on Extended 10 Problems")
    print("="*80)

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
        print(f"\n{'='*80}")
        print(f"Problem {i+1}/10: {problem_name}")
        print(f"{'='*80}")

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

    # 全体統計
    success_rate = (total_success / len(results)) * 100
    avg_tests = total_tests / len(results)

    print(f"\n{'='*80}")
    print("Overall Results")
    print(f"{'='*80}")
    print(f"Success Rate: {success_rate:.1f}% ({total_success}/{len(results)})")
    print(f"Average Tests per Problem: {avg_tests:.2f}")
    print(f"Total Time: {total_time:.2f}s")

    # カテゴリ別統計
    print(f"\n{'='*80}")
    print("Category Breakdown")
    print(f"{'='*80}")

    for category in ['algorithm', 'performance']:
        stats = category_stats[category]
        if stats['total'] > 0:
            cat_success_rate = (stats['success'] / stats['total']) * 100
            cat_avg_tests = sum(stats['tests']) / stats['total'] if stats['tests'] else 0.0
            print(f"\n{category.capitalize()}:")
            print(f"  Problems: {stats['total']}")
            print(f"  Success Rate: {cat_success_rate:.1f}% ({stats['success']}/{stats['total']})")
            print(f"  Average Tests: {cat_avg_tests:.2f}")

    # 結果を保存
    output = {
        'method': 'cbba_adaptive',
        'total_problems': len(results),
        'success_count': total_success,
        'success_rate': success_rate / 100,
        'avg_tests_per_problem': avg_tests,
        'total_time': total_time,
        'category_stats': category_stats,
        'results': results
    }

    output_path = str(OUTPUT_PATH)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")

    return output


if __name__ == '__main__':
    print("Starting Adaptive Threshold CBBA Evaluation...")
    results = evaluate_adaptive_cbba_on_extended()

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)

    # 終了コード
    sys.exit(0 if results['success_rate'] >= 0.8 else 1)
