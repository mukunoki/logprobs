"""
UGIR: Uncertainty-Guided Iterative Refinement for Code Generation

新規性:
1. Token-level不確実性に基づいて改善すべきコード行を特定
2. 不確実な部分のみを選択的に改善（全体改善より効率的）
3. 固定予算制約下での反復的改善

既存手法との違い:
- Self-Refine: 全体を一律に改善
- UGIR: 不確実な部分だけを選択的に改善
"""

import os
import json
import requests
from typing import List, Dict, Tuple
import numpy as np

# vLLM API設定
VLLM_API_URL = "http://localhost:8000/v1/completions"
VLLM_CHAT_API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "Ankushbl6/Qwen3.5-35B-A3B-AWQ-4bit")
VLLM_REQUEST_TIMEOUT = int(os.environ.get("VLLM_REQUEST_TIMEOUT", "600"))
VLLM_USE_CHAT = os.environ.get("VLLM_USE_CHAT", "0") == "1"


def _generate_chat_with_logprobs(prompt: str, n: int, temperature: float, max_tokens: int) -> List[Dict]:
    response = requests.post(
        VLLM_CHAT_API_URL,
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": 5,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=VLLM_REQUEST_TIMEOUT
    )

    if response.status_code != 200:
        raise Exception(f"Chat API error: {response.status_code}: {response.text[:500]}")

    data = response.json()
    candidates = []
    for choice in data.get("choices", []):
        message = choice.get("message") or {}
        text = message.get("content") or ""
        content_logprobs = ((choice.get("logprobs") or {}).get("content") or [])
        candidates.append({
            "text": text,
            "tokens": [item.get("token", "") for item in content_logprobs],
            "token_logprobs": [item.get("logprob") for item in content_logprobs],
            "finish_reason": choice.get("finish_reason"),
            "stop_reason": choice.get("stop_reason"),
        })
    return candidates

def generate_with_logprobs(prompt: str, n: int = 1, temperature: float = 0.8, max_tokens: int = 512) -> List[Dict]:
    """候補を生成し、token-level確率情報を取得"""
    if n <= 0:
        return []

    if VLLM_USE_CHAT:
        return _generate_chat_with_logprobs(prompt, n, temperature, max_tokens)

    response = requests.post(
        VLLM_API_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": 5,  # top-5確率を取得
        },
        timeout=VLLM_REQUEST_TIMEOUT
    )

    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code}")

    data = response.json()

    candidates = []
    for choice in data.get("choices", []):
        text = choice.get("text", "")
        logprobs_data = choice.get("logprobs", {})

        # token-level確率を抽出
        tokens = logprobs_data.get("tokens", [])
        token_logprobs = logprobs_data.get("token_logprobs", [])

        candidates.append({
            "text": text,
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "finish_reason": choice.get("finish_reason"),
            "stop_reason": choice.get("stop_reason"),
        })

    return candidates


def compute_token_uncertainty(candidate: Dict) -> List[float]:
    """各トークンの不確実性を計算（低い確率 = 高い不確実性）"""
    token_logprobs = candidate["token_logprobs"]

    # logprobを確率に変換
    probs = [np.exp(lp) if lp is not None else 0.5 for lp in token_logprobs]

    # 不確実性 = 1 - prob（確率が低いほど不確実性が高い）
    uncertainties = [1.0 - p for p in probs]

    return uncertainties


def identify_uncertain_lines(candidate: Dict, threshold: float = 0.5) -> List[int]:
    """
    不確実性が高い行を特定

    Args:
        candidate: 候補（text, tokens, token_logprobs）
        threshold: 不確実性閾値（これ以上なら「不確実」と判定）

    Returns:
        不確実な行番号のリスト
    """
    uncertainties = compute_token_uncertainty(candidate)
    tokens = candidate["tokens"]
    text = candidate["text"]

    # 行ごとの平均不確実性を計算
    lines = text.split('\n')
    line_uncertainties = []

    token_idx = 0
    for line_idx, line in enumerate(lines):
        # この行に含まれるトークンの不確実性を集計
        line_tokens = []
        while token_idx < len(tokens):
            token = tokens[token_idx]
            if '\n' in token:
                token_idx += 1
                break
            line_tokens.append(uncertainties[token_idx])
            token_idx += 1

        if line_tokens:
            avg_uncertainty = np.mean(line_tokens)
            line_uncertainties.append(avg_uncertainty)
        else:
            line_uncertainties.append(0.0)

    # 閾値以上の不確実性を持つ行を抽出
    uncertain_lines = [i for i, u in enumerate(line_uncertainties) if u > threshold]

    return uncertain_lines


def refine_uncertain_parts(code: str, uncertain_lines: List[int], problem_desc: str) -> str:
    """
    不確実な行を改善

    Args:
        code: 元のコード
        uncertain_lines: 不確実な行番号のリスト
        problem_desc: 問題の説明

    Returns:
        改善されたコード
    """
    lines = code.split('\n')

    if not uncertain_lines:
        return code

    # 不確実な行を強調したプロンプトを作成
    uncertain_line_texts = [f"Line {i+1}: {lines[i]}" for i in uncertain_lines if i < len(lines)]

    prompt = f"""以下のコードには不確実性が高い行があります。これらの行を改善してください。

問題: {problem_desc}

元のコード:
{code}

不確実性が高い行:
{chr(10).join(uncertain_line_texts)}

改善されたコードを出力してください（コードのみ、説明不要）:
"""

    # LLMで改善
    candidates = generate_with_logprobs(prompt, n=1, temperature=0.3, max_tokens=512)

    if candidates:
        return candidates[0]["text"].strip()
    else:
        return code


def ugir_pipeline(problem_desc: str, initial_code: str = None, max_iterations: int = 3, uncertainty_threshold: float = 0.5) -> Dict:
    """
    UGIR パイプライン

    Args:
        problem_desc: 問題の説明
        initial_code: 初期コード（Noneなら生成）
        max_iterations: 最大反復回数
        uncertainty_threshold: 不確実性閾値

    Returns:
        結果（最終コード、反復ごとの不確実性など）
    """
    # 初期コード生成
    if initial_code is None:
        prompt = f"以下の問題を解くPython関数を実装してください:\n\n{problem_desc}\n\nコードのみ出力:"
        candidates = generate_with_logprobs(prompt, n=1, temperature=0.8)
        current_code = candidates[0]["text"].strip()
        current_candidate = candidates[0]
    else:
        current_code = initial_code
        # 初期コードの不確実性を評価
        prompt = f"以下のコード:\n{current_code}"
        candidates = generate_with_logprobs(prompt, n=1, temperature=0.0)
        current_candidate = candidates[0]

    history = []

    for iteration in range(max_iterations):
        # 不確実な行を特定
        uncertain_lines = identify_uncertain_lines(current_candidate, threshold=uncertainty_threshold)

        # 全体の平均不確実性を計算
        uncertainties = compute_token_uncertainty(current_candidate)
        avg_uncertainty = np.mean(uncertainties) if uncertainties else 0.0

        history.append({
            "iteration": iteration,
            "code": current_code,
            "uncertain_lines": uncertain_lines,
            "avg_uncertainty": avg_uncertainty
        })

        print(f"Iteration {iteration}: {len(uncertain_lines)} uncertain lines, avg uncertainty: {avg_uncertainty:.3f}")

        # 不確実な行がなければ終了
        if not uncertain_lines:
            print("No uncertain lines found. Stopping.")
            break

        # 不確実な部分を改善
        improved_code = refine_uncertain_parts(current_code, uncertain_lines, problem_desc)

        # 改善後の不確実性を評価
        prompt_eval = f"以下のコード:\n{improved_code}"
        candidates_eval = generate_with_logprobs(prompt_eval, n=1, temperature=0.0)

        if candidates_eval:
            current_code = improved_code
            current_candidate = candidates_eval[0]
        else:
            print("Refinement failed. Stopping.")
            break

    return {
        "final_code": current_code,
        "history": history
    }


if __name__ == "__main__":
    # テスト: フィボナッチ数列
    problem = "n番目のフィボナッチ数を返す関数fibonacci(n)を実装してください。"

    print("=== UGIR Test ===")
    result = ugir_pipeline(problem, max_iterations=3)

    print("\n=== Final Code ===")
    print(result["final_code"])

    print("\n=== History ===")
    for h in result["history"]:
        print(f"Iteration {h['iteration']}: {len(h['uncertain_lines'])} uncertain lines, avg_uncertainty={h['avg_uncertainty']:.3f}")
