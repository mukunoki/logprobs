# topk

このプロジェクトは，LLMコード生成で得た複数候補をどの順序でコンパイル・テストするかを，token-level log確率で最適化できるかを評価するための実験・論文作業用リポジトリである．現在の論文は「生成回数削減」ではなく，固定候補集合に対する「テスト順序最適化」を対象にする．

## 現行の論文

- 論文本文: `paper/cbba_paper.tex`
- 論文PDF: `paper/cbba_paper.pdf`
- 現行結果: `results/selected20_9b_k10_by_problem_qwen35_9b_awq4_20260430_222029_100trials_w5_mt4096/`
- 候補単位のJSON: `results/selected20_9b_k10_by_problem_qwen35_9b_awq4_20260430_222029_100trials_w5_mt4096/by_problem/`

## 現行の実験設定

- モデル: `cyankiwi/Qwen3.5-9B-AWQ-4bit`
- 推論エンジン: vLLM OpenAI互換API
- 生成設定: temperature=0.8，`k=10`，max_tokens=4096
- 出力設定: thinking無効，Cコードのみを出力
- 試行数: 20ベンチマーク，各100 trials
- 候補生成: 同一プロンプトを独立に10回呼び出し，1呼び出しにつき1候補を得る
- 評価方式: 全候補のコンパイル・テスト結果とtoken logprobsを保存し，同一候補集合上で検証順序だけをオフライン比較する
- 主指標: 成功候補に到達するまでの平均テスト数
- 成功率: 全手法で同じ候補集合を使うため，手法差ではなく候補集合品質の補助指標として扱う

## 現行の比較手法

- `Gen`: API返却順にテストする生成順ベースライン
- `Mean`: 候補全体の平均token log確率に基づく順位付け
- `Tail`: token log確率の下位四分位に基づく順位付け
- `MinLogP`: 候補中の最小token log確率に基づく順位付け

現在の論文では，固定候補集合上の4手法のみを扱う．非採用となった旧方式の詳細はこのREADMEには残さない．

## ベンチマーク分類

20問を，事前難易度ではなく，観測されたlog確率信号の振る舞いに基づいて5問ずつ4ケースに分類する．

| ケース | ベンチマーク |
|---|---|
| LogP-Separable | `pareval_sort_ignore_zero_i32`, `utf8_validate_strict`, `pareval_largest_component_i32`, `csr_spmv_axpy_dot`, `transpose_strided_f64` |
| Tail-Sensitive | `pareval_convex_hull_perimeter_f64`, `topk_ignore_nan_f32`, `stencil3d_mixed_7pt`, `stencil2d_halo5`, `cholesky_spd_f64` |
| Weak-Signal | `crop2d_strided_u8`, `matrix_transpose_cache`, `rmsnorm_mixed`, `heap_sort_implementation`, `quadratic_roots_stable_f64` |
| Unreliable-Signal | `conv2d_3x3_multi_channel`, `radix_sort_u32_pairs`, `floyd_warshall_blocked`, `lower_tri_solve_strided_f64`, `banded_edit_distance_i32` |

## 主要結果

平均テスト数は以下の通りである．値が小さいほど，成功候補に早く到達できている．

| ケース | Gen | Mean | Tail | MinLogP |
|---|---:|---:|---:|---:|
| LogP-Separable | 3.51 | 2.70 | 2.65 | 2.79 |
| Tail-Sensitive | 6.42 | 5.73 | 5.67 | 5.90 |
| Weak-Signal | 2.86 | 2.72 | 2.72 | 2.45 |
| Unreliable-Signal | 6.56 | 6.78 | 6.74 | 6.50 |

全20問平均では，Gen 4.836，Mean 4.481，Tail 4.447，MinLogP 4.407である．ただし，論文では全20問平均よりも，問題群ごとの効き方の違いを重視する．

## 現行の結論

token log確率は，成功候補と失敗候補の相対的な順位付けに使える場合がある．特にLogP-SeparableとTail-SensitiveではTailが有効であり，Weak-SignalではMinLogPが補完的に効く．一方で，Unreliable-Signalでは高log確率でも意味的に誤る候補が多く，log確率だけでsemantic correctnessを判別することはできない．

## 現行の図生成

論文中の主要図は以下のスクリプトで生成する．

```bash
cd /home/mukunoki/bot/pocketnika2/work/topk/paper
python3 create_case_figures_20problems.py
python3 create_token_logp_feature_figure.py
python3 create_token_loss_distribution_figure.py
python3 create_sensitivity_figure.py
make -B
```

## 現行実験の再実行

主評価を再実行する場合は，長時間ジョブとしてバックグラウンド起動スクリプトを使う．

```bash
cd /home/mukunoki/bot/pocketnika2/work/topk
NUM_TRIALS=100 WORKERS=5 MAX_TOKENS=4096 \
  VLLM_MODEL_NAME=cyankiwi/Qwen3.5-9B-AWQ-4bit \
  bash scripts/run_9b_k10_selected20_by_problem_background.sh
```

バックグラウンド実行は，PID，ログ，メタ情報を `paper/logs/` に残す．前景で長時間実験を走らせない．

## 現在見るべきファイル

- `paper/cbba_paper.tex`: 論文本文
- `paper/cbba_paper.pdf`: ビルド済みPDF
- `paper/create_case_figures_20problems.py`: 20問4ケースの図生成
- `paper/create_token_logp_feature_figure.py`: MeanLogP/TailLogP分布図
- `paper/create_token_loss_distribution_figure.py`: token loss生存分布図
- `scripts/run_9b_k10_selected20_by_problem_background.sh`: 主評価のバックグラウンド実行
- `scripts/run_9b_k10_selected20_by_problem_all.sh`: 主評価の実処理
- `scripts/paper_benchmark_sets.py`: 現行ベンチマーク集合

## 古いMarkdownの扱い

現在の論文に不要な旧実験メモと自動生成サマリは `trash/obsolete_md_20260501/` に退避した．プロジェクト文書として参照するMarkdownは，このREADMEと最新結果サマリのみである．
