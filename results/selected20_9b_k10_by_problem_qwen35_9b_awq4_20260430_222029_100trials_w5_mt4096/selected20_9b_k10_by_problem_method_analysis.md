# Selected20 9B k=10 手法分析

このファイルは，現在の論文で用いる20問・100 trials・9Bモデル評価の最新サマリである．現行論文に不要な旧方式の詳細は含めない．

## 入力

- 結果ディレクトリ: `/home/mukunoki/bot/pocketnika2/work/topk/results/selected20_9b_k10_by_problem_qwen35_9b_awq4_20260430_222029_100trials_w5_mt4096`
- 候補JSONディレクトリ: `by_problem/`
- モデル: `cyankiwi/Qwen3.5-9B-AWQ-4bit`
- trials: 各ベンチマーク100
- ベンチマーク数: 20
- 候補数: 各trialで同一プロンプトを独立に10回呼び出す
- temperature: 0.8
- max_tokens: 4096
- workers: 5

## 比較手法

- `Gen`: API返却順に候補をテストする．
- `Mean`: 平均token log確率で順位付ける．
- `Tail`: token log確率の下位四分位で順位付ける．
- `MinLogP`: 候補中の最小token log確率で順位付ける．

すべての手法は同じ候補集合上で評価する．したがって，各trialの成功条件は同一であり，成功または全失敗までのテスト数だけが異なる．

## ケース別平均テスト数

| Case | Gen | Mean | Tail | MinLogP |
|---|---:|---:|---:|---:|
| LogP-Separable | 3.508 | 2.696 | 2.650 | 2.786 |
| Tail-Sensitive | 6.416 | 5.728 | 5.674 | 5.896 |
| Weak-Signal | 2.862 | 2.724 | 2.722 | 2.448 |
| Unreliable-Signal | 6.558 | 6.778 | 6.742 | 6.496 |
| Overall | 4.836 | 4.481 | 4.447 | 4.407 |

## 現行解釈

TailはLogP-SeparableとTail-Sensitiveで最も有効である．MinLogPは補完的であり，Weak-Signalで最も強い．Unreliable-Signalでは，高log確率候補でも意味的に誤ることが多く，改善は小さい，または悪化する．

現行論文では，token log確率を較正済みの正解確率ではなく，相対的な順位付け信号として扱う．主張の対象は固定候補集合上のテスト順序最適化であり，成功率改善，生成回数削減，壁時計時間削減ではない．

## ベンチマーク分類

| ケース | ベンチマーク |
|---|---|
| LogP-Separable | `pareval_sort_ignore_zero_i32`, `utf8_validate_strict`, `pareval_largest_component_i32`, `csr_spmv_axpy_dot`, `transpose_strided_f64` |
| Tail-Sensitive | `pareval_convex_hull_perimeter_f64`, `topk_ignore_nan_f32`, `stencil3d_mixed_7pt`, `stencil2d_halo5`, `cholesky_spd_f64` |
| Weak-Signal | `crop2d_strided_u8`, `matrix_transpose_cache`, `rmsnorm_mixed`, `heap_sort_implementation`, `quadratic_roots_stable_f64` |
| Unreliable-Signal | `conv2d_3x3_multi_channel`, `radix_sort_u32_pairs`, `floyd_warshall_blocked`, `lower_tri_solve_strided_f64`, `banded_edit_distance_i32` |
