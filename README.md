# topk

現行の論文作業は `paper/` と `scripts/` を正本にしています。論文に直接関係しない旧実験系は `trash/nonpaper_cleanup_20260427/` に退避済みです。

## 現在の主な置き場

- `paper/`: 論文本文，図，PDF．採用版は `paper/cbba_paper.tex` と `paper/cbba_paper.pdf`．
- `scripts/`: 論文用の実験実行・集計スクリプト．現行の正本は `run_paper12_k5_all.sh`，`run_cbba_paper.sh`，`summarize_paper_results.py`．
- `results/`: 論文実験と CBBA 系補助出力の置き場．現行 run は `results/paper12_cbba_*` を使う．
- `trash/`: 旧構成，旧結果，旧スクリプトの退避先．

## 論文パイプライン

```bash
bash scripts/run_cbba_paper.sh

python scripts/summarize_paper_results.py \
  --hybrid-dir results/paper12_k5_<timestamp>/hybrid \
  --output-dir results/paper12_cbba_<timestamp>

cd paper && make -B
```

## 運用メモ

- 現在の実験単位は `paper_easy4` / `paper_medium4` / `paper_hard4` / `paper12`。
- 一括実行は `bash scripts/run_paper12_k5_all.sh`。
- 背景実行は `bash scripts/run_paper12_k5_background.sh`。
- `NUM_TRIALS=10` 以上の長時間実験は，前景実行ではなく `run_paper12_k5_background.sh` を使う．対話セッションの中断や割り込みで，前景の子プロセスまで停止することがあるためである．
- 図生成は `paper/create_clean_figures_12problems.py` と `paper/create_sensitivity_figure.py`。
- 主結果は `results/paper12_cbba_<timestamp>/hybrid/hybrid_cbba_paper_easy4.json`、`hybrid_cbba_paper_medium4.json`、`hybrid_cbba_paper_hard4.json` を正本にする。
- `CBBA-C` / `CBBA-A` の生成数削減は，同一候補集合上でのオフライン再評価として扱っている。

## まず見る場所

- 論文本文: [cbba_paper.tex](/home/mukunoki/bot/pocketnika2/work/topk/paper/cbba_paper.tex)
- 論文 PDF: [cbba_paper.pdf](/home/mukunoki/bot/pocketnika2/work/topk/paper/cbba_paper.pdf)
- 論文実験: [run_cbba_paper.sh](/home/mukunoki/bot/pocketnika2/work/topk/scripts/run_cbba_paper.sh)
- 論文一括実験: [run_paper12_k5_all.sh](/home/mukunoki/bot/pocketnika2/work/topk/scripts/run_paper12_k5_all.sh)
- ベンチマーク定義: [paper_benchmark_sets.py](/home/mukunoki/bot/pocketnika2/work/topk/scripts/paper_benchmark_sets.py)
