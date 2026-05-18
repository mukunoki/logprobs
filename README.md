# Verilog Hybrid-UGT — paper, pipeline, and frozen experiment data

This branch (`verilog-paper`) is an orphan snapshot containing only the
self-contained artifacts of the paper *"Token log確率を用いたVerilog
コード生成候補の問題依存的不確実性誘導テスト"*: the LaTeX sources, the
reconstruction pipeline (`topk/paper/eval_pipeline/`), and the frozen
evaluation data needed to reproduce every number in the paper.

## Contents

```
topk/paper/                                  Paper sources + asset builders
  new.tex, references.bib, ipsj*.cls/sty/bst Paper text and IPSJ class
  verilog_*_table.tex                        Tables used by new.tex
  verilog_hybrid_*.eps                       Figures used by new.tex
  Makefile, rungs                            platex + dvipdfmx build
  new.pdf                                    Pre-built paper PDF
  eval_pipeline/                             Reconstruction modules
    parse_interface.py, select_eval_set.py
    rescore_logprobs.py, extract_span_signals.py
    generate_inputs.py, generate_testbench.py
    evaluate_candidate.py, run_full_eval.py
    build_v2_assets.py

topk/results/verilog_eval_formal_qwen35_9b_awq4_20260509_*/
  metadata.json                              Generation run config
  candidates.jsonl.gz.part-{aa,ab}           156k generated candidates
                                              (split into <100 MB chunks)
  aggregate_summary.csv/json, reference_check.csv, trial_summary.csv

topk/results/verilog_eval_b_subsample/
  index_full.jsonl.gz                        Eval set (31,842 CPOF cand.)
  meta_full.json
  prompt_logprobs.jsonl.gz                   Per-token logprobs (vLLM rescored)
  eval_full.jsonl.gz                         Per-(cand, method) eval result
  summary_full.csv                           Per-method det@k headline numbers
```

## Restoring the large data

The generation candidates are split into two compressed chunks
(GitHub blocks files > 100 MB). To reconstitute the original
`candidates.jsonl`:

```bash
cd topk/results/verilog_eval_formal_qwen35_9b_awq4_20260509_*
cat candidates.jsonl.gz.part-* > candidates.jsonl.gz
gunzip candidates.jsonl.gz
# yields candidates.jsonl (1.5 GB)
```

For the rescored logprobs and eval results:

```bash
cd topk/results/verilog_eval_b_subsample
gunzip -k prompt_logprobs.jsonl.gz eval_full.jsonl.gz index_full.jsonl.gz
```

## Reproducing the headline numbers

After restoring the data:

```bash
cd topk/paper
python3 -m eval_pipeline.build_v2_assets \
  --eval-jsonl ../results/verilog_eval_b_subsample/eval_full.jsonl
make
```

This rebuilds the four `.tex` tables and two `.eps` figures, then
compiles `new.pdf` via platex.

To re-run the evaluation from scratch (requires vLLM + iverilog):

```bash
# 1. rescore per-token logprobs (vLLM server at 127.0.0.1:8000)
python3 -m eval_pipeline.rescore_logprobs \
  --index-jsonl ../results/verilog_eval_b_subsample/index_full.jsonl \
  --candidates-jsonl ../results/verilog_eval_formal_*/candidates.jsonl \
  --out-jsonl ../results/verilog_eval_b_subsample/prompt_logprobs.jsonl \
  --workers 8

# 2. run the 5-method evaluation
python3 -m eval_pipeline.run_full_eval \
  --index-jsonl ../results/verilog_eval_b_subsample/index_full.jsonl \
  --candidates-jsonl ../results/verilog_eval_formal_*/candidates.jsonl \
  --prompt-logprobs-jsonl ../results/verilog_eval_b_subsample/prompt_logprobs.jsonl \
  --out-jsonl ../results/verilog_eval_b_subsample/eval_full.jsonl \
  --summary-csv ../results/verilog_eval_b_subsample/summary_full.csv \
  --methods random generic category span hybrid \
  --workers 8
```

## Headline results (A-plan full evaluation, n=31,842)

| Method       | det@1 | det@2 | det@4 | det@8 | det@16 | mean |
|--------------|-------|-------|-------|-------|--------|------|
| Random       | 0.496 | 0.649 | 0.762 | 0.826 | 0.845  | 4.42 |
| Generic      | 0.296 | 0.504 | 0.632 | 0.718 | 0.779  | 6.13 |
| Category-UGT | 0.351 | 0.522 | 0.659 | 0.709 | 0.788  | 5.98 |
| Span-UGT     | 0.496 | 0.646 | 0.737 | 0.818 | 0.839  | 4.57 |
| Hybrid-UGT   | 0.524 | 0.649 | 0.761 | 0.819 | 0.841  | 4.45 |

Hybrid-UGT − Random Δdet@1 (paired bootstrap, 95% CI):

| Tag           | n      | Δdet@1 | 95% CI            |
|---------------|--------|--------|-------------------|
| counter       | 2,375  | +0.146 | [+0.126, +0.166]  |
| posedge       | 13,867 | +0.069 | [+0.062, +0.076]  |
| fsm_state     | 3,889  | +0.059 | [+0.044, +0.073]  |
| arith_op      | 7,712  | +0.055 | [+0.044, +0.067]  |
| shift         | 3,139  | +0.053 | [+0.039, +0.067]  |
| case          | 8,519  | +0.032 | [+0.021, +0.043]  |
| case_mux      | 2,629  | +0.022 | [+0.002, +0.040]  |

`shift` at det@16 = −0.061 [−0.078, −0.044] is the only persistent
negative; Random uniform's broader bit-width coverage outperforms
Hybrid-UGT's part-select boundaries on shift/rotate problems when the
budget is large.
