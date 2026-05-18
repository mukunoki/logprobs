# Verilog UGT Evaluation Pipeline (Reconstruction)

Reconstructs the offline failure-detection evaluator described in `topk/paper/new.tex`
§4–§6 from the surviving candidate generation data
(`topk/results/verilog_eval_formal_qwen35_9b_awq4_20260509_*/candidates.jsonl`).

The original evaluator was lost during a prior archive cleanup. The candidate
generation phase is **not** reconstructed (the 4.5 GB `candidates.jsonl` already
contains all generated Verilog plus per-token log probabilities).

## Module layout

| File | Role |
|---|---|
| `parse_interface.py` | Parse `Prob*_ifc.txt` → `Interface` (signals, widths, clock, reset). |
| `extract_span_signals.py` | Map bottom-16 log-prob tokens to spans, extract focal signals / bit ranges / operators / case-state / reset-clock. |
| `generate_inputs.py` | Generate up to 16 validation-input vectors per method (`random`, `generic`, `category`, `span`, `hybrid`). |
| `generate_testbench.py` | Emit testbench wrapping `TopModule` (candidate) vs `RefModule` (reference) over a user-supplied input vector list. |
| `evaluate_candidate.py` | Compile (iverilog) + simulate (vvp) one (candidate, method) → `tests_to_detect`. |
| `run_full_eval.py` | Orchestrate (problem, trial, candidate, method) sweep with multiprocess pool, resume support. |

## Output schema

Per-candidate row (CSV / JSONL):

```
problem, trial, candidate, method,
detected (bool), tests_to_detect (int or null),
feature_group (str)
```

Aggregation is delegated to existing scripts in `topk/paper/`:
`analyze_hybrid_problem_dependence.py`, `build_full_eval_tables.py`,
`build_hybrid_per_problem_figure.py`.

## Method definitions (per new.tex §4.5)

| Method | Behavior |
|---|---|
| `random` | Uniform-random input vector per bit width. |
| `generic` | Fixed sequence: zero, all-one, one-hot, arithmetic boundaries, selector values, reset values. |
| `category` | Use low-confidence token **categories** only (no span extraction). |
| `span` | Span-UGT: extract focal signals/bits from low-conf spans; emit one-hot, max, alternating, boundary inputs. |
| `hybrid` | First 4 random smoke + remaining span-guided. |

All methods produce **at most 16** vectors. Same-vector dedup. Seed derived from
`hash(problem, trial, candidate, method_name)`.

## Reproduction goal

Match (within bootstrap CI) the paper's headline numbers:

- Hybrid det@16 = 0.821, Random det@16 = 0.827 (overall, n=31,842)
- counter tag: Δdet@16 = +0.089 [+0.074, +0.105]
- shift tag:   Δdet@16 = −0.156 [−0.173, −0.140]
