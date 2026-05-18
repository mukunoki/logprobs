"""Build a reproducible per-problem stratified sub-sample of CPOF candidates.

The full Verilog generation run produced 156,000 candidates. The paper's
evaluation uses the CPOF (compile-pass, official-fail) subset with
generation length <= 512 tokens (31,842 candidates). This script samples
roughly 1/RATIO of each problem's CPOF candidates so that downstream
re-scoring and per-method evaluation stays inside one day of GPU time
while preserving the natural per-problem distribution. The
per-feature/per-problem caps that caused pilot bias in the original work
are deliberately not used.

Reading:
  ``--candidates-jsonl`` (default: the May-2026 run output).

Output:
  ``--out-index`` JSONL with one ``{problem, trial, candidate}`` per line.
  ``--out-meta``  JSON  with seed, ratio, and per-problem counts.

The same ``(seed, ratio)`` always yields the same subsample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_CANDIDATES = (
    "/home/mukunoki/bot/pocketnika2/work/topk/results/"
    "verilog_eval_formal_qwen35_9b_awq4_20260509_100trials_k10_w8_mt16384_s12345/"
    "candidates.jsonl"
)
DEFAULT_MAX_TOKENS = 512


def _problem_rng(global_seed: int, problem: str) -> random.Random:
    """Per-problem RNG derived from a global seed.

    Using a hash of (seed, problem) instead of a single global RNG keeps the
    sample for each problem independent of iteration order over ``problems``.
    """
    digest = hashlib.sha256(f"{global_seed}|{problem}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def collect_cpof(candidates_path: Path, max_tokens: int) -> dict[str, list[dict]]:
    by_problem: dict[str, list[dict]] = defaultdict(list)
    with candidates_path.open() as f:
        for line in f:
            d = json.loads(line)
            if not d.get("compile_ok"):
                continue
            if d.get("success"):
                continue
            if d.get("tokens", 0) > max_tokens:
                continue
            by_problem[d["problem"]].append(
                {
                    "problem": d["problem"],
                    "trial": d["trial"],
                    "candidate": d["candidate"],
                    "tokens": d["tokens"],
                }
            )
    return by_problem


def sample_subset(
    by_problem: dict[str, list[dict]],
    ratio: int,
    seed: int,
) -> tuple[list[dict], dict[str, dict]]:
    chosen: list[dict] = []
    summary: dict[str, dict] = {}
    for problem in sorted(by_problem):
        rows = by_problem[problem]
        target = max(1, round(len(rows) / ratio))
        rng = _problem_rng(seed, problem)
        # Sort first so the ordering fed to rng is deterministic regardless of
        # the jsonl scan order.
        rows_sorted = sorted(rows, key=lambda r: (r["trial"], r["candidate"]))
        picked = rng.sample(rows_sorted, target)
        picked.sort(key=lambda r: (r["trial"], r["candidate"]))
        chosen.extend(picked)
        summary[problem] = {"cpof_count": len(rows), "picked": target}
    return chosen, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-jsonl", type=Path, default=Path(DEFAULT_CANDIDATES))
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--ratio", type=int, default=5,
                        help="Sample 1/RATIO of each problem's CPOF candidates")
    parser.add_argument("--seed", type=int, default=54321)
    parser.add_argument("--out-index", type=Path, required=True)
    parser.add_argument("--out-meta", type=Path, required=True)
    args = parser.parse_args()

    print(f"reading candidates from {args.candidates_jsonl}")
    by_problem = collect_cpof(args.candidates_jsonl, args.max_tokens)
    total_cpof = sum(len(v) for v in by_problem.values())
    print(f"  CPOF (tokens<={args.max_tokens}): {total_cpof} across {len(by_problem)} problems")

    chosen, summary = sample_subset(by_problem, args.ratio, args.seed)
    print(f"sampled {len(chosen)} candidates (ratio=1/{args.ratio}, seed={args.seed})")

    args.out_index.parent.mkdir(parents=True, exist_ok=True)
    with args.out_index.open("w") as f:
        for row in chosen:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {args.out_index}")

    meta = {
        "candidates_jsonl": str(args.candidates_jsonl),
        "max_tokens": args.max_tokens,
        "ratio": args.ratio,
        "seed": args.seed,
        "total_cpof": total_cpof,
        "total_sampled": len(chosen),
        "problems_with_cpof": len(by_problem),
        "per_problem": summary,
    }
    args.out_meta.write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out_meta}")


if __name__ == "__main__":
    main()
