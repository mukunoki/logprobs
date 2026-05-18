"""Re-score generated Verilog candidates with vLLM to recover per-token logprobs.

The original generation run only retained aggregated log-probability scalars
(mean / tail / min). Hybrid-UGT and its ablations need the per-token series.
This script asks the same vLLM server (with the same model) to recompute
``prompt_logprobs`` over ``problem_prompt + response_text`` and stores the
sequence corresponding to the response portion.

Input
-----
* ``--index-jsonl``: candidate identifiers ``(problem, trial, candidate)``
  selected by ``select_eval_set.py``.
* ``--candidates-jsonl``: the full candidates file (provides
  ``response_text`` per candidate).
* ``--dataset-dir``: VerilogEval dataset to read ``Prob*_prompt.txt``.

Output
------
``--out-jsonl`` with one row per candidate:
``{problem, trial, candidate, tokens: [...], logprobs: [...]}``

The output is resumable: rows already present in the output file are skipped
on the next invocation. This is critical because rescoring takes hours and we
do not want to lose progress on transient vLLM errors.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "cyankiwi/Qwen3.5-9B-AWQ-4bit"
DEFAULT_DATASET = (
    "/home/mukunoki/bot/pocketnika2/work/topk/external/verilog-eval/"
    "dataset_code-complete-iccad2023"
)


def _extract_actual(entry: dict[str, dict[str, Any]] | None) -> tuple[str | None, float | None]:
    """Pull the actually-emitted token's decoded string and logprob from one
    position in vLLM's ``prompt_logprobs`` payload.

    With ``prompt_logprobs=1`` the server returns the rank-1 prediction plus
    the actual token if it was not already rank 1. So either one entry (where
    rank 1 == actual) or two entries (in which case the actual has rank > 1).
    """
    if entry is None:
        return None, None
    if len(entry) == 1:
        info = next(iter(entry.values()))
        return info["decoded_token"], info["logprob"]
    for info in entry.values():
        if info["rank"] != 1:
            return info["decoded_token"], info["logprob"]
    # All entries are rank 1 (shouldn't happen with N=1, but fall back to first).
    info = next(iter(entry.values()))
    return info["decoded_token"], info["logprob"]


def get_prompt_token_count(
    session: requests.Session,
    base_url: str,
    model: str,
    prompt_text: str,
    timeout: float = 30.0,
) -> int:
    """Ask vLLM's /tokenize endpoint how many tokens the prompt occupies.

    The boundary between prompt and response in ``prompt_logprobs`` cannot be
    reliably derived from character counts because the first token's decoded
    string is hidden inside a ``None`` entry, and BPE may also merge characters
    across the prompt/response seam.
    """
    # /tokenize is served at the root (not /v1), so strip a trailing "/v1".
    root_url = base_url.removesuffix("/v1")
    r = session.post(
        f"{root_url}/tokenize",
        json={"model": model, "prompt": prompt_text},
        timeout=timeout,
    )
    r.raise_for_status()
    return int(r.json()["count"])


def rescore_one(
    session: requests.Session,
    base_url: str,
    model: str,
    prompt_text: str,
    response_text: str,
    prompt_token_count: int,
    timeout: float = 120.0,
) -> tuple[list[str], list[float]]:
    """Submit one (prompt, response) pair to vLLM and slice out the response logprobs."""
    full_text = prompt_text + response_text
    payload = {
        "model": model,
        "prompt": full_text,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 1,
        "prompt_logprobs": 1,
    }
    r = session.post(f"{base_url}/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    prompt_logprobs = data["choices"][0]["prompt_logprobs"]
    response_part = [_extract_actual(e) for e in prompt_logprobs[prompt_token_count:]]
    tokens = [tok or "" for tok, _ in response_part]
    logprobs = [lp if lp is not None else float("-inf") for _, lp in response_part]
    return tokens, logprobs


def load_subsample(path: Path) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            out.append((d["problem"], d["trial"], d["candidate"]))
    return out


def load_candidates_map(
    candidates_path: Path,
    wanted: set[tuple[str, int, int]],
) -> dict[tuple[str, int, int], dict]:
    """Stream candidates.jsonl once and keep only the rows we will rescore."""
    out: dict[tuple[str, int, int], dict] = {}
    with candidates_path.open() as f:
        for line in f:
            d = json.loads(line)
            key = (d["problem"], d["trial"], d["candidate"])
            if key in wanted:
                out[key] = d
    return out


def load_prompts(dataset_dir: Path, problems: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for problem in problems:
        p = dataset_dir / f"{problem}_prompt.txt"
        out[problem] = p.read_text(encoding="utf-8")
    return out


def load_completed(out_path: Path) -> set[tuple[str, int, int]]:
    done: set[tuple[str, int, int]] = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((d["problem"], d["trial"], d["candidate"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-jsonl", type=Path, required=True)
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET))
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent vLLM requests (server batches internally)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N pending candidates (smoke testing)")
    args = parser.parse_args()

    print(f"loading subsample index from {args.index_jsonl}")
    wanted = load_subsample(args.index_jsonl)
    wanted_set = set(wanted)
    print(f"  {len(wanted)} candidates requested")

    done = load_completed(args.out_jsonl)
    pending = [k for k in wanted if k not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"  {len(done)} already rescored, {len(pending)} pending")

    if not pending:
        print("nothing to do")
        return

    print(f"loading candidate rows for {len(pending)} pending entries...")
    rows = load_candidates_map(args.candidates_jsonl, set(pending))
    print(f"  matched {len(rows)} rows in candidates.jsonl")
    if len(rows) != len(pending):
        missing = [k for k in pending if k not in rows]
        print(f"  WARN: {len(missing)} pending entries not found; first: {missing[:3]}")

    problems = {k[0] for k in pending}
    prompts = load_prompts(args.dataset_dir, problems)
    print(f"  loaded {len(prompts)} prompts")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.out_jsonl.open("a", encoding="utf-8")

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    print("computing prompt token counts...")
    prompt_token_counts = {
        p: get_prompt_token_count(session, args.base_url, args.model, prompts[p])
        for p in sorted(problems)
    }
    print(f"  prompt token counts ready for {len(prompt_token_counts)} problems")

    def task(key: tuple[str, int, int]) -> dict:
        problem, trial, candidate = key
        row = rows.get(key)
        if row is None:
            return {"problem": problem, "trial": trial, "candidate": candidate,
                    "error": "candidate row missing"}
        try:
            tokens, logprobs = rescore_one(
                session, args.base_url, args.model,
                prompt_text=prompts[problem],
                response_text=row["response_text"],
                prompt_token_count=prompt_token_counts[problem],
            )
        except Exception as exc:
            return {"problem": problem, "trial": trial, "candidate": candidate,
                    "error": f"{type(exc).__name__}: {exc}"}
        return {
            "problem": problem,
            "trial": trial,
            "candidate": candidate,
            "tokens": tokens,
            "logprobs": logprobs,
        }

    t0 = time.time()
    n_ok = n_err = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(task, pending):
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            if "error" in result:
                n_err += 1
                if n_err <= 5:
                    print(f"  ERR {result['problem']} t{result['trial']} c{result['candidate']}: {result['error']}")
            else:
                n_ok += 1
            done_count = n_ok + n_err
            if done_count % 50 == 0 or done_count == len(pending):
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (len(pending) - done_count) / rate if rate > 0 else 0
                print(f"  [{done_count}/{len(pending)}] ok={n_ok} err={n_err} "
                      f"rate={rate:.1f}/s eta={eta/60:.1f}min")

    out_f.close()
    print(f"done. ok={n_ok}, err={n_err}, total={n_ok + n_err}")


if __name__ == "__main__":
    main()
