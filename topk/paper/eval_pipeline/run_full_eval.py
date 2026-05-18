"""Orchestrate the 5-method evaluation over the stratified CPOF subsample.

For every ``(problem, trial, candidate)`` in ``--index-jsonl`` and every
method in ``--methods``, runs:

  1. ``extract_span_signals.extract`` on the cached per-token logprobs,
  2. ``generate_inputs.generate_inputs`` with a per-method seed,
  3. ``evaluate_candidate.evaluate`` to compile+simulate the testbench.

Output is a JSONL with one row per ``(problem, trial, candidate, method)``:

::

    {
        "problem": ...,
        "trial": ...,
        "candidate": ...,
        "method": ...,
        "detected": true|false,
        "tests_to_detect": int|null,
        "n_vectors": int,
        "status": "ok"|"compile_fail"|"sim_fail"|"timeout"|"no_marker",
    }

The output is **resumable**: rows already present (matched by the same key)
are skipped. A separate CSV summary is produced at the end via
``--summary-csv``.

Compile and simulate dominate the wall-clock time; the orchestration uses
a process pool so each candidate spawns iverilog/vvp on a separate core.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Allow `python -m topk.paper.eval_pipeline.run_full_eval` from repo root.
from .evaluate_candidate import evaluate
from .extract_span_signals import extract
from .generate_inputs import derive_seed, generate_inputs
from .parse_interface import parse_interface_file


DEFAULT_METHODS = ("random", "generic", "category", "span", "hybrid")
DEFAULT_N_VECTORS = 16
DEFAULT_DATASET = (
    "/home/mukunoki/bot/pocketnika2/work/topk/external/verilog-eval/"
    "dataset_code-complete-iccad2023"
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _index_candidates(
    candidates_path: Path,
    wanted: set[tuple[str, int, int]],
) -> dict[tuple[str, int, int], dict]:
    rows: dict[tuple[str, int, int], dict] = {}
    with candidates_path.open() as f:
        for line in f:
            d = json.loads(line)
            key = (d["problem"], d["trial"], d["candidate"])
            if key in wanted:
                rows[key] = d
    return rows


def _index_logprobs(
    pl_path: Path,
    wanted: set[tuple[str, int, int]],
) -> dict[tuple[str, int, int], dict]:
    rows: dict[tuple[str, int, int], dict] = {}
    with pl_path.open() as f:
        for line in f:
            d = json.loads(line)
            if "error" in d:
                continue
            key = (d["problem"], d["trial"], d["candidate"])
            if key in wanted:
                rows[key] = d
    return rows


def _load_completed(out_path: Path) -> set[tuple[str, int, int, str]]:
    done: set[tuple[str, int, int, str]] = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((d["problem"], d["trial"], d["candidate"], d["method"]))
    return done


# ----- worker -----------------------------------------------------------


# Worker globals (set in init). Each child process holds these for its lifetime.
_WORKER_RESOURCES: dict[str, Any] = {}


def _worker_init(
    dataset_dir: str,
    candidates_path: str,
    pl_path: str,
    wanted_set: set[tuple[str, int, int]],
    n_vectors: int,
) -> None:
    pl_idx = _index_logprobs(Path(pl_path), wanted_set)
    cand_idx = _index_candidates(Path(candidates_path), wanted_set)
    # Pre-parse interfaces and load reference modules.
    problems = {k[0] for k in wanted_set}
    iface_map = {p: parse_interface_file(Path(dataset_dir) / f"{p}_ifc.txt") for p in problems}
    ref_map = {
        p: (Path(dataset_dir) / f"{p}_ref.sv").read_text(encoding="utf-8")
        for p in problems
    }
    _WORKER_RESOURCES.update({
        "pl_idx": pl_idx,
        "cand_idx": cand_idx,
        "iface_map": iface_map,
        "ref_map": ref_map,
        "n_vectors": n_vectors,
    })


def _worker_task(args: tuple) -> dict:
    problem, trial, candidate, method = args
    key = (problem, trial, candidate)
    out = {"problem": problem, "trial": trial, "candidate": candidate, "method": method}
    try:
        pl = _WORKER_RESOURCES["pl_idx"].get(key)
        cand = _WORKER_RESOURCES["cand_idx"].get(key)
        iface = _WORKER_RESOURCES["iface_map"][problem]
        ref_text = _WORKER_RESOURCES["ref_map"][problem]
        n_vectors = _WORKER_RESOURCES["n_vectors"]

        if pl is None or cand is None:
            out.update({"detected": False, "tests_to_detect": None, "n_vectors": 0,
                        "status": "missing_input"})
            return out

        feats = extract(cand["response_text"], pl["tokens"], pl["logprobs"], iface)
        seed = derive_seed(problem, trial, candidate, method)
        vectors = generate_inputs(method, iface, feats, n=n_vectors, seed=seed)
        result = evaluate(iface, cand["candidate_sv"], ref_text, vectors)
        out.update({
            "detected": result.detected,
            "tests_to_detect": result.tests_to_detect,
            "n_vectors": len(vectors),
            "status": result.status,
        })
        return out
    except Exception as exc:
        # Per-task failures must not kill the pool. Record the error and move on.
        out.update({"detected": False, "tests_to_detect": None, "n_vectors": 0,
                    "status": "worker_exception",
                    "error": f"{type(exc).__name__}: {exc}"})
        return out


# ----- summary ----------------------------------------------------------


def write_summary(out_jsonl: Path, summary_csv: Path, methods: list[str], max_k: int) -> None:
    """Aggregate per-method det@k and mean tests-to-detect."""
    rows = _load_jsonl(out_jsonl)
    if not rows:
        print("no rows to summarize")
        return
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    ks = [1, 2, 4, 8, 16]
    if max_k > 16:
        ks.append(max_k)
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "n_candidates",
                         *[f"det@{k}" for k in ks],
                         "mean_tests_to_detect"])
        for m in methods:
            data = by_method.get(m, [])
            n = len(data)
            row = [m, n]
            for k in ks:
                det_count = sum(
                    1 for r in data
                    if r.get("tests_to_detect") is not None
                    and r["tests_to_detect"] <= k
                )
                row.append(round(det_count / n, 4) if n else 0)
            ttd = [
                r["tests_to_detect"] if r.get("tests_to_detect") is not None else max_k + 1
                for r in data
            ]
            row.append(round(sum(ttd) / n, 4) if n else 0)
            writer.writerow(row)
    print(f"wrote {summary_csv}")


# ----- CLI --------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-jsonl", type=Path, required=True)
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--prompt-logprobs-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET))
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS),
                        choices=list(DEFAULT_METHODS))
    parser.add_argument("--n-vectors", type=int, default=DEFAULT_N_VECTORS)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N pending tasks")
    args = parser.parse_args()

    print(f"loading subsample index from {args.index_jsonl}")
    index_rows = _load_jsonl(args.index_jsonl)
    wanted = [(d["problem"], d["trial"], d["candidate"]) for d in index_rows]
    wanted_set = set(wanted)
    print(f"  {len(wanted)} candidates, methods={args.methods}, n_vectors={args.n_vectors}")

    done = _load_completed(args.out_jsonl)
    all_tasks = [(p, t, c, m) for (p, t, c) in wanted for m in args.methods]
    pending = [task for task in all_tasks if task not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"  {len(done)} already done, {len(all_tasks)} total tasks, "
          f"{len(pending)} pending this run")

    if not pending:
        write_summary(args.out_jsonl, args.summary_csv, args.methods, args.n_vectors)
        return

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.out_jsonl.open("a", encoding="utf-8")

    init_args = (
        str(args.dataset_dir),
        str(args.candidates_jsonl),
        str(args.prompt_logprobs_jsonl),
        wanted_set,
        args.n_vectors,
    )

    print(f"starting pool of {args.workers} workers...")
    t0 = time.time()
    n_done = 0
    n_detected = 0
    n_status_err = 0
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=init_args,
    ) as pool:
        for result in pool.map(_worker_task, pending, chunksize=4):
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            n_done += 1
            if result.get("detected"):
                n_detected += 1
            if result.get("status") not in {"ok"}:
                n_status_err += 1
            if n_done % 200 == 0 or n_done == len(pending):
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(pending) - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{len(pending)}] det={n_detected} "
                      f"non-ok={n_status_err} rate={rate:.1f}/s eta={eta/60:.1f}min")

    out_f.close()
    print(f"done. processed {n_done}, detected {n_detected}, non-ok {n_status_err}")

    write_summary(args.out_jsonl, args.summary_csv, args.methods, args.n_vectors)


if __name__ == "__main__":
    main()
