#!/usr/bin/env python3
"""Analyze same-prompt candidate pools for confidence-ordering methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("-inf")
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def tail_score(candidate: dict[str, Any]) -> float:
    return percentile([float(x) for x in candidate.get("token_logprobs", [])], 0.25)


def evaluate_order(candidates: list[dict[str, Any]], order: list[int]) -> dict[str, Any]:
    for tests, idx in enumerate(order, start=1):
        if candidates[idx]["success"]:
            return {"success": True, "tests": float(tests)}
    return {"success": False, "tests": float(len(order))}


def evaluate_tg_tail(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {"success": False, "tests": 0.0, "generated": 0.0}
    if candidates[0]["success"]:
        return {"success": True, "tests": 1.0, "generated": 1.0}
    rest = sorted(range(1, len(candidates)), key=lambda idx: (-tail_score(candidates[idx]), idx))
    for tests, idx in enumerate(rest, start=2):
        if candidates[idx]["success"]:
            return {"success": True, "tests": float(tests), "generated": float(len(candidates))}
    return {"success": False, "tests": float(len(candidates)), "generated": float(len(candidates))}


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def summarize_problem(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = {
        name: {"success": 0, "tests": [], "generated": []}
        for name in ["Gen", "Mean", "Tail", "TG-Tail"]
    }
    first_success = 0
    any_success = 0
    success_counts: list[int] = []
    success_conf: list[float] = []
    failure_conf: list[float] = []
    success_tail: list[float] = []
    failure_tail: list[float] = []

    for row in rows:
        candidates = row["candidates"]
        k = len(candidates)
        if candidates and candidates[0]["success"]:
            first_success += 1
        scount = sum(1 for cand in candidates if cand["success"])
        success_counts.append(scount)
        if scount:
            any_success += 1

        for cand in candidates:
            if cand["success"]:
                success_conf.append(float(cand["confidence"]))
                success_tail.append(tail_score(cand))
            else:
                failure_conf.append(float(cand["confidence"]))
                failure_tail.append(tail_score(cand))

        gen = evaluate_order(candidates, list(range(k)))
        methods["Gen"]["success"] += int(gen["success"])
        methods["Gen"]["tests"].append(gen["tests"])
        methods["Gen"]["generated"].append(float(k))

        mean_order = sorted(range(k), key=lambda idx: (-float(candidates[idx]["confidence"]), idx))
        mean_eval = evaluate_order(candidates, mean_order)
        methods["Mean"]["success"] += int(mean_eval["success"])
        methods["Mean"]["tests"].append(mean_eval["tests"])
        methods["Mean"]["generated"].append(float(k))

        tail_order = sorted(range(k), key=lambda idx: (-tail_score(candidates[idx]), idx))
        tail_eval = evaluate_order(candidates, tail_order)
        methods["Tail"]["success"] += int(tail_eval["success"])
        methods["Tail"]["tests"].append(tail_eval["tests"])
        methods["Tail"]["generated"].append(float(k))

        tg = evaluate_tg_tail(candidates)
        methods["TG-Tail"]["success"] += int(tg["success"])
        methods["TG-Tail"]["tests"].append(tg["tests"])
        methods["TG-Tail"]["generated"].append(tg["generated"])

    n = len(rows)
    return {
        "trials": n,
        "first_success_rate": first_success / n if n else 0.0,
        "any_success_rate": any_success / n if n else 0.0,
        "avg_success_candidates": mean(success_counts) if success_counts else 0.0,
        "mean_auc": auc(success_conf, failure_conf),
        "tail_auc": auc(success_tail, failure_tail),
        "methods": {
            name: {
                "success_rate": values["success"] / n if n else 0.0,
                "avg_tests": mean(values["tests"]) if values["tests"] else 0.0,
                "avg_generated": mean(values["generated"]) if values["generated"] else 0.0,
            }
            for name, values in methods.items()
        },
    }


def load_candidate_sets(input_path: str | None, input_dir: str | None) -> tuple[list[dict[str, Any]], str]:
    if bool(input_path) == bool(input_dir):
        raise ValueError("Specify exactly one of --input or --input-dir")
    if input_path:
        data = json.loads(Path(input_path).read_text(encoding="utf-8"))
        return list(data["candidate_sets"]), input_path

    rows: list[dict[str, Any]] = []
    source_dir = Path(input_dir or "")
    for path in sorted(source_dir.glob("*.json")):
        if path.name.startswith("manifest") or path.name.endswith("_analysis.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(data.get("candidate_sets", []))
    return rows, str(source_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-md")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    candidate_sets, input_label = load_candidate_sets(args.input, args.input_dir)
    by_problem: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_sets:
        by_problem.setdefault(row["problem_name"], []).append(row)

    summary = {
        problem: summarize_problem(rows)
        for problem, rows in sorted(by_problem.items())
    }

    lines = [
        "# Same-Prompt Method Analysis",
        "",
        f"Input: `{input_label}`",
        "",
        "| Problem | Trials | Any | First | Succ cand | Mean AUC | Tail AUC | Gen tests | Mean tests | Tail tests | TG-Tail tests | TG gen |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for problem, item in summary.items():
        methods = item["methods"]
        mean_auc = f"{item['mean_auc']:.3f}" if item["mean_auc"] is not None else "nan"
        tail_auc = f"{item['tail_auc']:.3f}" if item["tail_auc"] is not None else "nan"
        lines.append(
            f"| {problem} | {item['trials']} | {item['any_success_rate']*100:.1f}% | "
            f"{item['first_success_rate']*100:.1f}% | {item['avg_success_candidates']:.2f} | "
            f"{mean_auc} | {tail_auc}"
            f" | {methods['Gen']['avg_tests']:.2f} | {methods['Mean']['avg_tests']:.2f} | "
            f"{methods['Tail']['avg_tests']:.2f} | {methods['TG-Tail']['avg_tests']:.2f} | "
            f"{methods['TG-Tail']['avg_generated']:.2f} |"
        )

    text = "\n".join(lines) + "\n"
    if args.output_md:
        Path(args.output_md).write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
