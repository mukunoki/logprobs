#!/usr/bin/env python3
"""Merge multiple threshold_refinement_eval outputs into a single file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from threshold_refinement_eval import aggregate_results


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-name", type=str, default="paper12")
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    datasets = [load(path) for path in args.inputs]
    first = datasets[0]
    candidate_sets: list[dict[str, Any]] = []
    for data in datasets:
        candidate_sets.extend(data.get("candidate_sets", []))

    merged = dict(first)
    merged["benchmark"] = args.benchmark_name
    merged["num_problems"] = sum(int(data.get("num_problems", 0)) for data in datasets)
    merged["candidate_sets"] = sorted(
        candidate_sets,
        key=lambda row: (row["trial"], row["problem_index"], row["problem_name"]),
    )
    merged["is_partial"] = any(bool(data.get("is_partial")) for data in datasets)
    merged["summary"] = aggregate_results(merged["candidate_sets"], first["thresholds"]) if candidate_sets else {}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
