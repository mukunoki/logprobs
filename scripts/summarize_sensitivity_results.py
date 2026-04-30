#!/usr/bin/env python3
"""Summarize Easy-4 sensitivity runs for CBBA-C / CBBA-A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_hybrid_cbba import process_dataset


def parse_mapping(values: list[str], key_type: type[int] | type[float]) -> list[tuple[int | float, Path]]:
    parsed: list[tuple[int | float, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=PATH mapping, got: {value}")
        raw_key, raw_path = value.split("=", 1)
        parsed.append((key_type(raw_key), Path(raw_path)))
    return parsed


def summarize_run(path: Path, budget: int) -> dict[str, Any]:
    data = process_dataset("paper_easy4", path, budget)
    summary = data["summary"]
    return {
        "cbba_c": {
            "success_rate": summary["hybrid_conservative"]["success_rate"],
            "avg_generated": summary["hybrid_conservative"]["avg_generated_candidates_counted"],
            "avg_tests": summary["hybrid_conservative"]["avg_tests_executed"],
        },
        "cbba_a": {
            "success_rate": summary["hybrid_aggressive"]["success_rate"],
            "avg_generated": summary["hybrid_aggressive"]["avg_generated_candidates_counted"],
            "avg_tests": summary["hybrid_aggressive"]["avg_tests_executed"],
        },
        "instances": len(data["candidate_sets"]),
        "source": str(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-run", action="append", default=[], help="Mapping of budget k to JSON path: K=/path/to.json")
    parser.add_argument("--temp-run", action="append", default=[], help="Mapping of temperature to JSON path: TEMP=/path/to.json")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_runs = parse_mapping(args.k_run, int)
    temp_runs = parse_mapping(args.temp_run, float)

    result: dict[str, Any] = {
        "k_variation": [],
        "temperature_variation": [],
    }

    for budget, path in sorted(k_runs, key=lambda item: item[0]):
        row = summarize_run(path, budget)
        row["k"] = budget
        row["temperature"] = 0.8
        result["k_variation"].append(row)

    for temperature, path in sorted(temp_runs, key=lambda item: item[0]):
        row = summarize_run(path, 3)
        row["k"] = 3
        row["temperature"] = temperature
        result["temperature_variation"].append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
