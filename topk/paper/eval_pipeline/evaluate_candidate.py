"""Compile and simulate one (candidate, method) pair, recovering tests_to_detect.

Given the candidate source (``TopModule``), the reference source
(``RefModule``), and a list of ``InputVector``s, this module:

1. Writes top.sv, ref.sv, and tb.sv into a temporary directory.
2. Invokes ``iverilog`` to compile them into a vvp binary.
3. Invokes ``vvp`` to simulate.
4. Parses stdout for ``UGT_MISMATCH=<N>`` or ``UGT_PASS``.

Result: ``EvalResult(detected, tests_to_detect, status, log_tail)``.

``status`` is one of:
  - ``ok``           simulation completed and produced a marker.
  - ``compile_fail`` iverilog returned non-zero.
  - ``sim_fail``     vvp returned non-zero before printing a marker.
  - ``timeout``      simulation watchdog fired (or wall-clock timeout).
  - ``no_marker``    completed but printed neither marker.

The temp directory is deleted unless ``keep_tmp=True`` is passed, which is
useful for offline debugging of specific candidates.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .generate_inputs import InputVector
from .generate_testbench import render_testbench
from .parse_interface import Interface


DEFAULT_IVERILOG = "/home/mukunoki/bot/pocketnika2/work/topk/tools/oss-cad-suite/bin/iverilog"
DEFAULT_VVP = "/home/mukunoki/bot/pocketnika2/work/topk/tools/oss-cad-suite/bin/vvp"

_MISMATCH_RE = re.compile(r"^UGT_MISMATCH=(\d+)\s*$", re.MULTILINE)
_PASS_RE = re.compile(r"^UGT_PASS\s*$", re.MULTILINE)
_TIMEOUT_RE = re.compile(r"^UGT_TIMEOUT\s*$", re.MULTILINE)


@dataclass
class EvalResult:
    detected: bool
    tests_to_detect: int | None
    status: str
    log_tail: str


def _rename_module(src: str, old: str, new: str) -> str:
    """Replace the first ``module <old>`` declaration with ``module <new>``."""
    pattern = re.compile(rf"\bmodule\s+{re.escape(old)}\b")
    return pattern.sub(f"module {new}", src, count=1)


def evaluate(
    interface: Interface,
    candidate_sv: str,
    reference_sv: str,
    vectors: list[InputVector],
    *,
    iverilog: str = DEFAULT_IVERILOG,
    vvp: str = DEFAULT_VVP,
    timeout_sec: float = 60.0,
    keep_tmp: bool = False,
) -> EvalResult:
    """Compile + simulate; return EvalResult."""
    work = Path(tempfile.mkdtemp(prefix="ugt_eval_"))
    try:
        # Rename candidate's module if it still calls itself TopModule (it should).
        # The candidates already use TopModule, so no rename needed there.
        # The reference uses RefModule already.
        (work / "top.sv").write_text(candidate_sv, encoding="utf-8")
        (work / "ref.sv").write_text(reference_sv, encoding="utf-8")
        (work / "tb.sv").write_text(render_testbench(interface, vectors), encoding="utf-8")

        vvp_path = work / "sim.vvp"
        cp = subprocess.run(
            [iverilog, "-g2012", "-o", str(vvp_path), "-s", "ugt_testbench",
             str(work / "top.sv"), str(work / "ref.sv"), str(work / "tb.sv")],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if cp.returncode != 0:
            return EvalResult(
                detected=False,
                tests_to_detect=None,
                status="compile_fail",
                log_tail=(cp.stdout + cp.stderr)[-400:],
            )

        sp = subprocess.run(
            [vvp, str(vvp_path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        out = sp.stdout + sp.stderr
        m = _MISMATCH_RE.search(out)
        if m:
            return EvalResult(
                detected=True,
                tests_to_detect=int(m.group(1)) + 1,
                status="ok",
                log_tail=out[-400:],
            )
        if _PASS_RE.search(out):
            return EvalResult(
                detected=False,
                tests_to_detect=None,
                status="ok",
                log_tail=out[-400:],
            )
        if _TIMEOUT_RE.search(out):
            return EvalResult(
                detected=False,
                tests_to_detect=None,
                status="timeout",
                log_tail=out[-400:],
            )
        if sp.returncode != 0:
            return EvalResult(
                detected=False,
                tests_to_detect=None,
                status="sim_fail",
                log_tail=out[-400:],
            )
        return EvalResult(
            detected=False,
            tests_to_detect=None,
            status="no_marker",
            log_tail=out[-400:],
        )
    except subprocess.TimeoutExpired:
        return EvalResult(
            detected=False,
            tests_to_detect=None,
            status="timeout",
            log_tail="wall-clock timeout",
        )
    finally:
        if not keep_tmp:
            shutil.rmtree(work, ignore_errors=True)


def _main_demo() -> None:
    import json
    from pathlib import Path

    from .extract_span_signals import extract
    from .generate_inputs import derive_seed, generate_inputs
    from .parse_interface import parse_interface_file

    dataset = Path("/home/mukunoki/bot/pocketnika2/work/topk/external/verilog-eval/dataset_code-complete-iccad2023")
    pl_path = Path("/home/mukunoki/bot/pocketnika2/work/topk/results/verilog_eval_b_subsample/prompt_logprobs.jsonl")
    cand_path = Path("/home/mukunoki/bot/pocketnika2/work/topk/results/verilog_eval_formal_qwen35_9b_awq4_20260509_100trials_k10_w8_mt16384_s12345/candidates.jsonl")

    # Take the first valid candidate from the rescored set.
    sample = None
    with pl_path.open() as f:
        for line in f:
            d = json.loads(line)
            if "error" not in d:
                sample = d
                break
    assert sample is not None

    key = (sample["problem"], sample["trial"], sample["candidate"])
    cand = None
    with cand_path.open() as f:
        for line in f:
            row = json.loads(line)
            if (row["problem"], row["trial"], row["candidate"]) == key:
                cand = row
                break
    assert cand is not None

    iface = parse_interface_file(dataset / f"{key[0]}_ifc.txt")
    ref_text = (dataset / f"{key[0]}_ref.sv").read_text(encoding="utf-8")
    feats = extract(cand["response_text"], sample["tokens"], sample["logprobs"], iface)

    for method in ("random", "generic", "hybrid"):
        seed = derive_seed(*key, method)
        vectors = generate_inputs(method, iface, feats, n=16, seed=seed)
        result = evaluate(iface, cand["candidate_sv"], ref_text, vectors)
        print(f"{key} {method:8s}: status={result.status} detected={result.detected} "
              f"tests_to_detect={result.tests_to_detect}")


if __name__ == "__main__":
    _main_demo()
