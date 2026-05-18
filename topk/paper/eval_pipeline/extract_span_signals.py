"""Map low-logprob tokens to Verilog structural features used by UGT methods.

Hybrid-UGT, Span-UGT, and Category-UGT all start from the same primitive: a
list of low-confidence character spans inside the candidate code, where the
"low-confidence" set is the bottom-K tokens by log-probability. Each span is
the ``[token_offset - WINDOW, token_offset + WINDOW]`` substring of the
candidate response text.

This module produces, for one candidate:

* ``LowConfSpan`` records: ``(token, logprob, char_start, char_end, text)``,
  one per low-confidence token.
* ``ExtractedFeatures``: an aggregate over all spans capturing
  - focal signal names from ``Interface.driveable_inputs`` mentioned in any
    span,
  - explicit ``signal[N:M]`` / ``signal[N]`` part-selects,
  - flags for arithmetic, shift, case/selector, state-transition, and
    clock/reset constructs.

``ExtractedFeatures.categories`` is the set of low-confidence categories
(arith / shift / selector / state / reset / identifier / literal / other)
used by Category-UGT to choose stimulus templates without resolving specific
signal names.

The categorization rules are intentionally simple and explicit so that the
paper can describe them precisely. They are NOT a calibrated classifier;
they are pattern-matchers over the ~96-char neighborhood of each token.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parse_interface import Interface, Signal


DEFAULT_LOW_K = 16
DEFAULT_WINDOW = 96


# Patterns applied to span text. Order matters when both apply; keep them
# orthogonal and use ``set`` operations on the resulting tags.
_ARITH_OP_RE = re.compile(r"[+\-*/%](?!=)|[+\-*/%]=")
_SHIFT_OP_RE = re.compile(r"<<|>>")
_CASE_RE = re.compile(r"\bcase[zx]?\b|\?\s*[^:]+:")  # case statement OR ternary
_STATE_RE = re.compile(r"<=|always\s*@\s*\(\s*posedge|state\s*[<=]|next_state")
_RESET_CLOCK_RE = re.compile(r"\breset\b|\brst\b|\bclk\b|\bclock\b|\bposedge\b|\bnegedge\b", re.IGNORECASE)
_PARTSELECT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([+-]?\d+)\s*(?::\s*([+-]?\d+)\s*)?\]")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LITERAL_RE = re.compile(r"\b\d+'[bdho][0-9a-fxzA-FXZ_]+\b|\b\d+\b")


@dataclass
class LowConfSpan:
    """One low-confidence token + its neighborhood in the candidate code."""

    token_idx: int
    token: str
    logprob: float
    char_start: int  # span start (inclusive) in the response text
    char_end: int    # span end (exclusive)
    text: str        # response_text[char_start:char_end]


@dataclass
class PartSelect:
    """An observed ``signal[H:L]`` or ``signal[N]`` reference."""

    signal: str
    msb: int
    lsb: int  # for ``signal[N]`` we set msb == lsb == N


@dataclass
class ExtractedFeatures:
    """Aggregate features computed across all low-confidence spans."""

    spans: list[LowConfSpan] = field(default_factory=list)
    focal_signals: set[str] = field(default_factory=set)
    part_selects: list[PartSelect] = field(default_factory=list)
    has_arith: bool = False
    has_shift: bool = False
    has_selector: bool = False  # case / ternary / mux selector
    has_state: bool = False     # non-blocking assignment, posedge, state var
    has_reset_clock: bool = False
    categories: set[str] = field(default_factory=set)


def _select_low_conf_indices(
    tokens: list[str],
    logprobs: list[float],
    k: int,
) -> list[int]:
    """Indices of the ``k`` lowest-logprob non-trivial tokens.

    Filters out whitespace-only tokens and lone punctuation. They carry little
    information about Verilog structure and would otherwise dominate the
    bottom-K when the candidate has long indented blocks.
    """
    trivial = set("(){}[];,:")
    candidates = []
    for i, tok in enumerate(tokens):
        if not tok or not tok.strip():
            continue
        if tok.strip() in trivial:
            continue
        if logprobs[i] == float("-inf"):
            continue
        candidates.append((logprobs[i], i))
    candidates.sort()
    return [i for _, i in candidates[:k]]


def _token_char_offsets(tokens: list[str]) -> list[int]:
    """Return character start offsets for each token assuming concatenation."""
    offsets: list[int] = []
    cum = 0
    for tok in tokens:
        offsets.append(cum)
        cum += len(tok)
    return offsets


def _categorize_token(token: str) -> str:
    s = token.strip()
    if not s:
        return "other"
    if _ARITH_OP_RE.fullmatch(s) or s in {"+", "-", "*", "/", "%"}:
        return "arith"
    if s in {"<<", ">>"}:
        return "shift"
    if s in {"case", "casex", "casez", "?", ":"}:
        return "selector"
    if s in {"<=", "posedge", "negedge"}:
        return "state"
    if s.lower() in {"reset", "rst", "clk", "clock"}:
        return "reset_clock"
    if _LITERAL_RE.fullmatch(s):
        return "literal"
    if _IDENT_RE.fullmatch(s):
        return "identifier"
    return "other"


def extract(
    response_text: str,
    tokens: list[str],
    logprobs: list[float],
    interface: Interface,
    low_k: int = DEFAULT_LOW_K,
    window: int = DEFAULT_WINDOW,
) -> ExtractedFeatures:
    """Compute spans and aggregate features for one candidate.

    ``response_text`` must be the concatenation of ``tokens``; if the rescoring
    step honored its own slicing this is guaranteed. We do not assert it
    because some candidates start with markdown fences that introduce a tiny
    offset; instead, we treat token offsets as approximate and use the
    *response* text for span slicing (response_text could be a stripped
    version, but for the rescoring output it's identical).
    """
    feats = ExtractedFeatures()
    if not tokens or not logprobs or len(tokens) != len(logprobs):
        return feats

    low_idx = _select_low_conf_indices(tokens, logprobs, low_k)
    offsets = _token_char_offsets(tokens)
    signal_names = {s.name for s in interface.driveable_inputs}

    for i in low_idx:
        char_start = max(0, offsets[i] - window)
        char_end = min(len(response_text), offsets[i] + len(tokens[i]) + window)
        text = response_text[char_start:char_end]
        feats.spans.append(LowConfSpan(
            token_idx=i,
            token=tokens[i],
            logprob=logprobs[i],
            char_start=char_start,
            char_end=char_end,
            text=text,
        ))
        feats.categories.add(_categorize_token(tokens[i]))

        # Focal signal names (intersect span identifiers with interface inputs).
        idents = set(_IDENT_RE.findall(text))
        feats.focal_signals.update(idents & signal_names)

        # Part-selects mentioning interface signals.
        for m in _PARTSELECT_RE.finditer(text):
            name = m.group(1)
            if name not in signal_names:
                continue
            msb = int(m.group(2))
            lsb = int(m.group(3)) if m.group(3) is not None else msb
            feats.part_selects.append(PartSelect(signal=name, msb=msb, lsb=lsb))

        # Coarse flags.
        if _ARITH_OP_RE.search(text):
            feats.has_arith = True
        if _SHIFT_OP_RE.search(text):
            feats.has_shift = True
        if _CASE_RE.search(text):
            feats.has_selector = True
        if _STATE_RE.search(text):
            feats.has_state = True
        if _RESET_CLOCK_RE.search(text):
            feats.has_reset_clock = True

    # Deduplicate part-selects (signal, msb, lsb).
    seen: set[tuple[str, int, int]] = set()
    dedup: list[PartSelect] = []
    for ps in feats.part_selects:
        key = (ps.signal, ps.msb, ps.lsb)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(ps)
    feats.part_selects = dedup

    return feats


def _main_demo() -> None:
    """Smoke test against the first few rescored candidates."""
    import json
    from pathlib import Path

    from .parse_interface import parse_interface_file

    pl_path = Path("/home/mukunoki/bot/pocketnika2/work/topk/results/verilog_eval_b_subsample/prompt_logprobs.jsonl")
    cand_path = Path("/home/mukunoki/bot/pocketnika2/work/topk/results/verilog_eval_formal_qwen35_9b_awq4_20260509_100trials_k10_w8_mt16384_s12345/candidates.jsonl")
    dataset = Path("/home/mukunoki/bot/pocketnika2/work/topk/external/verilog-eval/dataset_code-complete-iccad2023")

    wanted: list[dict] = []
    if pl_path.exists():
        with pl_path.open() as f:
            for line in f:
                d = json.loads(line)
                if "error" not in d and len(wanted) < 3:
                    wanted.append(d)
                if len(wanted) >= 3:
                    break

    cand_map: dict[tuple, dict] = {}
    keys = {(d["problem"], d["trial"], d["candidate"]) for d in wanted}
    with cand_path.open() as f:
        for line in f:
            d = json.loads(line)
            k = (d["problem"], d["trial"], d["candidate"])
            if k in keys:
                cand_map[k] = d

    for d in wanted:
        key = (d["problem"], d["trial"], d["candidate"])
        iface = parse_interface_file(dataset / f"{d['problem']}_ifc.txt")
        cand = cand_map[key]
        feats = extract(cand["response_text"], d["tokens"], d["logprobs"], iface)
        print(f"=== {key} ===")
        print(f"  spans: {len(feats.spans)}")
        print(f"  focal_signals: {sorted(feats.focal_signals)}")
        print(f"  part_selects: {[(p.signal, p.msb, p.lsb) for p in feats.part_selects][:5]}")
        print(f"  flags: arith={feats.has_arith} shift={feats.has_shift} "
              f"selector={feats.has_selector} state={feats.has_state} "
              f"reset_clock={feats.has_reset_clock}")
        print(f"  categories: {sorted(feats.categories)}")
        if feats.spans:
            ex = feats.spans[0]
            print(f"  span[0]: token={ex.token!r} lp={ex.logprob:.2f}")
            print(f"    text: {ex.text[:120]!r}...")
        print()


if __name__ == "__main__":
    _main_demo()
