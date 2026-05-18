"""Generate up to ``n`` validation input vectors per UGT method.

Each vector assigns an integer value to every driveable input of the
interface. Clock signals are not sampled (the testbench drives them).
Reset signals, if present, are treated as ordinary driveable inputs; a
method that wants to exercise reset emits a vector with ``reset=1``.

Five methods are defined here, matching the paper:

  - ``random``: uniform random over each input's bit width.
  - ``generic``: a fixed library of structural stimuli (zero, all-one,
    one-hot, arithmetic boundaries, selector / reset variants).
  - ``category``: pick from the generic library according to low-conf token
    categories (no signal-name resolution).
  - ``span``: directed inputs built from ``ExtractedFeatures.focal_signals``,
    ``part_selects``, and flags.
  - ``hybrid``: 4 random smoke inputs followed by ``span``-style directed
    inputs to make ``n`` vectors total.

Vectors that produce an identical (signal, value) tuple are de-duplicated
in append order. Random number generators are derived from a
``hashlib.sha256`` digest of ``(problem, trial, candidate, method)`` so the
sequence is reproducible from the identifier alone.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from .extract_span_signals import ExtractedFeatures, PartSelect
from .parse_interface import Interface, Signal


DEFAULT_N_VECTORS = 16
DEFAULT_RANDOM_SMOKE = 4


@dataclass(frozen=True)
class InputVector:
    """Values for every driveable input. Field order follows ``Interface``."""

    values: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, int]:
        return dict(self.values)


def derive_seed(problem: str, trial: int, candidate: int, method: str) -> int:
    digest = hashlib.sha256(f"{problem}|{trial}|{candidate}|{method}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _max_val(signal: Signal) -> int:
    return (1 << signal.width) - 1


def _alternating(width: int) -> int:
    """0xAAAA... pattern truncated to ``width`` bits (bit 0 = 0)."""
    return ((1 << width) - 1) & 0xAAAAAAAAAAAAAAAA & ((1 << width) - 1)


def _one_hot(width: int, bit: int) -> int:
    if bit < 0 or bit >= width:
        return 0
    return 1 << bit


def _random_value(signal: Signal, rng: random.Random) -> int:
    if signal.width >= 63:
        return rng.randint(0, _max_val(signal))
    return rng.randrange(0, _max_val(signal) + 1)


def _vector(iface: Interface, values: dict[str, int]) -> InputVector:
    """Build an InputVector with every driveable input set (default 0)."""
    out: list[tuple[str, int]] = []
    for s in iface.driveable_inputs:
        v = values.get(s.name, 0) & _max_val(s)
        out.append((s.name, v))
    return InputVector(values=tuple(out))


# ----- per-method generators ---------------------------------------------


def _random_vector(iface: Interface, rng: random.Random) -> InputVector:
    return _vector(iface, {s.name: _random_value(s, rng) for s in iface.driveable_inputs})


def make_random(iface: Interface, n: int, rng: random.Random) -> list[InputVector]:
    return [_random_vector(iface, rng) for _ in range(n)]


def _generic_library(iface: Interface, rng: random.Random) -> list[InputVector]:
    """Fixed-order structural stimuli used by ``generic`` and ``category``."""
    inputs = iface.driveable_inputs
    out: list[InputVector] = []
    # All zero
    out.append(_vector(iface, {}))
    # All one (each signal at its max).
    out.append(_vector(iface, {s.name: _max_val(s) for s in inputs}))
    # One-hot on bit 0 across every signal.
    out.append(_vector(iface, {s.name: 1 if s.width >= 1 else 0 for s in inputs}))
    # One-hot on each signal's MSB.
    out.append(_vector(iface, {s.name: _one_hot(s.width, s.width - 1) for s in inputs}))
    # Alternating bit pattern.
    out.append(_vector(iface, {s.name: _alternating(s.width) for s in inputs}))
    # Arithmetic boundaries: each signal at ``max_val - 1`` and ``max_val // 2``.
    out.append(_vector(iface, {s.name: max(_max_val(s) - 1, 0) for s in inputs}))
    out.append(_vector(iface, {s.name: _max_val(s) // 2 for s in inputs}))
    # Selector values: enumerate small values 1..min(7, max).
    for v in (1, 2, 3, 5, 7):
        out.append(_vector(iface, {s.name: min(v, _max_val(s)) for s in inputs}))
    # Reset variants: if reset exists, assert it once and deassert with another shape.
    if iface.reset is not None:
        rname = iface.reset.name
        out.append(_vector(iface, {rname: 1}))
        out.append(_vector(iface, {rname: 1, **{s.name: _max_val(s) for s in inputs if s.name != rname}}))
        out.append(_vector(iface, {rname: 0, **{s.name: _one_hot(s.width, 0) for s in inputs if s.name != rname}}))
    # Random fill at end so the prefix is deterministic.
    while len(out) < 32:
        out.append(_random_vector(iface, rng))
    return out


def make_generic(iface: Interface, n: int, rng: random.Random) -> list[InputVector]:
    return _generic_library(iface, rng)[:n]


def make_category(
    iface: Interface,
    n: int,
    features: ExtractedFeatures,
    rng: random.Random,
) -> list[InputVector]:
    """Use only the *categories* of low-conf tokens, never the specific signals.

    The category set picks an ordering over the generic library so that
    arithmetic-heavy candidates get arithmetic-boundary inputs first, and so
    on. This is the closest faithful interpretation of the paper's
    description ("低確信度カテゴリだけを使う").
    """
    library = _generic_library(iface, rng)
    indexes: list[int] = []
    cats = features.categories

    def add(idx: int) -> None:
        if 0 <= idx < len(library) and idx not in indexes:
            indexes.append(idx)

    # Priority order driven by which categories appear in low-conf tokens.
    if "arith" in cats:
        # Arithmetic boundaries first.
        add(5)  # max - 1
        add(6)  # max // 2
        add(1)  # all one
        add(0)  # zero
    if "shift" in cats:
        add(4)  # alternating
        add(2)  # one-hot bit0
        add(3)  # one-hot MSB
    if "selector" in cats:
        for off in range(7, 12):
            add(off)
    if "state" in cats:
        add(2)
        add(3)
    if "reset_clock" in cats and iface.reset is not None:
        for off in (12, 13, 14):
            if off < len(library):
                add(off)
    # Fall through to the rest of the library.
    for idx in range(len(library)):
        add(idx)
    return [library[i] for i in indexes[:n]]


def _span_directed_vectors(
    iface: Interface,
    features: ExtractedFeatures,
    rng: random.Random,
    target_n: int,
) -> list[InputVector]:
    """Construct Span-UGT directed vectors in the fixed paper-described order."""
    inputs = iface.driveable_inputs
    focal_inputs = [s for s in inputs if s.name in features.focal_signals]
    if not focal_inputs:
        # No focal info — fall back to the generic library.
        return _generic_library(iface, rng)[:target_n]

    out: list[InputVector] = []
    # (a) two random-over-focal-only vectors: focal random, the rest zero.
    for _ in range(2):
        vals = {s.name: _random_value(s, rng) for s in focal_inputs}
        out.append(_vector(iface, vals))

    # (b) part-select-driven: one-hot on the extracted bits, max value, alternating.
    for ps in features.part_selects:
        s = iface.signal(ps.signal)
        if s is None:
            continue
        lo = min(ps.msb, ps.lsb)
        hi = max(ps.msb, ps.lsb)
        # Candidates occasionally contain malformed part-selects (negative
        # indices, or ranges larger than the declared signal). Skip them
        # rather than crashing the worker.
        if lo < 0 or hi < 0 or hi >= s.width:
            continue
        # Mask covering [lo, hi]
        mask = ((1 << (hi - lo + 1)) - 1) << lo
        # One-hot on the high bit of the slice.
        out.append(_vector(iface, {s.name: _one_hot(s.width, hi) & _max_val(s)}))
        # All bits in the slice set.
        out.append(_vector(iface, {s.name: mask & _max_val(s)}))
        # Alternating across the whole signal.
        out.append(_vector(iface, {s.name: _alternating(s.width)}))

    # (c) arithmetic boundaries: 0, 1, 2, mid, max-1.
    if features.has_arith:
        for v in (0, 1, 2):
            out.append(_vector(iface, {s.name: min(v, _max_val(s)) for s in focal_inputs}))
        out.append(_vector(iface, {s.name: _max_val(s) // 2 for s in focal_inputs}))
        out.append(_vector(iface, {s.name: max(_max_val(s) - 1, 0) for s in focal_inputs}))

    # (d) shift-op stimuli: alternating bits, single-bit-set across positions.
    if features.has_shift:
        for s in focal_inputs:
            out.append(_vector(iface, {s.name: _alternating(s.width)}))
            for bit in range(min(s.width, 4)):
                out.append(_vector(iface, {s.name: _one_hot(s.width, bit)}))

    # (e) reset / clock — drive reset high then low.
    if features.has_reset_clock and iface.reset is not None:
        rname = iface.reset.name
        out.append(_vector(iface, {rname: 1}))
        out.append(_vector(iface, {rname: 1, **{s.name: _random_value(s, rng) for s in focal_inputs if s.name != rname}}))
        out.append(_vector(iface, {rname: 0, **{s.name: _max_val(s) for s in focal_inputs if s.name != rname}}))

    # (f) padding with random over all inputs.
    while len(out) < target_n:
        out.append(_random_vector(iface, rng))
    return out


def make_span(
    iface: Interface,
    n: int,
    features: ExtractedFeatures,
    rng: random.Random,
) -> list[InputVector]:
    return _span_directed_vectors(iface, features, rng, n)[:n]


def make_hybrid(
    iface: Interface,
    n: int,
    features: ExtractedFeatures,
    rng: random.Random,
    random_smoke: int = DEFAULT_RANDOM_SMOKE,
) -> list[InputVector]:
    """Hybrid-UGT = ``random_smoke`` smoke vectors + Span-UGT directed vectors.

    When the interface has a reset signal (sequential circuits), we constrain
    the smoke prefix so that at least one early vector deasserts reset and at
    least one asserts it. Without this constraint, a uniform-random smoke
    sequence on a 1-bit reset has a 50/50 chance of leaving the design held
    in reset for the whole smoke phase, causing multi-cycle state-machine
    bugs (e.g., decade counter rollovers) to slip past the first vector. The
    constraint preserves the uniform-random spirit on all *non-reset* signals
    while making the smoke window a fair test for sequential candidates.
    """
    if random_smoke >= n:
        return make_random(iface, n, rng)
    rst = iface.reset
    smoke: list[InputVector] = []
    if rst is not None and random_smoke >= 2:
        # Force the first two smoke vectors to cover both reset polarities;
        # remaining smoke is uniform-random as before.
        for forced in (0, 1):
            vals = {s.name: _random_value(s, rng) for s in iface.driveable_inputs}
            vals[rst.name] = forced
            smoke.append(_vector(iface, vals))
        for _ in range(random_smoke - 2):
            smoke.append(_random_vector(iface, rng))
    else:
        smoke = [_random_vector(iface, rng) for _ in range(random_smoke)]
    directed = _span_directed_vectors(iface, features, rng, n - random_smoke)
    return (smoke + directed)[:n]


def _dedup(vectors: list[InputVector]) -> list[InputVector]:
    seen: set[InputVector] = set()
    out: list[InputVector] = []
    for v in vectors:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def generate_inputs(
    method: str,
    interface: Interface,
    features: ExtractedFeatures | None,
    n: int,
    seed: int,
) -> list[InputVector]:
    """Top-level entry point.

    ``features`` may be ``None`` for methods that ignore it (``random``,
    ``generic``); span / category / hybrid require a real ``ExtractedFeatures``.
    """
    rng = random.Random(seed)
    if method == "random":
        vectors = make_random(interface, n, rng)
    elif method == "generic":
        vectors = make_generic(interface, n, rng)
    elif method == "category":
        assert features is not None, "category requires features"
        vectors = make_category(interface, n, features, rng)
    elif method == "span":
        assert features is not None, "span requires features"
        vectors = make_span(interface, n, features, rng)
    elif method == "hybrid":
        assert features is not None, "hybrid requires features"
        vectors = make_hybrid(interface, n, features, rng)
    else:
        raise ValueError(f"unknown method: {method!r}")
    return _dedup(vectors)[:n]


def _main_demo() -> None:
    from pathlib import Path

    from .parse_interface import parse_interface_file

    dataset = Path("/home/mukunoki/bot/pocketnika2/work/topk/external/verilog-eval/dataset_code-complete-iccad2023")
    iface = parse_interface_file(dataset / "Prob080_timer_ifc.txt")
    print("Interface signals:", [(s.name, s.direction, s.width) for s in iface.signals])

    # Toy features mimicking what Span-UGT might extract for a timer
    feats = ExtractedFeatures(
        focal_signals={"load", "data"},
        part_selects=[PartSelect(signal="data", msb=9, lsb=0)],
        has_arith=True,
        has_state=True,
        has_reset_clock=False,
    )
    feats.categories = {"arith", "state", "identifier", "literal"}

    seed = derive_seed("Prob080_timer", 0, 0, "hybrid")
    for method in ("random", "generic", "category", "span", "hybrid"):
        vectors = generate_inputs(method, iface, feats, n=16, seed=seed)
        print(f"\n-- {method} ({len(vectors)} vectors) --")
        for i, v in enumerate(vectors):
            print(f"  {i}: {v.as_dict()}")


if __name__ == "__main__":
    _main_demo()
