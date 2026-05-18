"""Parse VerilogEval `Prob*_ifc.txt` into a structured Interface.

The IFC file holds the `TopModule` port list used by both the candidate and the
reference implementation. The downstream testbench generator needs:

  - bit width and direction of every port,
  - which input is the clock (so the testbench can drive it instead of
    sampling values for it),
  - which input is the reset (so reset behavior can be exercised explicitly).

Heuristics for clock/reset names follow VerilogEval conventions: `clk`/`clock`
for clock, names containing `rst`, `reset`, `areset`, `sreset` for reset. The
heuristics fail closed: if no clock is found the interface is treated as
combinational, which is correct for problems like `Prob001_zero`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# Verilog identifier patterns and reserved words we may encounter on a port line.
_KEYWORDS = {"input", "output", "inout", "reg", "wire", "logic", "signed"}
_CLOCK_NAMES = {"clk", "clock", "ck"}
_RESET_PATTERNS = (
    re.compile(r"(?:^|_)(rst|reset|areset|sreset|aresetn|resetn|rst_n|rstn)(?:$|_)", re.IGNORECASE),
)


@dataclass(frozen=True)
class Signal:
    """One port of TopModule."""

    name: str
    direction: str  # "input", "output", or "inout"
    is_reg: bool
    msb: int  # for `[3:0]` msb=3, lsb=0; for scalar msb=lsb=0
    lsb: int

    @property
    def width(self) -> int:
        return abs(self.msb - self.lsb) + 1

    @property
    def is_vector(self) -> bool:
        return not (self.msb == 0 and self.lsb == 0 and self.width == 1)

    def decl(self) -> str:
        """Re-emit the port as a Verilog declaration (for testbench wiring)."""
        parts = [self.direction]
        if self.is_reg:
            parts.append("reg")
        if self.is_vector or self.msb != self.lsb:
            parts.append(f"[{self.msb}:{self.lsb}]")
        parts.append(self.name)
        return " ".join(parts)


@dataclass
class Interface:
    """Parsed TopModule port list."""

    signals: list[Signal] = field(default_factory=list)

    @property
    def clock(self) -> Signal | None:
        for s in self.signals:
            if s.direction == "input" and s.name.lower() in _CLOCK_NAMES:
                return s
        return None

    @property
    def reset(self) -> Signal | None:
        for s in self.signals:
            if s.direction != "input":
                continue
            if any(pat.search(s.name) for pat in _RESET_PATTERNS):
                return s
        return None

    @property
    def is_sequential(self) -> bool:
        return self.clock is not None

    @property
    def driveable_inputs(self) -> list[Signal]:
        """Inputs whose value the testbench should sample. Excludes clock; reset
        is included because methods may drive reset transitions explicitly."""
        clock = self.clock
        return [s for s in self.signals if s.direction == "input" and s is not clock]

    @property
    def outputs(self) -> list[Signal]:
        return [s for s in self.signals if s.direction == "output"]

    def signal(self, name: str) -> Signal | None:
        for s in self.signals:
            if s.name == name:
                return s
        return None


def _strip_comments(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def _extract_port_block(text: str) -> str:
    """Return the substring between `module TopModule (` and the matching `);`."""
    m = re.search(r"module\s+TopModule\s*\(", text)
    if m is None:
        raise ValueError("module TopModule not found")
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    raise ValueError("unterminated TopModule port list")


def _split_ports(block: str) -> list[str]:
    """Split the port-list block on commas that are not inside brackets."""
    ports: list[str] = []
    depth = 0
    buf: list[str] = []
    for c in block:
        if c == "[":
            depth += 1
            buf.append(c)
        elif c == "]":
            depth -= 1
            buf.append(c)
        elif c == "," and depth == 0:
            entry = "".join(buf).strip()
            if entry:
                ports.append(entry)
            buf = []
        else:
            buf.append(c)
    tail = "".join(buf).strip()
    if tail:
        ports.append(tail)
    return ports


_RANGE_RE = re.compile(r"\[\s*([+-]?\d+)\s*:\s*([+-]?\d+)\s*\]")


def _parse_one_port(entry: str) -> Signal:
    """Parse a single port declaration like `output reg [3:0] q` or `input clk`."""
    # Pull out a [msb:lsb] range if present.
    msb = lsb = 0
    m = _RANGE_RE.search(entry)
    if m is not None:
        msb = int(m.group(1))
        lsb = int(m.group(2))
        entry = _RANGE_RE.sub(" ", entry)

    tokens = [t for t in re.split(r"\s+", entry.strip()) if t]
    if not tokens:
        raise ValueError(f"empty port declaration: {entry!r}")

    name = tokens[-1]
    qualifiers = set(tokens[:-1])

    direction = None
    for d in ("input", "output", "inout"):
        if d in qualifiers:
            direction = d
            break
    if direction is None:
        raise ValueError(f"no direction in port declaration: {entry!r}")

    is_reg = "reg" in qualifiers

    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"unexpected port name token: {name!r} from {entry!r}")

    return Signal(name=name, direction=direction, is_reg=is_reg, msb=msb, lsb=lsb)


def parse_interface_text(text: str) -> Interface:
    """Parse a raw IFC file body into an Interface."""
    cleaned = _strip_comments(text)
    block = _extract_port_block(cleaned)
    ports = _split_ports(block)
    return Interface(signals=[_parse_one_port(p) for p in ports])


def parse_interface_file(path: str | Path) -> Interface:
    return parse_interface_text(Path(path).read_text(encoding="utf-8"))


def _main_demo() -> None:
    """Smoke test: print parsed interfaces for a few representative problems."""
    import sys

    paths = sys.argv[1:]
    if not paths:
        dataset = Path(
            "/home/mukunoki/bot/pocketnika2/work/topk/external/verilog-eval/dataset_code-complete-iccad2023"
        )
        paths = [
            str(dataset / "Prob001_zero_ifc.txt"),
            str(dataset / "Prob035_count1to10_ifc.txt"),
            str(dataset / "Prob068_countbcd_ifc.txt"),
            str(dataset / "Prob115_shift18_ifc.txt"),
            str(dataset / "Prob128_fsm_ps2_ifc.txt"),
        ]
    for path in paths:
        iface = parse_interface_file(path)
        print(f"== {Path(path).name} ==")
        print(f"  clock={iface.clock.name if iface.clock else None}")
        print(f"  reset={iface.reset.name if iface.reset else None}")
        print(f"  sequential={iface.is_sequential}")
        for s in iface.signals:
            tag = ""
            if s is iface.clock:
                tag = " [CLOCK]"
            elif s is iface.reset:
                tag = " [RESET]"
            print(f"  {s.decl():35s} width={s.width}{tag}")
        print()


if __name__ == "__main__":
    _main_demo()
