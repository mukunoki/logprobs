"""Emit a Verilog testbench that drives TopModule and RefModule with a fixed
list of input vectors and prints the first mismatching vector index.

For combinational interfaces (no clock signal), the testbench applies each
vector, waits a small propagation delay, then strictly compares outputs.

For sequential interfaces, the testbench drives a clock and applies vectors
one per cycle. Outputs are compared after each posedge. If the interface has
a reset signal, the testbench drives reset high for a startup phase before
the vector sweep; methods may also assert reset inside the vector list.

The expected stdout markers are:

  - ``UGT_MISMATCH=<N>``   first cycle (0-indexed) where outputs diverged
  - ``UGT_PASS``           all vectors matched within the run

These markers are independent of any candidate-emitted ``$display`` calls.
"""
from __future__ import annotations

from .generate_inputs import InputVector
from .parse_interface import Interface, Signal


def _hex_lit(value: int, width: int) -> str:
    """Format a width-bit value as a Verilog sized literal in hex."""
    digits = max(1, (width + 3) // 4)
    return f"{width}'h{value:0{digits}x}"


def _port_declarations(iface: Interface) -> list[str]:
    """Generate reg/wire declarations for the testbench (one per signal)."""
    lines: list[str] = []
    for s in iface.signals:
        if s.direction == "input":
            # reg, driven by the testbench
            if s.is_vector or s.msb != s.lsb:
                lines.append(f"  reg [{s.msb}:{s.lsb}] {s.name};")
            else:
                lines.append(f"  reg {s.name};")
    # Outputs need separate wires for top and ref.
    for s in iface.outputs:
        for suffix in ("__top", "__ref"):
            if s.is_vector or s.msb != s.lsb:
                lines.append(f"  wire [{s.msb}:{s.lsb}] {s.name}{suffix};")
            else:
                lines.append(f"  wire {s.name}{suffix};")
    return lines


def _connections(iface: Interface, suffix: str) -> str:
    """Build the ``.port(net)`` connection list for one instantiation."""
    parts = []
    for s in iface.signals:
        if s.direction == "input":
            parts.append(f".{s.name}({s.name})")
        else:
            parts.append(f".{s.name}({s.name}{suffix})")
    return ", ".join(parts)


def _compare_expr(iface: Interface) -> str:
    """An expression that is non-zero when any output differs."""
    parts = [f"({s.name}__top !== {s.name}__ref)" for s in iface.outputs]
    return " || ".join(parts) if parts else "1'b0"


def _apply_vector(iface: Interface, vec: InputVector) -> list[str]:
    lines: list[str] = []
    values = vec.as_dict()
    for s in iface.driveable_inputs:
        v = values.get(s.name, 0)
        lines.append(f"    {s.name} = {_hex_lit(v, s.width)};")
    return lines


def render_testbench(
    iface: Interface,
    vectors: list[InputVector],
    timescale: str = "1ns/1ps",
    clock_period: int = 10,
    reset_cycles: int = 2,
    sim_finish_time: int = 1_000_000,
    seq_fill_cycles: int = 8,
) -> str:
    is_seq = iface.is_sequential
    clk = iface.clock
    rst = iface.reset

    lines: list[str] = []
    lines.append(f"`timescale {timescale}")
    lines.append("")
    lines.append("module ugt_testbench;")
    lines.extend(_port_declarations(iface))

    # Instantiations
    lines.append("")
    lines.append(f"  TopModule dut_top ({_connections(iface, '__top')});")
    lines.append(f"  RefModule dut_ref ({_connections(iface, '__ref')});")
    lines.append("")

    # Clock generation
    if is_seq:
        assert clk is not None
        lines.append(f"  initial {clk.name} = 1'b0;")
        lines.append(f"  always #{clock_period // 2} {clk.name} = ~{clk.name};")
        lines.append("")

    # Watchdog
    lines.append(f"  initial begin : watchdog")
    lines.append(f"    #{sim_finish_time};")
    lines.append('    $display("UGT_TIMEOUT");')
    lines.append("    $finish;")
    lines.append("  end")
    lines.append("")

    # Stimulus block
    lines.append("  integer i;")
    lines.append("  initial begin")
    # Default zero-init all driveable inputs.
    for s in iface.driveable_inputs:
        lines.append(f"    {s.name} = {_hex_lit(0, s.width)};")
    lines.append("")

    if is_seq and rst is not None:
        lines.append(f"    // Startup reset phase")
        lines.append(f"    {rst.name} = 1'b1;")
        for _ in range(reset_cycles):
            lines.append(f"    @(posedge {clk.name});")
        lines.append(f"    @(negedge {clk.name});")
        lines.append(f"    {rst.name} = 1'b0;")
        lines.append("")
    elif is_seq:
        # No reset signal — still wait a couple cycles so internal logic settles.
        lines.append(f"    @(posedge {clk.name});")
        lines.append(f"    @(posedge {clk.name});")
        lines.append("")

    # Apply each vector. For sequential circuits, hold the same inputs for
    # ``seq_fill_cycles`` additional clock edges so multi-cycle state
    # progressions (counter rollovers, FSM traversals) can manifest as output
    # mismatches; the detection index still refers to the vector that was
    # applied, not to the fill cycle.
    for i, vec in enumerate(vectors):
        lines.append(f"    // Vector {i}")
        if is_seq:
            lines.append(f"    @(negedge {clk.name});")
            lines.extend(_apply_vector(iface, vec))
            lines.append(f"    @(posedge {clk.name});")
            lines.append("    #1;")
            lines.append(f"    if ({_compare_expr(iface)}) begin")
            lines.append(f'      $display("UGT_MISMATCH=%0d", {i});')
            lines.append("      $finish;")
            lines.append("    end")
            for _ in range(seq_fill_cycles):
                lines.append(f"    @(posedge {clk.name});")
                lines.append("    #1;")
                lines.append(f"    if ({_compare_expr(iface)}) begin")
                lines.append(f'      $display("UGT_MISMATCH=%0d", {i});')
                lines.append("      $finish;")
                lines.append("    end")
        else:
            lines.extend(_apply_vector(iface, vec))
            lines.append("    #1;")
            lines.append(f"    if ({_compare_expr(iface)}) begin")
            lines.append(f'      $display("UGT_MISMATCH=%0d", {i});')
            lines.append("      $finish;")
            lines.append("    end")
        lines.append("")

    lines.append('    $display("UGT_PASS");')
    lines.append("    $finish;")
    lines.append("  end")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def _main_demo() -> None:
    from pathlib import Path

    from .parse_interface import parse_interface_file
    from .extract_span_signals import ExtractedFeatures, PartSelect
    from .generate_inputs import derive_seed, generate_inputs

    dataset = Path("/home/mukunoki/bot/pocketnika2/work/topk/external/verilog-eval/dataset_code-complete-iccad2023")

    print("# Combinational case (Prob004_vector2)")
    iface_c = parse_interface_file(dataset / "Prob004_vector2_ifc.txt")
    feats_c = ExtractedFeatures(
        focal_signals={"in"},
        part_selects=[PartSelect(signal="in", msb=7, lsb=0)],
        has_arith=False,
    )
    seed_c = derive_seed("Prob004_vector2", 0, 3, "hybrid")
    vecs_c = generate_inputs("hybrid", iface_c, feats_c, n=4, seed=seed_c)
    print(render_testbench(iface_c, vecs_c))

    print("# Sequential case (Prob080_timer)")
    iface_s = parse_interface_file(dataset / "Prob080_timer_ifc.txt")
    feats_s = ExtractedFeatures(focal_signals={"load", "data"}, has_arith=True)
    seed_s = derive_seed("Prob080_timer", 0, 0, "hybrid")
    vecs_s = generate_inputs("hybrid", iface_s, feats_s, n=3, seed=seed_s)
    print(render_testbench(iface_s, vecs_s))


if __name__ == "__main__":
    _main_demo()
