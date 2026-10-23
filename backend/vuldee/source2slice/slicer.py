"""Backward program slicing on LLVM IR (a pure-Python ``dg``-free slicer).

The original pipeline delegated to the ``dg`` ``llvm-slicer`` binary. Because
that toolchain is Linux-only and unmaintained, this module provides an
equivalent (data-flow) backward slicer that works on the textual LLVM IR that
``clang -S -emit-llvm`` emits.

It builds a def-use graph over SSA virtual registers and transitively includes
every instruction a seed instruction depends on. The result is a list of IR
lines in original source order - the semantic slice for one candidate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_DEF_RE = re.compile(r"^\s*(%\w+)\s*=")
_USE_RE = re.compile(r"%\w+")


@dataclass
class IrInstruction:
    index: int
    line: str
    defined: str | None
    uses: list[str]


def parse_instructions(ir_lines: list[str]) -> list[IrInstruction]:
    """Parse SSA def-use information out of a list of IR lines."""
    instrs: list[IrInstruction] = []
    for idx, line in enumerate(ir_lines):
        defined = None
        m = _DEF_RE.match(line)
        if m:
            defined = m.group(1)
        uses = _USE_RE.findall(line)
        if defined:
            uses = [u for u in uses if u != defined]
        instrs.append(IrInstruction(index=idx, line=line, defined=defined, uses=uses))
    return instrs


def backward_slice(ir_lines: list[str], seed_indices: list[int]) -> list[str]:
    """Return the IR lines (in original order) that ``seed_indices`` depend on."""
    return [ir_lines[i] for i in backward_slice_indices(ir_lines, seed_indices)]


def backward_slice_indices(ir_lines: list[str], seed_indices: list[int]) -> list[int]:
    """Return the indices of the IR lines that ``seed_indices`` depend on."""
    instrs = parse_instructions(ir_lines)
    def_site: dict[str, int] = {}
    for ins in instrs:
        if ins.defined:
            def_site[ins.defined] = ins.index

    included: set[int] = set()
    queue = list(seed_indices)
    while queue:
        idx = queue.pop()
        if idx in included:
            continue
        included.add(idx)
        ins = instrs[idx]
        for use in ins.uses:
            if use in def_site and def_site[use] not in included:
                queue.append(def_site[use])

    return sorted(included)
