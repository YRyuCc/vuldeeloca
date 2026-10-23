"""clang-driven semantic slicing helpers.

Bridges the source-level focus points with the LLVM IR: compiles C/C++ to
textual IR with debug line information, maps every IR instruction back to a
source line, and produces the semantic slice (a :class:`CorpusSample`) for a
single focus point.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

from vuldee.data.corpus import CorpusSample, build_sample_from_lines
from vuldee.source2slice.slicer import backward_slice_indices

_DBG_RE = re.compile(r"!dbg !(\d+)")
_LOC_RE = re.compile(r"^!(\d+)\s*=\s*!DILocation\(line:\s*(\d+)")

CLANG_FLAGS = [
    "-S", "-emit-llvm", "-g", "-O0", "-w",
    "-std=gnu89",
    "-Wno-error=implicit-function-declaration",
    "-Wno-error=implicit-int",
    "-Wno-error=int-conversion",
]


def find_clang() -> str:
    return os.environ.get("VULDEE_CLANG") or shutil.which("clang") or "clang"


def compile_to_ir(src_path: str, extra_includes: list[str] | None = None) -> list[str] | None:
    """Compile ``src_path`` to textual LLVM IR lines, or ``None`` on failure."""
    clang = find_clang()
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "out.ll")
        cmd = [clang, *CLANG_FLAGS]
        for inc in extra_includes or []:
            cmd += ["-I", inc]
        cmd += [src_path, "-o", out_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(out_path):
            return None
        with open(out_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()


def parse_debug_map(ir_lines: list[str]) -> dict[int, int]:
    """Map IR instruction line index -> source line number."""
    metadata: dict[str, int] = {}
    for line in ir_lines:
        m = _LOC_RE.match(line.strip())
        if m:
            metadata[m.group(1)] = int(m.group(2))

    line_to_source: dict[int, int] = {}
    for idx, line in enumerate(ir_lines):
        m = _DBG_RE.search(line)
        if m and m.group(1) in metadata:
            line_to_source[idx] = metadata[m.group(1)]
    return line_to_source


def focus_slice(
    ir_lines: list[str],
    line_to_source: dict[int, int],
    focus_line: int,
) -> tuple[list[str], list[int]] | None:
    """Return ``(slice_lines, ir_indices)`` for the focus at ``focus_line``."""
    seeds = [i for i, src_line in line_to_source.items() if src_line == focus_line]
    if not seeds:
        return None
    indices = backward_slice_indices(ir_lines, seeds)
    if not indices:
        return None
    return [ir_lines[i] for i in indices], indices


def build_slice_sample(
    ir_lines: list[str],
    line_to_source: dict[int, int],
    focus_line: int,
    flaw_lines: set[int],
    sample_id: str,
    focus_type: str,
    focus_name: str,
    source: str,
) -> CorpusSample | None:
    """Build a corpus sample from one focus point's semantic slice."""
    result = focus_slice(ir_lines, line_to_source, focus_line)
    if result is None:
        return None
    slice_lines, indices = result
    vuln_positions = [
        j + 1
        for j, ir_idx in enumerate(indices)
        if line_to_source.get(ir_idx) in flaw_lines
    ]
    return build_sample_from_lines(
        slice_lines,
        sample_id,
        vuln_line_numbers=vuln_positions,
        funcs=[focus_name],
        focus_type=focus_type,
        source=source,
    )
