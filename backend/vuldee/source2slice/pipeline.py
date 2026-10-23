"""Source-to-candidate pipeline used by the scanner.

For a source file this module extracts focus points and wraps each one into a
source-level candidate (a :class:`~vuldee.data.corpus.CorpusSample`) that can
be vectorized and scored by the model. When ``clang`` is available the
semantic IR slicer can be used instead; the source-level path keeps the scanner
usable out of the box.
"""

from __future__ import annotations

from vuldee.source2slice.focus import FocusPoint, extract_focus_points
from vuldee.data.corpus import CorpusSample, build_sample_from_lines


def candidate_from_source(
    path: str,
    source: str,
    focus: FocusPoint,
    window: int = 8,
) -> CorpusSample:
    """Build a source-level candidate around one focus point."""
    lines = source.splitlines()
    start = max(0, focus.line - 1 - window)
    end = min(len(lines), focus.line + window)
    window_lines = lines[start:end]

    return build_sample_from_lines(
        lines=window_lines,
        sample_id=f"{path}:{focus.line}:{focus.type.value}",
        vuln_line_numbers=[],
        funcs=[focus.name],
        focus_type=focus.type.value,
        source=path,
        start_line=start + 1,
    )


def extract_candidates(path: str, source: str, window: int = 8) -> list[CorpusSample]:
    return [
        candidate_from_source(path, source, fp, window)
        for fp in extract_focus_points(path, source)
    ]
