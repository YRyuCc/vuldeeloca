"""NVD ingestion from the extracted ``real-world programs`` pairs.

The archive stores each CVE as ``CVE-YYYY-NNNN_vul/<file>.c`` and
``CVE-YYYY-NNNN_patch/<file>.c``. The vulnerable lines are recovered by
diffing the vulnerable file against its patched counterpart, then the same
clang semantic-slicing path is applied.
"""

from __future__ import annotations

import difflib
import os
import pickle
import sys

from vuldee.data.corpus import build_sample_from_lines
from vuldee.source2slice.focus import extract_focus_points
from vuldee.source2slice.ir import build_slice_sample, compile_to_ir, parse_debug_map


def _changed_lines(vul_lines: list[str], patch_lines: list[str]) -> list[int]:
    sm = difflib.SequenceMatcher(None, vul_lines, patch_lines, autojunk=False)
    changed: list[int] = []
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            changed.extend(range(i1 + 1, i2 + 1))
    return changed


def _source_fallback(
    vul_path: str,
    source: str,
    focus,
    flaw_set: set[int],
    cve: str,
    basename: str,
) -> dict | None:
    """Build a source-window candidate when clang cannot compile the file."""
    lines = source.splitlines()
    start = max(0, focus.line - 1 - 8)
    end = min(len(lines), focus.line + 8)
    vuln_positions = [focus.line - start] if focus.line in flaw_set else []
    return build_sample_from_lines(
        lines[start:end],
        f"{cve}/{basename}:{focus.line}:{focus.type.value}",
        vuln_line_numbers=vuln_positions,
        funcs=[focus.name],
        focus_type=focus.type.value,
        source=vul_path,
        start_line=start + 1,
    ).to_dict()


def ingest_nvd(
    programs_root: str,
    out_path: str,
    limit: int = 200,
    max_per_file: int = 20,
) -> dict:
    """Process vulnerable/patched file pairs, slice, and save the corpus."""
    samples: list[dict] = []
    compiled = skipped = attempted = 0
    cves: list[str] = []

    for name in sorted(os.listdir(programs_root)):
        if name.endswith("_vul"):
            cves.append(name[:-4])

    for cve in cves:
        vul_dir = os.path.join(programs_root, cve + "_vul")
        patch_dir = os.path.join(programs_root, cve + "_patch")
        if not os.path.isdir(vul_dir) or not os.path.isdir(patch_dir):
            continue
        for basename in os.listdir(vul_dir):
            if not basename.endswith((".c", ".cpp", ".cc")):
                continue
            vul_path = os.path.join(vul_dir, basename)
            patch_path = os.path.join(patch_dir, basename)
            if not os.path.isfile(patch_path):
                continue
            with open(vul_path, "r", encoding="utf-8", errors="replace") as f:
                vul_source = f.read()
            with open(patch_path, "r", encoding="utf-8", errors="replace") as f:
                patch_source = f.read()
            flaw_set = set(_changed_lines(vul_source.splitlines(), patch_source.splitlines()))
            if not flaw_set:
                continue
            attempted += 1
            ir_lines = compile_to_ir(vul_path)
            use_ir = ir_lines is not None
            if use_ir:
                compiled += 1
                line_to_source = parse_debug_map(ir_lines)
            else:
                skipped += 1
            print(
                f"\r[{attempted}] {'compiled' if use_ir else 'source-fallback'} {len(samples)} samples",
                end="", flush=True,
            )
            made = 0
            for fp in extract_focus_points(vul_path, vul_source):
                if use_ir:
                    sample = build_slice_sample(
                        ir_lines,
                        line_to_source,
                        fp.line,
                        flaw_set,
                        f"{cve}/{basename}:{fp.line}:{fp.type.value}",
                        fp.type.value,
                        fp.name,
                        vul_path,
                    )
                    sample = sample.to_dict() if sample else None
                else:
                    sample = _source_fallback(
                        vul_path, vul_source, fp, flaw_set, cve, basename
                    )
                if sample is None:
                    continue
                samples.append(sample)
                made += 1
                if made >= max_per_file:
                    break
        if attempted >= limit:
            break

    print("", file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(samples, f)
    stats = {
        "samples": len(samples),
        "compiled_files": compiled,
        "skipped_files": skipped,
        "attempted_files": attempted,
    }
    print(stats)
    return stats
