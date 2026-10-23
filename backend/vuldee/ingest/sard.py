"""SARD ingestion: XML flaw metadata + clang semantic slicing.

Reimplements the original ``getVulLineForCounting.py`` + ``multiFileCompile.py``
step, producing corpus samples with vulnerability line labels from the real
SARD ``synthetic and academic programs`` archive.
"""

from __future__ import annotations

import os
import pickle
import sys
import xml.etree.ElementTree as ET

from vuldee.source2slice.focus import extract_focus_points
from vuldee.source2slice.ir import build_slice_sample, compile_to_ir, parse_debug_map


def _iter_testcases(xml_path: str):
    for _event, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag == "testcase":
            yield elem
            elem.clear()


def _parse_files(elem: ET.Element) -> tuple[str, list[dict]]:
    tid = elem.get("id", "")
    files: list[dict] = []
    for fe in elem.findall("file"):
        path = fe.get("path", "")
        basename = path.rsplit("/", 1)[-1]
        lines: list[int] = []
        for child in fe.iter():
            if child.tag in ("flaw", "mixed"):
                ln = int(child.get("line", "0") or "0")
                if ln > 0:
                    lines.append(ln)
        files.append({"basename": basename, "path": path, "flaws": sorted(set(lines))})
    return tid, files


def ingest_sard(
    programs_root: str,
    xml_path: str,
    out_path: str,
    limit: int = 300,
    max_per_file: int = 20,
    testcase_lib: str | None = None,
) -> dict:
    """Walk SARD testcases, slice each flawed file with clang, save corpus."""
    samples: list[dict] = []
    compiled = skipped = attempted = 0
    available_dirs = set(os.listdir(programs_root))

    for elem in _iter_testcases(xml_path):
        tid, files = _parse_files(elem)
        if not tid.isdigit():
            continue
        dirname = f"{int(tid):09d}"
        if dirname not in available_dirs:
            continue
        prog_dir = os.path.join(programs_root, dirname)

        for fi in files:
            if not fi["flaws"]:
                continue
            src_path = os.path.join(prog_dir, fi["basename"])
            if not os.path.isfile(src_path):
                continue
            attempted += 1
            ir_lines = compile_to_ir(src_path, [testcase_lib] if testcase_lib else None)
            if ir_lines is None:
                skipped += 1
                print(f"\r[{attempted}] skip {src_path}", end="", flush=True)
                continue
            compiled += 1
            print(f"\r[{attempted}] compiled {compiled} files, {len(samples)} samples", end="", flush=True)
            line_to_source = parse_debug_map(ir_lines)
            flaw_set = set(fi["flaws"])
            with open(src_path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
            made = 0
            for fp in extract_focus_points(src_path, source):
                sample = build_slice_sample(
                    ir_lines,
                    line_to_source,
                    fp.line,
                    flaw_set,
                    f"{tid}/{fi['basename']}:{fp.line}:{fp.type.value}",
                    fp.type.value,
                    fp.name,
                    src_path,
                )
                if sample is None:
                    continue
                samples.append(sample.to_dict())
                made += 1
                if made >= max_per_file:
                    break

        if compiled >= limit:
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
