"""Inference entrypoint: scan a C/C++ file and report vulnerable lines."""

from __future__ import annotations

import json
import os
from collections import defaultdict

import torch

from vuldee.config import Config
from vuldee.data.corpus import build_sample_from_lines
from vuldee.data.vectorize import vectorize_tokens, window_bounds
from vuldee.models import BGRULocator, locate
from vuldee.source2slice import FocusType, extract_focus_points
from vuldee.source2slice.ir import compile_to_ir, focus_slice, parse_debug_map


def load_artifacts(model_dir: str, cfg: Config, device: torch.device):
    from gensim.models import Word2Vec

    model = BGRULocator(
        vector_dim=cfg.vector_dim,
        hidden_size=cfg.hidden_size,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "locator.pt"), map_location=device))
    model.eval()
    w2v = Word2Vec.load(os.path.join(model_dir, "word2vec.model"))
    return model, w2v


def _run_sample(model, w2v, cfg, device, sample):
    vectors = vectorize_tokens(sample.tokens, w2v, cfg.vector_dim)
    if not vectors:
        return None
    start, end = window_bounds(len(vectors), cfg.maxlen)
    windowed = vectors[start:end]
    length = len(windowed)
    pad = cfg.maxlen - length
    vec_tensor = torch.tensor(
        [windowed + [[0.0] * cfg.vector_dim for _ in range(pad)]],
        dtype=torch.float32,
    ).to(device)
    lengths = torch.tensor([length], dtype=torch.long)
    place = torch.ones((1, cfg.maxlen), dtype=torch.float32).to(device)
    with torch.no_grad():
        probs, _score = model(vec_tensor, lengths, place)
    shifted_lines = [x - start for x in sample.linetokens if start <= x < end]
    return probs[0][:length], shifted_lines


def _scan_ir(path, source, model, w2v, cfg, device):
    ir_lines = compile_to_ir(path)
    if ir_lines is None:
        return None
    line_to_source = parse_debug_map(ir_lines)
    focuses = extract_focus_points(path, source)
    findings: dict[int, dict] = {}
    for fp in focuses:
        result = focus_slice(ir_lines, line_to_source, fp.line)
        if result is None:
            continue
        slice_lines, indices = result
        sample = build_sample_from_lines(
            slice_lines, f"{path}:{fp.line}", funcs=[fp.name], focus_type=fp.type.value
        )
        scored = _run_sample(model, w2v, cfg, device, sample)
        if scored is None:
            continue
        probs, shifted_lines = scored
        for line_idx in locate(probs, shifted_lines, cfg.k, cfg.threshold):
            source_line = line_to_source.get(indices[line_idx], fp.line)
            if source_line not in findings or findings[source_line]["score"] < probs[shifted_lines[line_idx]].item():
                findings[source_line] = {
                    "line": source_line,
                    "type": fp.type.value,
                    "name": fp.name,
                    "score": round(float(probs[shifted_lines[line_idx]].item()), 4),
                }
    return sorted(findings.values(), key=lambda x: x["line"])


def _scan_source(path, source, model, w2v, cfg, device):
    """Fallback: run once over the whole file and attribute by focus line."""
    focus_by_line: dict[int, list] = defaultdict(list)
    for fp in extract_focus_points(path, source):
        focus_by_line[fp.line].append(fp)

    sample = build_sample_from_lines(source.splitlines(), sample_id=path)
    scored = _run_sample(model, w2v, cfg, device, sample)
    if scored is None:
        return []
    probs, shifted_lines = scored
    flagged = locate(probs, shifted_lines, cfg.k, cfg.threshold)

    included = [tok for tok in shifted_lines]
    # reconstruct original line numbers: first included line index + line offset
    findings: list[dict] = []
    for line_idx in flagged:
        source_line = line_idx + 1  # whole-file sample starts at line 1
        focus_points = focus_by_line.get(source_line, [])
        order = {FocusType.FC: 0, FocusType.AD: 1, FocusType.PD: 2, FocusType.AE: 3}
        chosen = min(focus_points, key=lambda p: order.get(p.type, 9)) if focus_points else None
        findings.append(
            {
                "line": source_line,
                "type": chosen.type.value if chosen else "unknown",
                "name": chosen.name if chosen else "",
                "score": round(float(probs[included[line_idx]].item()), 4),
            }
        )
    return findings


def scan_file(
    path: str,
    model: BGRULocator,
    w2v,
    cfg: Config,
    device: torch.device,
) -> list[dict]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        source = f.read()

    ir_findings = _scan_ir(path, source, model, w2v, cfg, device)
    if ir_findings is not None:
        return ir_findings
    return _scan_source(path, source, model, w2v, cfg, device)


def scan_to_json(path: str, model_dir: str, cfg: Config, device: torch.device) -> str:
    model, w2v = load_artifacts(model_dir, cfg, device)
    findings = scan_file(path, model, w2v, cfg, device)
    return json.dumps({"file": os.path.abspath(path), "findings": findings})
