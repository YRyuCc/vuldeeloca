"""Corpus construction from program slices.

Reimplements the core loop of the original ``process_dataflow.py``: a slice is
read line by line, each line is tokenized and normalised, and we record

* the flat token stream (``tokens``),
* the token index where every line starts (``linetokens``),
* the token indices of vulnerable lines (``vlinenum``),
* the function names mentioned by the slice (``funcs``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vuldee.data.tokenizer import create_tokens, normalize_tokens


@dataclass
class CorpusSample:
    id: str
    tokens: list[str] = field(default_factory=list)
    linetokens: list[int] = field(default_factory=list)
    vlinenum: list[int] = field(default_factory=list)
    funcs: list[str] = field(default_factory=list)
    focus_type: str = "FC"
    source: str = ""
    start_line: int = 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tokens": self.tokens,
            "linetokens": self.linetokens,
            "vlinenum": self.vlinenum,
            "funcs": self.funcs,
            "focus_type": self.focus_type,
            "source": self.source,
            "start_line": self.start_line,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CorpusSample":
        return cls(
            id=d["id"],
            tokens=d["tokens"],
            linetokens=d["linetokens"],
            vlinenum=d["vlinenum"],
            funcs=d.get("funcs", []),
            focus_type=d.get("focus_type", "FC"),
            source=d.get("source", ""),
            start_line=d.get("start_line", 1),
        )


def build_sample_from_lines(
    lines: list[str],
    sample_id: str,
    vuln_line_numbers: list[int] | None = None,
    funcs: list[str] | None = None,
    focus_type: str = "FC",
    source: str = "",
    start_line: int = 1,
    normalize: bool = True,
) -> CorpusSample:
    """Build a :class:`CorpusSample` from already-tokenizable slice lines.

    ``vuln_line_numbers`` is a list of 1-based line numbers (relative to the
    start of ``lines``) that are vulnerable.
    """
    vuln = set(vuln_line_numbers or [])
    flat: list[str] = []
    linetokens: list[int] = []
    vlinenum: list[int] = []
    token_index = 0

    for offset, line in enumerate(lines, start=1):
        toks = create_tokens(line)
        if normalize:
            toks = normalize_tokens(toks)
        linetokens.append(token_index)
        if offset in vuln:
            vlinenum.append(token_index)
        flat.extend(toks)
        token_index += len(toks)

    return CorpusSample(
        id=sample_id,
        tokens=flat,
        linetokens=linetokens,
        vlinenum=vlinenum,
        funcs=list(funcs or []),
        focus_type=focus_type,
        source=source,
        start_line=start_line,
    )
