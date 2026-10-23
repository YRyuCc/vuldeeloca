"""Extraction of the four source/syntax vulnerability candidates (sSyVC).

The original project used hand-written libclang visitors (``ngetapi``,
``ngetarr``, ``ngetpoint``, ``ngetval``/``ngetbds``). This module reimplements
the same four focus types on top of the modern `tree-sitter` parser, which is a
pure-Python install and cross-platform:

* FC - library / API function calls,
* AD - array definitions,
* PD - pointer definitions,
* AE - arithmetic expressions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tree_sitter import Language, Node, Parser
import tree_sitter_c
import tree_sitter_cpp


class FocusType(str, Enum):
    FC = "FC"
    AD = "AD"
    PD = "PD"
    AE = "AE"


@dataclass
class FocusPoint:
    type: FocusType
    name: str
    line: int          # 1-based line number
    column: int        # 0-based start column
    end_column: int    # 0-based end column (exclusive)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "name": self.name,
            "line": self.line,
            "column": self.column,
            "end_column": self.end_column,
        }


_C_LANG = Language(tree_sitter_c.language())
_CPP_LANG = Language(tree_sitter_cpp.language())
_PARSERS = {
    ".c": Parser(_C_LANG),
    ".h": Parser(_C_LANG),
    ".cc": Parser(_CPP_LANG),
    ".cpp": Parser(_CPP_LANG),
    ".cxx": Parser(_CPP_LANG),
    ".hpp": Parser(_CPP_LANG),
    ".hh": Parser(_CPP_LANG),
}

_ARITHMETIC_OPERATORS = {
    "+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^",
    "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "|=", "^=",
}


def _parser_for(path: str) -> Parser | None:
    for suffix, parser in _PARSERS.items():
        if path.endswith(suffix):
            return parser
    return None


def _name_of(node: Node, source: bytes) -> str:
    for child in node.named_children:
        if child.type in {"identifier", "field_identifier", "type_identifier"}:
            return source[child.start_byte:child.end_byte].decode("utf-8", "replace")
    return source[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _walk(node: Node):
    yield node
    for child in node.children:
        yield from _walk(child)


def extract_focus_points(path: str, source: str) -> list[FocusPoint]:
    """Extract the four focus types from C/C++ source text."""
    parser = _parser_for(path)
    if parser is None:
        raise ValueError(f"unsupported source extension for focus extraction: {path}")

    raw = source.encode("utf-8")
    tree = parser.parse(raw)
    points: list[FocusPoint] = []

    for node in _walk(tree.root_node):
        if node.type == "call_expression":
            line = node.start_point[0] + 1
            points.append(
                FocusPoint(
                    type=FocusType.FC,
                    name=_name_of(node, raw),
                    line=line,
                    column=node.start_point[1],
                    end_column=node.end_point[1],
                )
            )
        elif node.type == "array_declarator":
            line = node.start_point[0] + 1
            points.append(
                FocusPoint(
                    type=FocusType.AD,
                    name=_name_of(node, raw),
                    line=line,
                    column=node.start_point[1],
                    end_column=node.end_point[1],
                )
            )
        elif node.type in {"pointer_declarator", "pointer_expression"}:
            line = node.start_point[0] + 1
            points.append(
                FocusPoint(
                    type=FocusType.PD,
                    name=_name_of(node, raw),
                    line=line,
                    column=node.start_point[1],
                    end_column=node.end_point[1],
                )
            )
        elif node.type in {"binary_expression", "assignment_expression"}:
            op = None
            for child in node.children:
                if child.type.endswith("_operator") or child.type in {
                    "operator", "+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^",
                }:
                    op = raw[child.start_byte:child.end_byte].decode("utf-8", "replace")
                    break
            if op in _ARITHMETIC_OPERATORS:
                line = node.start_point[0] + 1
                points.append(
                    FocusPoint(
                        type=FocusType.AE,
                        name=op,
                        line=line,
                        column=node.start_point[1],
                        end_column=node.end_point[1],
                    )
                )

    return points
