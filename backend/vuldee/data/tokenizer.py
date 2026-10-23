"""Tokenization and symbolic normalization.

This is a clean, dependency-free reimplementation of the original
``get_tokens.py`` / ``mapping.py`` logic:

* :func:`create_tokens` splits one line of (LLVM IR or C) source into tokens
  while keeping multi-character operators and string literals intact.
* :func:`normalize_tokens` performs the "symbolic abstraction" step: user
  defined function names become ``func_N`` and variables become ``variable_N``,
  while keywords, types, numbers and a curated list of dangerous API names are
  preserved verbatim.
"""

from __future__ import annotations

import re

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Multi-character operators are matched longest-first.
_MULTI_OPERATORS = (
    "...", "->", "<<", ">>", "&&", "||", "==", "!=", "<=", ">=",
    "+=", "-=", "*=", "/=", "%=", "|=", "&=", "^=", "++", "--", "::",
)

# C/C++ keywords and primitive type words that must never be renamed.
_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if",
    "inline", "int", "long", "register", "restrict", "return", "short",
    "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while", "bool", "class", "namespace",
    "new", "delete", "public", "private", "protected", "template", "this",
    "try", "catch", "throw", "using", "virtual", "wchar_t", "size_t",
}

# Dangerous / resource-related API names that carry vulnerability signal and
# therefore are preserved instead of being abstracted to ``func_N``.
_PRESERVED_FUNCS = {
    "strcpy", "strncpy", "strcat", "strncat", "sprintf", "snprintf", "vsprintf",
    "vsnprintf", "scanf", "fscanf", "sscanf", "gets", "fgets", "getenv",
    "memcpy", "memmove", "memset", "memcmp", "memchr", "malloc", "calloc",
    "realloc", "free", "kfree", "alloca", "strlen", "strcmp", "strchr",
    "strstr", "strtok", "strerror", "read", "write", "recv", "recvfrom",
    "send", "sendto", "open", "close", "fopen", "fclose", "fread", "fwrite",
    "printf", "fprintf", "syslog", "execve", "system", "popen", "dlopen",
    "mmap", "munmap", "stat", "access", "chmod", "chown", "socket", "bind",
    "connect", "listen", "accept", "main",
}


def create_tokens(line: str) -> list[str]:
    """Split ``line`` into tokens.

    Whitespace is ignored, string/char literals are single tokens, identifiers
    and numbers are maximal runs, and multi-character operators are preferred
    over single-character punctuation.
    """
    tokens: list[str] = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c.isspace():
            i += 1
            continue
        if c in "\"'":
            quote, j = c, i + 1
            while j < n:
                if line[j] == "\\":
                    j += 2
                    continue
                if line[j] == quote:
                    j += 1
                    break
                j += 1
            tokens.append(line[i:j])
            i = j
            continue
        if c.isalnum() or c in "_$":
            j = i
            while j < n and (line[j].isalnum() or line[j] in "_.$"):
                j += 1
            tokens.append(line[i:j])
            i = j
            continue
        for op in _MULTI_OPERATORS:
            if line.startswith(op, i):
                tokens.append(op)
                i += len(op)
                break
        else:
            tokens.append(c)
            i += 1
    return tokens


def _rename(token: str, mapping: dict[str, str], prefix: str) -> str:
    if token not in mapping:
        mapping[token] = f"{prefix}_{len(mapping)}"
    return mapping[token]


def normalize_tokens(tokens: list[str]) -> list[str]:
    """Abstract away user-defined names while preserving signal-bearing words.

    An identifier immediately followed by ``(`` is treated as a function and
    renamed to ``func_N`` (unless preserved); any other identifier becomes
    ``variable_N``. Keywords, types, numbers and literals are kept as-is.
    """
    func_map: dict[str, str] = {}
    var_map: dict[str, str] = {}
    out: list[str] = []
    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]
        if _IDENTIFIER_RE.match(tok) and tok not in _KEYWORDS:
            followed_by_paren = i + 1 < n and tokens[i + 1] == "("
            if followed_by_paren and tok not in _PRESERVED_FUNCS:
                out.append(_rename(tok, func_map, "func"))
            elif not followed_by_paren:
                out.append(_rename(tok, var_map, "variable"))
            else:
                out.append(tok)
            i += 1
            continue
        out.append(tok)
        i += 1
    return out
