"""Token vectors and fixed-length windowing (reimplements ``get_dl_input.py``).

Two responsibilities live here:

* :func:`vectorize_tokens` maps a token stream to embedding vectors (zero for
  out-of-vocabulary tokens),
* :func:`window_and_pad` applies the original ``cutdata``/``load_data_binary``
  policy: pad short sequences with zero vectors, and for over-long sequences
  keep a ``maxlen`` window centred on the first vulnerable token.
"""

from __future__ import annotations

from gensim.models import Word2Vec


def vectorize_tokens(tokens: list[str], model: Word2Vec, vector_dim: int = 30) -> list[list[float]]:
    wv = model.wv
    vectors: list[list[float]] = []
    for tok in tokens:
        if tok in wv:
            vectors.append([float(x) for x in wv[tok]])
        else:
            vectors.append([0.0] * vector_dim)
    return vectors


def window_and_pad(
    vectors: list[list[float]],
    place: list[int],
    maxlen: int,
    vector_dim: int,
    focus: int | None = None,
) -> tuple[list[list[float]], list[int], int]:
    """Return ``(vectors, place, length)`` with shape exactly ``maxlen``.

    ``place`` is a per-token 0/1 vector marking vulnerable tokens. When the
    input is longer than ``maxlen`` the window is centred on ``focus`` (the
    first vulnerable token index); otherwise the sequence is right-padded with
    zero vectors.
    """
    length = min(len(vectors), maxlen)

    if len(vectors) <= maxlen:
        pad = maxlen - len(vectors)
        out_vec = vectors + [[0.0] * vector_dim for _ in range(pad)]
        out_place = place + [0] * pad
        return out_vec, out_place, length

    center = focus if focus is not None else len(vectors) // 2
    start = int(center - round(maxlen / 2.0))
    if start < 0:
        start = 0
    end = start + maxlen
    if end > len(vectors):
        start = len(vectors) - maxlen
        end = len(vectors)
    return vectors[start:end], place[start:end], maxlen


def build_place(linetokens: list[int], vlinenum: list[int], maxlen: int) -> list[int]:
    """Build the per-token vulnerable-token indicator from line metadata.

    Mirrors ``sample_place_sequence``: vulnerable line ranges are marked with
    ones; if there is no vulnerable line the whole sequence is marked, which
    lets the model learn to suppress activations on benign samples.
    """
    if not vlinenum:
        return [1] * maxlen

    bounds = [x for x in linetokens if x < maxlen] + [maxlen]
    place = [0] * maxlen
    for ptr in vlinenum:
        if ptr >= maxlen:
            continue
        # first bound strictly greater than ptr
        right = maxlen
        for b in bounds:
            if b > ptr:
                right = b
                break
        for i in range(ptr, right):
            place[i] = 1
    return place


def window_bounds(seq_len: int, maxlen: int, focus: int | None = None) -> tuple[int, int]:
    """Return the ``(start, end)`` token window for a fixed-length model input."""
    if seq_len <= maxlen:
        return 0, seq_len
    center = focus if focus is not None else seq_len // 2
    start = int(center - round(maxlen / 2.0))
    if start < 0:
        start = 0
    end = start + maxlen
    if end > seq_len:
        start = seq_len - maxlen
        end = seq_len
    return start, end
