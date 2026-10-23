"""Word2Vec training with gensim (reimplements ``create_word2vecmodel.py``).

Uses the current gensim 4.x API (``vector_size``/``epochs`` instead of the
removed ``size``/``iter``).
"""

from __future__ import annotations

from collections.abc import Iterable
import os

from gensim.models import Word2Vec


def iter_corpus(corpus_dir: str) -> Iterable[list[str]]:
    """Yield token lists from every corpus pickle in ``corpus_dir`` (recursive)."""
    import pickle

    for root, _dirs, files in os.walk(corpus_dir):
        for fn in files:
            if not fn.endswith(".pkl"):
                continue
            path = os.path.join(root, fn)
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, list) and data and isinstance(data[0], list):
                for sample in data:
                    yield sample


def train_word2vec(
    corpus: Iterable[list[str]],
    model_path: str,
    vector_size: int = 30,
    window: int = 5,
    sg: int = 1,
    hs: int = 0,
    negative: int = 7,
    min_count: int = 1,
    epochs: int = 3,
    workers: int = 1,
) -> Word2Vec:
    """Train and persist a Word2Vec model on token streams."""
    model = Word2Vec(
        sentences=list(corpus),
        vector_size=vector_size,
        window=window,
        sg=sg,
        hs=hs,
        negative=negative,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
    )
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    model.save(model_path)
    return model
