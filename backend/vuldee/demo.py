"""Synthetic dataset generator + quick end-to-end training demo.

This keeps the project runnable without the original NVD/SARD corpora: it
generates small vulnerable and benign C snippets, then drives them through the
same tokenizer -> word2vec -> vectorize -> dataset -> model pipeline.
"""

from __future__ import annotations

import os
import pickle

from vuldee.config import Config
from vuldee.data.corpus import build_sample_from_lines
from vuldee.data.word2vec import train_word2vec
from vuldee.train import (
    evaluate_locator,
    save_config,
    stratified_split,
    train_locator,
    vectorize_samples,
)


def _vulnerable_fc(i: int) -> dict:
    lines = [
        "void process(char *src) {",
        "    char buf[16];",
        "    strcpy(buf, src);",
        "    printf(\"%s\\n\", buf);",
        "}",
    ]
    return build_sample_from_lines(
        lines, f"vuln_fc_{i}", vuln_line_numbers=[3], funcs=["process", "strcpy"],
        focus_type="FC",
    ).to_dict()


def _safe_fc(i: int) -> dict:
    lines = [
        "void process(char *src) {",
        "    char buf[16];",
        "    strncpy(buf, src, sizeof(buf) - 1);",
        "    buf[15] = '\\0';",
        "}",
    ]
    return build_sample_from_lines(
        lines, f"safe_fc_{i}", vuln_line_numbers=[], funcs=["process"],
        focus_type="FC",
    ).to_dict()


def _vulnerable_ad(i: int) -> dict:
    lines = [
        "void fill(int idx) {",
        "    int arr[8];",
        "    arr[idx] = 1;",
        "}"
    ]
    return build_sample_from_lines(
        lines, f"vuln_ad_{i}", vuln_line_numbers=[3], funcs=["fill"],
        focus_type="AD",
    ).to_dict()


def _safe_ad(i: int) -> dict:
    lines = [
        "void fill(int idx) {",
        "    int arr[8];",
        "    if (idx < 8) arr[idx] = 1;",
        "}"
    ]
    return build_sample_from_lines(
        lines, f"safe_ad_{i}", vuln_line_numbers=[], funcs=["fill"],
        focus_type="AD",
    ).to_dict()


def _vulnerable_pd(i: int) -> dict:
    lines = [
        "void deref(int *p) {",
        "    int x = *p;",
        "    printf(\"%d\", x);",
        "}"
    ]
    return build_sample_from_lines(
        lines, f"vuln_pd_{i}", vuln_line_numbers=[2], funcs=["deref"],
        focus_type="PD",
    ).to_dict()


def _safe_pd(i: int) -> dict:
    lines = [
        "void deref(int *p) {",
        "    if (p != 0) {",
        "        int x = *p;",
        "    }",
        "}"
    ]
    return build_sample_from_lines(
        lines, f"safe_pd_{i}", vuln_line_numbers=[], funcs=["deref"],
        focus_type="PD",
    ).to_dict()


def generate_samples(n_per_class: int = 40) -> list[dict]:
    samples: list[dict] = []
    for i in range(n_per_class):
        samples.append(_vulnerable_fc(i))
        samples.append(_safe_fc(i))
        samples.append(_vulnerable_ad(i))
        samples.append(_safe_ad(i))
        samples.append(_vulnerable_pd(i))
        samples.append(_safe_pd(i))
    return samples


def run_demo(out_dir: str, cfg: Config, device) -> None:
    os.makedirs(out_dir, exist_ok=True)
    samples = generate_samples()
    corpus_path = os.path.join(out_dir, "demo_corpus.pkl")
    with open(corpus_path, "wb") as f:
        pickle.dump(samples, f)

    w2v_path = os.path.join(out_dir, "word2vec.model")
    w2v = train_word2vec(
        (s["tokens"] for s in samples), w2v_path, vector_size=cfg.vector_dim, epochs=5
    )
    vec_samples = vectorize_samples(samples, w2v, cfg.vector_dim)

    train_samples, val_samples = stratified_split(vec_samples, seed=cfg.seed)

    model = train_locator(train_samples, cfg, out_dir, device)
    save_config(out_dir, cfg)
    metrics = evaluate_locator(model, val_samples, cfg, device)
    print(metrics)
    return metrics
