"""Command-line interface: ``vuldee preprocess|train|scan|demo``."""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import torch

from vuldee.config import Config


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cmd_demo(args) -> int:
    from vuldee.demo import run_demo

    cfg = Config(epochs=args.epochs, maxlen=args.maxlen, vector_dim=args.vector_dim)
    run_demo(args.out_dir, cfg, _device())
    return 0


def cmd_train(args) -> int:
    from gensim.models import Word2Vec
    from vuldee.train import (
        evaluate_locator,
        load_corpus,
        save_config,
        stratified_split,
        train_locator,
        vectorize_samples,
    )

    cfg = Config(
        epochs=args.epochs,
        maxlen=args.maxlen,
        vector_dim=args.vector_dim,
        class_balance=not args.no_class_balance,
    )
    samples = load_corpus(args.corpus)
    w2v = Word2Vec.load(args.w2v)
    vec_samples = vectorize_samples(samples, w2v, cfg.vector_dim)
    train_set, val_set = stratified_split(vec_samples, seed=cfg.seed)
    model = train_locator(train_set, cfg, args.out_dir, _device())
    save_config(args.out_dir, cfg)
    os.makedirs(args.out_dir, exist_ok=True)
    for w2v_file in glob.glob(args.w2v + "*"):
        suffix = w2v_file[len(args.w2v):]
        shutil.copy2(w2v_file, os.path.join(args.out_dir, "word2vec.model" + suffix))
    metrics = evaluate_locator(model, val_set, cfg, _device())
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_scan(args) -> int:
    from vuldee.infer import scan_to_json

    cfg_path = os.path.join(args.model_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = Config(**json.load(f))
    else:
        cfg = Config(maxlen=args.maxlen, vector_dim=args.vector_dim)
    print(scan_to_json(args.file, args.model_dir, cfg, _device()))
    return 0


def cmd_w2v(args) -> int:
    from vuldee.data.word2vec import train_word2vec
    from vuldee.train import load_corpus

    samples = load_corpus(args.corpus)
    train_word2vec(
        (s["tokens"] for s in samples),
        args.out,
        vector_size=args.vector_dim,
        epochs=args.epochs,
    )
    print(f"saved word2vec to {args.out}")
    return 0


def cmd_ingest_sard(args) -> int:
    from vuldee.ingest.sard import ingest_sard

    ingest_sard(
        args.programs_root,
        args.xml,
        args.out,
        limit=args.limit,
        max_per_file=args.max_per_file,
        testcase_lib=args.testcase_lib,
    )
    return 0


def cmd_ingest_nvd(args) -> int:
    from vuldee.ingest.nvd import ingest_nvd

    ingest_nvd(args.programs_root, args.out, limit=args.limit, max_per_file=args.max_per_file)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vuldee")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="generate a synthetic dataset and train a demo model")
    demo.add_argument("--out-dir", required=True)
    demo.add_argument("--epochs", type=int, default=3)
    demo.add_argument("--maxlen", type=int, default=900)
    demo.add_argument("--vector-dim", type=int, default=30)
    demo.set_defaults(func=cmd_demo)

    train = sub.add_parser("train", help="train the locator on a prepared corpus")
    train.add_argument("--corpus", required=True, help="pkl file with corpus samples")
    train.add_argument("--w2v", required=True, help="trained word2vec model")
    train.add_argument("--out-dir", required=True)
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--maxlen", type=int, default=900)
    train.add_argument("--vector-dim", type=int, default=30)
    train.add_argument("--no-class-balance", action="store_true", help="disable class-balanced sampling")
    train.set_defaults(func=cmd_train)

    scan = sub.add_parser("scan", help="scan a C/C++ file for vulnerabilities")
    scan.add_argument("--file", required=True)
    scan.add_argument("--model-dir", required=True, help="dir containing locator.pt + word2vec.model")
    scan.add_argument("--maxlen", type=int, default=900)
    scan.add_argument("--vector-dim", type=int, default=30)
    scan.set_defaults(func=cmd_scan)

    w2v = sub.add_parser("w2v", help="train word2vec on a corpus pkl")
    w2v.add_argument("--corpus", required=True)
    w2v.add_argument("--out", required=True)
    w2v.add_argument("--vector-dim", type=int, default=30)
    w2v.add_argument("--epochs", type=int, default=5)
    w2v.set_defaults(func=cmd_w2v)

    sard = sub.add_parser("ingest-sard", help="build a corpus from real SARD data with clang slicing")
    sard.add_argument("--programs-root", required=True)
    sard.add_argument("--xml", required=True)
    sard.add_argument("--out", required=True)
    sard.add_argument("--limit", type=int, default=300)
    sard.add_argument("--max-per-file", type=int, default=20)
    sard.add_argument("--testcase-lib", default=None)
    sard.set_defaults(func=cmd_ingest_sard)

    nvd = sub.add_parser("ingest-nvd", help="build a corpus from real NVD programs with clang slicing")
    nvd.add_argument("--programs-root", required=True)
    nvd.add_argument("--out", required=True)
    nvd.add_argument("--limit", type=int, default=200)
    nvd.add_argument("--max-per-file", type=int, default=20)
    nvd.set_defaults(func=cmd_ingest_nvd)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
