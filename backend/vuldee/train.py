"""Model training and evaluation (reimplements the ``main`` training loops).

Kept intentionally framework-light so it is easy to inspect and modify:
dataset assembly, label derivation, the locator training loop and the
detection + localization evaluation are all separate functions.
"""

from __future__ import annotations

import json
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from vuldee.config import Config
from vuldee.data.dataset import VulDataset, collate_fn
from vuldee.data.vectorize import build_place, vectorize_tokens, window_and_pad, window_bounds
from vuldee.models import BGRULocator, confusion_counts, precision_recall_f1


def load_corpus(path: str) -> list[dict]:
    """Load corpus samples from a pkl file (a list of sample dicts)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return [data]
    return list(data)


def compute_label(sample: dict) -> int:
    if not sample.get("vlinenum"):
        return 0
    for fn in sample.get("funcs", []):
        if "good" in fn:
            return 0
    return 1


def vectorize_samples(samples: list[dict], w2v, vector_dim: int) -> list[dict]:
    out: list[dict] = []
    for s in samples:
        vecs = vectorize_tokens(s["tokens"], w2v, vector_dim)
        label = s.get("label", compute_label(s))
        out.append(
            {
                "id": s["id"],
                "vectors": vecs,
                "linetokens": s.get("linetokens", []),
                "vlinenum": s.get("vlinenum", []),
                "label": label,
                "focus_type": s.get("focus_type", "FC"),
            }
        )
    return out


def stratified_split(
    samples: list[dict],
    val_ratio: float = 0.2,
    seed: int = 2018,
) -> tuple[list[dict], list[dict]]:
    """Shuffle and split while keeping the positive ratio equal in both parts."""
    rng = random.Random(seed)
    positives = [s for s in samples if s["label"] == 1]
    negatives = [s for s in samples if s["label"] == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    n_val_pos = int(len(positives) * val_ratio)
    n_val_neg = int(len(negatives) * val_ratio)
    val = positives[:n_val_pos] + negatives[:n_val_neg]
    train = positives[n_val_pos:] + negatives[n_val_neg:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _to_tensors(sample: dict, cfg: Config):
    place = build_place(sample["linetokens"], sample["vlinenum"], maxlen=len(sample["vectors"]))
    vectors, place, length = window_and_pad(
        sample["vectors"],
        place,
        cfg.maxlen,
        cfg.vector_dim,
        focus=sample["vlinenum"][0] if sample["vlinenum"] else None,
    )
    return (
        torch.tensor([vectors], dtype=torch.float32),
        torch.tensor([place], dtype=torch.float32),
        torch.tensor([length], dtype=torch.long),
    )


def train_locator(
    train_samples: list[dict],
    cfg: Config,
    out_dir: str,
    device: torch.device,
) -> BGRULocator:
    """Train the BGRU locator and save its weights."""
    model = BGRULocator(
        vector_dim=cfg.vector_dim,
        hidden_size=cfg.hidden_size,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    dataset = VulDataset(train_samples, cfg.maxlen, cfg.vector_dim)

    labels = [s["label"] for s in train_samples]
    n_pos = int(sum(labels))
    n_neg = len(labels) - n_pos
    sampler = None
    if cfg.class_balance and n_pos > 0 and n_neg > 0:
        weights = [1.0 / n_pos if lbl == 1 else 1.0 / n_neg for lbl in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
    print(
        f"train samples={len(labels)} pos={n_pos} neg={n_neg} "
        f"class_balance={cfg.class_balance}",
        flush=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=collate_fn,
    )

    model.train()
    for epoch in range(cfg.epochs):
        total_loss = 0.0
        for batch in loader:
            vectors = batch["vectors"].to(device)
            place = batch["place"].to(device)
            lengths = batch["length"]
            labels = batch["label"].to(device)
            _probs, score = model(vectors, lengths, place)
            loss = torch.nn.functional.binary_cross_entropy(
                score.clamp(1e-7, 1 - 1e-7), labels
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * vectors.size(0)
        print(
            f"epoch {epoch + 1}/{cfg.epochs} loss={total_loss / len(dataset):.4f}",
            flush=True,
        )

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "locator.pt"))
    return model


def evaluate_locator(
    model: BGRULocator,
    samples: list[dict],
    cfg: Config,
    device: torch.device,
) -> dict:
    """Return detection metrics plus the line-level localization metrics."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    tp_l = fp_l = fn_l = tn_l = 0

    with torch.no_grad():
        for sample in samples:
            vectors, place, lengths = _to_tensors(sample, cfg)
            probs, score = model(vectors.to(device), lengths, place.to(device))
            label = sample["label"]
            pred = 1 if float(score.item()) > cfg.threshold else 0
            y_true.append(label)
            y_pred.append(pred)

            start, end = window_bounds(
                len(sample["vectors"]), cfg.maxlen,
                focus=sample["vlinenum"][0] if sample["vlinenum"] else None,
            )
            shifted_lines = [x - start for x in sample["linetokens"] if start <= x < end]
            shifted_vuln = [x - start for x in sample["vlinenum"] if start <= x < end]
            true_lines = [
                shifted_lines.index(v) for v in shifted_vuln if v in shifted_lines
            ]

            if pred == 0 and label == 0:
                tn_l += 1
            elif pred == 0 and label == 1:
                fn_l += 1
            elif pred == 1 and label == 0:
                fp_l += 1
            elif pred == 1 and label == 1:
                from vuldee.models import locate

                pred_lines = locate(
                    probs[0][: int(lengths.item())], shifted_lines, cfg.k, cfg.threshold
                )
                if set(pred_lines) & set(true_lines):
                    tp_l += 1
                else:
                    fn_l += 1

    tp, fp, fn, tn = confusion_counts(np.array(y_true), np.array(y_pred))
    prf = precision_recall_f1(np.array(y_true), np.array(y_pred))
    return {
        "samples": len(samples),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "TP_l": tp_l, "FP_l": fp_l, "FN_l": fn_l, "TN_l": tn_l,
    }


def save_config(out_dir: str, cfg: Config) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)
