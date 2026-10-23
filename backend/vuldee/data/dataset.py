"""PyTorch dataset / collation for the BGRU models.

Reimplements ``generator_of_data`` plus ``sample_place_sequence`` from the
original code, but as an idiomatic modern :class:`torch.utils.data.Dataset`.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from vuldee.data.vectorize import build_place, window_and_pad


class VulDataset(Dataset):
    """Dataset of fixed-length vectorized candidate slices."""

    def __init__(
        self,
        samples: list[dict],
        maxlen: int,
        vector_dim: int,
    ):
        self.maxlen = maxlen
        self.vector_dim = vector_dim
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        place = build_place(
            s.get("linetokens", []),
            s.get("vlinenum", []),
            maxlen=len(s["vectors"]),
        )
        vectors, place, length = window_and_pad(
            s["vectors"],
            place,
            self.maxlen,
            self.vector_dim,
            focus=s["vlinenum"][0] if s.get("vlinenum") else None,
        )
        return {
            "vectors": torch.tensor(vectors, dtype=torch.float32),
            "place": torch.tensor(place, dtype=torch.float32),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(s["label"], dtype=torch.float32),
            "id": s.get("id", ""),
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "vectors": torch.stack([b["vectors"] for b in batch]),
        "place": torch.stack([b["place"] for b in batch]),
        "length": torch.stack([b["length"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "id": [b["id"] for b in batch],
    }
