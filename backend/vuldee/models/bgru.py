"""Bidirectional GRU detector (reimplements ``bgru.py``).

Detection-only model: variable-length input is masked via packed sequences,
the final hidden state of a stacked BiGRU is passed through a dense sigmoid.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class BGRU(nn.Module):
    def __init__(
        self,
        vector_dim: int = 30,
        hidden_size: int = 512,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=vector_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return per-sample vulnerability probabilities in ``[0, 1]``."""
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        # hidden: (num_layers * 2, batch, hidden_size)
        hidden = hidden.view(self.n_layers, 2, x.size(0), self.hidden_size)[-1]
        last = torch.cat([hidden[0], hidden[1]], dim=-1)
        return torch.sigmoid(self.classifier(self.dropout(last))).squeeze(-1)
