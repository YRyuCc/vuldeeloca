"""BGRU detector + localizer (reimplements ``bgru_threshold.py``).

The token-level head emits a per-token vulnerability probability; the scalar
training objective multiplies it by the vulnerable-token mask and keeps the
k-max value (equivalent to the Keras ``Multiply`` + ``KMaxPooling(k=1)`` +
``GlobalAveragePooling1D`` stack). :func:`locate` then converts a token-level
probability sequence back into vulnerable line numbers using the original
k-max-per-line threshold rule.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BGRULocator(nn.Module):
    def __init__(
        self,
        vector_dim: int = 30,
        hidden_size: int = 512,
        n_layers: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.rnns = nn.ModuleList()
        in_dim = vector_dim
        for _ in range(n_layers):
            self.rnns.append(
                nn.GRU(in_dim, hidden_size, batch_first=True, bidirectional=True)
            )
            in_dim = hidden_size * 2
        self.token_head = nn.Linear(hidden_size * 2, 1)

    def _run_gru(self, gru: nn.GRU, x: torch.Tensor, lengths: torch.Tensor, total: int):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=total)
        return out

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        place: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(token_probs, scalar_score)``.

        ``token_probs`` has shape ``(batch, maxlen)``; ``scalar_score`` is the
        k-max of ``place * token_probs`` per sample.
        """
        maxlen = x.size(1)
        h = x
        for gru in self.rnns:
            h = self._run_gru(gru, h, lengths, maxlen)
            h = self.dropout(h)
        logits = self.token_head(h).squeeze(-1)      # (batch, maxlen)
        probs = torch.sigmoid(logits)
        masked = place * probs
        score, _ = masked.max(dim=1)
        return probs, score


def locate(
    token_probs: torch.Tensor | np.ndarray,
    linetokens: list[int],
    k: int = 1,
    threshold: float = 0.5,
) -> list[int]:
    """Convert token probabilities into 0-based vulnerable line indices.

    Reimplements ``sample_threshold_windows``: for each source line the top-k
    token probabilities are averaged; a line whose average exceeds
    ``threshold`` is reported.
    """
    if isinstance(token_probs, torch.Tensor):
        probs = token_probs.detach().cpu().numpy().reshape(-1)
    else:
        probs = np.asarray(token_probs, dtype=float).reshape(-1)

    n = len(probs)
    bounds = [b for b in linetokens if b < n]
    if not bounds:
        return []
    if bounds[-1] < n:
        bounds = bounds + [n]

    result: list[int] = []
    for i in range(len(bounds) - 1):
        left, right = bounds[i], bounds[i + 1]
        if left >= right:
            continue
        window = probs[left:right]
        topk = np.sort(window)[::-1][:k]
        if float(topk.mean()) > threshold:
            result.append(i)
    return result
