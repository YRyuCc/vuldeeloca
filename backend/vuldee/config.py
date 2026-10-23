"""Central configuration shared by every module.

The default values mirror the original VulDeeLocator hyper-parameters so the
reimplementation is behaviourally comparable while staying configurable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    vector_dim: int = 30        # word2vec embedding size
    maxlen: int = 900           # max tokens per candidate slice
    batch_size: int = 64        # training minibatch size
    hidden_size: int = 512      # GRU hidden units per direction
    n_layers: int = 2           # number of (bidirectional) GRU layers
    dropout: float = 0.4        # dropout rate between recurrent layers
    k: int = 1                  # k-max value used by the line localizer
    threshold: float = 0.5      # line activation threshold
    epochs: int = 4             # training epochs
    learning_rate: float = 1e-3
    class_balance: bool = True  # balance positive/negative samples per batch
    seed: int = 2018


DEFAULT = Config()
