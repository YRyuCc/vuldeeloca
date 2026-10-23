# vuldee backend

The Python engine for the VulDee VSCode extension. It is a modular,
dependency-light reimplementation of the VulDeeLocator pipeline:

- `vuldee.source2slice` - focus extraction + program slicing,
- `vuldee.data` - tokenization, corpus, word2vec, vectorization, datasets,
- `vuldee.models` - BGRU detection/localization models and metrics,
- `vuldee.train` / `vuldee.infer` / `vuldee.cli` - training, inference, CLI.

Install with `pip install -e .` inside a virtual/conda environment.
