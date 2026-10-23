"""VulDee: a modern, modular reimplementation of the VulDeeLocator detector.

The package is split into three independent layers:

* :mod:`vuldee.source2slice` - focus-point extraction and program slicing,
* :mod:`vuldee.data` - tokenization, corpus building, word2vec and datasets,
* :mod:`vuldee.models` - BGRU detection/localization models and metrics.

The command-line entrypoint lives in :mod:`vuldee.cli`.
"""

__version__ = "0.1.0"
