"""Real-corpus ingestion (SARD and NVD) into the training corpus format."""

from vuldee.ingest.sard import ingest_sard
from vuldee.ingest.nvd import ingest_nvd

__all__ = ["ingest_sard", "ingest_nvd"]
