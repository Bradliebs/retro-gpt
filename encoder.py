"""Encoder wrapper for the memory service.

Loads MiniLM once at process start and serves embeddings from memory. Lazy-loaded
so that import-time is fast (sentence-transformers loads in ~1-3s).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class EncoderSingleton:
    """One MiniLM instance per process."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None  # lazy-loaded on first encode

    @property
    def dim(self) -> int:
        # Hardcoded for MiniLM-L6-v2. We could query the model but lazy-loading
        # means we'd force a load just to know the dim, which is silly.
        if self.model_name == "all-MiniLM-L6-v2":
            return 384
        # Lazy fallback: load and query
        self._ensure_loaded()
        return self._model.get_sentence_embedding_dimension()

    def _ensure_loaded(self) -> None:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(self.model_name, device=dev)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a batch of texts to (N, D) float32."""
        self._ensure_loaded()
        emb = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we whiten ourselves; raw output preferred
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """Convenience: encode a single text, return (D,) float32."""
        return self.encode([text])[0]
