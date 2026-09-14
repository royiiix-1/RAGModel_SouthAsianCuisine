from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class Embedder(Protocol):
    model_id: str
    revision: str

    def encode_documents(self, texts: Sequence[str]) -> Any: ...

    def encode_query(self, text: str) -> Any: ...


class SentenceTransformerEmbedder:
    """Lazy, revision-pinned sentence-transformers adapter."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        device: str = "auto",
        batch_size: int = 64,
        local_files_only: bool = False,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed; install the project dependencies"
                ) from exc
            kwargs: dict[str, Any] = {
                "revision": self.revision,
                "trust_remote_code": False,
                "local_files_only": self.local_files_only,
            }
            if self.device != "auto":
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_id, **kwargs)
        return self._model

    def encode_documents(self, texts: Sequence[str]) -> Any:
        model = self._load()
        return model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

    def encode_query(self, text: str) -> Any:
        # BGE v1.5 recommends an instruction for short-query-to-passage retrieval.
        query = f"Represent this sentence for searching relevant passages: {text}"
        model = self._load()
        return model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
