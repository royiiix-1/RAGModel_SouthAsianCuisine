from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol

from .artifacts import ArtifactStore
from .config import Settings
from .embeddings import Embedder, SentenceTransformerEmbedder
from .models import SearchResult

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_QUERY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "ancient",
    "are",
    "believed",
    "cuisine",
    "dish",
    "do",
    "does",
    "for",
    "from",
    "how",
    "historical",
    "in",
    "is",
    "of",
    "origin",
    "period",
    "preparation",
    "prepare",
    "prepared",
    "root",
    "savory",
    "savoury",
    "snack",
    "the",
    "to",
    "traditional",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> set[str]:
    return {
        _stem(token.lower())
        for token in _TOKEN_RE.findall(text)
        if token.lower() not in _QUERY_STOP_WORDS
    }


def _without_header(text: str) -> str:
    return re.sub(r"^\[Topic:.*?\]\s*", "", text, count=1).strip()


class Reranker(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.local_files_only = local_files_only
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed; install project dependencies"
                ) from exc
            kwargs: dict[str, Any] = {
                "revision": self.revision,
                "local_files_only": self.local_files_only,
            }
            if self.device != "auto":
                kwargs["device"] = self.device
            self._model = CrossEncoder(self.model_id, **kwargs)
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores = self._load().predict([(query, passage) for passage in passages])
        return [float(score) for score in scores]


class Retriever:
    def __init__(
        self,
        settings: Settings,
        store: ArtifactStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        document_frequency: Counter[str] = Counter()
        for record in store.records:
            document_frequency.update(
                _tokens(
                    f"{record.metadata.topic} {record.metadata.section} {record.text}"
                )
            )
        self._document_frequency = document_frequency
        self._record_count = max(1, len(store.records))

    def _metadata_boost(self, query: str, record: Any) -> float:
        query_terms = _tokens(query)
        metadata_terms = _tokens(
            f"{record.metadata.topic} {record.metadata.section}"
        )
        rare_limit = max(8, int(self._record_count * 0.005))
        matched = {
            term
            for term in query_terms & metadata_terms
            if self._document_frequency[term] <= rare_limit
        }
        if not matched:
            return 0.0
        weights = sorted(
            (
                math.log((self._record_count + 1) / (self._document_frequency[term] + 1))
                + 1.0
                for term in matched
            ),
            reverse=True,
        )
        return weights[0] + (0.15 * sum(weights[1:]))

    @classmethod
    def from_settings(cls, settings: Settings) -> Retriever:
        store = ArtifactStore.load(settings)
        embedder = SentenceTransformerEmbedder(
            settings.embedding_model,
            settings.embedding_revision,
            device=settings.device,
            batch_size=settings.embedding_batch_size,
            local_files_only=settings.offline_mode,
        )
        reranker: Reranker | None = None
        if settings.use_reranker:
            reranker = CrossEncoderReranker(
                settings.reranker_model,
                settings.reranker_revision,
                device=settings.device,
                local_files_only=settings.offline_mode,
            )
        return cls(settings, store, embedder, reranker)

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[SearchResult]:
        requested_k = self.settings.top_k if top_k is None else top_k
        if not 1 <= requested_k <= 10:
            raise ValueError("top_k must be between 1 and 10")

        query_vector = self.embedder.encode_query(query)
        raw_candidates = [
            (record, score)
            for record, score in self.store.search(query_vector, self.settings.candidate_k)
            if score >= self.settings.minimum_similarity
        ]
        children_by_parent: dict[str, list[tuple[Any, float]]] = {}
        best_child_by_parent: dict[str, tuple[Any, float]] = {}
        for record, score in raw_candidates:
            children_by_parent.setdefault(record.parent_id, []).append((record, score))
            existing = best_child_by_parent.get(record.parent_id)
            if existing is None or score > existing[1]:
                best_child_by_parent[record.parent_id] = (record, score)
        candidates = list(best_child_by_parent.values())
        if not candidates:
            return []

        ranked: list[tuple[Any, float, float | None, float]]
        if self.reranker is not None:
            rerank_scores = self.reranker.score(
                query, [record.text for record, _ in candidates]
            )
            ranked = [
                (
                    record,
                    similarity,
                    rerank,
                    rerank + self._metadata_boost(query, record) + (0.5 * similarity),
                )
                for (record, similarity), rerank in zip(
                    candidates, rerank_scores, strict=True
                )
            ]
            ranked.sort(
                key=lambda item: item[3],
                reverse=True,
            )
        else:
            ranked = [
                (
                    record,
                    similarity,
                    None,
                    similarity + self._metadata_boost(query, record),
                )
                for record, similarity in candidates
            ]
            ranked.sort(key=lambda item: item[3], reverse=True)

        selected: list[tuple[Any, float, float | None, float]] = []
        seen_parents: set[str] = set()
        seen_texts: set[str] = set()
        for record, similarity, rerank_score, ranking_score in ranked:
            normalized_text = " ".join(record.text.lower().split())
            if record.parent_id in seen_parents or normalized_text in seen_texts:
                continue
            seen_parents.add(record.parent_id)
            seen_texts.add(normalized_text)
            selected.append((record, similarity, rerank_score, ranking_score))
            if len(selected) == requested_k:
                break

        return [
            SearchResult(
                source_id=f"S{position}",
                chunk_id=record.chunk_id,
                parent_id=record.parent_id,
                text=record.parent_text,
                metadata=record.metadata,
                similarity_score=similarity,
                rerank_score=rerank_score,
                ranking_score=ranking_score,
                matched_text=self._focused_children(
                    record.parent_text, children_by_parent[record.parent_id]
                ),
            )
            for position, (record, similarity, rerank_score, ranking_score) in enumerate(
                selected, start=1
            )
        ]

    @staticmethod
    def _focused_children(
        parent_text: str, candidates: list[tuple[Any, float]]
    ) -> str:
        strongest = sorted(candidates, key=lambda item: item[1], reverse=True)[:2]
        parent_body = _without_header(parent_text)
        child_bodies = [_without_header(record.text) for record, _ in strongest]
        child_bodies.sort(
            key=lambda child: (
                parent_body.find(child) if parent_body.find(child) >= 0 else len(parent_body)
            )
        )
        return "\n...\n".join(dict.fromkeys(child_bodies))
