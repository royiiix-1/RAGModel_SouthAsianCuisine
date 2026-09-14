from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    source: str
    topic: str
    section: str
    url: str
    captured_at: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "source": self.source,
            "topic": self.topic,
            "section": self.section,
            "url": self.url,
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceMetadata:
        return cls(
            source=str(data["source"]),
            topic=str(data["topic"]),
            section=str(data["section"]),
            url=str(data["url"]),
            captured_at=str(data["captured_at"]) if data.get("captured_at") else None,
        )


@dataclass(frozen=True, slots=True)
class RawDocument:
    document_id: str
    text: str
    metadata: SourceMetadata


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: str
    parent_id: str
    document_id: str
    text: str
    parent_text: str
    metadata: SourceMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "document_id": self.document_id,
            "text": self.text,
            "parent_text": self.parent_text,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkRecord:
        return cls(
            chunk_id=str(data["chunk_id"]),
            parent_id=str(data["parent_id"]),
            document_id=str(data["document_id"]),
            text=str(data["text"]),
            parent_text=str(data["parent_text"]),
            metadata=SourceMetadata.from_dict(data["metadata"]),
        )


@dataclass(frozen=True, slots=True)
class SearchResult:
    source_id: str
    chunk_id: str
    parent_id: str
    text: str
    metadata: SourceMetadata
    similarity_score: float
    rerank_score: float | None = None
    ranking_score: float | None = None
    matched_text: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.metadata.topic,
            "section": self.metadata.section,
            "publisher": self.metadata.source,
            "url": self.metadata.url,
            "excerpt": self.matched_text or self.text,
            "similarity_score": round(self.similarity_score, 6),
            "rerank_score": (
                round(self.rerank_score, 6) if self.rerank_score is not None else None
            ),
            "ranking_score": (
                round(self.ranking_score, 6) if self.ranking_score is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class AnswerResult:
    request_id: str
    answer: str
    grounded: bool
    model: str
    sources: tuple[SearchResult, ...]
    retrieval_ms: float
    generation_ms: float

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "answer": self.answer,
            "grounded": self.grounded,
            "model": self.model,
            "sources": [source.to_public_dict() for source in self.sources],
            "timings_ms": {
                "retrieval": round(self.retrieval_ms, 2),
                "generation": round(self.generation_ms, 2),
            },
        }
