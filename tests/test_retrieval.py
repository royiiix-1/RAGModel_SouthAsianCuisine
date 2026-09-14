from __future__ import annotations

import unittest
from collections.abc import Sequence
from typing import Any

from south_asian_cuisine_rag.config import Settings
from south_asian_cuisine_rag.models import ChunkRecord, SourceMetadata
from south_asian_cuisine_rag.retrieval import Retriever


def chunk(chunk_id: str, parent_id: str, topic: str) -> ChunkRecord:
    metadata = SourceMetadata(
        source="Test",
        topic=topic,
        section="Procedure",
        url=f"https://example.test/{chunk_id}",
    )
    return ChunkRecord(
        chunk_id=chunk_id,
        parent_id=parent_id,
        document_id=f"doc_{chunk_id}",
        text=f"Child {topic}",
        parent_text=f"Parent {topic}",
        metadata=metadata,
    )


class FakeStore:
    def __init__(self, values: list[tuple[ChunkRecord, float]]) -> None:
        self.values = values
        self.records = [record for record, _ in values]

    def search(self, query_vector: Any, k: int) -> list[tuple[ChunkRecord, float]]:
        return self.values[:k]


class FakeEmbedder:
    model_id = "test"
    revision = "test"

    def encode_query(self, text: str) -> list[list[float]]:
        return [[1.0]]

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


class FakeReranker:
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [float(index) for index, _ in enumerate(passages)]


class RetrievalTests(unittest.TestCase):
    def test_threshold_reranking_and_parent_deduplication(self) -> None:
        values = [
            (chunk("a", "same_parent", "A"), 0.91),
            (chunk("b", "same_parent", "A continuation"), 0.89),
            (chunk("c", "other_parent", "C"), 0.80),
            (chunk("d", "weak_parent", "D"), 0.10),
        ]
        retriever = Retriever(
            Settings(use_reranker=True),
            FakeStore(values),  # type: ignore[arg-type]
            FakeEmbedder(),
            FakeReranker(),
        )
        results = retriever.retrieve("query", top_k=3)
        self.assertEqual(
            set(item.parent_id for item in results), {"other_parent", "same_parent"}
        )
        self.assertEqual([item.source_id for item in results], ["S1", "S2"])

    def test_returns_empty_when_all_candidates_are_weak(self) -> None:
        retriever = Retriever(
            Settings(use_reranker=False),
            FakeStore([(chunk("d", "weak", "D"), 0.1)]),  # type: ignore[arg-type]
            FakeEmbedder(),
        )
        self.assertEqual(retriever.retrieve("unrelated"), [])

    def test_focus_combines_two_strongest_children_in_parent_order(self) -> None:
        focused = Retriever._focused_children(
            "[Topic: Test | Section: Test]\nFirst part. Second part. Third part.",
            [
                (chunk("c", "p", "Third part."), 0.8),
                (chunk("a", "p", "First part."), 0.9),
                (chunk("b", "p", "Second part."), 0.7),
            ],
        )
        self.assertLess(focused.find("First"), focused.find("Third"))
        self.assertNotIn("Second", focused)


if __name__ == "__main__":
    unittest.main()
