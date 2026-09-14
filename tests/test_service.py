from __future__ import annotations

import unittest
from typing import Any

from south_asian_cuisine_rag.config import GENERATION_MODEL_ID, Settings
from south_asian_cuisine_rag.models import SearchResult, SourceMetadata
from south_asian_cuisine_rag.prompts import FALLBACK_ANSWER
from south_asian_cuisine_rag.service import RagService


def result() -> SearchResult:
    return SearchResult(
        source_id="S1",
        chunk_id="chunk_1",
        parent_id="parent_1",
        text="Toast fenugreek and onion seeds before grinding them into a fine powder.",
        metadata=SourceMetadata(
            source="Wikibooks",
            topic="Tandoori Masala",
            section="Procedure",
            url="https://example.test/tandoori",
        ),
        similarity_score=0.91,
    )


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[SearchResult]:
        return self.results[:top_k] if top_k else self.results


class FakeGenerator:
    model_id = GENERATION_MODEL_ID

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def generate(self, messages: Any) -> str:
        self.calls += 1
        return self.answer


class ServiceTests(unittest.TestCase):
    def test_no_retrieval_returns_fallback_without_generation(self) -> None:
        generator = FakeGenerator("This must not be returned")
        service = RagService(Settings(), FakeRetriever([]), generator)  # type: ignore[arg-type]
        answer = service.answer("What is the capital of France?")
        self.assertEqual(answer.answer, FALLBACK_ANSWER)
        self.assertFalse(answer.grounded)
        self.assertEqual(generator.calls, 0)

    def test_supported_answer_is_returned(self) -> None:
        generator = FakeGenerator(
            "Fenugreek and onion seeds are toasted before being ground into a fine powder [S1]."
        )
        service = RagService(Settings(), FakeRetriever([result()]), generator)  # type: ignore[arg-type]
        answer = service.answer("How are fenugreek and onion seeds prepared?")
        self.assertTrue(answer.grounded)
        self.assertIn("toasted", answer.answer)

    def test_unsupported_answer_is_rejected(self) -> None:
        generator = FakeGenerator("Saturn has bright rings and many moons.")
        service = RagService(Settings(), FakeRetriever([result()]), generator)  # type: ignore[arg-type]
        answer = service.answer("How are fenugreek and onion seeds prepared?")
        self.assertFalse(answer.grounded)
        self.assertEqual(answer.answer, FALLBACK_ANSWER)


if __name__ == "__main__":
    unittest.main()
