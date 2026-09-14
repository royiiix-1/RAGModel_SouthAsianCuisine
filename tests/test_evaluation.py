from __future__ import annotations

import unittest

from south_asian_cuisine_rag.evaluation import (
    required_fact_recall,
    source_hit,
    source_rank,
    token_f1,
)
from south_asian_cuisine_rag.models import AnswerResult, SearchResult, SourceMetadata


class EvaluationTests(unittest.TestCase):
    def test_token_f1_counts_repeated_tokens(self) -> None:
        self.assertAlmostEqual(token_f1("rice rice lentils", "rice lentils"), 0.8)

    def test_source_hit_uses_returned_provenance(self) -> None:
        source = SearchResult(
            source_id="S1",
            chunk_id="c",
            parent_id="p",
            text="content",
            metadata=SourceMetadata(
                source="Wikipedia",
                topic="Andhra cuisine",
                section="Breakfast",
                url="https://example.test",
            ),
            similarity_score=0.8,
        )
        result = AnswerResult("r", "answer", True, "model", (source,), 1.0, 2.0)
        self.assertTrue(source_hit("Wikipedia - Andhra cuisine", result))
        self.assertEqual(source_rank("Wikipedia - Andhra cuisine", result), 1)

    def test_required_fact_recall_accepts_declared_alternatives(self) -> None:
        facts = (("toast", "toasted"), ("grind", "ground"), ("fenugreek",))
        score = required_fact_recall(
            facts, "The fenugreek is toasted and then ground into powder."
        )
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
