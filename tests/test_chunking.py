from __future__ import annotations

import unittest

from south_asian_cuisine_rag.chunking import chunk_documents, split_spans
from south_asian_cuisine_rag.models import RawDocument, SourceMetadata


class ChunkingTests(unittest.TestCase):
    def test_split_spans_is_bounded_and_terminates(self) -> None:
        text = " ".join(f"word-{index}." for index in range(400))
        spans = split_spans(text, max_chars=180, overlap=30)
        self.assertGreater(len(spans), 1)
        self.assertTrue(all(len(piece) <= 180 for _, _, piece in spans))
        self.assertEqual(spans[-1][1], len(text))
        self.assertTrue(
            all(start == 0 or text[start - 1].isspace() for start, _, _ in spans)
        )

    def test_parent_ids_do_not_collide_across_documents(self) -> None:
        metadata = SourceMetadata(
            source="Test",
            topic="Same topic",
            section="Same section",
            url="https://example.test/page",
        )
        documents = [
            RawDocument("doc_one", "First paragraph with one culinary fact." * 20, metadata),
            RawDocument("doc_two", "Second paragraph with another culinary fact." * 20, metadata),
        ]
        chunks = chunk_documents(
            documents,
            parent_size=300,
            parent_overlap=30,
            child_size=120,
            child_overlap=20,
        )
        parent_documents: dict[str, set[str]] = {}
        for chunk in chunks:
            parent_documents.setdefault(chunk.parent_id, set()).add(chunk.document_id)
        self.assertTrue(parent_documents)
        self.assertTrue(all(len(document_ids) == 1 for document_ids in parent_documents.values()))


if __name__ == "__main__":
    unittest.main()
