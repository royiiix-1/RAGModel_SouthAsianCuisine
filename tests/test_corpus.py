from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from south_asian_cuisine_rag.corpus import load_corpus
from south_asian_cuisine_rag.errors import CorpusValidationError


def record(text: str = "A sufficiently detailed culinary fact.") -> dict[str, object]:
    return {
        "text": text,
        "metadata": {
            "source": "Test source",
            "topic": "Test topic",
            "section": "Test section",
            "url": "https://example.test/source",
        },
    }


class CorpusTests(unittest.TestCase):
    def test_loads_deduplicates_and_filters_boilerplate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text(
                json.dumps([record(), record(), record("- Read more")]), encoding="utf-8"
            )
            documents, report = load_corpus(Path(directory))
        self.assertEqual(len(documents), 1)
        self.assertEqual(report.input_records, 3)
        self.assertEqual(report.duplicate_documents, 1)
        self.assertEqual(report.filtered_documents, 1)

    def test_rejects_missing_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            payload = record()
            del payload["metadata"]["url"]  # type: ignore[index]
            path.write_text(json.dumps([payload]), encoding="utf-8")
            with self.assertRaises(CorpusValidationError):
                load_corpus(Path(directory))


if __name__ == "__main__":
    unittest.main()
