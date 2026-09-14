from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    import faiss  # noqa: F401
    import numpy as np

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from south_asian_cuisine_rag.artifacts import RECORDS_FILENAME, ArtifactStore
from south_asian_cuisine_rag.config import Settings
from south_asian_cuisine_rag.errors import ArtifactError
from south_asian_cuisine_rag.indexing import build_index


class FakeEmbedder:
    model_id = "BAAI/bge-small-en-v1.5"
    revision = "test"

    def encode_documents(self, texts: Sequence[str]) -> Any:
        vectors = []
        for index, _ in enumerate(texts):
            vector = [0.0, 0.0, 0.0]
            vector[index % 3] = 1.0
            vectors.append(vector)
        return np.asarray(vectors, dtype="float32")  # type: ignore[possibly-undefined]

    def encode_query(self, text: str) -> Any:
        return np.asarray([[1.0, 0.0, 0.0]], dtype="float32")  # type: ignore[possibly-undefined]


def write_corpus(path: Path) -> None:
    records = [
        {
            "text": "Toast the seeds before grinding them into a fine powder.",
            "metadata": {
                "source": "Test",
                "topic": "Spice preparation",
                "section": "Procedure",
                "url": "https://example.test/one",
            },
        },
        {
            "text": "Cooked chickpeas are mixed with potato, onion, tomato, and spices.",
            "metadata": {
                "source": "Test",
                "topic": "Chana chaat",
                "section": "Procedure",
                "url": "https://example.test/two",
            },
        },
    ]
    path.write_text(json.dumps(records), encoding="utf-8")


@unittest.skipUnless(HAS_FAISS, "FAISS test dependencies are not installed")
class ArtifactTests(unittest.TestCase):
    def test_build_load_and_checksum_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            write_corpus(raw / "corpus.json")
            settings = replace(
                Settings(),
                project_root=root,
                raw_data_dir=raw,
                artifact_dir=root / "artifacts",
                parent_chunk_size=300,
                parent_chunk_overlap=30,
                child_chunk_size=120,
                child_chunk_overlap=20,
            )
            manifest, stats = build_index(settings, embedder=FakeEmbedder())
            store = ArtifactStore.load(settings)
            results = store.search([[1.0, 0.0, 0.0]], 1)

            self.assertEqual(manifest.chunk_count, stats["child_chunks"])
            self.assertEqual(len(results), 1)

            records_path = settings.artifact_dir / RECORDS_FILENAME
            records_path.write_text(
                records_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8"
            )
            with self.assertRaises(ArtifactError):
                ArtifactStore.load(settings)


if __name__ == "__main__":
    unittest.main()
