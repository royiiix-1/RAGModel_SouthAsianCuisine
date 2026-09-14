from __future__ import annotations

import unittest

from south_asian_cuisine_rag.config import GENERATION_MODEL_ID, Settings
from south_asian_cuisine_rag.errors import ConfigurationError


class SettingsTests(unittest.TestCase):
    def test_generation_model_is_locked(self) -> None:
        with self.assertRaises(ConfigurationError):
            Settings.from_env({"RAG_GENERATION_MODEL": "some/other-model"})

    def test_defaults_preserve_required_model(self) -> None:
        settings = Settings.from_env({})
        self.assertEqual(settings.generation_model, GENERATION_MODEL_ID)
        self.assertEqual(len(settings.generation_revision), 40)
        self.assertEqual(len(settings.embedding_revision), 40)

    def test_chunk_configuration_is_validated(self) -> None:
        with self.assertRaises(ConfigurationError):
            Settings(child_chunk_size=500, parent_chunk_size=400).validate()

    def test_generation_sources_cannot_exceed_retrieval_count(self) -> None:
        with self.assertRaises(ConfigurationError):
            Settings(top_k=2, max_generation_sources=3).validate()


if __name__ == "__main__":
    unittest.main()
