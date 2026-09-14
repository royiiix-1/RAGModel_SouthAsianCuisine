from __future__ import annotations

import ast
import json
import unittest

from south_asian_cuisine_rag.config import GENERATION_MODEL_ID, PROJECT_ROOT


class NotebookTests(unittest.TestCase):
    def test_demo_notebook_is_clean_and_syntactically_valid(self) -> None:
        path = PROJECT_ROOT / "notebooks/Interactive_Demo.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 10)

        code = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            source = "".join(cell["source"])
            ast.parse(source)
            code.append(source)

        combined = "\n".join(code)
        self.assertIn("RagService", combined)
        self.assertIn("ArtifactStore", combined)
        self.assertNotIn("class BaselineRAG", combined)
        notebook_text = path.read_text(encoding="utf-8")
        self.assertIn(GENERATION_MODEL_ID, notebook_text)


if __name__ == "__main__":
    unittest.main()
