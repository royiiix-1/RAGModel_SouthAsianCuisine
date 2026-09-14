from __future__ import annotations

import hashlib
import json
import unittest

from south_asian_cuisine_rag.config import PROJECT_ROOT


def digest(path: str) -> str:
    return hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()


class EvaluationBaselineTests(unittest.TestCase):
    def test_recorded_baseline_matches_evaluation_inputs(self) -> None:
        baseline = json.loads(
            (PROJECT_ROOT / "data/evaluation/baseline_v2.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            baseline["inputs"]["benchmark_sha256"],
            digest("data/evaluation/benchmark_v2.csv"),
        )
        self.assertEqual(
            baseline["inputs"]["no_answer_sha256"],
            digest("data/evaluation/no_answer.jsonl"),
        )


if __name__ == "__main__":
    unittest.main()
