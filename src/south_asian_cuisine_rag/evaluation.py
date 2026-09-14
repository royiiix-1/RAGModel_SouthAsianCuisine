from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import AnswerResult
from .prompts import FALLBACK_ANSWER
from .service import RagService

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    query_id: str
    question: str
    gold_answer: str
    source_reference: str
    expected_url: str = ""
    required_facts: tuple[tuple[str, ...], ...] = ()
    expect_answer: bool = True


def normalize_answer(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def token_f1(gold: str, predicted: str) -> float:
    gold_tokens = _TOKEN_RE.findall(gold.lower())
    predicted_tokens = _TOKEN_RE.findall(re.sub(r"\[S\d+\]", "", predicted.lower()))
    if not gold_tokens or not predicted_tokens:
        return float(gold_tokens == predicted_tokens)
    overlap = sum((Counter(gold_tokens) & Counter(predicted_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def required_fact_recall(
    required_facts: tuple[tuple[str, ...], ...], predicted: str
) -> float | None:
    if not required_facts:
        return None
    normalized_prediction = normalize_answer(predicted)
    matched = 0
    for alternatives in required_facts:
        if any(
            normalize_answer(alternative) in normalized_prediction
            for alternative in alternatives
        ):
            matched += 1
    return matched / len(required_facts)


def source_rank(
    reference: str, result: AnswerResult, *, expected_url: str = ""
) -> int | None:
    if expected_url:
        normalized_url = expected_url.rstrip("/")
        for position, source in enumerate(result.sources, start=1):
            if source.metadata.url.rstrip("/") == normalized_url:
                return position
        return None
    reference_tokens = set(_TOKEN_RE.findall(reference.lower()))
    generic = {
        "around",
        "blog",
        "cookbook",
        "cuisine",
        "cuisines",
        "wikipedia",
        "wikibooks",
        "world",
    }
    distinctive = reference_tokens - generic
    for position, source in enumerate(result.sources, start=1):
        haystack = normalize_answer(
            f"{source.metadata.source} {source.metadata.topic} {source.metadata.section}"
        )
        if distinctive and any(token in haystack for token in distinctive):
            return position
    return None


def source_hit(reference: str, result: AnswerResult, *, expected_url: str = "") -> bool:
    return source_rank(reference, result, expected_url=expected_url) is not None


def _parse_required_facts(value: str) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(alternative.strip() for alternative in group.split("~") if alternative.strip())
        for group in value.split("|")
        if group.strip()
    )


def load_benchmark(path: Path) -> list[EvaluationExample]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        EvaluationExample(
            query_id=str(row["id"]),
            question=str(row["query"]),
            gold_answer=str(row["gold_answer"]),
            source_reference=str(row.get("source_reference", "")),
            expected_url=str(row.get("expected_url", "")),
            required_facts=_parse_required_facts(str(row.get("required_facts", ""))),
        )
        for row in rows
    ]


def load_no_answer_cases(path: Path) -> list[EvaluationExample]:
    examples: list[EvaluationExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                EvaluationExample(
                    query_id=str(row.get("id", f"NO_ANSWER_{line_number:03d}")),
                    question=str(row["query"]),
                    gold_answer=FALLBACK_ANSWER,
                    source_reference="",
                    expected_url="",
                    required_facts=(),
                    expect_answer=False,
                )
            )
    return examples


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[max(index, 0)]


def evaluate(
    service: RagService,
    examples: list[EvaluationExample],
    *,
    limit: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    selected = examples if limit is None else examples[:limit]
    rows: list[dict[str, Any]] = []
    for position, example in enumerate(selected, start=1):
        result = service.answer(example.question)
        answered = result.answer != FALLBACK_ANSWER
        retrieved_source_rank = (
            source_rank(
                example.source_reference,
                result,
                expected_url=example.expected_url,
            )
            if example.expect_answer
            and (example.source_reference or example.expected_url)
            else None
        )
        fact_recall = required_fact_recall(example.required_facts, result.answer)
        rows.append(
            {
                "query_id": example.query_id,
                "query": example.question,
                "gold_answer": example.gold_answer,
                "prediction": result.answer,
                "expect_answer": example.expect_answer,
                "answered": answered,
                "grounded": result.grounded,
                "token_f1": token_f1(example.gold_answer, result.answer),
                "exact_match": normalize_answer(example.gold_answer)
                == normalize_answer(result.answer),
                "source_hit": retrieved_source_rank is not None,
                "source_rank": retrieved_source_rank,
                "required_fact_recall": fact_recall,
                "latency_ms": result.retrieval_ms + result.generation_ms,
                "sources": [source.to_public_dict() for source in result.sources],
            }
        )
        if progress is not None:
            progress(position, len(selected), example.query_id)

    answer_rows = [row for row in rows if row["expect_answer"]]
    fact_rows = [row for row in answer_rows if row["required_fact_recall"] is not None]
    no_answer_rows = [row for row in rows if not row["expect_answer"]]
    latencies = [float(row["latency_ms"]) for row in rows]
    aggregate = {
        "examples": len(rows),
        "answer_examples": len(answer_rows),
        "no_answer_examples": len(no_answer_rows),
        "mean_token_f1": (
            sum(float(row["token_f1"]) for row in answer_rows) / len(answer_rows)
            if answer_rows
            else 0.0
        ),
        "exact_match": (
            sum(bool(row["exact_match"]) for row in answer_rows) / len(answer_rows)
            if answer_rows
            else 0.0
        ),
        "mean_required_fact_recall": (
            sum(float(row["required_fact_recall"]) for row in fact_rows) / len(fact_rows)
            if fact_rows
            else None
        ),
        "full_required_fact_coverage": (
            sum(float(row["required_fact_recall"]) == 1.0 for row in fact_rows)
            / len(fact_rows)
            if fact_rows
            else None
        ),
        "source_hit_rate": (
            sum(row["source_hit"] is True for row in answer_rows) / len(answer_rows)
            if answer_rows
            else 0.0
        ),
        "source_hit_at_1": (
            sum(row["source_rank"] == 1 for row in answer_rows) / len(answer_rows)
            if answer_rows
            else 0.0
        ),
        "source_mrr": (
            sum(
                1.0 / int(row["source_rank"])
                for row in answer_rows
                if row["source_rank"] is not None
            )
            / len(answer_rows)
            if answer_rows
            else 0.0
        ),
        "no_answer_accuracy": (
            sum(not bool(row["answered"]) for row in no_answer_rows) / len(no_answer_rows)
            if no_answer_rows
            else None
        ),
        "grounded_rate": (
            sum(bool(row["grounded"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": service.settings.generation_model,
        "model_revision": service.settings.generation_revision,
        "aggregate": aggregate,
        "results": rows,
    }


def write_evaluation(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
