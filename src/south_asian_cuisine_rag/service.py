from __future__ import annotations

import re
import time
import uuid

from .config import Settings
from .errors import QueryValidationError
from .generation import QwenGenerator, TextGenerator
from .models import AnswerResult
from .prompts import (
    FALLBACK_ANSWER,
    build_messages,
    constrain_answer,
    grounding_token_precision,
)
from .retrieval import Retriever

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def validate_query(query: str, *, max_chars: int) -> str:
    normalized = " ".join(query.split())
    if len(normalized) < 3:
        raise QueryValidationError("Question must contain at least 3 characters")
    if len(normalized) > max_chars:
        raise QueryValidationError(f"Question exceeds the {max_chars}-character limit")
    if _CONTROL_CHARACTERS.search(query):
        raise QueryValidationError("Question contains unsupported control characters")
    return normalized


class RagService:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        generator: TextGenerator,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.generator = generator

    @classmethod
    def from_settings(cls, settings: Settings) -> RagService:
        settings.validate()
        return cls(settings, Retriever.from_settings(settings), QwenGenerator(settings))

    def answer(self, question: str, *, top_k: int | None = None) -> AnswerResult:
        query = validate_query(question, max_chars=self.settings.max_query_chars)
        request_id = uuid.uuid4().hex

        retrieval_started = time.perf_counter()
        sources = self.retriever.retrieve(query, top_k=top_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        if not sources:
            return AnswerResult(
                request_id=request_id,
                answer=FALLBACK_ANSWER,
                grounded=False,
                model=self.generator.model_id,
                sources=(),
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
            )

        generation_sources = sources[: self.settings.max_generation_sources]
        messages = build_messages(
            query, generation_sources, max_context_chars=self.settings.max_context_chars
        )
        generation_started = time.perf_counter()
        generated_answer = self.generator.generate(messages).strip()
        generation_ms = (time.perf_counter() - generation_started) * 1000

        answer = constrain_answer(
            generated_answer,
            generation_sources,
            max_sentences=self.settings.max_answer_sentences,
            max_chars=self.settings.max_answer_chars,
        )

        precision = grounding_token_precision(answer, generation_sources)
        grounded = bool(answer) and precision >= self.settings.minimum_grounding_precision
        if not grounded:
            answer = FALLBACK_ANSWER

        return AnswerResult(
            request_id=request_id,
            answer=answer,
            grounded=grounded,
            model=self.generator.model_id,
            sources=tuple(sources),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )
