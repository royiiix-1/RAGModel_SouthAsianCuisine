from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

try:
    import httpx

    from south_asian_cuisine_rag.api import create_app

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from south_asian_cuisine_rag.config import GENERATION_MODEL_ID, Settings
from south_asian_cuisine_rag.models import AnswerResult


class FakeService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        manifest = SimpleNamespace(created_at="2026-01-01T00:00:00+00:00", chunk_count=3)
        self.retriever = SimpleNamespace(store=SimpleNamespace(manifest=manifest))

    def answer(self, question: str, *, top_k: int | None = None) -> AnswerResult:
        return AnswerResult(
            "request", "A grounded answer.", True, GENERATION_MODEL_ID, (), 1.0, 2.0
        )


@unittest.skipUnless(HAS_FASTAPI, "FastAPI test dependencies are not installed")
class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_endpoint_requires_configured_api_key(self) -> None:
        settings = replace(Settings(), api_key="test-secret")
        application = create_app(settings, service=FakeService(settings))  # type: ignore[arg-type]
        async with application.router.lifespan_context(application):
            transport = httpx.ASGITransport(app=application)  # type: ignore[possibly-undefined]
            async with httpx.AsyncClient(  # type: ignore[possibly-undefined]
                transport=transport, base_url="http://test"
            ) as client:
                denied = await client.post(
                    "/v1/answers", json={"question": "A valid question?"}
                )
                allowed = await client.post(
                    "/v1/answers",
                    headers={"X-API-Key": "test-secret"},
                    json={"question": "A valid question?"},
                )
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.json()["model"], GENERATION_MODEL_ID)

    async def test_health_is_public(self) -> None:
        settings = replace(Settings(), api_key="test-secret")
        application = create_app(settings, service=FakeService(settings))  # type: ignore[arg-type]
        async with application.router.lifespan_context(application):
            transport = httpx.ASGITransport(app=application)  # type: ignore[possibly-undefined]
            async with httpx.AsyncClient(  # type: ignore[possibly-undefined]
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.get("/health/live")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
