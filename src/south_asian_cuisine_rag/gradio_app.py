from __future__ import annotations

from typing import Any

from .config import Settings
from .service import RagService


def create_interface(
    settings: Settings | None = None, *, service: RagService | None = None
) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install the UI extra with `pip install -e .[ui]`") from exc

    active_settings = settings or Settings.from_env()
    rag_service = service or RagService.from_settings(active_settings)

    def answer(question: str) -> tuple[str, str]:
        result = rag_service.answer(question)
        source_lines = [
            f"{source.source_id}. [{source.metadata.topic}]({source.metadata.url}) — "
            f"{source.metadata.source}, {source.metadata.section}"
            for source in result.sources
        ]
        source_text = "\n\n".join(source_lines)
        return result.answer, source_text or "No supporting source passed the threshold."

    return gr.Interface(
        fn=answer,
        inputs=gr.Textbox(
            label="Question",
            lines=3,
            max_lines=6,
            placeholder="Ask a factual question about South Asian cuisine",
        ),
        outputs=[
            gr.Markdown(label="Grounded answer"),
            gr.Markdown(label="Sources"),
        ],
        title="South Asian Cuisine Encyclopedia",
        description=(
            "Answers are generated locally with Qwen2.5-0.5B-Instruct and include "
            "retrievable source links."
        ),
        examples=[
            ["How are fenugreek and onion seeds prepared for Tandoori Masala?"],
            ["How is chana chaat prepared in Pakistani cuisine?"],
        ],
        flagging_mode="never",
    )
