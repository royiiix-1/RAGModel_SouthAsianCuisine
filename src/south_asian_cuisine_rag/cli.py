from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from .artifacts import ArtifactStore
from .config import Settings
from .corpus import load_corpus

app = typer.Typer(
    no_args_is_help=True,
    help="Build, validate, evaluate, and serve the South Asian cuisine RAG system.",
)


def _echo_json(payload: object, *, ensure_ascii: bool = False) -> None:
    text = json.dumps(payload, ensure_ascii=ensure_ascii, indent=2) + "\n"
    try:
        typer.echo(text, nl=False)
    except UnicodeEncodeError:
        # Windows terminals may still expose a legacy code page such as GBK.
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.buffer.flush()


@app.command("validate-corpus")
def validate_corpus() -> None:
    """Validate raw JSON records without loading ML models."""
    settings = Settings.from_env()
    _, report = load_corpus(settings.raw_data_dir, strict=True)
    _echo_json({"status": "valid", **asdict(report)})


@app.command("build-index")
def build_index_command() -> None:
    """Create a verified FAISS index from the versioned raw corpus."""
    from .indexing import build_index

    settings = Settings.from_env()
    manifest, stats = build_index(settings)
    _echo_json({"status": "built", "stats": stats, "manifest": manifest.to_dict()})


@app.command("verify-index")
def verify_index_command() -> None:
    """Verify artifact checksums, source hashes, dimensions, and model revision."""
    settings = Settings.from_env()
    store = ArtifactStore.load(settings)
    _echo_json({"status": "valid", "manifest": store.manifest.to_dict()})


@app.command("ask")
def ask(
    question: Annotated[str, typer.Argument(help="Question to answer")],
    top_k: Annotated[int | None, typer.Option(min=1, max=10)] = None,
) -> None:
    """Ask one question and print a JSON response with traceable sources."""
    from .service import RagService

    result = RagService.from_settings(Settings.from_env()).answer(question, top_k=top_k)
    _echo_json(result.to_public_dict())


@app.command("evaluate")
def evaluate_command(
    output: Annotated[
        Path, typer.Option(help="Evaluation result JSON path")
    ] = Path("evaluation-results/latest.json"),
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    include_no_answer: Annotated[bool, typer.Option()] = True,
) -> None:
    """Run the held-out-style benchmark and no-answer regression set."""
    from .evaluation import (
        evaluate,
        load_benchmark,
        load_no_answer_cases,
        write_evaluation,
    )
    from .service import RagService

    settings = Settings.from_env()
    examples = load_benchmark(settings.project_root / "data/evaluation/benchmark_v2.csv")
    if include_no_answer:
        examples.extend(
            load_no_answer_cases(
                settings.project_root / "data/evaluation/no_answer.jsonl"
            )
        )
    payload = evaluate(
        RagService.from_settings(settings),
        examples,
        limit=limit,
        progress=lambda current, total, query_id: typer.echo(
            f"[{current}/{total}] {query_id}", err=True
        ),
    )
    output_path = output if output.is_absolute() else settings.project_root / output
    write_evaluation(payload, output_path)
    _echo_json(payload["aggregate"])


@app.command("serve")
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
) -> None:
    """Run the FastAPI service."""
    import uvicorn

    settings = Settings.from_env()
    if host not in {"127.0.0.1", "localhost", "::1"} and not settings.api_key:
        raise typer.BadParameter(
            "RAG_API_KEY is required when binding to a non-loopback address"
        )
    uvicorn.run(
        "south_asian_cuisine_rag.api:app",
        host=host,
        port=port,
        workers=1,
        log_config=None,
    )


@app.command("ui")
def ui(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 7860,
) -> None:
    """Run the optional local Gradio question-and-answer interface."""
    from .gradio_app import create_interface

    create_interface().launch(server_name=host, server_port=port, share=False)


if __name__ == "__main__":
    app()
