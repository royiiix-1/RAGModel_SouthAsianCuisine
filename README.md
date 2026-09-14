# South Asian Cuisine RAG

A local, source-grounded question-answering service for South Asian culinary
knowledge. Version 2 replaces the original all-in-one notebook with a tested
Python package, verified FAISS artifacts, a FastAPI service, a CLI, and an
optional Gradio interface.

> **Model invariant:** generation always uses
> [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
> at the pinned revision in `config.py`. Configuration that requests another
> generation model is rejected at startup.

## What changed

- Stable document, parent, and child IDs remove the legacy parent-collision bug.
- Indexes are rebuilt from raw inputs and protected by model, source, and file hashes.
- Retrieval has an evidence threshold, optional cross-encoder reranking, and refusal.
- Answers return stable source IDs, URLs, sections, excerpts, scores, and timings.
- The API has readiness/liveness endpoints, bounded concurrency, optional API-key
  authentication, input limits, structured logs, and no question-body logging.
- Evaluation separates token accuracy, source retrieval, no-answer behavior, and latency.
- The original notebook and results remain under `legacy/` for historical comparison only.

## Architecture

```text
versioned raw JSON
        |
        v
validate + deterministic parent/child chunking
        |
        v
revision-pinned BGE embeddings -> checksummed FAISS artifact
        |
        v
similarity threshold -> optional cross-encoder rerank -> bounded context
        |
        v
Qwen2.5-0.5B-Instruct -> grounding check -> answer + traceable sources
        |
        +---- FastAPI / CLI / local Gradio UI
```

See [docs/architecture.md](docs/architecture.md) for the trust boundaries and
artifact lifecycle.

## Requirements

- Python 3.10, 3.11, or 3.12 (**3.11 recommended; Python 3.13 is not supported**)
- Approximately 2 GB free disk space for Python packages and pinned model caches
- More memory is required when the optional reranker and Qwen are loaded together
- Internet access on the first index build and first generated answer

CUDA is optional. CPU mode works but generation is slower.

## Local environment and dependency installation

Use **one** environment manager. Do not create a `.venv` on top of an existing
Conda/venv directory with a different Python version: compiled packages such as
NumPy, regex, PyTorch, and FAISS will then be incompatible with the interpreter.

### Windows: Conda (recommended if your prompt shows `(base)`)

```powershell
conda create --name south-asian-rag python=3.11 -y
conda activate south-asian-rag
python --version
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,ui]"
python -m pip check
rag-cuisine validate-corpus
rag-cuisine build-index
rag-cuisine verify-index
jupyter lab notebooks\Interactive_Demo.ipynb
```

For contributors, install `.[dev,notebook,ui]` instead.

### Windows: standard `.venv`

First confirm that the Python launcher can find Python 3.11:

```powershell
py -0p
py -3.11 --version
```

Then run the checked setup script from the repository root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows.ps1
```

If `.venv` already exists but was created by another Python version, recreate
only this project's environment explicitly:

```powershell
.\scripts\setup_windows.ps1 -Recreate
```

The script verifies Python before making changes, installs the Notebook and UI
libraries, runs `pip check`, and validates the corpus. Use `-Dev` to include test
and lint dependencies.

If `py` is unavailable but you have a supported standalone interpreter, pass its
absolute path, for example `-PythonPath "C:\Python311\python.exe"`.

Manual equivalent:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version                 # must report 3.10.x, 3.11.x, or 3.12.x
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,ui]"
python -m pip check
rag-cuisine validate-corpus
rag-cuisine build-index
rag-cuisine verify-index
```

### macOS/Linux

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python --version
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[notebook,ui]'
.venv/bin/python -m pip check
.venv/bin/rag-cuisine validate-corpus
.venv/bin/rag-cuisine build-index
.venv/bin/rag-cuisine verify-index
```

The first build downloads the pinned BGE embedding model. The Qwen model is
loaded lazily on the first question, so health checks do not allocate the full
generation model.

### Fixing an incompatible existing `.venv`

If installation reports `requires a different Python: 3.13.x not in
'<3.13,>=3.10'`, the environment was created with Python 3.13. If NumPy or regex
then reports `cp312` binaries under Python 3.13, the same directory contains
packages from two Python versions. Do not reinstall individual libraries into
that mixed environment. Recreate it with the script above, or create the clean
Python 3.11 Conda environment. Model caches live outside `.venv`, so recreating
the project environment does not normally redownload cached model weights.

## Run

Local API:

```bash
rag-cuisine serve
```

Then call:

```bash
curl -X POST http://127.0.0.1:8000/v1/answers \
  -H "Content-Type: application/json" \
  -d '{"question":"How is chana chaat prepared in Pakistani cuisine?"}'
```

If `RAG_API_KEY` is set, also send `X-API-Key`. The CLI refuses to bind to a
non-loopback address without an API key.

Other interfaces:

```bash
rag-cuisine ask "How are fenugreek and onion seeds prepared for Tandoori Masala?"
rag-cuisine ui
```

The UI is deliberately a factual Q&A interface, not a misleading stateful chat.

### Visual Notebook demo

```bash
jupyter lab notebooks/Interactive_Demo.ipynb
```

The production demo notebook walks through corpus statistics, collision-free
chunking, artifact verification, retrieval scores, the grounded Qwen response,
source links, evaluation charts, and an `ipywidgets` question box. It imports the
same package used by FastAPI; no RAG implementation is duplicated in the notebook.
The original all-in-one notebook remains under `legacy/` for audit only.

## Evaluate

```bash
rag-cuisine evaluate --output evaluation-results/latest.json
python scripts/check_evaluation.py evaluation-results/latest.json
```

The release gate checks answer token F1, retrieval Hit@1/MRR, and no-answer
accuracy separately. It does not reuse the legacy mean-of-sentences semantic
metric that favored one-sentence prompts. The curated v2 set has only 20
answerable questions; passing it is necessary but not sufficient for release.

See [docs/evaluation.md](docs/evaluation.md) for the verified v2 metrics and
their limitations.

## Docker

Set a key, build the index once in the named volume, then start the API:

```bash
export RAG_API_KEY='replace-with-a-long-random-value'
docker compose --profile tools run --rm indexer
docker compose up --build api
```

PowerShell uses `$env:RAG_API_KEY = "..."` for the first command. See
[docs/deployment.md](docs/deployment.md) before exposing the service publicly.

## Configuration

Copy `.env.example` into your secret/configuration system; the application does
not automatically load `.env` files. Important settings:

| Variable | Default | Purpose |
|---|---|---|
| `RAG_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `RAG_OFFLINE_MODE` | `false` | Use only already-cached pinned model files |
| `RAG_USE_RERANKER` | `true` | Enable the pinned cross-encoder |
| `RAG_MINIMUM_SIMILARITY` | `0.65` | Reject weak vector matches |
| `RAG_TOP_K` | `5` | Final source count |
| `RAG_MAX_GENERATION_SOURCES` | `1` | Highest-ranked parent shown to Qwen |
| `RAG_REQUEST_CONCURRENCY` | `1` | Concurrent model generations |
| `RAG_MAX_ANSWER_SENTENCES` | `2` | Post-generation relevance bound |
| `RAG_API_KEY` | unset | Required for non-loopback CLI serving |

Changing any embedding or chunking setting invalidates the current index and
requires `rag-cuisine build-index`. The runtime detects this automatically.
`requirements.lock` pins the complete production dependency graph used by the
container; regenerate it deliberately after dependency review and rerun evaluation.

## Known release blockers

- The inherited corpus lacks capture timestamps and needs a reviewed refresh pipeline.
- The 45-question benchmark needs a larger independent test set and human factuality review.
- The repository owner must select a code license and complete the third-party corpus review.
- Load, abuse, prompt-injection, and food-safety acceptance tests must pass in the target environment.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [SECURITY.md](SECURITY.md),
and [docs/production-checklist.md](docs/production-checklist.md). These blockers
are explicit so the repository cannot again be mistaken for a release-ready build.
