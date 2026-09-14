# Changelog

## 2.0.0

- Replaced the notebook runtime with a modular Python package.
- Locked generation to Qwen2.5-0.5B-Instruct and pinned model revisions.
- Added deterministic collision-free chunk IDs and checksummed FAISS artifacts.
- Added evidence thresholds, refusal behavior, reranking, and source traceability.
- Added FastAPI, CLI, optional Gradio, evaluation gates, tests, Docker, and CI.
- Moved the original experiment and result files under `legacy/`.
