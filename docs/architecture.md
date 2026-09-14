# Architecture and trust boundaries

## Runtime components

1. `corpus.py` validates source text and mandatory provenance fields.
2. `chunking.py` creates deterministic parent and child records. IDs include the
   document identity, character offsets, and text, so unrelated paragraphs cannot
   collapse under one parent ID.
3. `indexing.py` embeds child chunks with the pinned BGE revision.
4. `artifacts.py` writes `index.faiss`, `records.jsonl`, and `manifest.json`.
   The manifest is replaced last and contains source hashes, artifact hashes,
   dimensions, counts, configuration fingerprint, and model revision.
5. `retrieval.py` rejects candidates below the cosine threshold, optionally
   reranks them, deduplicates only genuinely identical parents, and returns full
   provenance.
6. `generation.py` loads one pinned Qwen instance lazily and runs deterministic
   generation. Remote model code is disabled.
7. `service.py` bounds questions and context, skips generation when retrieval has
   no evidence, and refuses answers with very low lexical support.
8. `api.py`, `cli.py`, and `gradio_app.py` expose the same service implementation.

## Trust boundaries

- User questions are untrusted and length-limited.
- Corpus text is untrusted even when it comes from a known publisher. It is
  explicitly labelled as data in the prompt, and source markup is escaped.
- Model output is untrusted. It receives a lightweight grounding check and is
  always returned with the exact retrieved sources for independent verification.
- Artifact files are untrusted until all hashes, counts, dimensions, source
  inputs, and model/configuration identities pass validation.

The lexical grounding check is a guardrail, not a factuality proof. Production
acceptance still requires human-reviewed evaluation and monitoring.

## Build and runtime separation

Index building is an offline operation. The API never scrapes websites or
silently rebuilds its index. A corpus update must follow this sequence:

1. capture reviewed content with URLs and timestamps;
2. run corpus validation and data-quality review;
3. build a new index artifact;
4. run the full benchmark and release gate;
5. deploy the corpus and matching artifact together;
6. retain the prior version for rollback.

The current filesystem implementation is appropriate for a single instance. A
multi-instance deployment should publish immutable, versioned artifact bundles
to controlled object storage and mount the same verified version into every pod.
