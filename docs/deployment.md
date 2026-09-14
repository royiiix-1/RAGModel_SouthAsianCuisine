# Deployment guide

## Minimum production topology

- TLS reverse proxy or ingress
- one API process per allocated Qwen model instance
- persistent read-only model cache after prefetch
- immutable corpus and matching FAISS artifact
- external request rate limiting
- centralized collection of JSON logs
- alerts on readiness, 5xx rate, latency, refusal rate, and memory pressure

Do not use multiple Uvicorn workers in one container: every worker would load a
separate model. Scale with separately measured containers instead.

## Build

```bash
docker compose build
docker compose --profile tools run --rm indexer
```

Index creation downloads the pinned BGE revision and writes a manifest into the
`rag-artifacts` volume. `verify-index` can be run in the same image before rollout.

## Start

```bash
export RAG_API_KEY='a-long-random-secret-from-your-secret-manager'
docker compose up -d api
```

The liveness endpoint only confirms that the process is running. The readiness
endpoint returns 503 until corpus/index validation succeeds. Model weights are
loaded on the first real request; warm the instance with a controlled query
before adding it to live traffic.

## Rollback

An index cannot be mixed with a different corpus, embedding revision, or chunking
configuration. Roll back the container image, corpus, and artifact manifest as a
single release unit.

## Capacity

Start with `RAG_REQUEST_CONCURRENCY=1`. Measure peak memory and p95 latency on the
actual CPU/GPU before increasing it. The small model reduces resource needs but
does not make concurrent Transformer generation free or thread-safe by default.
