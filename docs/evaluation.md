# Evaluation

## Benchmark policy

The inherited 45-question file is retained as `benchmark_legacy.csv`, but it is
not a release gate. Several of its gold answers assert details absent from their
source records, so optimizing against them would reward hallucination.

`benchmark_v2.csv` contains 20 manually traceable answerable questions. Every
row names an exact source URL and required fact groups; alternatives such as
`yogurt~yoghurt` are accepted. Eight separate no-answer and instruction-attack
queries must be refused before generation.

Metrics are deliberately separate:

- token F1 measures wording overlap and remains diagnostic;
- required-fact recall measures whether supported answer facts are present;
- full-fact coverage measures complete answers;
- source Hit@1 and MRR measure retrieval independently of generation;
- no-answer accuracy measures refusal before generation;
- p50/p95 report observed end-to-end latency.

## Verified v2 baseline

The baseline in `data/evaluation/baseline_v2.json` was produced on Windows CPU
with the pinned models and the checksummed 3,831-chunk index:

| Metric | Result | Release floor |
|---|---:|---:|
| Token F1 | 0.563 | 0.35 |
| Required-fact recall | 0.930 | 0.75 |
| Full fact coverage | 0.750 | 0.60 |
| Source Hit@1 | 1.000 | 0.75 |
| Source MRR | 1.000 | 0.85 |
| No-answer accuracy | 1.000 | 0.80 |
| Latency p50 | 2.96 s | informational |
| Latency p95 | 14.13 s | informational |

The checked-in gate passes this baseline. This is a regression result, not a
claim of universal factual accuracy: the set is small and derived from the same
corpus. Public release still requires a larger independent human-reviewed set,
target-hardware load testing, and safety review.

## Reproduce

```bash
rag-cuisine verify-index
rag-cuisine evaluate --output evaluation-results/v2.json
python scripts/check_evaluation.py evaluation-results/v2.json
```

Set `RAG_OFFLINE_MODE=true` after all pinned model snapshots are cached to prove
that evaluation does not depend on mutable remote files.
