# Production acceptance checklist

The service is not approved for public release until every required item is complete.

## Correctness

- [ ] Parent/child ID and artifact tests pass.
- [ ] Full benchmark passes `scripts/check_evaluation.py`.
- [ ] A separate, unseen human-reviewed test set meets the agreed target.
- [ ] No-answer, ambiguous, multi-hop, and conflicting-source cases are covered.
- [ ] The ten inherited demo questions are manually re-reviewed.

## Safety and security

- [ ] API key, TLS, rate limits, and request-size limits are enforced at the edge.
- [ ] Prompt-injection and malicious-corpus tests pass.
- [ ] Dependency and container scans have no unaccepted critical findings.
- [ ] Food allergy, food safety, health, and religious-compliance disclaimers are approved.
- [ ] Secrets are supplied through a secret manager and do not appear in logs.

## Reliability

- [ ] Target-hardware load test establishes concurrency, memory, and p95/p99 latency.
- [ ] Readiness, error rate, latency, refusal rate, and resource alerts exist.
- [ ] Corpus and artifact rollback has been exercised.
- [ ] Model and artifact caches survive routine restarts.

## Data and legal

- [ ] Project license selected by the repository owner.
- [ ] Corpus licenses/terms and attribution reviewed.
- [ ] Every new source record includes URL and capture timestamp.
- [ ] Content refresh, correction, and deletion procedures are documented.
