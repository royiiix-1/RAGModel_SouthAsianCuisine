# Security policy

## Deployment requirements

- Set a long random `RAG_API_KEY` before binding to a non-loopback interface.
- Terminate TLS at a trusted reverse proxy and apply network-level rate limits.
- Keep model and embedding revisions pinned; do not enable remote model code.
- Treat all indexed pages as untrusted data and review corpus refreshes.
- Do not log full questions, generated answers, API keys, or source contents.
- Rebuild artifacts after any source or index-configuration change.

This service is a culinary information tool, not medical or nutritional advice.
Do not use generated output as the sole authority for allergies, food safety,
religious compliance, or health decisions.

Report suspected vulnerabilities privately to the repository owner. Do not put
credentials, personal data, or exploit details in a public issue.
