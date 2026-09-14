# Data snapshot

`raw/` contains the inherited Wikipedia, Wikibooks, and Around the World in 80
Cuisines records. The v2 runtime treats these JSON files as immutable source
inputs and records their SHA-256 hashes in every generated index manifest.

Important limitations of the inherited snapshot:

- it has no capture timestamps;
- ten URLs configured in the legacy notebook produced no records;
- upstream pages can change independently of this repository;
- the external text remains subject to its publishers' terms and licenses.

Do not silently replace these files. A refresh must be reviewed, preserve each
record's source URL and capture time, rebuild the index, and rerun evaluation.
`evaluation/benchmark_legacy.csv` is the inherited 45-question benchmark and is
retained only for comparison because several gold answers are not supported by
their source text. `evaluation/benchmark_v2.csv` is a curated regression set with
exact source URLs and required fact groups. `evaluation/no_answer.jsonl` adds
cases that should be refused.
