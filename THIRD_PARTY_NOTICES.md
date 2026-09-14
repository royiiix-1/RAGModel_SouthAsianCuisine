# Third-party notices and release blocker

The application code does not yet declare a project license. The repository
owner must choose and add one before public distribution; this refactor does not
make that legal decision on the owner's behalf.

The inherited corpus contains excerpts derived from Wikipedia, Wikibooks, and
the Around the World in 80 Cuisines blog. Source URLs are retained in every raw
record and returned by the API. Before a public launch, the owner must verify
the applicable licenses/terms, attribution format, redistribution rights, and
deletion/update process for every source family.

Runtime models are separately licensed by their publishers:

- `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0 model repository)
- `BAAI/bge-small-en-v1.5` (MIT model repository)
- `cross-encoder/ms-marco-MiniLM-L6-v2` (Apache-2.0 model repository)

Refer to the exact pinned model revisions in `config.py` and review the model
repositories before redistributing model weights.
