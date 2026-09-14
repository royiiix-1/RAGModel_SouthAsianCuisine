# Notebook demonstration

Open `Interactive_Demo.ipynb` from the repository root:

```bash
jupyter lab notebooks/Interactive_Demo.ipynb
```

Run the cells from top to bottom. On a fresh clone, the artifact cell builds the
verified BGE/FAISS index. The first answer cell downloads and loads the pinned
Qwen2.5-0.5B-Instruct revision; subsequent questions reuse the loaded models.

Set `RAG_OFFLINE_MODE=true` before starting Jupyter only after all three pinned
model snapshots are cached. The notebook intentionally contains no scraping,
chunking, retrieval, or generation implementation: it calls the production
package so that the demonstration and API cannot silently diverge.
