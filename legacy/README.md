# Legacy experiment

`RAG_Pipeline_Codebase.ipynb` and `original-results/` are preserved for audit and
historical comparison. They are not imported, executed, packaged, or deployed by
version 2.

Known legacy defects include parent ID collisions, silent ingestion gaps,
unversioned artifacts, biased evaluation, incorrect P4/P6 winner documentation,
and outputs without stable provenance. Do not copy legacy indexes back into
`artifacts/current`; build a fresh verified index with `rag-cuisine build-index`.
