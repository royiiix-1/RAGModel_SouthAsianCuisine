# South Asian Culinary RAG - Technical Overview

## Project Objective
This system provides grounded culinary information using a Retrieval-Augmented Generation (RAG) framework. The core challenge was optimizing a small-scale LLM (Qwen2.5-0.5B) to handle specialized South Asian datasets.

## Core Component Logic
* **Data Source**: Scraped raw content from Wikipedia, Wikibooks, and specialized blogs.
* **Chunking**: Implemented a Hierarchical (Parent-Child) strategy. Small chunks are used for precise vector search, while larger parent contexts are fed to the LLM for better coherence.
* **Retrieval**: Two-stage process using BGE-small-en embeddings (HNSW index) and a Cross-Encoder re-ranker (MS-MARCO) to filter top-k results.
* **Inference**: Uses Qwen2.5-0.5B-Instruct.

## Experimental Results (Ablation Study)
We conducted 6 decoupled experiments (P1-P6) to isolate the impact of Chunking, Retrieval, and Prompting.

### Key Performance Insight: P5 vs P6
* **Winner**: **Pipeline 5** (Advanced Retrieval + Advanced Chunking + Baseline Prompt).
* **Observation**: Pipeline 6 (Ultimate RAG) utilized an Advanced CoT Prompt which actually decreased performance. 
* **Conclusion**: For 0.5B parameter models, high-quality context is more beneficial than complex reasoning prompts. Small models can be "overwhelmed" by long Chain-of-Thought instructions.

## Directory Guide
* `data/raw/`: Raw scraped JSON corpus (Deliverable requirement).
* `data/chunked/`: Processed fragments (800-char vs Parent-Child).
* `data/indices/`: Pre-computed FAISS vector stores.
* `results/payloads/`: Generated answers for all 6 test pipelines.
* `results/plots/`: Performance charts for the poster.

## Demonstration Instructions (3-Minute Limit)
Do not run scraping or evaluation cells during the live demo. Use the pre-loaded inference cells at the end of the notebook:
1. **Interactive CLI**: Fast terminal-based Q&A.
2. **Gradio GUI**: Web-based interface showing the generated answer and the specific sources retrieved.

## Presentation Talking Points
1. **Decoupling**: We isolated every variable to prove exactly why our architecture works.
2. **Re-ranking Power**: The Cross-Encoder was critical in ensuring the small LLM received only relevant information.
3. **Model Capacity**: Discussing why P5 beat P6 shows advanced understanding of LLM attention constraints.
