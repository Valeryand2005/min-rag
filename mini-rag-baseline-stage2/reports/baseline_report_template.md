# Mini-RAG Stage 2 — Baseline Report

**Innopolis University · Mini-RAG: A Lightweight Retrieval-Augmented Generation System**

---

## 1. Setup

- **Dataset:** neural-bridge/rag-dataset-12000 (context, question, answer)
- **Chunk size:** 256 tokens
- **Embedding model:** sentence-transformers/all-MiniLM-L6-v2
- **Vector store:** FAISS (IndexFlatIP)
- **Top-k:** 5
- **Generator:** google/gemma-2b-it (CPU)
- **Test set size:** ___ (e.g. 500–1000)

---

## 2. Metrics

| System | Exact Match (EM) | Token-level F1 |
|--------|------------------|----------------|
| Baseline (generation only) | | |
| Mini-RAG (retrieval + generation) | | |

*(Fill after running `python -m src.evaluate`.)*

---

## 3. Qualitative Examples (5)

*(Paste or summarize from `data/evaluation_examples.json`.)*

| # | Question | Ground Truth | Baseline Prediction | Mini-RAG Prediction | Baseline F1 | Mini-RAG F1 |
|---|----------|--------------|---------------------|---------------------|-------------|-------------|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |

---

## 4. Conclusions (1–2 paragraphs)

- Compare baseline vs Mini-RAG.
- Note effect of retrieval on EM/F1 and on quality of answers.
- Mention limitations (model size, CPU-only, subset size).

---

*Report length: about 1–2 pages.*
