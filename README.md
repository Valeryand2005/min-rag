# Mini-RAG: Stage 2 — Baseline Implementation

**Innopolis University · Mini-RAG: A Lightweight Retrieval-Augmented Generation System**

## Project Structure

```
mini-rag/
├── data/                    # Dataset and artifacts (created by scripts)
│   ├── chunked_docs.json    # Chunked documents (256 tokens)
│   ├── test_set.json        # Test questions + ground truth
│   ├── faiss.index          # FAISS index
│   ├── chunk_metadata.pkl   # Chunk texts for retrieval
│   ├── baseline_predictions.json
│   ├── mini_rag_predictions.json
│   └── evaluation_examples.json
├── src/
│   ├── __init__.py
│   ├── preprocess.py              # Chunking (256 tokens)
│   ├── embed_and_index.py        # all-MiniLM-L6-v2 + FAISS
│   ├── baseline_generation_only.py # Baseline without retrieval
│   ├── mini_rag.py                # Full Mini-RAG (retrieval + generation)
│   └── evaluate.py                # EM, F1, qualitative examples
├── reports/
│   └── baseline_report_template.md
├── requirements.txt
└── README.md
```

---

## How to Run

Run from the project root `mini-rag/`

### 0. Prepare enviroment 
```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```


### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess dataset

```bash
python -m src.preprocess
```

- Downloads `neural-bridge/rag-dataset-12000`
- Chunks contexts to **256 tokens** (tokenizer: all-MiniLM-L6-v2)
- Saves chunked docs and test set into `data/` (default: 2500 docs, 1000 test examples)

### 3. Build embeddings and FAISS index

```bash
python -m src.embed_and_index
```

- Embeds chunks with **sentence-transformers/all-MiniLM-L6-v2**
- Builds **FAISS** 
- Saves index and chunk metadata in `data/`

### 4. Baseline

```bash
python -m src.baseline_generation_only
```

- Loads 
- Prompt: `Question: {question}\nAnswer:`
- Writes `data/baseline_predictions.json`

### 5. Mini-RAG (retrieval + generation)

```bash
python -m src.mini_rag
```

- Retrieves **top-5** chunks per question
- Prompt: context + question 
- Writes `data/mini_rag_predictions.json`

### 6. Evaluation

```bash
python -m src.evaluate
```

- Computes **Exact Match (EM)** and **token-level F1**
- Calculates semantic similarity using all-MiniLM-L6-v2 embeddings
- Saves 5 qualitative examples to `data/evaluation_examples.json`
- Prints metrics and table 

---

## Configuration

- **Dataset:** neural-bridge/rag-dataset-12000
- **Chunk size:** 256 tokens
- **Embedding:** sentence-transformers/all-MiniLM-L6-v2
- **Vector store:** FAISS (faiss-cpu, IndexFlatIP)
- **Top-k:** 5
- **Generator (default):** distilgpt2 set `GENERATOR_MODEL` in `baseline_generation_only.py` and `mini_rag.py`.
- **Default scale:** 2500 documents, 1000 test examples (tunable in `src/preprocess.py`)
