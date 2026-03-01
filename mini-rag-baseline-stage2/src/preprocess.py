"""
My own implementation for Innopolis University Mini-RAG project (Stage 2 Baseline)
Heavily modified and cleaned from srbhr/Local-RAG-with-Ollama.
- Removed all LangChain, Reflex, UI
- Added separate generation-only baseline from scratch
- Chunk size exactly 256 tokens as in proposal
- Clean script structure for fair comparison
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

from datasets import load_dataset
from tqdm import tqdm

# --- Constants (proposal) ---
DATASET_NAME = "neural-bridge/rag-dataset-12000"
CHUNK_SIZE_TOKENS = 256
MAX_DOCUMENTS = 2500  # 2000–3000 for speed
MAX_TEST_EXAMPLES = 1000
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKED_DOCS_PATH = DATA_DIR / "chunked_docs.json"
TEST_SET_PATH = DATA_DIR / "test_set.json"


def chunk_text_by_tokens(text: str, tokenizer: Any, max_tokens: int) -> List[str]:
    """
    Split text into chunks of at most max_tokens tokens.
    Uses simple word/sentence boundary when possible.
    """
    if not text or not text.strip():
        return []
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return [text] if text.strip() else []
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        start = end
    return chunks


def run_preprocess() -> None:
    """Load dataset, chunk contexts to 256 tokens, save to data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset '{DATASET_NAME}'...")
    dataset = load_dataset(DATASET_NAME, split="train")
    total_rows = len(dataset)
    # Use first MAX_DOCUMENTS for indexing, rest can be test or we split by index
    n_docs = min(MAX_DOCUMENTS, total_rows)
    n_test = min(MAX_TEST_EXAMPLES, total_rows - n_docs)
    if n_test <= 0:
        n_docs = max(1, total_rows - MAX_TEST_EXAMPLES)
        n_test = total_rows - n_docs

    # Train split: rows [0 : n_docs] — for chunking and index
    # Test split: rows [n_docs : n_docs + n_test] — for evaluation
    train_data = dataset.select(range(n_docs))
    test_data = dataset.select(range(n_docs, n_docs + n_test))

    # Tokenizer for chunk size (same as embedding model for consistency)
    from transformers import AutoTokenizer
    print("Loading tokenizer for chunking (all-MiniLM-L6-v2)...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    all_chunks: List[Dict[str, Any]] = []
    doc_id = 0
    for row in tqdm(train_data, desc="Chunking"):
        context = row.get("context") or ""
        if not context.strip():
            continue
        chunks = chunk_text_by_tokens(context, tokenizer, CHUNK_SIZE_TOKENS)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"doc{doc_id}_chunk{i}",
                "text": chunk,
                "doc_index": doc_id,
            })
        doc_id += 1

    with open(CHUNKED_DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=0)

    test_list = [
        {
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "context": row.get("context", ""),
        }
        for row in test_data
    ]
    with open(TEST_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(test_list, f, ensure_ascii=False, indent=0)

    print(f"Saved {len(all_chunks)} chunks to {CHUNKED_DOCS_PATH}")
    print(f"Saved {len(test_list)} test examples to {TEST_SET_PATH}")


if __name__ == "__main__":
    run_preprocess()
