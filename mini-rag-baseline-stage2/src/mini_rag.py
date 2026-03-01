"""
My own implementation for Innopolis University Mini-RAG project (Stage 2 Baseline)
Heavily modified and cleaned from srbhr/Local-RAG-with-Ollama.
- Removed all LangChain, Reflex, UI
- Added separate generation-only baseline from scratch
- Chunk size exactly 256 tokens as in proposal
- Clean script structure for fair comparison
"""

import json
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# --- Constants (proposal) ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Default: open model. For Gemma/Phi-3: run `huggingface-cli login` and set GENERATOR_MODEL.
GENERATOR_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TOP_K = 5
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKED_DOCS_PATH = DATA_DIR / "chunked_docs.json"
TEST_SET_PATH = DATA_DIR / "test_set.json"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
CHUNK_META_PATH = DATA_DIR / "chunk_metadata.pkl"
MINI_RAG_PREDICTIONS_PATH = DATA_DIR / "mini_rag_predictions.json"
MAX_NEW_TOKENS = 150


def get_prompt_rag(context: str, question: str) -> str:
    """RAG prompt: context + question."""
    return (
        "You are a helpful assistant. Use ONLY the following context to answer the question.\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Answer:"
    )


def run_mini_rag() -> None:
    """Load index + test set, retrieve top-k, generate with context, save predictions."""
    if not FAISS_INDEX_PATH.exists() or not CHUNK_META_PATH.exists():
        raise FileNotFoundError("Run embed_and_index first.")
    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(f"Run preprocess first. Missing {TEST_SET_PATH}")

    # FAISS C++ on Windows can fail on paths with unicode; copy to temp then read
    try:
        index = faiss.read_index(FAISS_INDEX_PATH.resolve().as_posix())
    except RuntimeError:
        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
            tmp_path = tmp.name
        shutil.copy(FAISS_INDEX_PATH, tmp_path)
        index = faiss.read_index(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
    with open(CHUNK_META_PATH, "rb") as f:
        chunks: List[Dict[str, Any]] = pickle.load(f)
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_set: List[Dict[str, Any]] = json.load(f)

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embed_model.max_seq_length = 512

    print(f"Loading generator '{GENERATOR_MODEL}'...")
    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_MODEL,
        device_map="cpu",
        low_cpu_mem_usage=True,
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    results: List[Dict[str, Any]] = []
    for item in tqdm(test_set, desc="Mini-RAG"):
        question = item.get("question") or ""
        q_emb = embed_model.encode([question], normalize_embeddings=True)
        q_emb = np.array(q_emb).astype(np.float32)
        scores, indices = index.search(q_emb, min(TOP_K, index.ntotal))
        top_chunks = [chunks[i]["text"] for i in indices[0]]
        context = "\n\n".join(top_chunks)
        prompt = get_prompt_rag(context, question)
        out = pipe(prompt, return_full_text=False)
        answer = (out[0]["generated_text"] if out else "").strip()
        results.append({
            "question": question,
            "prediction": answer,
            "ground_truth": item.get("answer", ""),
        })

    with open(MINI_RAG_PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=0)
    print(f"Saved {len(results)} Mini-RAG predictions to {MINI_RAG_PREDICTIONS_PATH}")


if __name__ == "__main__":
    run_mini_rag()
