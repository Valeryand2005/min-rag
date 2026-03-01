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
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Constants (proposal) ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKED_DOCS_PATH = DATA_DIR / "chunked_docs.json"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
CHUNK_META_PATH = DATA_DIR / "chunk_metadata.pkl"


def run_embed_and_index() -> None:
    """Load chunked docs, embed with all-MiniLM-L6-v2, build FAISS IndexFlatIP, save."""
    if not CHUNKED_DOCS_PATH.exists():
        raise FileNotFoundError(f"Run preprocess first. Missing {CHUNKED_DOCS_PATH}")

    with open(CHUNKED_DOCS_PATH, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)
    texts = [c["text"] for c in chunks]

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # Normalize for cosine = inner product
    model.max_seq_length = 512

    print("Embedding chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings).astype(np.float32)
    d = embeddings.shape[1]
    n = embeddings.shape[0]

    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    index_path = FAISS_INDEX_PATH.resolve()
    path_str = index_path.as_posix()  # forward slashes for FAISS C++ on Windows
    try:
        faiss.write_index(index, path_str)
    except RuntimeError as e:
        if "could not open" in str(e).lower() or "no such file" in str(e).lower():
            # Windows: path with unicode/spaces can fail; write to temp then copy
            with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
                tmp_path = tmp.name
            faiss.write_index(index, tmp_path)
            import shutil
            shutil.copy(tmp_path, index_path)
            Path(tmp_path).unlink(missing_ok=True)
        else:
            raise

    with open(CHUNK_META_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved FAISS index (n={n}, d={d}) to {FAISS_INDEX_PATH}")
    print(f"Saved chunk metadata to {CHUNK_META_PATH}")


if __name__ == "__main__":
    run_embed_and_index()
