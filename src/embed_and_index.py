"""
Build embeddings and FAISS index from chunked docs.
Uses normalized embeddings so IndexFlatIP = cosine sim.
"""

import json
import pickle
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKED_DOCS_PATH = DATA_DIR / "chunked_docs.json"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
CHUNK_META_PATH = DATA_DIR / "chunk_metadata.pkl"


def run_embed_and_index(
    embedding_model=None,
    chunked_docs_path=None,
    data_dir=None,
    faiss_index_path=None,
    chunk_meta_path=None,
):
    # use defaults if nothing is passed
    embedding_model = embedding_model or EMBEDDING_MODEL_NAME
    chunked_path = chunked_docs_path or CHUNKED_DOCS_PATH
    data_dir = data_dir or DATA_DIR
    index_path = faiss_index_path or FAISS_INDEX_PATH
    meta_path = chunk_meta_path or CHUNK_META_PATH

    # check that chunked docs exist
    if not chunked_path.exists():
        raise FileNotFoundError(
            f"Run preprocess first. No chunk file found at {chunked_path}"
        )

    # load chunked documents
    with open(chunked_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    # take only text from each chunk
    texts = [chunk["text"] for chunk in chunks]

    # load embedding model
    print(f"Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)
    model.max_seq_length = 512 

    # create embeddings
    print("Creating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,  # for cosine similarity
    )

    # convert to numpy float32 faiss needs
    embeddings = np.array(embeddings).astype(np.float32)
    # embedding size
    dim = embeddings.shape[1]
    # number of chunks
    count = embeddings.shape[0]

    # create FAISS index
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # make sure folder exists
    data_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_path.resolve()

    # save index (sometimes faiss fails if path is weird)
    path_str = index_path.as_posix()
    try:
        faiss.write_index(index, path_str)
    except RuntimeError as e:
        # fallback: save to temp file and then copy
        if "could not open" in str(e).lower() or "no such file" in str(e).lower():
            with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
                tmp_path = tmp.name

            faiss.write_index(index, tmp_path)

            import shutil
            shutil.copy(tmp_path, index_path)

            Path(tmp_path).unlink(missing_ok=True)
        else:
            raise

    # save metadata -- chunks
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved FAISS index (count={count}, dim={dim}) -> {index_path}")
    print(f"Saved chunk metadata -> {meta_path}")


if __name__ == "__main__":
    run_embed_and_index()