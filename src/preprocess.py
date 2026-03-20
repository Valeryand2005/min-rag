import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME = "neural-bridge/rag-dataset-12000"
CHUNK_SIZE_TOKENS = 256
MAX_DOCUMENTS = 2500   # keep dataset small for faster indexing
MAX_TEST_EXAMPLES = 1000

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKED_DOCS_PATH = DATA_DIR / "chunked_docs.json"
TEST_SET_PATH = DATA_DIR / "test_set.json"


def get_paths(data_dir):
    # return paths for chunked docs and test set
    return data_dir / "chunked_docs.json", data_dir / "test_set.json"


def chunk_text_by_tokens(text, tokenizer, max_tokens):
    # split text into token-based chunks
    if not text or not text.strip():
        return []

    tokens = tokenizer.encode(text, add_special_tokens=False)

    # if text is small enough, keep it as one chunk
    if len(tokens) <= max_tokens:
        return [text] if text.strip() else []

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]

        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        if chunk_text.strip():
            chunks.append(chunk_text.strip())

        start = end

    return chunks


def run_preprocess(
    chunk_size=None,
    tokenizer_model=None,
    data_dir=None,
    chunked_docs_path=None,
    test_set_path=None,
):
    # set defaults
    chunk_size = chunk_size or CHUNK_SIZE_TOKENS
    tokenizer_model = tokenizer_model or "sentence-transformers/all-MiniLM-L6-v2"
    data_dir = data_dir or DATA_DIR
    chunked_path = chunked_docs_path or (data_dir / "chunked_docs.json")
    test_path = test_set_path or (data_dir / "test_set.json")

    # create folder if needed
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading dataset '{DATASET_NAME}'...")
    dataset = load_dataset(DATASET_NAME, split="train")

    total_rows = len(dataset)

    # split dataset into train/test
    n_docs = min(MAX_DOCUMENTS, total_rows)
    n_test = min(MAX_TEST_EXAMPLES, total_rows - n_docs)

    if n_test <= 0:
        n_docs = max(1, total_rows - MAX_TEST_EXAMPLES)
        n_test = total_rows - n_docs

    train_data = dataset.select(range(n_docs))
    test_data = dataset.select(range(n_docs, n_docs + n_test))

    # tokenizer for chunking
    from transformers import AutoTokenizer

    print(f"loading tokenizer ({tokenizer_model})...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    all_chunks = []
    doc_id = 0

    # build chunks
    for row in tqdm(train_data, desc="chunking"):
        context = row.get("context") or ""

        if not context.strip():
            continue

        chunks = chunk_text_by_tokens(context, tokenizer, chunk_size)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"doc{doc_id}_chunk{i}",
                "text": chunk,
                "doc_index": doc_id,
            })

        doc_id += 1

    # save chunks
    with open(chunked_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=0)

    # build test set
    test_list = [
        {
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "context": row.get("context", ""),
        }
        for row in test_data
    ]

    # save test set
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_list, f, ensure_ascii=False, indent=0)

    print(f"saved {len(all_chunks)} chunks to {chunked_path}")
    print(f"saved {len(test_list)} test examples to {test_path}")


if __name__ == "__main__":
    run_preprocess()