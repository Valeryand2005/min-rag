import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME = "neural-bridge/rag-dataset-12000"
CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 64  # overlap between consecutive chunks to avoid losing context at boundaries
MAX_DOCUMENTS = 2500   # documents to index
MAX_TEST_EXAMPLES = 1000  # test QA pairs drawn from the SAME documents that are indexed

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKED_DOCS_PATH = DATA_DIR / "chunked_docs.json"
TEST_SET_PATH = DATA_DIR / "test_set.json"


def get_paths(data_dir):
    # return paths for chunked docs and test set
    return data_dir / "chunked_docs.json", data_dir / "test_set.json"


def chunk_text_by_tokens(text, tokenizer, max_tokens, overlap=None):
    if not text or not text.strip():
        return []

    if overlap is None:
        overlap = CHUNK_OVERLAP_TOKENS

    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_tokens:
        return [text] if text.strip() else []

    chunks = []
    step = max(1, max_tokens - overlap)
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]

        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        if chunk_text.strip():
            chunks.append(chunk_text.strip())

        if end == len(tokens):
            break

        start += step

    return chunks


def run_preprocess(
    chunk_size=None,
    chunk_overlap=None,
    tokenizer_model=None,
    data_dir=None,
    chunked_docs_path=None,
    test_set_path=None,
):
    # set defaults
    chunk_size = chunk_size or CHUNK_SIZE_TOKENS
    chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP_TOKENS
    tokenizer_model = tokenizer_model or "sentence-transformers/all-MiniLM-L6-v2"
    data_dir = data_dir or DATA_DIR
    chunked_path = chunked_docs_path or (data_dir / "chunked_docs.json")
    test_path = test_set_path or (data_dir / "test_set.json")

    # create folder if needed
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading dataset '{DATASET_NAME}'...")
    dataset = load_dataset(DATASET_NAME, split="train")

    total_rows = len(dataset)

    # Use the same pool of documents for both indexing and test questions.
    # Previously the test set was taken from documents AFTER the indexed ones,
    # meaning the retriever could never find the right context (recall@5 = 0).
    # Now: index first MAX_DOCUMENTS docs, and build test QA pairs from those
    # same docs (capped at MAX_TEST_EXAMPLES).
    n_docs = min(MAX_DOCUMENTS, total_rows)
    docs_data = dataset.select(range(n_docs))

    # tokenizer for chunking
    from transformers import AutoTokenizer

    print(f"loading tokenizer ({tokenizer_model})...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    all_chunks = []
    test_list = []
    doc_id = 0

    # build chunks and test set from the same documents
    for row in tqdm(docs_data, desc="chunking"):
        context = row.get("context") or ""
        question = row.get("question", "")
        answer = row.get("answer", "")

        if not context.strip():
            continue

        chunks = chunk_text_by_tokens(context, tokenizer, chunk_size, overlap=chunk_overlap)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"doc{doc_id}_chunk{i}",
                "text": chunk,
                "doc_index": doc_id,
            })

        # only keep rows that have a question and answer for the test set
        if question.strip() and answer.strip() and len(test_list) < MAX_TEST_EXAMPLES:
            test_list.append({
                "question": question,
                "answer": answer,
                "context": context,
            })

        doc_id += 1

    # save chunks
    with open(chunked_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=0)

    # save test set
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_list, f, ensure_ascii=False, indent=0)

    print(f"chunk_size={chunk_size}, overlap={chunk_overlap}")
    print(f"saved {len(all_chunks)} chunks to {chunked_path}")
    print(f"saved {len(test_list)} test examples to {test_path}")


if __name__ == "__main__":
    run_preprocess()