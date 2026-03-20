import json
import pickle
import shutil
import tempfile
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL = "distilgpt2"
TOP_K = 5
DATA_DIR = Path(file).resolve().parent.parent / "data"
CHUNKED_DOCS_PATH = DATA_DIR / "chunked_docs.json"
TEST_SET_PATH = DATA_DIR / "test_set.json"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
CHUNK_META_PATH = DATA_DIR / "chunk_metadata.pkl"
MINI_RAG_PREDICTIONS_PATH = DATA_DIR / "mini_rag_predictions.json"
MAX_NEW_TOKENS = 100
MAX_CONTEXT_LENGTH = 400  # limit context so model input is not too big

# needed for some embedding models (like bge)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def get_prompt_gpt2(context, question):
    # simple prompt for gpt2-like models
    return f"Context: {context}\nQuestion: {question}\nAnswer:"


def get_prompt_chat(context, question, model_name):
    # build prompt depending on model format
    instruction = (
        "Use ONLY the following context to answer the question. "
        "If the context does not contain the answer, say so briefly.\n\n"
        f"Context: {context}\n\nQuestion: {question}"
    )

    if "TinyLlama" in model_name:
        return f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{instruction}\n<|assistant|>\n"

    if "phi" in model_name.lower():
        return f"Instruct: {instruction}\nOutput:"

    if "gemma" in model_name.lower():
        return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"

    return get_prompt_gpt2(context, question)


def truncate_context(context, tokenizer, max_length=400):
    # cut context so it fits into model input
    tokens = tokenizer.encode(context, truncation=True, max_length=max_length)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def _encode_query(embed_model, question, model_name):
    # convert question to embedding
    if "bge" in model_name.lower():
        question = BGE_QUERY_PREFIX + question

    q_emb = embed_model.encode([question], normalize_embeddings=True)
    return np.array(q_emb).astype(np.float32)


def _extract_chat_answer(text, model_name):
    # clean extra tokens from chat models output
    if "TinyLlama" in model_name:
        for stop in ["<|endoftext|>", "<|user|>", "<|assistant|>"]:
            if stop in text:
                text = text.split(stop)[0]

    if "phi" in model_name.lower():
        for stop in ["\nInstruct:", "\nOutput:"]:
            if stop in text:
                text = text.split(stop)[0]

    if "gemma" in model_name.lower():
        for stop in ["<end_of_turn>", "<start_of_turn>"]:
            if stop in text:
                text = text.split(stop)[0]

    return text.strip()


def run_mini_rag(
    embedding_model=None,
    generator_model=None,
    use_chat_format=False,
    test_set_path=None,
    faiss_index_path=None,
    chunk_meta_path=None,
    output_path=None,
    max_examples=None,
):
    # set defaults if nothing passed
    embedding_model = embedding_model or EMBEDDING_MODEL_NAME
    generator_model = generator_model or GENERATOR_MODEL
    test_path = test_set_path or TEST_SET_PATH
    index_path = faiss_index_path or FAISS_INDEX_PATH
    meta_path = chunk_meta_path or CHUNK_META_PATH
    out_path = output_path or MINI_RAG_PREDICTIONS_PATH

    # check files exist
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Run embed_and_index first.")

    if not test_path.exists():
        raise FileNotFoundError(f"Run preprocess first. Missing {test_path}")

    # load faiss index (fix for windows path issues)
    try:
        index = faiss.read_index(index_path.resolve().as_posix())
    except RuntimeError:
        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
            tmp_path = tmp.nameshutil.copy(index_path, tmp_path)
        index = faiss.read_index(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

    # load chunks and test data
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)

    with open(test_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    # limit number of examples if needed
    if max_examples is not None and max_examples > 0:
        test_set = test_set[:max_examples]

    print(f"loading embedding model '{embedding_model}'...")
    embed_model = SentenceTransformer(embedding_model)
    embed_model.max_seq_length = 512

    print(f"loading generator '{generator_model}'...")
    model = AutoModelForCausalLM.from_pretrained(
        generator_model,
        device_map="cpu",
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(generator_model, trust_remote_code=True)

    # fix for models without pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    # choose prompt format
    def get_prompt(c, q):
        return get_prompt_chat(c, q, generator_model) if use_chat_format else get_prompt_gpt2(c, q)

    results = []

    # main loop
    for item in tqdm(test_set, desc="mini-rag"):
        question = item.get("question") or ""

        # get embedding for question
        q_emb = _encode_query(embed_model, question, embedding_model)

        # search top-k chunks
        scores, indices = index.search(q_emb, min(TOP_K, index.ntotal))
        top_chunks = [chunks[i]["text"] for i in indices[0]]

        # build context
        context = "\n\n".join(top_chunks)
        context = truncate_context(context, tokenizer, MAX_CONTEXT_LENGTH)

        # generate answer
        prompt = get_prompt(context, question)
        out = pipe(prompt, return_full_text=False)

        answer = (out[0]["generated_text"] if out else "").strip()

        # clean output for chat models
        if use_chat_format:
            answer = _extract_chat_answer(answer, generator_model)

        results.append({
            "question": question,
            "prediction": answer,
            "ground_truth": item.get("answer", ""),
        })

    # save results
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=0)

    print(f"saved {len(results)} mini-rag predictions to {out_path}")


if name == "main":
    run_mini_rag()
