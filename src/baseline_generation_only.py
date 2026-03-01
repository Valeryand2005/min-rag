"""
My own implementation for Innopolis University Mini-RAG project (Stage 2 Baseline)
Heavily modified and cleaned from srbhr/Local-RAG-with-Ollama.
- Removed all LangChain, Reflex, UI
- Added separate generation-only baseline from scratch
- Chunk size exactly 256 tokens as in proposal
- Clean script structure for fair comparison
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# --- Constants (proposal) ---
# Default: open model, no Hugging Face login. For Gemma/Phi-3: run `huggingface-cli login` and set GENERATOR_MODEL.
GENERATOR_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_SET_PATH = DATA_DIR / "test_set.json"
BASELINE_PREDICTIONS_PATH = DATA_DIR / "baseline_predictions.json"
MAX_NEW_TOKENS = 150


def get_prompt_baseline(question: str) -> str:
    """Baseline: no retrieval. Prompt: Question: {question}\nAnswer:"""
    return f"Question: {question}\nAnswer:"


def run_baseline_generation_only() -> None:
    """Load test set, generate answers with generator only (no retrieval), save predictions."""
    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(f"Run preprocess first. Missing {TEST_SET_PATH}")

    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_set: List[Dict[str, Any]] = json.load(f)

    print(f"Loading generator '{GENERATOR_MODEL}' (CPU-friendly)...")
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
    for item in tqdm(test_set, desc="Baseline generation"):
        question = item.get("question") or ""
        prompt = get_prompt_baseline(question)
        out = pipe(prompt, return_full_text=False)
        answer = (out[0]["generated_text"] if out else "").strip()
        results.append({
            "question": question,
            "prediction": answer,
            "ground_truth": item.get("answer", ""),
        })

    with open(BASELINE_PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=0)
    print(f"Saved {len(results)} baseline predictions to {BASELINE_PREDICTIONS_PATH}")


if __name__ == "__main__":
    run_baseline_generation_only()
