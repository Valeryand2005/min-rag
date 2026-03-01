import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import warnings

from tqdm import tqdm
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('baseline_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# GENERATOR_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

GENERATOR_MODEL = "distilgpt2"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_SET_PATH = DATA_DIR / "test_set.json"
BASELINE_PREDICTIONS_PATH = DATA_DIR / "baseline_predictions.json"
MAX_NEW_TOKENS = 100
BATCH_SIZE = 100


def get_prompt_baseline(question: str) -> str:
    """Baseline: no retrieval. Prompt: Question: {question}\nAnswer:"""
    return f"Question: {question}\nAnswer:"


def setup_hf_token():
    """Setup Hugging Face token if available"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        try:
            from huggingface_hub import get_token
            hf_token = get_token()
        except:
            pass
    
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        logger.info("HF_TOKEN configured")
    else:
        logger.warning("No HF_TOKEN found. Using unauthenticated requests (rate limits apply)")


def run_baseline_generation_only() -> None:
    """Load test set, generate answers with generator only (no retrieval), save predictions."""
    logger.info("Starting baseline generation")
    
    # Setup HF token
    setup_hf_token()
    
    # Check test set
    if not TEST_SET_PATH.exists():
        error_msg = f"Run preprocess first. Missing {TEST_SET_PATH}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load test data
    logger.info(f"Loading test set from {TEST_SET_PATH}")
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_set: List[Dict[str, Any]] = json.load(f)
    logger.info(f"Loaded {len(test_set)} test items")

    # Load model
    logger.info(f"Loading generator '{GENERATOR_MODEL}' (CPU-friendly)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_MODEL,
            device_map="cpu",
            low_cpu_mem_usage=True,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            GENERATOR_MODEL,
            trust_remote_code=True,
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Setup pipeline
    logger.info(f"Setting up pipeline (batch_size={BATCH_SIZE}, max_new_tokens={MAX_NEW_TOKENS})")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=BATCH_SIZE,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Generate answers
    results: List[Dict[str, Any]] = []
    logger.info("Starting generation")
    
    for idx, item in enumerate(tqdm(test_set, desc="Baseline generation")):
        question = item.get("question") or ""
        prompt = get_prompt_baseline(question)
        
        try:
            out = pipe(prompt, return_full_text=False)
            answer = (out[0]["generated_text"] if out else "").strip()
        except Exception as e:
            logger.error(f"Error generating answer for item {idx}: {e}")
            answer = ""
        
        results.append({
            "question": question,
            "prediction": answer,
            "ground_truth": item.get("answer", ""),
        })
        
        # Log progress every 100 items
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(test_set)} items")

    # Save results
    logger.info(f"Saving {len(results)} predictions to {BASELINE_PREDICTIONS_PATH}")
    try:
        with open(BASELINE_PREDICTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Results saved successfully")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise

    # Log sample
    if results:
        logger.info("Sample prediction:")
        logger.info(f"  Question: {results[0]['question'][:100]}...")
        logger.info(f"  Prediction: {results[0]['prediction'][:100]}...")
        logger.info(f"  Ground truth: {results[0]['ground_truth'][:100]}...")


if __name__ == "__main__":
    try:
        run_baseline_generation_only()
        logger.info("Script completed successfully")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)