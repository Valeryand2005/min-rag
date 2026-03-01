"""
My own implementation for Innopolis University Mini-RAG project (Stage 2 Baseline)
Heavily modified and cleaned from srbhr/Local-RAG-with-Ollama.
- Removed all LangChain, Reflex, UI
- Added separate generation-only baseline from scratch
- Chunk size exactly 256 tokens as in proposal
- Clean script structure for fair comparison
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

# --- Constants ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASELINE_PREDICTIONS_PATH = DATA_DIR / "baseline_predictions.json"
MINI_RAG_PREDICTIONS_PATH = DATA_DIR / "mini_rag_predictions.json"
REPORT_EXAMPLES_PATH = DATA_DIR / "evaluation_examples.json"


def normalize_answer(s: str) -> str:
    """Lowercase, strip, remove punctuation, collapse whitespace."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(pred: str, gold: str) -> float:
    """Exact Match (EM): 1 if normalized pred == normalized gold else 0."""
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def token_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    """Token-level F1: precision, recall, F1 between token sets."""
    pred_tokens = set(normalize_answer(pred).split())
    gold_tokens = set(normalize_answer(gold).split())
    if not gold_tokens:
        return (1.0, 1.0, 1.0) if not pred_tokens else (0.0, 0.0, 0.0)
    if not pred_tokens:
        return (0.0, 0.0, 0.0)
    common = pred_tokens & gold_tokens
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gold_tokens)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return (prec, rec, f1)


def run_evaluate() -> None:
    """Compute EM and token F1 for baseline and Mini-RAG, save 5 qualitative examples."""
    if not BASELINE_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Run baseline_generation_only first. Missing {BASELINE_PREDICTIONS_PATH}")
    if not MINI_RAG_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Run mini_rag first. Missing {MINI_RAG_PREDICTIONS_PATH}")

    with open(BASELINE_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        baseline: List[Dict[str, Any]] = json.load(f)
    with open(MINI_RAG_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        mini_rag: List[Dict[str, Any]] = json.load(f)

    n = min(len(baseline), len(mini_rag))
    baseline = baseline[:n]
    mini_rag = mini_rag[:n]

    # Metrics
    em_baseline = sum(exact_match(b["prediction"], b["ground_truth"]) for b in baseline) / n
    em_rag = sum(exact_match(m["prediction"], m["ground_truth"]) for m in mini_rag) / n
    f1_baseline = sum(token_f1(b["prediction"], b["ground_truth"])[2] for b in baseline) / n
    f1_rag = sum(token_f1(m["prediction"], m["ground_truth"])[2] for m in mini_rag) / n

    print("--- Metrics (n={}) ---".format(n))
    print("Baseline (generation only): EM = {:.4f}, F1 = {:.4f}".format(em_baseline, f1_baseline))
    print("Mini-RAG (retrieval + generation): EM = {:.4f}, F1 = {:.4f}".format(em_rag, f1_rag))

    # 5 qualitative examples (first 5 or pick mixed good/bad by F1)
    indices = list(range(min(5, n)))
    examples = []
    for i in indices:
        examples.append({
            "question": baseline[i]["question"],
            "ground_truth": baseline[i]["ground_truth"],
            "baseline_prediction": baseline[i]["prediction"],
            "mini_rag_prediction": mini_rag[i]["prediction"],
            "baseline_f1": token_f1(baseline[i]["prediction"], baseline[i]["ground_truth"])[2],
            "mini_rag_f1": token_f1(mini_rag[i]["prediction"], mini_rag[i]["ground_truth"])[2],
        })

    with open(REPORT_EXAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"Saved 5 examples to {REPORT_EXAMPLES_PATH}")

    # Also print table for report
    print("\n--- 5 examples (for report) ---")
    for ex in examples:
        print("Q:", ex["question"][:80] + "..." if len(ex["question"]) > 80 else ex["question"])
        print("GT:", ex["ground_truth"][:80] + "..." if len(ex["ground_truth"]) > 80 else ex["ground_truth"])
        print("Baseline:", ex["baseline_prediction"][:80] + "..." if len(ex["baseline_prediction"]) > 80 else ex["baseline_prediction"])
        print("Mini-RAG:", ex["mini_rag_prediction"][:80] + "..." if len(ex["mini_rag_prediction"]) > 80 else ex["mini_rag_prediction"])
        print("F1 baseline / Mini-RAG:", ex["baseline_f1"], "/", ex["mini_rag_f1"])
        print("-" * 40)


if __name__ == "__main__":
    run_evaluate()
