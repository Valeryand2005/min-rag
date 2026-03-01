import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASELINE_PREDICTIONS_PATH = DATA_DIR / "baseline_predictions.json"
MINI_RAG_PREDICTIONS_PATH = DATA_DIR / "mini_rag_predictions.json"
REPORT_EXAMPLES_PATH = DATA_DIR / "evaluation_examples.json"
CHUNK_METADATA_PATH = DATA_DIR / "chunk_metadata.pkl"  
NUMBER_EVALUATIONS = 5

similarity_model = SentenceTransformer('all-MiniLM-L6-v2')


def normalize_answer(s: str) -> str:
    """Lowercase, strip, remove punctuation, collapse whitespace."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    
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


def semantic_similarity(pred: str, gold: str) -> float:
    """Semantic similarity using sentence embeddings."""
    if not pred or not gold:
        return 0.0
    emb_pred = similarity_model.encode(pred, convert_to_tensor=True)
    emb_gold = similarity_model.encode(gold, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb_pred, emb_gold).item()


def faithfulness_score(answer: str, context: str) -> float:
    """Measure how faithful answer is to context using BERTScore."""
    if not answer or not context:
        return 0.0
    try:
        P, R, F1 = bert_score([answer], [context], lang="en", verbose=False)
        return F1.item()
    except:
        return semantic_similarity(answer, context)


def retrieval_recall_at_k(retrieved_chunks: List[str], relevant_context: str, k: int = 5) -> float:
    """Check if relevant context is in top-k retrieved chunks."""
    if not retrieved_chunks or not relevant_context:
        return 0.0
    

    rel_emb = similarity_model.encode(relevant_context, convert_to_tensor=True)
    
    for i, chunk in enumerate(retrieved_chunks[:k]):
        chunk_emb = similarity_model.encode(chunk, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(rel_emb, chunk_emb).item()
        if sim > 0.8: 
            return 1.0
    return 0.0


def run_enhanced_evaluate() -> None:
    """Compute comprehensive metrics for RAG evaluation."""
    if not BASELINE_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Run baseline_generation_only first. Missing {BASELINE_PREDICTIONS_PATH}")
    if not MINI_RAG_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Run mini_rag first. Missing {MINI_RAG_PREDICTIONS_PATH}")

    with open(BASELINE_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        baseline: List[Dict[str, Any]] = json.load(f)
    with open(MINI_RAG_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        mini_rag: List[Dict[str, Any]] = json.load(f)

    if NUMBER_EVALUATIONS == 0 or NUMBER_EVALUATIONS is None:
        n = min(len(baseline), len(mini_rag))
    else:
        n = NUMBER_EVALUATIONS

    baseline = baseline[:n]
    mini_rag = mini_rag[:n]
    
    print(f"\n{'='*60}")
    print(f"Evaluating {n} examples")
    print(f"{'='*60}\n")


    metrics = {
        'exact_match': {'baseline': [], 'rag': []},
        'token_f1': {'baseline': [], 'rag': []},
        'semantic_sim': {'baseline': [], 'rag': []},
        'faithfulness': {'baseline': [], 'rag': []},
        'retrieval_recall': [],
    }

    for i in range(n):
        # Basic metrics
        em_b = exact_match(baseline[i]["prediction"], baseline[i]["ground_truth"])
        em_r = exact_match(mini_rag[i]["prediction"], mini_rag[i]["ground_truth"])
        
        f1_b = token_f1(baseline[i]["prediction"], baseline[i]["ground_truth"])[2]
        f1_r = token_f1(mini_rag[i]["prediction"], mini_rag[i]["ground_truth"])[2]
        
        sim_b = semantic_similarity(baseline[i]["prediction"], baseline[i]["ground_truth"])
        sim_r = semantic_similarity(mini_rag[i]["prediction"], mini_rag[i]["ground_truth"])
        
        faith_b = faithfulness_score(baseline[i]["prediction"], baseline[i].get("context", ""))
        faith_r = faithfulness_score(mini_rag[i]["prediction"], mini_rag[i].get("retrieved_context", ""))
      
        if "retrieved_chunks" in mini_rag[i]:
            recall = retrieval_recall_at_k(
                mini_rag[i]["retrieved_chunks"], 
                mini_rag[i].get("context", ""),
                k=5
            )
            metrics['retrieval_recall'].append(recall)
    
        metrics['exact_match']['baseline'].append(em_b)
        metrics['exact_match']['rag'].append(em_r)
        metrics['token_f1']['baseline'].append(f1_b)
        metrics['token_f1']['rag'].append(f1_r)
        metrics['semantic_sim']['baseline'].append(sim_b)
        metrics['semantic_sim']['rag'].append(sim_r)
        metrics['faithfulness']['baseline'].append(faith_b)
        metrics['faithfulness']['rag'].append(faith_r)

    results = {
        'exact_match': {
            'baseline': np.mean(metrics['exact_match']['baseline']),
            'rag': np.mean(metrics['exact_match']['rag']),
        },
        'token_f1': {
            'baseline': np.mean(metrics['token_f1']['baseline']),
            'rag': np.mean(metrics['token_f1']['rag']),
        },
        'semantic_similarity': {
            'baseline': np.mean(metrics['semantic_sim']['baseline']),
            'rag': np.mean(metrics['semantic_sim']['rag']),
        },
        'faithfulness': {
            'baseline': np.mean(metrics['faithfulness']['baseline']),
            'rag': np.mean(metrics['faithfulness']['rag']),
        },
        'retrieval_recall@5': np.mean(metrics['retrieval_recall']) if metrics['retrieval_recall'] else 0.0,
    }


    print("EVALUATION RESULTS")
    
    print(f"\n{'Metric':<25} {'Baseline':<15} {'Mini-RAG':<15} {'Improvement':<15}")
    
    for metric in ['exact_match', 'token_f1', 'semantic_similarity', 'faithfulness']:
        base = results[metric]['baseline']
        rag = results[metric]['rag']
        imp = rag - base
        print(f"{metric.replace('_', ' ').title():<25} {base:<15.4f} {rag:<15.4f} {imp:<+15.4f}")
    
    print(f"\nRetrieval Recall@5: {results['retrieval_recall@5']:.4f}")

    results['examples'] = []
    for i in range(min(5, n)):
        results['examples'].append({
            'question': baseline[i]['question'][:100] + "...",
            'ground_truth': baseline[i]['ground_truth'][:100] + "...",
            'baseline': baseline[i]['prediction'][:100] + "...",
            'rag': mini_rag[i]['prediction'][:100] + "...",
            'f1_baseline': metrics['token_f1']['baseline'][i],
            'f1_rag': metrics['token_f1']['rag'][i],
            'semantic_baseline': metrics['semantic_sim']['baseline'][i],
            'semantic_rag': metrics['semantic_sim']['rag'][i],
        })
    
    with open(REPORT_EXAMPLES_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    run_enhanced_evaluate()