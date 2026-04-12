import numpy as np
import json
from pathlib import Path
import re
import argparse
from sentence_transformers import SentenceTransformer, util


def normalize_answer(s):

    if s is None:
        return ""

    if not isinstance(s, str):
        s = str(s)

    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)

    return s


def exact_match(pred, gold):
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


# f1 for tokens
def token_f1(pred, gold):

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


# recall@k
# did the correct context appear in top-k retrieved chunks
def recall_at_k(retrieved_chunks, gold_context, k=5):
    if not retrieved_chunks or not gold_context:
        return 0.0

    gold_norm = normalize_answer(gold_context)

    for chunk in retrieved_chunks[:k]:
        chunk_norm = normalize_answer(chunk)
        if gold_norm[:100] in chunk_norm or chunk_norm[:100] in gold_norm:
            return 1.0

    return 0.0


# mean reciprocal rank for a single query
# 1/rank if relevant chunk is found, else 0
def reciprocal_rank(retrieved_chunks, gold_context):
    if not retrieved_chunks or not gold_context:
        return 0.0

    gold_norm = normalize_answer(gold_context)

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        chunk_norm = normalize_answer(chunk)
        if gold_norm[:100] in chunk_norm or chunk_norm[:100] in gold_norm:
            return 1.0 / rank

    return 0.0


# cosine similarity
def semantic_similarity(pred, gold, model):

    if not pred or not gold:
        return 0.0

    emb_pred = model.encode(pred, convert_to_tensor=True)
    emb_gold = model.encode(gold, convert_to_tensor=True)

    return util.pytorch_cos_sim(emb_pred, emb_gold).item()


def evaluate_predictions(
    predictions_path,
    n_examples=0,
    similarity_model_name="sentence-transformers/all-MiniLM-L6-v2",
):

    if not predictions_path.exists():
        raise FileNotFoundError(f"file not found: {predictions_path}")

    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # number of examples
    n = len(data) if n_examples <= 0 else min(n_examples, len(data))
    data = data[:n]

    # semantic model
    sim_model = SentenceTransformer(similarity_model_name)

    em_scores = []
    f1_scores = []
    sim_scores = []
    recall_scores = []
    rr_scores = []

    for item in data:
        pred = item.get("prediction", "")
        gold = item.get("ground_truth", "")

        em_scores.append(exact_match(pred, gold))

        _, _, f1 = token_f1(pred, gold)
        f1_scores.append(f1)

        sim_scores.append(semantic_similarity(pred, gold, sim_model))

        # retrieval metrics only available when retrieved_chunks is saved
        chunks = item.get("retrieved_chunks", [])
        context = item.get("context", "")

        if chunks and context:
            recall_scores.append(recall_at_k(chunks, context, k=5))
            rr_scores.append(reciprocal_rank(chunks, context))

    result = {
        "exact_match": float(np.mean(em_scores)),
        "token_f1": float(np.mean(f1_scores)),
        "semantic_similarity": float(np.mean(sim_scores)),
        "n_examples": n,
    }

    # add retrieval metrics only if we have data for them
    if recall_scores:
        result["recall_at_5"] = float(np.mean(recall_scores))
        result["mrr"] = float(np.mean(rr_scores))

    return result


def run_evaluate_experiment(
    predictions_path,
    n_examples=5,
    output_json=None,
):

    results = evaluate_predictions(predictions_path, n_examples=n_examples)

    print(f"\t results ({results['n_examples']} examples) ---")
    print(f"exact match:        {results['exact_match']:.4f}")
    print(f"f1 score:           {results['token_f1']:.4f}")
    print(f"semantic similarity:{results['semantic_similarity']:.4f}")

    # retrieval metrics print when available
    if "recall_at_5" in results:
        print(f"recall@5:           {results['recall_at_5']:.4f}")
        print(f"mrr:                {results['mrr']:.4f}")

    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, help="json file with predictions")
    parser.add_argument("-n", "--n_examples", type=int, default=5)
    parser.add_argument("-o", "--output", type=Path, default=None)

    args = parser.parse_args()

    run_evaluate_experiment(args.predictions, args.n_examples, args.output)
