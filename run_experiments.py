import sys
from pathlib import Path
import json
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from experiment_config import (
    ALL_EXPERIMENTS,
    QUICK_EXPERIMENTS,
    ExperimentConfig,
    DATA_DIR,
)
from preprocess import run_preprocess
from embed_and_index import run_embed_and_index
from mini_rag import run_mini_rag
from baseline_generation_only import run_no_retrieval
from evaluate_experiment import evaluate_predictions



def run_single_experiment(config: ExperimentConfig, n_eval: int = 5) -> list:
    # returns two rows: [rag_metrics, no_retrieval_metrics]
    exp_dir = config.exp_dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    chunked_path = exp_dir / "chunked_docs.json"
    test_path = exp_dir / "test_set.json"
    index_path = exp_dir / "faiss.index"
    meta_path = exp_dir / "chunk_metadata.pkl"
    pred_path = exp_dir / "mini_rag_predictions.json"
    no_ret_path = exp_dir / "no_retrieval_predictions.json"
    metrics_path = exp_dir / "metrics.json"

    print(f"\n{'='*60}")
    print(f"Experiment: {config.name}")
    print(f"  embedding={config.embedding_model}")
    print(f"  chunk_size={config.chunk_size}")
    print(f"  generator={config.generator_model}")
    print(f"{'='*60}\n")

    run_preprocess(
        chunk_size=config.chunk_size,
        tokenizer_model=config.tokenizer_model,
        data_dir=exp_dir,
        chunked_docs_path=chunked_path,
        test_set_path=test_path,
    )

    run_embed_and_index(
        embedding_model=config.embedding_model,
        chunked_docs_path=chunked_path,
        data_dir=exp_dir,
        faiss_index_path=index_path,
        chunk_meta_path=meta_path,
    )

    run_mini_rag(
        embedding_model=config.embedding_model,
        generator_model=config.generator_model,
        use_chat_format=config.use_chat_format,
        test_set_path=test_path,
        faiss_index_path=index_path,
        chunk_meta_path=meta_path,
        output_path=pred_path,
        max_examples=n_eval if n_eval > 0 and n_eval < 1000 else None,
    )

    # no retrieval
    run_no_retrieval(
        generator_model=config.generator_model,
        test_set_path=test_path,
        output_path=no_ret_path,
        max_examples=n_eval if n_eval > 0 and n_eval < 1000 else None,
    )

    metrics = evaluate_predictions(pred_path, n_examples=n_eval)
    metrics["experiment"] = config.name
    metrics["embedding_model"] = config.embedding_model
    metrics["chunk_size"] = config.chunk_size
    metrics["generator_model"] = config.generator_model
    metrics["mode"] = "rag"

    metrics_no_ret = evaluate_predictions(no_ret_path, n_examples=n_eval)
    metrics_no_ret["experiment"] = config.name + " (no retrieval)"
    metrics_no_ret["embedding_model"] = "-"
    metrics_no_ret["chunk_size"] = config.chunk_size
    metrics_no_ret["generator_model"] = config.generator_model
    metrics_no_ret["mode"] = "no_retrieval"

    with open(metrics_path, "w") as f:
        json.dump({"rag": metrics, "no_retrieval": metrics_no_ret}, f, indent=2)

    return [metrics, metrics_no_ret]



def run_all_experiments(experiments: list = None, n_eval: int = 5, quick: bool = False) -> list:
    experiments = experiments or (QUICK_EXPERIMENTS if quick else ALL_EXPERIMENTS)
    results = []
    for config in experiments:
        try:
            # returns [rag_row, no_retrieval_row]
            rows = run_single_experiment(config, n_eval=n_eval)
            results.extend(rows)
        except Exception as e:
            print(f"Experiment {config.name} FAILED: {e}")
            results.append({
                "experiment": config.name,
                "error": str(e),
                "exact_match": None,
                "token_f1": None,
                "semantic_similarity": None,
            })
    return results


def print_results_table(results: list) -> None:
    print("\n   experiment results\n")

    for r in results:
        if "error" in r:
            print(f"- {r['experiment']} -> FAILED")
            print(f"  error: {r['error']}")
            continue

        em = r.get("exact_match")
        f1 = r.get("token_f1")
        sim = r.get("semantic_similarity")
        mode = r.get("mode", "rag")

        print(f"- {r['experiment']}  [{mode}]")
        print(f"  exact match: {em:.4f}" if em is not None else "  exact match: N/A")
        print(f"  f1:          {f1:.4f}" if f1 is not None else "  f1: N/A")
        print(f"  similarity:  {sim:.4f}" if sim is not None else "  similarity: N/A")

        # retrieval metrics only present for rag rows
        if "recall_at_5" in r:
            print(f"  recall@5:    {r['recall_at_5']:.4f}")
            print(f"  mrr:         {r['mrr']:.4f}")

        print()



def save_results_table(results: list, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)


    # Markdown table
    md_path = path.with_suffix(".md")
    lines = [
        "# Mini-RAG Experiment Results",
        "",
        "| Experiment | Mode | Exact Match | Token F1 | Semantic Similarity | Recall@5 | MRR |",
        "|------------|------|-------------|----------|---------------------|----------|-----|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['experiment']} | - | FAILED | - | - | - | - |")
        else:
            em = r.get("exact_match", 0)
            f1 = r.get("token_f1", 0)
            sim = r.get("semantic_similarity", 0)
            recall = r.get("recall_at_5")
            mrr = r.get("mrr")
            mode = r.get("mode", "rag")

            em_s = f"{em:.4f}" if em is not None else "N/A"
            f1_s = f"{f1:.4f}" if f1 is not None else "N/A"
            sim_s = f"{sim:.4f}" if sim is not None else "N/A"
            recall_s = f"{recall:.4f}" if recall is not None else "-"
            mrr_s = f"{mrr:.4f}" if mrr is not None else "-"

            lines.append(f"| {r['experiment']} | {mode} | {em_s} | {f1_s} | {sim_s} | {recall_s} | {mrr_s} |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {json_path} and {md_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run only quick experiments (baseline + chunk sizes)")
    parser.add_argument("--skip", type=str, nargs="*", default=[], help="Skip experiments by name (e.g. gen_gemma-2b)")
    parser.add_argument("-n", "--n_eval", type=int, default=5, help="Number of examples to evaluate (0=all)")
    parser.add_argument("-o", "--output", type=Path, default=DATA_DIR / "experiment_results")
    args = parser.parse_args()

    experiments = QUICK_EXPERIMENTS if args.quick else ALL_EXPERIMENTS
    experiments = [e for e in experiments if e.name not in args.skip]

    n_eval = args.n_eval if args.n_eval > 0 else 1000
    results = run_all_experiments(experiments=experiments, n_eval=n_eval, quick=False)
    print_results_table(results)
    save_results_table(results, args.output)
