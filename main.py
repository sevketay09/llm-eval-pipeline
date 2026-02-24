"""
Main entry point for LLM Evaluation Pipeline
"""
import argparse
import sys
import yaml
from dotenv import load_dotenv

load_dotenv()  # Load .env file before anything else

from pipeline_runner import EvaluationPipeline
from utils.evaluation_store import DEFAULT_STORE_PATH


def _load_available_suites() -> list[str]:
    """Load suite names from config/tests.yaml for CLI validation."""
    default_suites = [
        "smoke", "full", "full_local", "fintech_only", "advanced", "benchmarks",
        "regression", "dashboard_custom", "embedding_basic", "embedding_full", "embedding_turkish"
    ]

    try:
        with open("config/tests.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        suites = config.get("test_suites", {})
        if isinstance(suites, dict) and suites:
            return sorted(suites.keys())
    except Exception:
        pass

    return default_suites


def main():
    available_suites = _load_available_suites()
    default_suite = "full" if "full" in available_suites else available_suites[0]

    parser = argparse.ArgumentParser(
        description="LLM Evaluation Pipeline - Türkçe ve Fintech odaklı"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model keys to evaluate (e.g., gpt4o-azure llama-3-70b-vllm)"
    )
    
    parser.add_argument(
        "--suite",
        type=str,
        default=default_suite,
        choices=available_suites,
        help="Test suite to run"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output file path (default: {DEFAULT_STORE_PATH})"
    )

    parser.add_argument(
        "--parallel-models",
        action="store_true",
        help="Run models in parallel (one process per model)"
    )

    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="Max parallel workers for --parallel-models (default: number of models)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/models.yaml",
        help="Path to models config file"
    )
    
    parser.add_argument(
        "--judge",
        type=str,
        default=None,
        help="Model key to use as judge (default: from config, fallback to first evaluated model)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Global temperature override for all selected LLM"
    )

    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=None,
        help="Global top_p override for all selected LLM"
    )

    parser.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=None,
        help="Global max_tokens override for all selected LLM"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("Initializing evaluation pipeline...")
    runtime_overrides = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    # Use --judge if specified, otherwise let pipeline use config default
    pipeline = EvaluationPipeline(
        config_path=args.config,
        judge_model_key=args.judge,
        runtime_overrides=runtime_overrides
    )
    
    # Decide output path early for incremental saves
    if args.output:
        output_path = args.output
    else:
        output_path = DEFAULT_STORE_PATH

    # Run evaluation
    try:
        if args.parallel_models:
            results = pipeline.run_full_evaluation_parallel(
                model_keys=args.models,
                test_suite=args.suite,
                output_path=output_path,
                max_workers=args.parallel_workers
            )
        else:
            results = pipeline.run_full_evaluation(
                model_keys=args.models,
                test_suite=args.suite,
                output_path=output_path
            )
        
        # Print summary
        pipeline.print_summary()

        # Save results
        pipeline.save_results(output_path)

        # Check if models pass thresholds
        thresholds = pipeline.test_config.get("thresholds", {})
        suite_config = pipeline.test_config.get("test_suites", {}).get(args.suite, {})
        configured_tests = suite_config.get("tests", [])

        passed = []
        failed = []
        run_had_test_errors = False

        # Embedding suites use test-specific thresholds
        if args.suite.startswith("embedding"):
            sts_threshold = thresholds.get("embedding_spearman_correlation", 0.70)
            xsts_threshold = thresholds.get("embedding_spearman_correlation_crosslingual", 0.68)
            ndcg_threshold = thresholds.get("embedding_ndcg_at_10", 0.75)
            hardneg_ndcg_threshold = thresholds.get("embedding_ndcg_at_10_hardneg", 0.80)
            clustering_threshold = thresholds.get("embedding_clustering_accuracy", 0.80)
            regulatory_clustering_threshold = thresholds.get("embedding_clustering_accuracy_regulatory", 0.82)

            embedding_tests = [
                test_name for test_name in configured_tests
                if isinstance(test_name, str) and test_name.startswith("embedding_")
            ]

            for model_key, model_data in results["models"].items():
                tests = model_data.get("tests", {})
                model_pass = True
                model_has_error = False

                # Required embedding tests must exist and contain summaries
                for embedding_test in embedding_tests:
                    test_data = tests.get(embedding_test)
                    if not isinstance(test_data, dict):
                        model_has_error = True
                        run_had_test_errors = True
                        break
                    if test_data.get("error"):
                        model_has_error = True
                        run_had_test_errors = True
                        break
                    if "summary" not in test_data:
                        model_has_error = True
                        run_had_test_errors = True
                        break

                if model_has_error:
                    failed.append(model_key)
                    continue

                if "embedding_sts" in tests and "summary" in tests["embedding_sts"]:
                    score = tests["embedding_sts"]["summary"].get("spearman_correlation", 0)
                    model_pass = model_pass and (score >= sts_threshold)

                if "embedding_sts_crosslingual" in tests and "summary" in tests["embedding_sts_crosslingual"]:
                    score = tests["embedding_sts_crosslingual"]["summary"].get("spearman_correlation", 0)
                    model_pass = model_pass and (score >= xsts_threshold)

                if "embedding_retrieval" in tests and "summary" in tests["embedding_retrieval"]:
                    ndcg_data = tests["embedding_retrieval"]["summary"].get("ndcg", {})
                    ndcg_at_10 = ndcg_data.get(10, ndcg_data.get("10", 0)) if isinstance(ndcg_data, dict) else 0
                    model_pass = model_pass and (ndcg_at_10 >= ndcg_threshold)

                if "embedding_retrieval_hardneg" in tests and "summary" in tests["embedding_retrieval_hardneg"]:
                    ndcg_data = tests["embedding_retrieval_hardneg"]["summary"].get("ndcg", {})
                    ndcg_at_10 = ndcg_data.get(10, ndcg_data.get("10", 0)) if isinstance(ndcg_data, dict) else 0
                    model_pass = model_pass and (ndcg_at_10 >= hardneg_ndcg_threshold)

                if "embedding_clustering" in tests and "summary" in tests["embedding_clustering"]:
                    acc = tests["embedding_clustering"]["summary"].get("avg_accuracy", 0)
                    model_pass = model_pass and (acc >= clustering_threshold)

                if "embedding_clustering_regulatory" in tests and "summary" in tests["embedding_clustering_regulatory"]:
                    acc = tests["embedding_clustering_regulatory"]["summary"].get("avg_accuracy", 0)
                    model_pass = model_pass and (acc >= regulatory_clustering_threshold)

                if model_pass:
                    passed.append(model_key)
                else:
                    failed.append(model_key)

            print("\n📌 Embedding thresholds:")
            print(f"  - embedding_spearman_correlation >= {sts_threshold}")
            print(f"  - embedding_spearman_correlation_crosslingual >= {xsts_threshold}")
            print(f"  - embedding_ndcg_at_10 >= {ndcg_threshold}")
            print(f"  - embedding_ndcg_at_10_hardneg >= {hardneg_ndcg_threshold}")
            print(f"  - embedding_clustering_accuracy >= {clustering_threshold}")
            print(f"  - embedding_clustering_accuracy_regulatory >= {regulatory_clustering_threshold}")

        else:
            overall_threshold = thresholds.get("overall_score", 0.70)

            for model_key, comparison in results["summary"]["model_comparison"].items():
                model_tests = results.get("models", {}).get(model_key, {}).get("tests", {})
                model_has_error = any(
                    isinstance(test_data, dict) and test_data.get("error")
                    for test_data in model_tests.values()
                )
                if model_has_error:
                    run_had_test_errors = True
                    failed.append(model_key)
                    continue

                if comparison["overall_score"] >= overall_threshold:
                    passed.append(model_key)
                else:
                    failed.append(model_key)

            print(f"\n📌 Overall threshold: {overall_threshold}")

        if passed:
            print(f"\n✅ Models passing threshold: {', '.join(passed)}")
        if failed:
            print(f"\n❌ Models below threshold: {', '.join(failed)}")

        if run_had_test_errors:
            print("\n❌ Evaluation completed with test errors (model threshold pass is disabled when test execution fails).")
            sys.exit(2)

        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
