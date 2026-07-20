"""
Programmatic Evaluate API - Clean Python interface for running evaluations.

Usage:
    from evaluate_api import evaluate, evaluate_single

    # Run a full suite
    results = evaluate(
        models=["gpt4o-azure", "qwen-31-onprem"],
        suite="smoke",
    )

    # Run a single test on a single model
    result = evaluate_single(
        model="gpt4o-azure",
        test="turkish_grammar",
        max_samples=5,
    )
"""
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# override=True ensures .env values always win over stale OS env vars
# (e.g. NO_PROXY set without .denizbank.com in a parent shell)
load_dotenv(override=True)

from pipeline_runner import EvaluationPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def evaluate(
    models: List[str],
    suite: str = "smoke",
    tests: Optional[List[str]] = None,
    config_path: str = "config/models.yaml",
    judge_model: Optional[str] = None,
    output_path: Optional[str] = None,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    custom_dataset_path: Optional[str] = None,
    custom_dataset_name: Optional[str] = None,
    custom_dataset_kind: Optional[str] = None,
    run=None,
) -> Dict[str, Any]:
    """Run a full evaluation suite programmatically.
    
    Args:
        models: List of model keys to evaluate.
        suite: Test suite name (from config/tests.yaml).
        tests: Optional subset of test names to run within the suite.
        config_path: Path to models.yaml config.
        judge_model: Optional judge model override.
        output_path: Optional file path to save results.
        parallel: Whether to run models in parallel.
        max_workers: Max parallel workers (only if parallel=True).
        temperature: Global temperature override.
        top_p: Global top_p override.
        max_tokens: Global max_tokens override.
        
    Returns:
        Dict with full evaluation results (same as pipeline output).
    """
    runtime_overrides = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    pipeline = EvaluationPipeline(
        config_path=config_path,
        judge_model_key=judge_model,
        runtime_overrides=runtime_overrides,
        run=run,
    )

    if custom_dataset_path:
        results = pipeline.run_custom_dataset_evaluation(
            model_keys=models,
            dataset_path=custom_dataset_path,
            dataset_name=custom_dataset_name,
            dataset_kind=custom_dataset_kind,
            output_path=output_path,
            parallel=parallel,
            max_workers=max_workers,
        )
    elif parallel:
        results = pipeline.run_full_evaluation_parallel(
            model_keys=models,
            test_suite=suite,
            selected_tests=tests,
            output_path=output_path,
            max_workers=max_workers,
        )
    else:
        results = pipeline.run_full_evaluation(
            model_keys=models,
            test_suite=suite,
            selected_tests=tests,
            output_path=output_path,
        )

    if output_path:
        pipeline.save_results(output_path)
        logger.info(f"Results saved to {output_path}")

    return results


def evaluate_single(
    model: str,
    test: str,
    max_samples: Optional[int] = None,
    config_path: str = "config/models.yaml",
    judge_model: Optional[str] = None,
    temperature: Optional[float] = None,
    run=None,
) -> Dict[str, Any]:
    """Run a single test on a single model.
    
    Lightweight alternative to evaluate() for quick checks.
    
    Args:
        model: Model key to evaluate.
        test: Test name (e.g., 'turkish_grammar').
        max_samples: Limit dataset size (None = all).
        config_path: Path to models config.
        judge_model: Optional judge model override.
        temperature: Temperature override.
        
    Returns:
        Dict with test results for the specified model+test.
    """
    runtime_overrides = {"temperature": temperature} if temperature is not None else {}

    pipeline = EvaluationPipeline(
        config_path=config_path,
        judge_model_key=judge_model,
        runtime_overrides=runtime_overrides,
        run=run,
    )

    # Build test mapping from registry
    test_mapping = pipeline._build_test_mapping()
    if test not in test_mapping:
        available = sorted(test_mapping.keys())
        raise ValueError(f"Unknown test '{test}'. Available: {available}")

    dataset_path, test_func = test_mapping[test]
    
    # Load dataset
    dataset = pipeline.load_dataset(
        dataset_path,
        max_samples,
        test_name=test,
        test_func=test_func,
    )
    
    # Initialize model
    model_adapter = pipeline.initialize_model(model)
    model_adapter.reset_stats()
    
    # Initialize judge if needed
    judge = None
    if not test.startswith("embedding_"):
        judge = pipeline.initialize_judge()
    
    # Determine test function signature and call
    if test.startswith("embedding_"):
        result = test_func(model_adapter, dataset, test)
    elif test_func.__name__ in ("run_qa_test", "run_reasoning_test"):
        result = test_func(model_adapter, dataset, judge, test)
    else:
        result = test_func(model_adapter, dataset, judge, test)
    
    return {
        "model": model,
        "test": test,
        "result": result,
    }


def list_available_tests() -> List[str]:
    """List all registered test names from task_registry.yaml."""
    import yaml
    registry_path = Path("config/task_registry.yaml")
    if registry_path.exists():
        with open(registry_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return sorted(data.get("tasks", {}).keys())
    return []


def list_available_suites() -> List[str]:
    """List all test suite names from config/tests.yaml."""
    import yaml
    try:
        with open("config/tests.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        return sorted(config.get("test_suites", {}).keys())
    except FileNotFoundError:
        return []
