#!/usr/bin/env python3
"""
Contamination Runner — test seti sızıntı (memorization) kontrolü.

Kullanım:
    python run_contamination.py --model demo-model --dataset eval_datasets/regression/golden.json
    python run_contamination.py --model gpt-4o --dataset eval_datasets/regression/golden.json --sample 20
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import yaml

from adapters.unified_adapter import UnifiedLLMAdapter
from analysis.contamination import DEFAULT_THRESHOLD, ContaminationChecker


def load_config(config_path: str = "config/models.yaml") -> dict:
    """Load models config with ${ENV_VAR} expansion (same as run_arena)."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config_str = yaml.dump(config)
    for key, value in os.environ.items():
        config_str = config_str.replace(f"${{{key}}}", value)
    return yaml.safe_load(config_str)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test-set contamination check (continuation probe)")
    parser.add_argument("--model", required=True, help="Model key from config/models.yaml")
    parser.add_argument("--dataset", required=True, help="Path to a JSON case list")
    parser.add_argument("--sample", type=int, default=None, help="Max cases to probe")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Flag threshold (rouge_l)")
    parser.add_argument("--output", default=None, help="Output JSON path (default: reports/contamination_<ts>.json)")
    args = parser.parse_args()

    config = load_config()
    if args.model not in config["models"]:
        raise SystemExit(f"Model '{args.model}' not found in config")
    adapter = UnifiedLLMAdapter(dict(config["models"][args.model]), model_key=args.model)

    with open(args.dataset, encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        raise SystemExit("Dataset must be a JSON list of cases")

    checker = ContaminationChecker(adapter, threshold=args.threshold)
    report = checker.run(cases, max_samples=args.sample)
    report["model"] = args.model
    report["dataset"] = args.dataset
    report["timestamp"] = datetime.now().isoformat()

    summary = report["summary"]
    print(f"\nModel: {args.model} | dataset: {args.dataset}")
    print(f"Probed cases : {summary['n_cases']}")
    print(f"Flagged      : {summary['flagged']} (rate: {summary['contamination_rate']})")
    print(f"Mean sim     : {summary['mean_similarity']} (threshold: {summary['threshold']})")
    print(f"Verdict      : {summary['verdict']}")

    output = args.output or f"reports/contamination_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
