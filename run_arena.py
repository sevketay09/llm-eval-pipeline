#!/usr/bin/env python3
"""
Arena Runner - Model Karşılaştırma Aracı
İki LLM aynı sorular üzerinde yarıştırır ve ComparativeEvaluator ile kazananı belirler.

Kullanım:
    python run_arena.py --model-a gpt-4o --model-b qwen-32b --dataset turkish_creativity
    python run_arena.py --model-a gpt-4o --model-b qwen-32b --all  # Tüm veri setleri
"""
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from tqdm import tqdm

from adapters.unified_adapter import UnifiedLLMAdapter
from evaluators.comparative_eval import ComparativeEvaluator


class ArenaRunner:
    """Run head-to-head comparisons between two models"""
    
    def __init__(self, config_path: str = "config/models.yaml", judge_model_key: str = None):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize judge
        self.judge_model_key = judge_model_key or self._get_default_judge()
        print(f"Using judge: {self.judge_model_key}")
        self.judge_adapter = self._initialize_adapter(self.judge_model_key)
        self.evaluator = ComparativeEvaluator(self.judge_adapter)
    
    def _load_config(self) -> Dict:
        """Load models config"""
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _get_default_judge(self) -> str:
        """Get default judge model from config"""
        if "default_judge" in self.config:
            return self.config["default_judge"]
        # Try to find a good judge model
        for key in ["gpt-4o", "gpt-4", "claude-3-opus"]:
            if key in self.config["models"]:
                return key
        # Fallback to first model
        return list(self.config["models"].keys())[0]
    
    def _initialize_adapter(self, model_key: str) -> UnifiedLLMAdapter:
        """Initialize a model adapter"""
        if model_key not in self.config["models"]:
            raise ValueError(f"Model '{model_key}' not found in config")
        
        model_config = self.config["models"][model_key]
        return UnifiedLLMAdapter(
            model_name=model_config["model_name"],
            provider=model_config["provider"],
            api_key=model_config.get("api_key"),
            base_url=model_config.get("base_url")
        )
    
    def load_dataset(self, dataset_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
        """Load test dataset"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        return data
    
    def run_arena(
        self,
        model_a_key: str,
        model_b_key: str,
        dataset_path: str,
        max_samples: int = None,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Run arena comparison between two models.
        
        Returns:
            {
                "model_a": str,
                "model_b": str,
                "results": List[Dict],
                "summary": {
                    "model_a_wins": int,
                    "model_b_wins": int,
                    "ties": int,
                    "model_a_win_rate": float,
                    "model_b_win_rate": float,
                    "tie_rate": float
                }
            }
        """
        print(f"\n{'='*80}")
        print(f"ARENA MODE: {model_a_key} vs {model_b_key}")
        print(f"Dataset: {dataset_path}")
        print(f"{'='*80}\n")
        
        # Initialize adapters
        adapter_a = self._initialize_adapter(model_a_key)
        adapter_b = self._initialize_adapter(model_b_key)
        
        # Load dataset
        dataset = self.load_dataset(dataset_path, max_samples)
        print(f"Loaded {len(dataset)} test cases\n")
        
        results = []
        model_a_wins = 0
        model_b_wins = 0
        ties = 0
        
        # Run comparisons
        for i, item in enumerate(tqdm(dataset, desc="Arena battles")):
            question = item.get("question") or item.get("prompt") or item.get("input", "")
            
            if not question:
                print(f"Warning: No question found in item {i}, skipping")
                continue
            
            # Get responses from both models
            try:
                messages = [{"role": "user", "content": question}]
                
                response_a_data = adapter_a.generate(messages, temperature=0.7, max_tokens=1000)
                response_a = response_a_data.get("content", "")
                
                response_b_data = adapter_b.generate(messages, temperature=0.7, max_tokens=1000)
                response_b = response_b_data.get("content", "")
                
            except Exception as e:
                print(f"Error getting responses for item {i}: {e}")
                continue
            
            # Compare responses
            try:
                comparison = self.evaluator.compare(
                    question=question,
                    response_a=response_a,
                    response_b=response_b,
                    model_a_name=model_a_key,
                    model_b_name=model_b_key
                )
            except Exception as e:
                print(f"Error comparing responses for item {i}: {e}")
                continue
            
            # Count wins
            winner = comparison.get("winner", "Tie")
            if winner == "A":
                model_a_wins += 1
            elif winner == "B":
                model_b_wins += 1
            else:
                ties += 1
            
            # Store result
            result = {
                "test_id": item.get("id", f"test_{i+1}"),
                "question": question,
                "response_a": response_a,
                "response_b": response_b,
                "winner": winner,
                "reasoning": comparison.get("reasoning", ""),
                "score_difference": comparison.get("score_difference", 0)
            }
            results.append(result)
        
        # Calculate summary
        total = len(results)
        summary = {
            "model_a": model_a_key,
            "model_b": model_b_key,
            "total_battles": total,
            "model_a_wins": model_a_wins,
            "model_b_wins": model_b_wins,
            "ties": ties,
            "model_a_win_rate": (model_a_wins / total * 100) if total > 0 else 0,
            "model_b_win_rate": (model_b_wins / total * 100) if total > 0 else 0,
            "tie_rate": (ties / total * 100) if total > 0 else 0
        }
        
        arena_result = {
            "timestamp": datetime.now().isoformat(),
            "model_a": model_a_key,
            "model_b": model_b_key,
            "dataset": dataset_path,
            "judge": self.judge_model_key,
            "results": results,
            "summary": summary
        }
        
        # Save results
        if output_path:
            self._save_results(arena_result, output_path)
        
        # Print summary
        self._print_summary(summary)
        
        return arena_result
    
    def run_multi_dataset_arena(
        self,
        model_a_key: str,
        model_b_key: str,
        dataset_paths: List[str],
        max_samples: int = None,
        output_dir: str = "reports/arena"
    ) -> Dict[str, Any]:
        """
        Run arena on multiple datasets and aggregate results.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        all_results = []
        dataset_summaries = []
        
        total_a_wins = 0
        total_b_wins = 0
        total_ties = 0
        
        for dataset_path in dataset_paths:
            print(f"\n{'#'*80}")
            print(f"# Running arena on: {dataset_path}")
            print(f"{'#'*80}\n")
            
            result = self.run_arena(
                model_a_key=model_a_key,
                model_b_key=model_b_key,
                dataset_path=dataset_path,
                max_samples=max_samples,
                output_path=None  # Don't save individual results yet
            )
            
            all_results.append(result)
            dataset_summaries.append({
                "dataset": dataset_path,
                "summary": result["summary"]
            })
            
            total_a_wins += result["summary"]["model_a_wins"]
            total_b_wins += result["summary"]["model_b_wins"]
            total_ties += result["summary"]["ties"]
        
        # Aggregate summary
        total_battles = total_a_wins + total_b_wins + total_ties
        aggregate_summary = {
            "model_a": model_a_key,
            "model_b": model_b_key,
            "total_datasets": len(dataset_paths),
            "total_battles": total_battles,
            "model_a_wins": total_a_wins,
            "model_b_wins": total_b_wins,
            "ties": total_ties,
            "model_a_win_rate": (total_a_wins / total_battles * 100) if total_battles > 0 else 0,
            "model_b_win_rate": (total_b_wins / total_battles * 100) if total_battles > 0 else 0,
            "tie_rate": (ties / total_battles * 100) if total_battles > 0 else 0,
            "dataset_results": dataset_summaries
        }
        
        # Save aggregate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/arena_{model_a_key}_vs_{model_b_key}_{timestamp}.json"
        
        final_result = {
            "timestamp": datetime.now().isoformat(),
            "aggregate_summary": aggregate_summary,
            "individual_results": all_results
        }
        
        self._save_results(final_result, output_path)
        
        # Print final summary
        print(f"\n{'='*80}")
        print("AGGREGATE ARENA RESULTS")
        print(f"{'='*80}")
        self._print_summary(aggregate_summary)
        
        return final_result
    
    def _save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to: {output_path}")
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print arena summary in a nice format"""
        print(f"\n{'='*80}")
        print("ARENA SCORECARD")
        print(f"{'='*80}")
        print(f"{summary['model_a']:30s} : {summary['model_a_wins']:3d} wins ({summary['model_a_win_rate']:.1f}%)")
        print(f"{summary['model_b']:30s} : {summary['model_b_wins']:3d} wins ({summary['model_b_win_rate']:.1f}%)")
        print(f"{'Ties':30s} : {summary['ties']:3d} ({summary['tie_rate']:.1f}%)")
        print(f"{'='*80}")
        print(f"Total Battles: {summary['total_battles']}")
        
        # Determine winner
        if summary['model_a_wins'] > summary['model_b_wins']:
            winner = summary['model_a']
            margin = summary['model_a_win_rate'] - summary['model_b_win_rate']
            print(f"\n🏆 WINNER: {winner} (+{margin:.1f}%)")
        elif summary['model_b_wins'] > summary['model_a_wins']:
            winner = summary['model_b']
            margin = summary['model_b_win_rate'] - summary['model_a_win_rate']
            print(f"\n🏆 WINNER: {winner} (+{margin:.1f}%)")
        else:
            print(f"\n🤝 TIE: Both models performed equally")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Arena Runner - LLM Comparison")
    parser.add_argument("--model-a", required=True, help="First model key from config")
    parser.add_argument("--model-b", required=True, help="Second model key from config")
    parser.add_argument("--dataset", help="Single dataset path")
    parser.add_argument("--all", action="store_true", help="Run on all benchmark datasets")
    parser.add_argument("--max-samples", type=int, help="Maximum samples per dataset")
    parser.add_argument("--judge", help="Judge model key (default: auto-select)")
    parser.add_argument("--output-dir", default="reports/arena", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize arena runner
    runner = ArenaRunner(judge_model_key=args.judge)
    
    # Determine datasets
    if args.all:
        # Default benchmark datasets
        datasets = [
            "eval_datasets/benchmark/turkish_grammar.json",
            "eval_datasets/benchmark/turkish_creativity.json",
            "eval_datasets/benchmark/turkish_reasoning.json",
            "eval_datasets/benchmark/turkish_nuance.json",
            "eval_datasets/fintech/fintech_knowledge.json",
            "eval_datasets/fintech/fintech_calculations.json"
        ]
        runner.run_multi_dataset_arena(
            model_a_key=args.model_a,
            model_b_key=args.model_b,
            dataset_paths=datasets,
            max_samples=args.max_samples,
            output_dir=args.output_dir
        )
    elif args.dataset:
        # Single dataset
        runner.run_arena(
            model_a_key=args.model_a,
            model_b_key=args.model_b,
            dataset_path=args.dataset,
            max_samples=args.max_samples,
            output_path=f"{args.output_dir}/arena_{args.model_a}_vs_{args.model_b}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    else:
        print("Error: Please specify --dataset or --all")
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
