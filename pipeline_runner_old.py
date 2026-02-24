"""
Main Pipeline Runner
Tüm testleri çalıştırır ve sonuçları toplar
"""
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml
from tqdm import tqdm

from adapters.unified_adapter import UnifiedLLMAdapter
from evaluators.llm_judge import LLMJudgeEvaluator
from evaluators.accuracy_eval import AccuracyEvaluator, FunctionCallingEvaluator


class EvaluationPipeline:
    """Main evaluation pipeline"""
    
    def __init__(self, config_path: str = "config/models.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.test_config = self._load_test_config()
        
        # Initialize adapters
        self.adapters = {}
        self.judge_adapter = None
        
        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "summary": {}
        }
    
    def _load_config(self) -> Dict:
        """Load model configuration"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Replace environment variables
        config_str = yaml.dump(config)
        for key, value in os.environ.items():
            config_str = config_str.replace(f"${{{key}}}", value)
        
        return yaml.safe_load(config_str)
    
    def _load_test_config(self) -> Dict:
        """Load test configuration"""
        with open("config/tests.yaml", 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def initialize_model(self, model_key: str) -> UnifiedLLMAdapter:
        """Initialize a model adapter"""
        if model_key not in self.config["models"]:
            raise ValueError(f"Model {model_key} not found in config")
        
        if model_key not in self.adapters:
            model_config = self.config["models"][model_key]
            self.adapters[model_key] = UnifiedLLMAdapter(model_config)
        
        return self.adapters[model_key]
    
    def initialize_judge(self):
        """Initialize judge model"""
        judge_key = self.config.get("judge_model", {}).get("model_key", "gpt4o-azure")
        self.judge_adapter = self.initialize_model(judge_key)
        return LLMJudgeEvaluator(self.judge_adapter)
    
    def load_dataset(self, dataset_path: str, max_samples: Optional[int] = None) -> List[Dict]:
        """Load test dataset"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples and max_samples != "all":
            data = data[:max_samples]
        
        return data
    
    def run_qa_test(
        self,
        model: UnifiedLLMAdapter,
        dataset: List[Dict],
        judge: LLMJudgeEvaluator,
        test_name: str
    ) -> Dict[str, Any]:
        """Run Q&A test"""
        results = []
        
        print(f"\n{'='*60}")
        print(f"Running {test_name} on {model.model_name}")
        print(f"{'='*60}")
        
        for item in tqdm(dataset, desc=test_name):
            # Generate answer
            messages = [
                {"role": "system", "content": "Sen yardımksever bir asistansın. Soruları Türkçe olarak açık ve doğru şekilde cevapla."},
                {"role": "user", "content": item["question"]}
            ]
            
            response = model.generate(messages)
            
            # Evaluate with different metrics
            accuracy_score = AccuracyEvaluator.evaluate(
                response['content'],
                item.get("expected_answer", ""),
                eval_type="auto"
            )
            
            # LLM Judge evaluation
            relevance_eval = judge.evaluate(
                "relevance",
                item["question"],
                response['content']
            )
            
            fluency_eval = judge.evaluate(
                "turkish_fluency",
                item["question"],
                response['content']
            )
            
            if "expected_answer" in item:
                accuracy_judge = judge.evaluate(
                    "accuracy",
                    item["question"],
                    response['content'],
                    item["expected_answer"]
                )
            else:
                accuracy_judge = {"score": accuracy_score["score"], "reasoning": "Automatic"}
            
            result = {
                "id": item.get("id", "unknown"),
                "category": item.get("category", "general"),
                "question": item["question"],
                "expected_answer": item.get("expected_answer", "N/A"),
                "model_answer": response['content'],
                "scores": {
                    "accuracy": accuracy_score["score"],
                    "accuracy_judge": accuracy_judge["score"],
                    "relevance": relevance_eval["score"],
                    "turkish_fluency": fluency_eval["score"],
                },
                "latency": response['latency'],
                "cost": response['cost'],
                "tokens": response['usage']
            }
            
            results.append(result)
        
        # Calculate aggregates
        avg_scores = {
            "accuracy": sum(r["scores"]["accuracy"] for r in results) / len(results),
            "accuracy_judge": sum(r["scores"]["accuracy_judge"] for r in results) / len(results),
            "relevance": sum(r["scores"]["relevance"] for r in results) / len(results),
            "turkish_fluency": sum(r["scores"]["turkish_fluency"] for r in results) / len(results),
        }
        
        avg_latency = sum(r["latency"] for r in results) / len(results)
        total_cost = sum(r["cost"] for r in results)
        
        return {
            "test_name": test_name,
            "results": results,
            "summary": {
                "total_tests": len(results),
                "avg_scores": avg_scores,
                "avg_latency": avg_latency,
                "total_cost": total_cost,
                "overall_score": sum(avg_scores.values()) / len(avg_scores)
            }
        }
    
    def run_reasoning_test(
        self,
        model: UnifiedLLMAdapter,
        dataset: List[Dict],
        judge: LLMJudgeEvaluator,
        test_name: str
    ) -> Dict[str, Any]:
        """Run reasoning test"""
        results = []
        
        print(f"\n{'='*60}")
        print(f"Running {test_name} on {model.model_name}")
        print(f"{'='*60}")
        
        for item in tqdm(dataset, desc=test_name):
            messages = [
                {"role": "system", "content": "Sen mantıksal düşünme konusunda uzman bir asistansın. Problemleri adım adım çöz ve muhakemeni açıkla."},
                {"role": "user", "content": item["question"]}
            ]
            
            response = model.generate(messages)
            
            # Evaluate reasoning quality
            reasoning_eval = judge.evaluate(
                "reasoning_quality",
                item["question"],
                response['content'],
                item.get("expected_reasoning", "")
            )
            
            # Check answer accuracy
            accuracy_score = AccuracyEvaluator.evaluate(
                response['content'],
                item.get("expected_answer", ""),
                eval_type="auto"
            )
            
            result = {
                "id": item.get("id"),
                "category": item.get("category"),
                "question": item["question"],
                "expected_reasoning": item.get("expected_reasoning", "N/A"),
                "expected_answer": item.get("expected_answer", "N/A"),
                "model_answer": response['content'],
                "scores": {
                    "reasoning_quality": reasoning_eval["score"],
                    "answer_accuracy": accuracy_score["score"],
                },
                "latency": response['latency'],
                "cost": response['cost']
            }
            
            results.append(result)
        
        avg_scores = {
            "reasoning_quality": sum(r["scores"]["reasoning_quality"] for r in results) / len(results),
            "answer_accuracy": sum(r["scores"]["answer_accuracy"] for r in results) / len(results),
        }
        
        return {
            "test_name": test_name,
            "results": results,
            "summary": {
                "total_tests": len(results),
                "avg_scores": avg_scores,
                "avg_latency": sum(r["latency"] for r in results) / len(results),
                "total_cost": sum(r["cost"] for r in results),
                "overall_score": sum(avg_scores.values()) / len(avg_scores)
            }
        }
    
    def run_function_calling_test(
        self,
        model: UnifiedLLMAdapter,
        dataset: List[Dict],
        judge: LLMJudgeEvaluator,
        test_name: str
    ) -> Dict[str, Any]:
        """Run function calling test"""
        results = []
        
        print(f"\n{'='*60}")
        print(f"Running {test_name} on {model.model_name}")
        print(f"{'='*60}")
        
        for item in tqdm(dataset, desc=test_name):
            messages = [
                {"role": "system", "content": "Sen bir finans asistanısın. Kullanıcının talebini yerine getirmek için uygun araçları kullan."},
                {"role": "user", "content": item["prompt"]}
            ]
            
            response = model.generate(
                messages,
                tools=item.get("available_tools")
            )
            
            # Evaluate function calling
            fc_eval = FunctionCallingEvaluator.evaluate(
                response.get('tool_calls'),
                item.get("expected_tool", ""),
                item.get("expected_params", {})
            )
            
            result = {
                "id": item.get("id"),
                "category": item.get("category"),
                "prompt": item["prompt"],
                "expected_tool": item.get("expected_tool"),
                "expected_params": item.get("expected_params"),
                "tool_calls": response.get('tool_calls'),
                "scores": {
                    "tool_selection": fc_eval["tool_selection_score"],
                    "parameter_extraction": fc_eval["parameter_score"],
                    "overall": fc_eval["overall_score"]
                },
                "latency": response['latency'],
                "cost": response['cost']
            }
            
            results.append(result)
        
        avg_scores = {
            "tool_selection": sum(r["scores"]["tool_selection"] for r in results) / len(results),
            "parameter_extraction": sum(r["scores"]["parameter_extraction"] for r in results) / len(results),
            "overall": sum(r["scores"]["overall"] for r in results) / len(results),
        }
        
        return {
            "test_name": test_name,
            "results": results,
            "summary": {
                "total_tests": len(results),
                "avg_scores": avg_scores,
                "avg_latency": sum(r["latency"] for r in results) / len(results),
                "total_cost": sum(r["cost"] for r in results),
                "overall_score": avg_scores["overall"]
            }
        }
    
    def run_agentic_test(
        self,
        model: UnifiedLLMAdapter,
        dataset: List[Dict],
        judge: LLMJudgeEvaluator,
        test_name: str
    ) -> Dict[str, Any]:
        """Run agentic workflow test"""
        results = []
        
        print(f"\n{'='*60}")
        print(f"Running {test_name} on {model.model_name}")
        print(f"{'='*60}")
        
        for item in tqdm(dataset, desc=test_name):
            messages = [
                {"role": "system", "content": "Sen akıllı bir finans asistanısın. Karmaşık görevleri planlayıp adım adım çöz."},
                {"role": "user", "content": item["task"]}
            ]
            
            response = model.generate(messages, max_tokens=2000)
            
            # Evaluate plan quality
            plan_eval = judge.evaluate(
                "agentic_plan_quality",
                item["task"],
                response['content'],
                context={
                    "task": item["task"],
                    "available_tools": item.get("available_tools", []),
                    "plan": response['content']
                }
            )
            
            result = {
                "id": item.get("id"),
                "category": item.get("category"),
                "task": item["task"],
                "model_response": response['content'],
                "scores": {
                    "plan_quality": plan_eval["score"],
                },
                "latency": response['latency'],
                "cost": response['cost']
            }
            
            results.append(result)
        
        avg_plan_quality = sum(r["scores"]["plan_quality"] for r in results) / len(results)
        
        return {
            "test_name": test_name,
            "results": results,
            "summary": {
                "total_tests": len(results),
                "avg_scores": {
                    "plan_quality": avg_plan_quality
                },
                "avg_latency": sum(r["latency"] for r in results) / len(results),
                "total_cost": sum(r["cost"] for r in results),
                "overall_score": avg_plan_quality
            }
        }
    
    def run_full_evaluation(
        self,
        model_keys: List[str],
        test_suite: str = "full"
    ) -> Dict[str, Any]:
        """Run complete evaluation pipeline"""
        
        print(f"\n{'='*80}")
        print(f"STARTING EVALUATION PIPELINE - Test Suite: {test_suite}")
        print(f"Models: {', '.join(model_keys)}")
        print(f"{'='*80}\n")
        
        # Initialize judge
        judge = self.initialize_judge()
        
        # Test definitions
        test_mapping = {
            "turkish_qa": ("eval_datasets/benchmark/turkish_qa.json", self.run_qa_test),
            "turkish_reasoning": ("eval_datasets/benchmark/turkish_reasoning.json", self.run_reasoning_test),
            "fintech_knowledge": ("eval_datasets/fintech/fintech_knowledge.json", self.run_qa_test),
            "fintech_calculations": ("eval_datasets/fintech/fintech_calculations.json", self.run_reasoning_test),
            "function_calling": ("eval_datasets/function_calling/function_calling_tests.json", self.run_function_calling_test),
            "agentic_workflows": ("eval_datasets/agentic/agentic_workflows.json", self.run_agentic_test),
        }
        
        # Get tests for this suite
        suite_config = self.test_config["test_suites"].get(test_suite, {})
        tests_to_run = suite_config.get("tests", [])
        max_samples = suite_config.get("max_samples", "all")
        
        # Run evaluation for each model
        for model_key in model_keys:
            print(f"\n{'#'*80}")
            print(f"# EVALUATING MODEL: {model_key}")
            print(f"{'#'*80}")
            
            model = self.initialize_model(model_key)
            model.reset_stats()
            
            model_results = {
                "model_name": model.model_name,
                "provider": model.provider,
                "tests": {},
                "overall_metrics": {}
            }
            
            # Run each test
            for test_name in tests_to_run:
                if test_name not in test_mapping:
                    print(f"Warning: Test {test_name} not found")
                    continue
                
                dataset_path, test_func = test_mapping[test_name]
                
                try:
                    dataset = self.load_dataset(dataset_path, max_samples)
                    test_result = test_func(model, dataset, judge, test_name)
                    model_results["tests"][test_name] = test_result
                except Exception as e:
                    print(f"Error running {test_name}: {str(e)}")
                    model_results["tests"][test_name] = {
                        "error": str(e)
                    }
            
            # Add model stats
            model_results["overall_metrics"] = model.get_stats()
            
            # Calculate weighted overall score
            weights = self.test_config.get("metric_weights", {})
            total_score = 0
            total_weight = 0
            
            for test_name, test_result in model_results["tests"].items():
                if "summary" in test_result and "overall_score" in test_result["summary"]:
                    weight = weights.get(test_name, 1.0)
                    total_score += test_result["summary"]["overall_score"] * weight
                    total_weight += weight
            
            if total_weight > 0:
                model_results["overall_metrics"]["weighted_score"] = total_score / total_weight
            
            self.results["models"][model_key] = model_results
        
        # Generate comparison summary
        self.results["summary"] = self._generate_summary()
        
        return self.results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comparison summary across models"""
        summary = {
            "model_comparison": {},
            "best_performers": {},
            "recommendations": []
        }
        
        # Compare models
        for model_key, model_data in self.results["models"].items():
            summary["model_comparison"][model_key] = {
                "overall_score": model_data["overall_metrics"].get("weighted_score", 0),
                "avg_latency": model_data["overall_metrics"].get("latency_avg", 0),
                "total_cost": model_data["overall_metrics"].get("total_cost", 0),
                "latency_p95": model_data["overall_metrics"].get("latency_p95", 0),
            }
        
        # Find best performers per category
        for test_name in ["turkish_qa", "function_calling", "agentic_workflows"]:
            best_model = None
            best_score = 0
            
            for model_key, model_data in self.results["models"].items():
                if test_name in model_data["tests"]:
                    test_data = model_data["tests"][test_name]
                    if "summary" in test_data:
                        score = test_data["summary"].get("overall_score", 0)
                        if score > best_score:
                            best_score = score
                            best_model = model_key
            
            if best_model:
                summary["best_performers"][test_name] = {
                    "model": best_model,
                    "score": best_score
                }
        
        return summary
    
    def save_results(self, output_path: str = "reports/evaluation_results.json"):
        """Save results to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*80}\n")
    
    def print_summary(self):
        """Print evaluation summary"""
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}\n")
        
        for model_key, comparison in self.results["summary"]["model_comparison"].items():
            print(f"\n{model_key}:")
            print(f"  Overall Score: {comparison['overall_score']:.3f}")
            print(f"  Avg Latency: {comparison['avg_latency']:.2f}s")
            print(f"  P95 Latency: {comparison['latency_p95']:.2f}s")
            print(f"  Total Cost: ${comparison['total_cost']:.4f}")
        
        print(f"\n{'='*80}")
        print("BEST PERFORMERS BY CATEGORY")
        print(f"{'='*80}\n")
        
        for category, data in self.results["summary"]["best_performers"].items():
            print(f"{category}: {data['model']} (score: {data['score']:.3f})")
