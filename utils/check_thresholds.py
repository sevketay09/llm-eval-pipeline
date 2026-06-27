"""
Threshold Checker
Evaluation sonuçlarının threshold'ları geçip geçmediğini kontrol eder
CI/CD pipeline'larında kullanılabilir
"""
import json
import sys
import yaml
from pathlib import Path


def load_results(filepath):
    """Load evaluation results"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_thresholds(config_path='config/tests.yaml'):
    """Load threshold configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('thresholds', {})


def check_thresholds(results, thresholds):
    """
    Check if models pass thresholds
    
    Returns:
        (passed_models, failed_models, details)
    """
    passed = []
    failed = []
    details = {}
    
    overall_threshold = thresholds.get('overall_score', 0.70)
    accuracy_threshold = thresholds.get('accuracy', 0.75)
    fc_threshold = thresholds.get('function_calling_accuracy', 0.80)
    fluency_threshold = thresholds.get('turkish_fluency', 0.85)
    latency_threshold = thresholds.get('max_latency_p95', 10.0)
    emb_sts_threshold = thresholds.get('embedding_spearman_correlation', 0.70)
    emb_xsts_threshold = thresholds.get('embedding_spearman_correlation_crosslingual', 0.68)
    emb_ndcg_threshold = thresholds.get('embedding_ndcg_at_10', 0.75)
    emb_ndcg_hardneg_threshold = thresholds.get('embedding_ndcg_at_10_hardneg', 0.80)
    emb_clustering_threshold = thresholds.get('embedding_clustering_accuracy', 0.80)
    emb_clustering_reg_threshold = thresholds.get('embedding_clustering_accuracy_regulatory', 0.82)

    def get_ndcg_at_10(summary):
        ndcg = summary.get('ndcg', {})
        if isinstance(ndcg, dict):
            return ndcg.get(10, ndcg.get('10', 0))
        return 0
    
    for model_key, model_data in results['models'].items():
        checks = {
            'overall_score': False,
            'accuracy': False,
            'function_calling': False,
            'turkish_fluency': False,
            'latency': False,
            'embedding_sts': True,
            'embedding_sts_crosslingual': True,
            'embedding_retrieval': True,
            'embedding_retrieval_hardneg': True,
            'embedding_clustering': True,
            'embedding_clustering_regulatory': True
        }
        
        # Overall score
        overall_score = model_data['overall_metrics'].get('weighted_score', 0)
        checks['overall_score'] = overall_score >= overall_threshold
        
        # Accuracy
        if 'turkish_qa' in model_data['tests']:
            avg_accuracy = model_data['tests']['turkish_qa']['summary']['avg_scores'].get('accuracy', 0)
            checks['accuracy'] = avg_accuracy >= accuracy_threshold
        
        # Function calling
        if 'function_calling' in model_data['tests']:
            fc_score = model_data['tests']['function_calling']['summary']['overall_score']
            checks['function_calling'] = fc_score >= fc_threshold
        
        # Turkish fluency
        if 'turkish_qa' in model_data['tests']:
            fluency = model_data['tests']['turkish_qa']['summary']['avg_scores'].get('turkish_fluency', 0)
            checks['turkish_fluency'] = fluency >= fluency_threshold
        
        # Latency
        latency_p95 = model_data['overall_metrics'].get('latency_p95', 0)
        checks['latency'] = latency_p95 <= latency_threshold

        # Embedding STS
        if 'embedding_sts' in model_data['tests'] and 'summary' in model_data['tests']['embedding_sts']:
            sts_score = model_data['tests']['embedding_sts']['summary'].get('spearman_correlation', 0)
            checks['embedding_sts'] = sts_score >= emb_sts_threshold

        # Embedding STS crosslingual
        if 'embedding_sts_crosslingual' in model_data['tests'] and 'summary' in model_data['tests']['embedding_sts_crosslingual']:
            xsts_score = model_data['tests']['embedding_sts_crosslingual']['summary'].get('spearman_correlation', 0)
            checks['embedding_sts_crosslingual'] = xsts_score >= emb_xsts_threshold

        # Embedding retrieval
        if 'embedding_retrieval' in model_data['tests'] and 'summary' in model_data['tests']['embedding_retrieval']:
            ndcg_at_10 = get_ndcg_at_10(model_data['tests']['embedding_retrieval']['summary'])
            checks['embedding_retrieval'] = ndcg_at_10 >= emb_ndcg_threshold

        # Embedding retrieval hard negatives
        if 'embedding_retrieval_hardneg' in model_data['tests'] and 'summary' in model_data['tests']['embedding_retrieval_hardneg']:
            ndcg_hardneg_at_10 = get_ndcg_at_10(model_data['tests']['embedding_retrieval_hardneg']['summary'])
            checks['embedding_retrieval_hardneg'] = ndcg_hardneg_at_10 >= emb_ndcg_hardneg_threshold

        # Embedding clustering
        if 'embedding_clustering' in model_data['tests'] and 'summary' in model_data['tests']['embedding_clustering']:
            clustering_acc = model_data['tests']['embedding_clustering']['summary'].get('avg_accuracy', 0)
            checks['embedding_clustering'] = clustering_acc >= emb_clustering_threshold

        # Embedding clustering regulatory
        if 'embedding_clustering_regulatory' in model_data['tests'] and 'summary' in model_data['tests']['embedding_clustering_regulatory']:
            clustering_reg_acc = model_data['tests']['embedding_clustering_regulatory']['summary'].get('avg_accuracy', 0)
            checks['embedding_clustering_regulatory'] = clustering_reg_acc >= emb_clustering_reg_threshold
        
        details[model_key] = {
            'checks': checks,
            'values': {
                'overall_score': overall_score,
                'latency_p95': latency_p95
            }
        }
        
        # Use embedding-focused checks when any embedding test exists; otherwise use legacy checks
        has_embedding_tests = any(k.startswith('embedding_') for k in model_data.get('tests', {}).keys())

        if has_embedding_tests:
            embedding_check_keys = [
                'embedding_sts',
                'embedding_sts_crosslingual',
                'embedding_retrieval',
                'embedding_retrieval_hardneg',
                'embedding_clustering',
                'embedding_clustering_regulatory'
            ]
            available_embedding_checks = [key for key in embedding_check_keys if key in model_data['tests']]
            evaluated_embedding_checks = [checks[key] for key in available_embedding_checks]
            model_passed = all(evaluated_embedding_checks) if evaluated_embedding_checks else False
            relevant_checks = available_embedding_checks
        else:
            legacy_check_keys = ['overall_score', 'accuracy', 'function_calling', 'turkish_fluency', 'latency']
            model_passed = all(checks[k] for k in legacy_check_keys)
            relevant_checks = legacy_check_keys

        details[model_key]['relevant_checks'] = relevant_checks

        if model_passed:
            passed.append(model_key)
        else:
            failed.append(model_key)
    
    return passed, failed, details


def print_report(passed, failed, details, thresholds):
    """Print threshold check report"""
    print("\n" + "="*80)
    print("THRESHOLD CHECK REPORT")
    print("="*80 + "\n")
    
    print("Thresholds:")
    for key, value in thresholds.items():
        print(f"  {key}: {value}")
    
    print("\n" + "-"*80 + "\n")
    
    for model_key, detail in details.items():
        status = "✅ PASSED" if model_key in passed else "❌ FAILED"
        print(f"{model_key}: {status}")
        print(f"  Overall Score: {detail['values']['overall_score']:.3f}")
        print(f"  Latency P95: {detail['values']['latency_p95']:.2f}s")
        
        relevant_checks = detail.get('relevant_checks', list(detail['checks'].keys()))
        failed_checks = [k for k in relevant_checks if not detail['checks'].get(k, False)]
        if failed_checks:
            print(f"  Failed checks: {', '.join(failed_checks)}")
        print()
    
    print("-"*80)
    
    if passed:
        print(f"\n✅ {len(passed)} model(s) passed: {', '.join(passed)}")
    if failed:
        print(f"\n❌ {len(failed)} model(s) failed: {', '.join(failed)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_thresholds.py <results_file.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"Error: File not found: {results_file}")
        sys.exit(1)
    
    # Load data
    results = load_results(results_file)
    thresholds = load_thresholds()
    
    # Check thresholds
    passed, failed, details = check_thresholds(results, thresholds)
    
    # Print report
    print_report(passed, failed, details, thresholds)
    
    # Exit with appropriate code for CI/CD
    if failed:
        sys.exit(1)  # Fail CI/CD if any model failed
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
