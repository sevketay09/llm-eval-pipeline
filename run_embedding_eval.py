"""
Embedding Model Evaluation Runner
Türkçe embedding modellerin performansını test eden runner script
"""
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
from adapters.embedding_adapter import UnifiedEmbeddingAdapter
from utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "config/models.yaml"):
    """Load embedding models configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('embedding_models', {})


def load_dataset(dataset_path: str):
    """Load test dataset"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_sts_test(model: UnifiedEmbeddingAdapter, dataset):
    """Run Semantic Textual Similarity test"""
    from evaluators.embedding_eval import SemanticSimilarityEvaluator
    import numpy as np
    from tqdm import tqdm
    
    logger.info(f"Running STS test on {model.model_name}")
    
    embeddings1 = []
    embeddings2 = []
    expected_scores = []
    
    for item in tqdm(dataset, desc="Generating embeddings"):
        emb1 = model.encode([item['sentence1']], normalize=True)
        emb2 = model.encode([item['sentence2']], normalize=True)
        
        embeddings1.append(emb1['embeddings'][0])
        embeddings2.append(emb2['embeddings'][0])
        expected_scores.append(item['similarity_score'])
    
    embeddings1 = np.array(embeddings1)
    embeddings2 = np.array(embeddings2)
    
    results = SemanticSimilarityEvaluator.evaluate(
        embeddings1,
        embeddings2,
        expected_scores
    )
    
    return results


def run_retrieval_test(model: UnifiedEmbeddingAdapter, dataset):
    """Run information retrieval test"""
    from evaluators.embedding_eval import RetrievalEvaluator
    import numpy as np
    from tqdm import tqdm
    
    logger.info(f"Running retrieval test on {model.model_name}")
    
    query_embeddings = []
    doc_embeddings_list = []
    relevance_labels = []
    
    for item in tqdm(dataset, desc="Generating embeddings"):
        query_emb = model.encode([item['query']], normalize=True)
        
        all_docs = (
            item['positive_docs'] +
            item.get('hard_negatives', []) +
            item.get('random_negatives', [])
        )
        doc_embs = model.encode(all_docs, normalize=True)
        
        labels = (
            [1] * len(item['positive_docs']) +
            [0] * len(item.get('hard_negatives', [])) +
            [0] * len(item.get('random_negatives', []))
        )
        
        query_embeddings.append(query_emb['embeddings'][0])
        doc_embeddings_list.append(doc_embs['embeddings'])
        relevance_labels.append(labels)
    
    query_embeddings = np.array(query_embeddings)
    
    results = RetrievalEvaluator.evaluate(
        query_embeddings,
        doc_embeddings_list,
        relevance_labels
    )
    
    return results


def run_clustering_test(model: UnifiedEmbeddingAdapter, dataset):
    """Run term clustering test"""
    from evaluators.embedding_eval import ClusteringEvaluator
    import numpy as np
    from tqdm import tqdm
    
    logger.info(f"Running clustering test on {model.model_name}")
    
    clustering_results = []
    
    for item in tqdm(dataset, desc="Generating embeddings"):
        term_emb = model.encode([item['term']], normalize=True)
        similar_embs = model.encode(item['similar_terms'], normalize=True)
        dissimilar_embs = model.encode(item['dissimilar_terms'], normalize=True)
        
        result = ClusteringEvaluator.evaluate_term_clustering(
            term_emb['embeddings'][0],
            similar_embs['embeddings'],
            dissimilar_embs['embeddings']
        )
        
        clustering_results.append(result)
    
    aggregated = ClusteringEvaluator.aggregate_clustering_results(clustering_results)
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Embedding Model Evaluation')
    parser.add_argument('--model', type=str, required=True, help='Model key from config')
    parser.add_argument('--test', type=str, choices=['sts', 'retrieval', 'clustering', 'all'], default='all')
    parser.add_argument('--output', type=str, default='reports', help='Output directory')
    
    args = parser.parse_args()
    
    # Load config
    models_config = load_config()
    
    if args.model not in models_config:
        logger.error(f"Model {args.model} not found in config")
        return
    
    # Initialize embedding model
    model_config = models_config[args.model]
    model = UnifiedEmbeddingAdapter(model_config, model_key=args.model)
    
    logger.info(f"Initialized model: {model.model_name}")
    logger.info(f"Provider: {model.provider}")
    logger.info(f"Embedding dimension: {model.embedding_dim}")
    
    # Run tests
    results = {
        "model_key": args.model,
        "model_name": model.model_name,
        "provider": model.provider,
        "embedding_dim": model.embedding_dim,
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    if args.test in ['sts', 'all']:
        logger.info("Running Turkish STS test...")
        sts_dataset = load_dataset('eval_datasets/embedding/turkish_sts.json')
        sts_results = run_sts_test(model, sts_dataset)
        results['tests']['sts'] = sts_results
        logger.info(f"STS Spearman: {sts_results['spearman_correlation']:.4f}")
    
    if args.test in ['retrieval', 'all']:
        logger.info("Running Turkish retrieval test...")
        retrieval_dataset = load_dataset('eval_datasets/embedding/turkish_retrieval.json')
        retrieval_results = run_retrieval_test(model, retrieval_dataset)
        results['tests']['retrieval'] = retrieval_results
        logger.info(f"Retrieval NDCG@10: {retrieval_results['ndcg'][10]:.4f}")
    
    if args.test in ['clustering', 'all']:
        logger.info("Running fintech domain clustering test...")
        clustering_dataset = load_dataset('eval_datasets/embedding/fintech_domain.json')
        clustering_results = run_clustering_test(model, clustering_dataset)
        results['tests']['clustering'] = clustering_results
        logger.info(f"Clustering accuracy: {clustering_results['avg_accuracy']:.4f}")
    
    # Add model stats
    results['model_stats'] = model.get_stats()
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"embedding_eval_{args.model}_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print(f"EMBEDDING EVALUATION SUMMARY: {model.model_name}")
    print("="*80)
    
    if 'sts' in results['tests']:
        sts = results['tests']['sts']
        print(f"\nSemantic Textual Similarity:")
        print(f"  Spearman Correlation: {sts['spearman_correlation']:.4f}")
        print(f"  Pearson Correlation:  {sts['pearson_correlation']:.4f}")
        print(f"  MAE:                  {sts['mae']:.4f}")
    
    if 'retrieval' in results['tests']:
        retr = results['tests']['retrieval']
        print(f"\nInformation Retrieval:")
        print(f"  NDCG@1:  {retr['ndcg'][1]:.4f}")
        print(f"  NDCG@5:  {retr['ndcg'][5]:.4f}")
        print(f"  NDCG@10: {retr['ndcg'][10]:.4f}")
        print(f"  MRR:     {retr['mrr']:.4f}")
        print(f"  MAP:     {retr['map']:.4f}")
    
    if 'clustering' in results['tests']:
        clust = results['tests']['clustering']
        print(f"\nDomain-Specific Clustering:")
        print(f"  Accuracy:          {clust['avg_accuracy']:.4f}")
        print(f"  Separation Margin: {clust['avg_separation_margin']:.4f}")
        print(f"  Pass Rate:         {clust['pass_rate']:.4f}")
    
    print(f"\nPerformance:")
    print(f"  Total Embeddings: {results['model_stats']['total_embeddings']}")
    print(f"  Avg Latency:      {results['model_stats']['avg_latency']:.3f}s")
    print(f"  P95 Latency:      {results['model_stats']['p95_latency']:.3f}s")
    
    print("="*80)


if __name__ == "__main__":
    main()
