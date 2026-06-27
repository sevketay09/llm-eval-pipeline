"""
Quick Example Script
Pipeline'ı hızlıca test etmek için örnek
"""
import os
import sys

# Set dummy env vars for testing
os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://dummy.openai.azure.com/'
os.environ['AZURE_OPENAI_KEY'] = 'dummy-key'
os.environ['OPENAI_API_KEY'] = 'dummy-key'
os.environ['ANTHROPIC_API_KEY'] = 'dummy-key'

from pipeline_runner import EvaluationPipeline


def example_single_model():
    """Example: Test single model"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Model Evaluation")
    print("="*80 + "\n")
    
    pipeline = EvaluationPipeline()
    
    # Run smoke test for one model
    results = pipeline.run_full_evaluation(
        model_keys=['gpt4o-azure'],
        test_suite='smoke'
    )
    
    pipeline.print_summary()
    pipeline.save_results('reports/example_single_model.json')
    
    print("\n✅ Example 1 completed!")


def example_model_comparison():
    """Example: Compare multiple models"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Model Comparison")
    print("="*80 + "\n")
    
    pipeline = EvaluationPipeline()
    
    # Compare three models
    results = pipeline.run_full_evaluation(
        model_keys=['gpt4o-azure', 'gpt4-turbo-azure', 'claude-sonnet-45'],
        test_suite='smoke'
    )
    
    pipeline.print_summary()
    pipeline.save_results('reports/example_comparison.json')
    
    print("\n✅ Example 2 completed!")


def example_fintech_only():
    """Example: Test only fintech capabilities"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Fintech-Only Evaluation")
    print("="*80 + "\n")
    
    pipeline = EvaluationPipeline()
    
    # Test fintech suite
    results = pipeline.run_full_evaluation(
        model_keys=['gpt4o-azure', 'llama-3-70b-vllm'],
        test_suite='fintech_only'
    )
    
    pipeline.print_summary()
    pipeline.save_results('reports/example_fintech.json')
    
    print("\n✅ Example 3 completed!")


def example_custom_test():
    """Example: Run custom test on specific dataset"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Test")
    print("="*80 + "\n")
    
    from adapters.unified_adapter import UnifiedLLMAdapter
    from evaluators.llm_judge import LLMJudgeEvaluator
    import json
    
    pipeline = EvaluationPipeline()
    
    # Initialize model and judge
    model = pipeline.initialize_model('gpt4o-azure')
    judge = pipeline.initialize_judge()
    
    # Load specific dataset
    with open('eval_datasets/fintech/fintech_knowledge.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)[:3]  # First 3 questions only
    
    # Run test
    results = pipeline.run_qa_test(
        model=model,
        dataset=dataset,
        judge=judge,
        test_name="custom_fintech_test"
    )
    
    print("\nTest Results:")
    print(f"Overall Score: {results['summary']['overall_score']:.3f}")
    print(f"Avg Latency: {results['summary']['avg_latency']:.2f}s")
    print(f"Total Cost: ${results['summary']['total_cost']:.4f}")
    
    print("\n✅ Example 4 completed!")


def show_menu():
    """Interactive menu"""
    print("\n" + "="*80)
    print("LLM EVALUATION PIPELINE - EXAMPLES")
    print("="*80)
    print("\n1. Single Model Evaluation (smoke test)")
    print("2. Model Comparison (3 models)")
    print("3. Fintech-Only Evaluation")
    print("4. Custom Test (specific dataset)")
    print("5. Exit")
    print()
    
    choice = input("Select example (1-5): ").strip()
    
    if choice == '1':
        example_single_model()
    elif choice == '2':
        example_model_comparison()
    elif choice == '3':
        example_fintech_only()
    elif choice == '4':
        example_custom_test()
    elif choice == '5':
        print("\nGoodbye!")
        sys.exit(0)
    else:
        print("\n❌ Invalid choice!")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                   LLM EVALUATION PIPELINE - EXAMPLES                         ║
║                   Türkçe ve Fintech Odaklı Model Testi                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

NOT: Bu örnekler dummy API key'ler kullanır. Gerçek testler için .env dosyasını
     düzenleyip gerçek API key'lerinizi ekleyin.
""")
    
    if len(sys.argv) > 1:
        # Command line argument
        example = sys.argv[1]
        if example == '1':
            example_single_model()
        elif example == '2':
            example_model_comparison()
        elif example == '3':
            example_fintech_only()
        elif example == '4':
            example_custom_test()
        else:
            print(f"Unknown example: {example}")
    else:
        # Interactive mode
        while True:
            show_menu()
            input("\nPress Enter to continue...")
