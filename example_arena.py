#!/usr/bin/env python3
"""
Example: Arena Runner Usage
Bu örnek iki modeli karşılaştırmayı gösterir.
"""
import sys
from run_arena import ArenaRunner

def main():
    print("="*80)
    print("Arena Runner - Usage Example")
    print("="*80)
    
    # NOTE: Bu örnekte kullanılan model isimleri config/models.yaml
    # dosyanızda tanımlanmış olmalıdır.
    
    model_a = "gpt-4o"  # İlk model
    model_b = "gpt-4o-mini"  # İkinci model
    dataset = "eval_datasets/benchmark/turkish_creativity.json"
    
    print(f"\nModels to compare:")
    print(f"  Model A: {model_a}")
    print(f"  Model B: {model_b}")
    print(f"  Dataset: {dataset}")
    print(f"\nNote: Make sure these models are defined in config/models.yaml")
    
    try:
        # Initialize arena runner
        print("\nInitializing Arena Runner...")
        runner = ArenaRunner()
        
        # Run arena on a single dataset
        print("\nRunning arena comparison...")
        result = runner.run_arena(
            model_a_key=model_a,
            model_b_key=model_b,
            dataset_path=dataset,
            max_samples=5,  # Test with just 5 samples
            output_path="reports/arena/test_arena.json"
        )
        
        print("\n✅ Arena comparison completed!")
        print(f"\nWinner: {result['summary']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTips:")
        print("1. Check that models are defined in config/models.yaml")
        print("2. Ensure API keys are set (if using cloud models)")
        print("3. Dataset path is correct")
        return 1
    
    print("\n" + "="*80)
    print("For actual usage, run:")
    print("  python run_arena.py --model-a MODEL_A --model-b MODEL_B --dataset DATASET")
    print("  python run_arena.py --model-a MODEL_A --model-b MODEL_B --all")
    print("="*80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
