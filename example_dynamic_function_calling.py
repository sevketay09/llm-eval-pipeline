#!/usr/bin/env python3
"""
Example: Dynamic Function Calling Evaluation
Bu örnek mock tool execution environment ile function calling testini gösterir.
"""
import sys
from adapters.unified_adapter import UnifiedLLMAdapter
from evaluators.dynamic_function_eval import DynamicFunctionCallingEvaluator

def main():
    print("="*80)
    print("Dynamic Function Calling Evaluation - Example")
    print("="*80)
    
    # Initialize model (adjust according to your config)
    print("\n1. Initializing model...")
    try:
        # Example with OpenAI (replace with your actual model)
        adapter = UnifiedLLMAdapter(
            model_name="gpt-4o-mini",
            provider="openai",
            api_key=None  # Will use environment variable
        )
        print("✅ Model initialized")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        print("Note: Update this example with your actual model configuration")
        return 1
    
    # Initialize evaluator (without judge for this simple example)
    print("\n2. Initializing evaluator...")
    evaluator = DynamicFunctionCallingEvaluator(judge_adapter=None)
    print("✅ Evaluator initialized")
    
    # Test scenario
    print("\n3. Running test scenario...")
    test_scenario = {
        "id": "weather_test",
        "prompt": "İstanbul'da hava nasıl? Bana bilgi ver.",
        "expected_tools": ["get_weather"],
        "expected_order": False,
        "expected_outcome": "İstanbul için hava durumu bilgisi verilmelidir",
        "max_turns": 3
    }
    
    print(f"   Prompt: {test_scenario['prompt']}")
    
    try:
        result = evaluator.evaluate_tool_chain(adapter, test_scenario)
        
        print("\n4. Results:")
        print(f"   ✅ Success: {result['success']}")
        print(f"   🔧 Tools Match: {result.get('tools_match', False)}")
        print(f"   🔄 Turns: {result['turns']}")
        print(f"   📞 Tool Calls Made: {len(result['tool_calls'])}")
        
        if result['tool_calls']:
            print("\n5. Tool Calls:")
            for i, tc in enumerate(result['tool_calls'], 1):
                print(f"   Call {i}: {tc['tool_name']}({tc['arguments']})")
        
        if result['execution_results']:
            print("\n6. Execution Results:")
            for i, exec_result in enumerate(result['execution_results'], 1):
                print(f"   Result {i}: Success={exec_result['success']}")
                if exec_result['success']:
                    print(f"              {exec_result['result']}")
        
        print(f"\n7. Final Response:")
        print(f"   {result['final_response'][:200]}...")
        
        if result.get('errors'):
            print(f"\n⚠️ Errors: {result['errors']}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*80)
    print("Example completed!")
    print("="*80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
