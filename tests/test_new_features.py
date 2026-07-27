#!/usr/bin/env python3
"""
Test script for new features:
1. Needle in Haystack
2. Tool Error Recovery  
3. Parallel Tool Execution
"""
import json
from utils.mock_tools import MockToolEnvironment


def test_mock_tools():
    """Test new mock tools"""
    print("=" * 60)
    print("Testing New Mock Tools")
    print("=" * 60)
    
    env = MockToolEnvironment()
    
    # Test search_flights
    print("\n1. Testing search_flights...")
    result = env.execute_tool("search_flights", {
        "from_city": "Istanbul",
        "to_city": "London"
    })
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Found {result['result']['results_count']} flights")
    
    # Test search_product
    print("\n2. Testing search_product...")
    result = env.execute_tool("search_product", {
        "query": "iPhone 15"
    })
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Found {result['result']['results_count']} products")
    
    # Test check_calendar
    print("\n3. Testing check_calendar...")
    result = env.execute_tool("check_calendar", {
        "date": "2024-04-15"
    })
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Availability: {result['result']['availability']}")
        print(f"   Free slots: {len(result['result']['free_slots'])}")
    
    # Test translate_text
    print("\n4. Testing translate_text...")
    result = env.execute_tool("translate_text", {
        "text": "elma",
        "source_lang": "tr",
        "target_lang": "en"
    })
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Translation: {result['result']['original_text']} -> {result['result']['translated_text']}")
    
    # Test book_flight
    print("\n5. Testing book_flight...")
    result = env.execute_tool("book_flight", {
        "flight_id": "FL1234",
        "passenger_name": "John Doe",
        "seat_preference": "window"
    })
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Booking ID: {result['result']['booking_id']}")
        print(f"   Confirmation: {result['result']['confirmation_code']}")


def test_error_recovery():
    """Test error simulation"""
    print("\n" + "=" * 60)
    print("Testing Error Recovery Mechanism")
    print("=" * 60)
    
    # Create env with error config
    error_config = {
        "unreliable_weather_api": {
            "fail_until_attempt": 3
        }
    }
    
    env = MockToolEnvironment(error_simulation_config=error_config)
    
    print("\n1. Testing unreliable_weather_api with 3 retries...")
    for attempt in range(4):
        result = env.execute_tool("unreliable_weather_api", {"city": "Istanbul"})
        print(f"   Attempt {attempt + 1}: {'SUCCESS' if result['success'] else 'FAILED'}")
        if not result['success']:
            print(f"   Error: {result.get('error', 'Unknown')}")


def test_needle_dataset():
    """Test needle in haystack dataset"""
    print("\n" + "=" * 60)
    print("Testing Needle in Haystack Dataset")
    print("=" * 60)
    
    with open("eval_datasets/rag/needle_in_haystack.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print(f"\nDataset loaded: {len(dataset)} test cases")
    
    # Show summary
    difficulties = {}
    lengths = {}
    positions = {}
    
    for test in dataset:
        diff = test.get("difficulty", "unknown")
        difficulties[diff] = difficulties.get(diff, 0) + 1
        
        length = test.get("context_length", "unknown")
        lengths[length] = lengths.get(length, 0) + 1
        
        pos = test.get("needle_position", "unknown")
        positions[pos] = positions.get(pos, 0) + 1
    
    print(f"\nBy difficulty: {difficulties}")
    print(f"By length: {lengths}")
    print(f"By position: {positions}")


def test_error_recovery_dataset():
    """Test error recovery dataset"""
    print("\n" + "=" * 60)
    print("Testing Error Recovery Dataset")
    print("=" * 60)
    
    with open("eval_datasets/function_calling/tool_error_recovery_tests.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print(f"\nDataset loaded: {len(dataset)} test cases")
    
    # Show summary
    test_types = {}
    difficulties = {}
    
    for test in dataset:
        t_type = test.get("test_type", "unknown")
        test_types[t_type] = test_types.get(t_type, 0) + 1
        
        diff = test.get("difficulty", "unknown")
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    print(f"\nBy test type: {test_types}")
    print(f"By difficulty: {difficulties}")


def test_parallel_dataset():
    """Test parallel tool execution dataset"""
    print("\n" + "=" * 60)
    print("Testing Parallel Tool Execution Dataset")
    print("=" * 60)
    
    with open("eval_datasets/function_calling/parallel_tool_tests.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print(f"\nDataset loaded: {len(dataset)} test cases")
    
    # Show summary
    parallel_count = sum(1 for t in dataset if t.get("is_parallel", False))
    sequential_count = len(dataset) - parallel_count
    
    difficulties = {}
    for test in dataset:
        diff = test.get("difficulty", "unknown")
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    print(f"\nParallel scenarios: {parallel_count}")
    print(f"Sequential scenarios: {sequential_count}")
    print(f"By difficulty: {difficulties}")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("NEW FEATURES TEST SUITE")
    print("=" * 60)
    
    try:
        test_mock_tools()
        test_error_recovery()
        test_needle_dataset()
        test_error_recovery_dataset()
        test_parallel_dataset()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
