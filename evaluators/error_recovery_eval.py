"""
Tool Error Recovery Evaluator
Modellerin tool hatalarını nasıl handle ettiğini test eder.
- Retry logic
- Fallback strategies
- Error message comprehension
"""
import concurrent.futures
import json
from typing import Dict, Any, List, Optional
from adapters.unified_adapter import UnifiedLLMAdapter
from utils.mock_tools import MockToolEnvironment


class ToolErrorRecoveryEvaluator:
    """
    Evaluate model's ability to handle tool errors and recover gracefully.
    """
    
    def __init__(self, judge_adapter: Optional[UnifiedLLMAdapter] = None):
        self.judge = judge_adapter
    
    def evaluate_retry_behavior(
        self,
        adapter: UnifiedLLMAdapter,
        scenario: Dict[str, Any],
        max_turns: int = 10
    ) -> Dict[str, Any]:
        """
        Test if model retries after transient errors.
        
        scenario format:
        {
            "prompt": str,
            "tool_name": str,
            "error_config": {
                "fail_until_attempt": int  # Fail on first N-1 attempts
            },
            "expected_behavior": str
        }
        """
        # Setup mock environment with error simulation
        tool_name = scenario["tool_name"]
        error_config = scenario.get("error_config", {})
        
        mock_env = MockToolEnvironment(
            error_simulation_config={
                tool_name: error_config
            }
        )
        
        # Get available tools
        available_tools = mock_env.get_tool_definitions()
        
        # Initial conversation
        conversation_history = [
            {"role": "system", "content": "Sen yardımcı bir asistansın. Tool hatalarıyla karşılaştığında mantıklı bir şekilde handle et."},
            {"role": "user", "content": scenario["prompt"]}
        ]
        
        tool_calls_made = []
        retries = 0
        max_retries = 5
        success = False
        
        for turn in range(max_turns):
            # Call model
            try:
                response = adapter.generate(
                    conversation_history,
                    temperature=0.0,
                    max_tokens=1000,
                    tools=available_tools
                )
            except Exception as e:
                break
            
            # Check for tool calls
            tool_calls = response.get("tool_calls", [])
            
            if not tool_calls:
                # Model gave up or provided final answer
                final_response = response.get("content", "")
                break
            
            # Execute tool calls
            assistant_message = {
                "role": "assistant",
                "content": response.get("content", ""),
                "tool_calls": []
            }
            
            for tool_call in tool_calls:
                tc_tool_name = tool_call.get("function", {}).get("name")
                tc_args_str = tool_call.get("function", {}).get("arguments", "{}")
                
                try:
                    tc_args = json.loads(tc_args_str) if isinstance(tc_args_str, str) else tc_args_str
                except json.JSONDecodeError:
                    tc_args = {}
                
                # Track if this is a retry of the same tool
                if tool_calls_made and tool_calls_made[-1]["tool_name"] == tc_tool_name:
                    retries += 1
                
                # Execute
                exec_result = mock_env.execute_tool(tc_tool_name, tc_args)
                
                tool_calls_made.append({
                    "tool_name": tc_tool_name,
                    "arguments": tc_args,
                    "success": exec_result["success"],
                    "error": exec_result.get("error"),
                    "attempt_number": exec_result.get("attempt_number", 1)
                })
                
                if exec_result["success"]:
                    success = True
                
                # Add to conversation
                assistant_message["tool_calls"].append({
                    "id": tool_call.get("id", f"call_{turn}"),
                    "type": "function",
                    "function": {
                        "name": tc_tool_name,
                        "arguments": json.dumps(tc_args)
                    }
                })
                
                conversation_history.append(assistant_message)
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", f"call_{turn}"),
                    "name": tc_tool_name,
                    "content": json.dumps(exec_result)
                })
                
                if exec_result["success"]:
                    break  # Success, no need for more attempts
            
            if success:
                break
            
            if retries >= max_retries:
                break
        
        # Evaluate retry behavior
        expected_retries = error_config.get("fail_until_attempt", 2) - 1
        retry_attempted = retries > 0
        eventually_succeeded = success
        
        return {
            "success": eventually_succeeded,
            "retry_attempted": retry_attempted,
            "retry_count": retries,
            "expected_retries": expected_retries,
            "total_tool_calls": len(tool_calls_made),
            "tool_calls": tool_calls_made,
            "evaluation": {
                "did_retry": retry_attempted,
                "appropriate_retry_count": retries >= expected_retries if retry_attempted else False,
                "eventually_succeeded": eventually_succeeded
            },
            "test_id": scenario.get("id", "unknown")
        }
    
    def evaluate_fallback_strategy(
        self,
        adapter: UnifiedLLMAdapter,
        scenario: Dict[str, Any],
        max_turns: int = 10
    ) -> Dict[str, Any]:
        """
        Test if model uses fallback tools when primary tool fails.
        
        scenario format:
        {
            "prompt": str,
            "primary_tool": str,
            "fallback_tools": List[str],
            "primary_tool_error_config": {...},
            "expected_outcome": str
        }
        """
        primary_tool = scenario["primary_tool"]
        fallback_tools = scenario.get("fallback_tools", [])
        error_config = scenario.get("primary_tool_error_config", {})
        
        # Setup mock with primary tool failing
        mock_env = MockToolEnvironment(
            error_simulation_config={
                primary_tool: error_config
            }
        )
        
        available_tools = mock_env.get_tool_definitions()
        
        conversation_history = [
            {"role": "system", "content": "Sen akıllı bir asistansın. Bir tool çalışmazsa alternatif yöntemler dene."},
            {"role": "user", "content": scenario["prompt"]}
        ]
        
        tool_calls_made = []
        used_fallback = False
        final_response = ""
        
        for turn in range(max_turns):
            try:
                response = adapter.generate(
                    conversation_history,
                    temperature=0.0,
                    max_tokens=1000,
                    tools=available_tools
                )
            except Exception:
                break
            
            tool_calls = response.get("tool_calls", [])
            
            if not tool_calls:
                final_response = response.get("content", "")
                break
            
            assistant_message = {
                "role": "assistant",
                "content": response.get("content", ""),
                "tool_calls": []
            }
            
            for tool_call in tool_calls:
                tc_tool_name = tool_call.get("function", {}).get("name")
                tc_args_str = tool_call.get("function", {}).get("arguments", "{}")
                
                try:
                    tc_args = json.loads(tc_args_str) if isinstance(tc_args_str, str) else tc_args_str
                except json.JSONDecodeError:
                    tc_args = {}
                
                # Check if using fallback
                if tc_tool_name in fallback_tools:
                    used_fallback = True
                
                exec_result = mock_env.execute_tool(tc_tool_name, tc_args)
                
                tool_calls_made.append({
                    "tool_name": tc_tool_name,
                    "arguments": tc_args,
                    "success": exec_result["success"],
                    "is_fallback": tc_tool_name in fallback_tools
                })
                
                assistant_message["tool_calls"].append({
                    "id": tool_call.get("id", f"call_{turn}"),
                    "type": "function",
                    "function": {
                        "name": tc_tool_name,
                        "arguments": json.dumps(tc_args)
                    }
                })
                
                conversation_history.append(assistant_message)
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", f"call_{turn}"),
                    "name": tc_tool_name,
                    "content": json.dumps(exec_result)
                })
        
        # Evaluate
        tried_primary = any(tc["tool_name"] == primary_tool for tc in tool_calls_made)
        primary_failed = any(tc["tool_name"] == primary_tool and not tc["success"] for tc in tool_calls_made)
        
        return {
            "success": used_fallback or not primary_failed,
            "tried_primary_tool": tried_primary,
            "primary_tool_failed": primary_failed,
            "used_fallback": used_fallback,
            "tool_calls": tool_calls_made,
            "final_response": final_response,
            "evaluation": {
                "appropriate_fallback": used_fallback if primary_failed else True,
                "recovered_from_error": used_fallback and primary_failed
            },
            "test_id": scenario.get("id", "unknown")
        }
    
    def evaluate_error_comprehension(
        self,
        adapter: UnifiedLLMAdapter,
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test if model understands error messages and responds appropriately.
        
        scenario format:
        {
            "prompt": str,
            "tool_name": str,
            "error_config": {
                "error_type": str,
                "error_message": str
            },
            "expected_response_contains": List[str],
            "should_not_contain": List[str]
        }
        """
        tool_name = scenario["tool_name"]
        error_config = scenario.get("error_config", {})
        
        # Force tool to fail with specific error
        error_config["fail_on_attempts"] = [1]
        
        mock_env = MockToolEnvironment(
            error_simulation_config={
                tool_name: error_config
            }
        )
        
        available_tools = mock_env.get_tool_definitions()
        
        conversation_history = [
            {"role": "system", "content": "Sen yardımcı bir asistansın. Hataları anla ve kullanıcıya açıkla."},
            {"role": "user", "content": scenario["prompt"]}
        ]
        
        # First turn - tool will fail
        try:
            response = adapter.generate(
                conversation_history,
                temperature=0.0,
                max_tokens=1000,
                tools=available_tools
            )
        except Exception as e:
            return {"success": False, "error": str(e), "test_id": scenario.get("id")}
        
        tool_calls = response.get("tool_calls", [])
        
        if tool_calls:
            tool_call = tool_calls[0]
            tc_tool_name = tool_call.get("function", {}).get("name")
            tc_args_str = tool_call.get("function", {}).get("arguments", "{}")
            
            try:
                tc_args = json.loads(tc_args_str) if isinstance(tc_args_str, str) else tc_args_str
            except json.JSONDecodeError:
                tc_args = {}
            
            exec_result = mock_env.execute_tool(tc_tool_name, tc_args)
            
            # Feed error back to model
            conversation_history.append({
                "role": "assistant",
                "content": response.get("content", ""),
                "tool_calls": [{
                    "id": tool_call.get("id", "call_1"),
                    "type": "function",
                    "function": {
                        "name": tc_tool_name,
                        "arguments": json.dumps(tc_args)
                    }
                }]
            })
            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id", "call_1"),
                "name": tc_tool_name,
                "content": json.dumps(exec_result)
            })
            
            # Get model's response to error
            try:
                error_response = adapter.generate(
                    conversation_history,
                    temperature=0.0,
                    max_tokens=500
                )
                final_response = error_response.get("content", "")
            except Exception as e:
                return {"success": False, "error": str(e), "test_id": scenario.get("id")}
        else:
            final_response = response.get("content", "")
        
        # Check if response is appropriate
        expected_contains = scenario.get("expected_response_contains", [])
        should_not_contain = scenario.get("should_not_contain", [])
        
        response_lower = final_response.lower()
        
        contains_expected = all(exp.lower() in response_lower for exp in expected_contains)
        contains_prohibited = any(prob.lower() in response_lower for prob in should_not_contain)
        
        return {
            "success": contains_expected and not contains_prohibited,
            "final_response": final_response,
            "error_received": error_config.get("error_message", ""),
            "evaluation": {
                "understood_error": contains_expected,
                "appropriate_response": not contains_prohibited,
                "mentioned_error_details": contains_expected
            },
            "test_id": scenario.get("id", "unknown")
        }


def evaluate_tool_error_recovery(
    adapter: UnifiedLLMAdapter,
    test_scenarios: List[Dict[str, Any]],
    judge_adapter: Optional[UnifiedLLMAdapter] = None
) -> Dict[str, Any]:
    """
    Run tool error recovery evaluation on multiple scenarios.
    
    Returns comprehensive results on retry behavior, fallback strategies,
    and error comprehension.
    """
    evaluator = ToolErrorRecoveryEvaluator(judge_adapter)

    results = []
    retry_tests = []
    fallback_tests = []
    comprehension_tests = []

    def _run_scenario(i, scenario):
        print(f"Running error recovery test {i+1}/{len(test_scenarios)}: {scenario.get('id', 'unknown')}")
        test_type = scenario.get("test_type", "retry")
        if test_type == "retry":
            return test_type, evaluator.evaluate_retry_behavior(adapter, scenario)
        elif test_type == "fallback":
            return test_type, evaluator.evaluate_fallback_strategy(adapter, scenario)
        elif test_type == "comprehension":
            return test_type, evaluator.evaluate_error_comprehension(adapter, scenario)
        return None, None

    # Scenarios are independent (each drives its own model/tool-error conversation),
    # so run them concurrently instead of blocking one at a time on model latency.
    # Original scenario order is preserved below for `results` and the per-type lists.
    indexed: Dict[int, Any] = {}
    if test_scenarios:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(test_scenarios))) as pool:
            futures = {pool.submit(_run_scenario, i, scenario): i for i, scenario in enumerate(test_scenarios)}
            for future in concurrent.futures.as_completed(futures):
                indexed[futures[future]] = future.result()

    for i in sorted(indexed):
        test_type, result = indexed[i]
        if test_type is None:
            continue
        if test_type == "retry":
            retry_tests.append(result)
        elif test_type == "fallback":
            fallback_tests.append(result)
        elif test_type == "comprehension":
            comprehension_tests.append(result)
        results.append(result)
    
    # Calculate summary
    total_tests = len(results)
    successful = sum(1 for r in results if r.get("success", False))
    
    retry_success = sum(1 for r in retry_tests if r.get("success", False))
    fallback_success = sum(1 for r in fallback_tests if r.get("success", False))
    comprehension_success = sum(1 for r in comprehension_tests if r.get("success", False))
    
    return {
        "test_results": results,
        "summary": {
            "total_tests": total_tests,
            "successful": successful,
            "success_rate": successful / total_tests if total_tests > 0 else 0.0,
            "retry_tests": {
                "total": len(retry_tests),
                "successful": retry_success,
                "success_rate": retry_success / len(retry_tests) if retry_tests else 0.0
            },
            "fallback_tests": {
                "total": len(fallback_tests),
                "successful": fallback_success,
                "success_rate": fallback_success / len(fallback_tests) if fallback_tests else 0.0
            },
            "comprehension_tests": {
                "total": len(comprehension_tests),
                "successful": comprehension_success,
                "success_rate": comprehension_success / len(comprehension_tests) if comprehension_tests else 0.0
            }
        }
    }
