"""
Dynamic Function Calling Evaluator
Modellerin function calling yeteneklerini dinamik execution ile test eder.
"""
import json
from typing import Dict, Any, List, Optional
from adapters.unified_adapter import UnifiedLLMAdapter
from utils.mock_tools import get_mock_environment


class DynamicFunctionCallingEvaluator:
    """
    Evaluate function calling with actual tool execution.
    Model generates tool calls, we execute them, and feed results back to the model.
    """
    
    def __init__(self, judge_adapter: Optional[UnifiedLLMAdapter] = None):
        self.mock_env = get_mock_environment()
        self.judge = judge_adapter
    
    def evaluate_multi_turn_tool_use(
        self,
        adapter: UnifiedLLMAdapter,
        initial_prompt: str,
        available_tools: Optional[List[Dict[str, Any]]] = None,
        max_turns: int = 5,
        expected_outcome: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Multi-turn evaluation with tool execution loop.
        
        Args:
            adapter: LLM adapter to test
            initial_prompt: User's initial request
            available_tools: List of available tools (if None, uses all from mock env)
            max_turns: Maximum conversation turns
            expected_outcome: Expected final outcome for judging
        
        Returns:
            {
                "success": bool,
                "turns": int,
                "tool_calls": List[Dict],
                "execution_results": List[Dict],
                "final_response": str,
                "judge_score": float (if judge available),
                "errors": List[str]
            }
        """
        # Get available tools
        if available_tools is None:
            available_tools = self.mock_env.get_tool_definitions()
        
        conversation_history = [
            {"role": "system", "content": "Sen yardımcı bir asistansın. Görevini tamamlamak için gerekli araçları kullanabilirsin."},
            {"role": "user", "content": initial_prompt}
        ]
        
        tool_calls_made = []
        execution_results = []
        errors = []
        turns = 0
        
        for turn in range(max_turns):
            turns += 1
            
            # Call model with tools
            try:
                response = adapter.generate(
                    conversation_history,
                    temperature=0.0,
                    max_tokens=1000,
                    tools=available_tools
                )
            except Exception as e:
                errors.append(f"Turn {turn+1}: Generation error - {str(e)}")
                break

            if response.get("error"):
                # A real infrastructure failure (retries exhausted) — don't
                # score an empty/absent tool trace as if the model chose not
                # to call tools. Callers must check `generation_error` and
                # exclude this item.
                return {
                    "generation_error": response["error"],
                    "success": False,
                    "turns": turns,
                    "tool_calls": [],
                    "execution_results": [],
                    "final_response": "",
                    "judge_score": None,
                    "judge_reasoning": None,
                    "errors": errors,
                    "conversation_history": conversation_history,
                }

            # Check if model wants to call tools
            tool_calls = response.get("tool_calls", [])
            
            if not tool_calls:
                # Model provided final answer without tools
                final_response = response.get("content", "")
                conversation_history.append({
                    "role": "assistant",
                    "content": final_response
                })
                break
            
            # Execute tool calls
            assistant_message = {
                "role": "assistant",
                "content": response.get("content", ""),
                "tool_calls": []
            }

            tool_results_to_append = []

            for tool_call in tool_calls:
                # Adapter normalizes to flat {id, name, arguments}; also accept
                # raw OpenAI {id, type, function: {name, arguments}} format.
                tool_name = (
                    tool_call.get("name")
                    or tool_call.get("function", {}).get("name")
                )
                raw_args = (
                    tool_call.get("arguments")
                    or tool_call.get("function", {}).get("arguments", "{}")
                )

                # Skip malformed tool calls (name is required by vLLM)
                if not tool_name:
                    errors.append(f"Turn {turn+1}: Skipping tool call with no name")
                    continue

                # Parse arguments
                try:
                    tool_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except json.JSONDecodeError:
                    errors.append(f"Turn {turn+1}: Invalid JSON in tool arguments for {tool_name}")
                    tool_args = {}

                tool_call_id = tool_call.get("id") or f"call_{turn}_{tool_name}"

                # Record tool call
                tool_calls_made.append({
                    "turn": turn + 1,
                    "tool_name": tool_name,
                    "arguments": tool_args
                })

                # Execute tool
                execution_result = self.mock_env.execute_tool(tool_name, tool_args)
                execution_results.append(execution_result)

                # Accumulate into assistant message (OpenAI wire format)
                assistant_message["tool_calls"].append({
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args)
                    }
                })

                # Collect tool result messages – append after assistant message
                tool_results_to_append.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": json.dumps(execution_result)
                })

            # Append assistant message once (outside tool_call loop)
            if assistant_message["tool_calls"]:
                conversation_history.append(assistant_message)
                for tool_result_msg in tool_results_to_append:
                    conversation_history.append(tool_result_msg)
        
        # Get final response if not already obtained
        final_response = ""
        if turns == max_turns and tool_calls:
            # Model hit max turns, ask for final answer
            try:
                final_gen = adapter.generate(
                    conversation_history + [{"role": "user", "content": "Lütfen sonucu özetle."}],
                    temperature=0.0,
                    max_tokens=500
                )
                if final_gen.get("error"):
                    # Only the closing summary call failed — the turns
                    # already completed (tool_calls_made/execution_results)
                    # are still real data, so degrade gracefully here rather
                    # than excluding the whole item.
                    errors.append(f"Final response error: {final_gen['error']}")
                    final_response = "[Error getting final response]"
                else:
                    final_response = final_gen.get("content", "") or ""
            except Exception as e:
                errors.append(f"Final response error: {str(e)}")
                final_response = "[Error getting final response]"
        else:
            # Get last assistant message
            for msg in reversed(conversation_history):
                if msg["role"] == "assistant" and msg.get("content"):
                    final_response = msg["content"]
                    break
        
        # Judge the result
        judge_score = None
        judge_reasoning = None
        
        if self.judge and expected_outcome:
            judge_score, judge_reasoning = self._judge_tool_use_outcome(
                initial_prompt=initial_prompt,
                tool_calls=tool_calls_made,
                execution_results=execution_results,
                final_response=final_response,
                expected_outcome=expected_outcome
            )
        
        return {
            "success": len(errors) == 0 and len(tool_calls_made) > 0,
            "turns": turns,
            "tool_calls": tool_calls_made,
            "execution_results": execution_results,
            "final_response": final_response,
            "judge_score": judge_score,
            "judge_reasoning": judge_reasoning,
            "errors": errors,
            "conversation_history": conversation_history
        }
    
    def _judge_tool_use_outcome(
        self,
        initial_prompt: str,
        tool_calls: List[Dict],
        execution_results: List[Dict],
        final_response: str,
        expected_outcome: str
    ) -> tuple[float, str]:
        """
        Use LLM judge to evaluate if tool use achieved expected outcome.
        
        Returns:
            (score, reasoning) where score is between 0 and 1
        """
        judge_prompt = f"""
Bir yapay zeka modelinin tool kullanımını değerlendiriyorsun.

Kullanıcı İsteği:
{initial_prompt}

Model'in Yaptığı Tool Çağrıları:
{json.dumps(tool_calls, indent=2, ensure_ascii=False)}

Tool Execution Sonuçları:
{json.dumps(execution_results, indent=2, ensure_ascii=False)}

Model'in Final Yanıtı:
{final_response}

Beklenen Sonuç:
{expected_outcome}

Aşağıdaki kriterlere göre değerlendir:
1. Model doğru tool'ları seçti mi?
2. Tool parametreleri doğru mu?
3. Tool chain (ardışık tool kullanımı) mantıklı mı?
4. Final yanıt kullanıcının isteğini karşılıyor mu?
5. Beklenen sonuca ulaşıldı mı?

JSON formatında yanıt ver:
{{"score": <0.0-1.0 arası>, "reasoning": "<detaylı açıklama>"}}
"""
        
        messages = [
            {"role": "system", "content": "Sen objektif bir AI değerlendirme uzmanısın."},
            {"role": "user", "content": judge_prompt}
        ]
        
        try:
            result = self.judge.generate(messages, temperature=0.0, max_tokens=500)
            response_text = result.get("content", "")
            
            # Extract JSON
            import re
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("score", 0.0), data.get("reasoning", "")
            else:
                return 0.5, "Could not parse judge response"
        
        except Exception as e:
            return 0.5, f"Judge error: {str(e)}"
    
    def evaluate_tool_chain(
        self,
        adapter: UnifiedLLMAdapter,
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a specific tool chain scenario.
        
        scenario format:
        {
            "prompt": str,
            "available_tools": List[Dict],
            "expected_tools": List[str],  # Expected tools to be called
            "expected_order": bool,  # Whether order matters
            "expected_outcome": str,
            "is_parallel": bool  # Whether parallel execution is expected
        }
        """
        result = self.evaluate_multi_turn_tool_use(
            adapter=adapter,
            initial_prompt=scenario["prompt"],
            available_tools=scenario.get("available_tools"),
            max_turns=scenario.get("max_turns", 5),
            expected_outcome=scenario.get("expected_outcome")
        )
        if result.get("generation_error"):
            # Pass the signal through unchanged — computing tools_match/
            # parallel_execution from an empty trace we never actually
            # observed would fabricate a "didn't call the right tools"
            # verdict for what was really an infrastructure failure.
            return result

        # Check if expected tools were called
        expected_tools = scenario.get("expected_tools", [])
        called_tools = [tc["tool_name"] for tc in result["tool_calls"]]
        
        if scenario.get("expected_order", False):
            # Order matters
            tools_match = called_tools == expected_tools
        else:
            # Order doesn't matter
            tools_match = set(called_tools) == set(expected_tools)
        
        result["tools_match"] = tools_match
        result["expected_tools"] = expected_tools
        result["called_tools"] = called_tools
        
        # Check for parallel execution if expected
        if scenario.get("is_parallel", False):
            parallel_result = self._evaluate_parallel_execution(
                result["tool_calls"],
                result["conversation_history"]
            )
            result["parallel_execution"] = parallel_result
        
        return result
    
    def _evaluate_parallel_execution(
        self,
        tool_calls: List[Dict],
        conversation_history: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate if model executed tools in parallel when possible.
        
        Returns:
            {
                "detected_parallel": bool,
                "parallel_groups": List[List[Dict]],
                "efficiency_score": float,
                "details": str
            }
        """
        # Group tool calls by turn
        turn_groups = {}
        for tc in tool_calls:
            turn = tc.get("turn", 1)
            if turn not in turn_groups:
                turn_groups[turn] = []
            turn_groups[turn].append(tc)
        
        # Find parallel calls (multiple tools in same turn)
        parallel_groups = [tools for tools in turn_groups.values() if len(tools) > 1]
        
        detected_parallel = len(parallel_groups) > 0
        total_tools = len(tool_calls)
        parallel_tools = sum(len(group) for group in parallel_groups)
        
        # Calculate efficiency
        # If tools could be parallelized but weren't, score is lower
        efficiency_score = 0.0
        
        if detected_parallel:
            # Model used parallel execution
            efficiency_score = min(1.0, parallel_tools / total_tools)
            details = f"Model executed {parallel_tools}/{total_tools} tools in parallel across {len(parallel_groups)} turn(s)"
        else:
            # Model did not use parallel execution
            if total_tools >= 2:
                # Check if tools were independent (could have been parallelized)
                independent_tools = self._check_tool_independence(tool_calls)
                if independent_tools > 2:
                    efficiency_score = 0.3  # Penalty for not parallelizing
                    details = f"Model executed {total_tools} tools sequentially. {independent_tools} tools were independent and could be parallelized."
                else:
                    efficiency_score = 1.0  # Tools were dependent, sequential is correct
                    details = f"Model correctly executed {total_tools} dependent tools sequentially"
            else:
                efficiency_score = 1.0
                details = "Only 1 tool called, parallelization not applicable"
        
        return {
            "detected_parallel": detected_parallel,
            "parallel_groups": parallel_groups,
            "efficiency_score": efficiency_score,
            "details": details,
            "total_tools": total_tools,
            "parallel_tools": parallel_tools
        }
    
    def _check_tool_independence(self, tool_calls: List[Dict]) -> int:
        """
        Check how many tools are independent (don't depend on previous results).
        
        Simple heuristic: Tools with the same name and different arguments
        are likely independent (e.g., get_weather for different cities).
        """
        independent_count = 0
        
        # Group by tool name
        tool_groups = {}
        for tc in tool_calls:
            tool_name = tc.get("tool_name", "")
            if tool_name not in tool_groups:
                tool_groups[tool_name] = []
            tool_groups[tool_name].append(tc)
        
        # Count independent calls
        for tool_name, calls in tool_groups.items():
            if len(calls) > 1:
                # Multiple calls to same tool with different args = likely independent
                independent_count += len(calls)
        
        return independent_count


def evaluate_dynamic_function_calling(
    adapter: UnifiedLLMAdapter,
    test_cases: List[Dict[str, Any]],
    judge_adapter: Optional[UnifiedLLMAdapter] = None
) -> Dict[str, Any]:
    """
    Run dynamic function calling evaluation on multiple test cases.
    
    Args:
        adapter: LLM to test
        test_cases: List of test scenarios
        judge_adapter: Optional judge for evaluation
    
    Returns:
        Evaluation results with scores and details
    """
    evaluator = DynamicFunctionCallingEvaluator(judge_adapter)
    
    results = []
    total_score = 0.0
    successful_calls = 0
    total_calls = 0
    
    for i, test_case in enumerate(test_cases):
        print(f"Running test case {i+1}/{len(test_cases)}: {test_case.get('id', 'unknown')}")
        
        result = evaluator.evaluate_tool_chain(adapter, test_case)
        
        # Calculate score
        score = 0.0
        if result["success"]:
            score += 0.3
        if result.get("tools_match"):
            score += 0.3
        if result.get("judge_score") is not None:
            score += result["judge_score"] * 0.4
        else:
            score += 0.2  # partial credit if no judge
        
        result["score"] = score
        result["test_id"] = test_case.get("id", f"test_{i+1}")
        results.append(result)
        
        total_score += score
        if result["success"]:
            successful_calls += 1
        total_calls += 1
    
    return {
        "test_results": results,
        "summary": {
            "total_tests": total_calls,
            "successful": successful_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0.0,
            "average_score": total_score / total_calls if total_calls > 0 else 0.0
        }
    }
