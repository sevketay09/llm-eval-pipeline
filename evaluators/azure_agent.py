"""
Azure Agent Evaluators - Wrapper for azure-ai-evaluation Agent evaluators.

Provides TaskAdherence, ToolCallAccuracy, ResponseCompleteness, and IntentResolution
evaluations for agentic workflows. Requires Azure OpenAI model_config (same as azure_quality.py).

These evaluators assess multi-turn agent conversations including tool calls,
function outputs, and final responses against the original user query/intent.
"""
import os
from typing import Dict, Any, List, Optional, Union
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from azure.ai.evaluation import (
        TaskAdherenceEvaluator,
        ToolCallAccuracyEvaluator,
        ResponseCompletenessEvaluator,
        IntentResolutionEvaluator,
    )
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False
    logger.warning("azure-ai-evaluation not installed. Agent evaluators unavailable.")


def _get_model_config() -> Optional[Dict[str, str]]:
    """Build model_config from environment variables."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    key = os.environ.get("AZURE_OPENAI_KEY")
    deployment = os.environ.get(
        "AZURE_OPENAI_DEPLOYMENT_NAME_PTU",
        os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_PR", "")
    )
    if endpoint and key and deployment:
        return {
            "azure_endpoint": endpoint,
            "api_key": key,
            "azure_deployment": deployment,
            "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        }
    return None


def is_agent_eval_available() -> bool:
    """Check if Agent evaluators can be used (SDK + env vars)."""
    return _SDK_AVAILABLE and _get_model_config() is not None


class AzureAgentEvaluator:
    """Wrapper for Azure AI Agent Evaluators.

    Evaluates agentic conversations across 4 dimensions:
    - TaskAdherence: Did the agent follow its instructions and complete the task?
    - ToolCallAccuracy: Were tool calls made with correct parameters?
    - ResponseCompleteness: Does the final response address all parts of the query?
    - IntentResolution: Was the user's original intent resolved?

    Usage:
        evaluator = AzureAgentEvaluator()

        # For full agent conversation (list of messages with tool calls)
        result = evaluator.evaluate_conversation(
            query=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the weather in Istanbul?"}
            ],
            response=[
                {"role": "assistant", "content": [{"type": "tool_call", "tool_call": {...}}]},
                {"role": "tool", "content": [{"type": "tool_result", "tool_result": "..."}]},
                {"role": "assistant", "content": [{"type": "text", "text": "It is 25°C in Istanbul."}]}
            ]
        )

        # For simple query/response (non-agent, still useful)
        result = evaluator.evaluate_simple(
            query="What is 2+2?",
            response="4"
        )
    """

    def __init__(self, model_config: Optional[Dict[str, str]] = None):
        """Initialize with Azure OpenAI model_config.

        Args:
            model_config: Dict with azure_endpoint, api_key, azure_deployment, api_version.
                         If None, reads from environment variables.
        """
        if not _SDK_AVAILABLE:
            raise ImportError("azure-ai-evaluation package required.")

        config = model_config or _get_model_config()
        if not config:
            raise ValueError(
                "Azure OpenAI config required. Set AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_KEY, and Azure deployment env vars."
            )

        self._task_adherence = TaskAdherenceEvaluator(model_config=config)
        self._tool_call_accuracy = ToolCallAccuracyEvaluator(model_config=config)
        self._response_completeness = ResponseCompletenessEvaluator(model_config=config)
        self._intent_resolution = IntentResolutionEvaluator(model_config=config)

    @staticmethod
    def _build_messages(
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
    ) -> List[dict]:
        """Build a flat list of message dicts from query + response, dropping empty content."""
        messages: List[dict] = []

        if isinstance(query, list):
            for msg in query:
                if isinstance(msg, dict) and msg.get("content"):
                    messages.append(msg)
        elif isinstance(query, str) and query.strip():
            messages.append({"role": "user", "content": query.strip()})

        if isinstance(response, list):
            for msg in response:
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if content and (isinstance(content, list) or (isinstance(content, str) and content.strip())):
                        messages.append(msg)
        elif isinstance(response, str) and response.strip():
            messages.append({"role": "assistant", "content": response.strip()})

        return messages

    @staticmethod
    def _conversation_dict(messages: List[dict]) -> Dict[str, Any]:
        """Wrap a message list into the ``{"messages": [...]}`` dict the SDK expects."""
        return {"messages": messages}

    @staticmethod
    def _extract_query_str(query: Union[str, List[dict]]) -> str:
        """Extract the first user-content string from query (for ToolCallAccuracy)."""
        if isinstance(query, str):
            return query.strip()
        if isinstance(query, list):
            for msg in query:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
        return ""

    def evaluate_task_adherence(
        self,
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
    ) -> Dict[str, Any]:
        """Evaluate if agent adhered to its task/instructions."""
        try:
            messages = self._build_messages(query, response)
            result = self._task_adherence(conversation=self._conversation_dict(messages))
            return {
                "score": result.get("task_adherence", 0.0),
                "reasoning": result.get("task_adherence_reason", ""),
                "result": result.get("task_adherence_result", ""),
                "raw": result,
            }
        except Exception as e:
            logger.warning(f"TaskAdherence evaluation failed: {e}")
            return {"score": 0.0, "reasoning": f"Error: {e}", "result": "error", "raw": {}}

    def evaluate_tool_call_accuracy(
        self,
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
        tool_definitions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate accuracy of tool calls made by the agent.

        ToolCallAccuracyEvaluator uses the separate ``query`` / ``response``
        interface (not ``conversation``), plus an optional ``tool_definitions`` list.
        """
        try:
            query_str = self._extract_query_str(query)
            response_messages = self._build_messages([], response)  # response portion only
            kwargs: Dict[str, Any] = {
                "query": query_str or "N/A",
                "response": response_messages,
            }
            if tool_definitions:
                kwargs["tool_definitions"] = tool_definitions
            result = self._tool_call_accuracy(**kwargs)
            return {
                "score": result.get("tool_call_accuracy", 0.0),
                "reasoning": result.get("tool_call_accuracy_reason", ""),
                "result": result.get("tool_call_accuracy_result", ""),
                "raw": result,
            }
        except Exception as e:
            logger.warning(f"ToolCallAccuracy evaluation failed: {e}")
            return {"score": 0.0, "reasoning": f"Error: {e}", "result": "error", "raw": {}}

    def evaluate_response_completeness(
        self,
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
    ) -> Dict[str, Any]:
        """Evaluate if the response fully addresses the query."""
        try:
            messages = self._build_messages(query, response)
            result = self._response_completeness(conversation=self._conversation_dict(messages))
            return {
                "score": result.get("response_completeness", 0.0),
                "reasoning": result.get("response_completeness_reason", ""),
                "result": result.get("response_completeness_result", ""),
                "raw": result,
            }
        except Exception as e:
            logger.warning(f"ResponseCompleteness evaluation failed: {e}")
            return {"score": 0.0, "reasoning": f"Error: {e}", "result": "error", "raw": {}}

    def evaluate_intent_resolution(
        self,
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
    ) -> Dict[str, Any]:
        """Evaluate if the user's intent was resolved."""
        try:
            messages = self._build_messages(query, response)
            result = self._intent_resolution(conversation=self._conversation_dict(messages))
            return {
                "score": result.get("intent_resolution", 0.0),
                "reasoning": result.get("intent_resolution_reason", ""),
                "result": result.get("intent_resolution_result", ""),
                "raw": result,
            }
        except Exception as e:
            logger.warning(f"IntentResolution evaluation failed: {e}")
            return {"score": 0.0, "reasoning": f"Error: {e}", "result": "error", "raw": {}}

    def evaluate_all(
        self,
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
        tool_definitions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run all agent evaluations.

        Returns:
            Dict with all 4 dimension scores + aggregate.
        """
        task = self.evaluate_task_adherence(query, response)
        tool = self.evaluate_tool_call_accuracy(query, response, tool_definitions=tool_definitions)
        completeness = self.evaluate_response_completeness(query, response)
        intent = self.evaluate_intent_resolution(query, response)

        scores = [task["score"], tool["score"], completeness["score"], intent["score"]]
        valid_scores = [s for s in scores if s > 0]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        return {
            "task_adherence": task,
            "tool_call_accuracy": tool,
            "response_completeness": completeness,
            "intent_resolution": intent,
            "aggregate_score": avg_score,
        }

    def evaluate_simple(
        self,
        query: str,
        response: str,
    ) -> Dict[str, Any]:
        """Simplified evaluation for non-agent query/response pairs.

        Runs only ResponseCompleteness and IntentResolution (no tool calls).
        """
        completeness = self.evaluate_response_completeness(query, response)
        intent = self.evaluate_intent_resolution(query, response)

        scores = [completeness["score"], intent["score"]]
        valid_scores = [s for s in scores if s > 0]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        return {
            "response_completeness": completeness,
            "intent_resolution": intent,
            "aggregate_score": avg_score,
        }
