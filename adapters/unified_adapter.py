"""
Unified LLM Adapter
Supports: OpenAI API, Azure OpenAI, vLLM, Ollama, LM Studio (OpenAI compatible), Anthropic Claude
"""
import re
import time
import json
import asyncio
import ssl
import certifi
import threading
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic
import tiktoken
import logging as _logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from utils.logger import get_logger

logger = get_logger(__name__)


class UnifiedLLMAdapter:
    """Unified interface for all LLM providers"""
    
    def __init__(self, config: Dict[str, Any], model_key: Optional[str] = None):
        self.config = config
        self.model_key = model_key or config.get("model_key", "unknown")
        self.provider = config.get("provider", "openai")
        self.model_name = config["model_name"]
        self.supports_function_calling = config.get("supports_function_calling", False)
        self.supports_response_format = config.get("supports_response_format", True)
        self.quirks = config.get("quirks", [])
        
        logger.debug(f"Initializing adapter for {self.model_name} (provider: {self.provider})")
        
        # Initialize appropriate client
        if self.provider == "mock":
            # Offline demo provider: deterministic canned responses, no network.
            self.client = None
        elif self.provider == "anthropic":
            self.client = Anthropic(api_key=config["api_key"])
        elif self.provider == "azure" or "azure" in config.get("base_url", "").lower() or "api_version" in config:
            # Azure OpenAI
            endpoint = config["base_url"].rstrip("/")
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=config["api_key"],
                api_version=config.get("api_version", "2024-02-15-preview")
            )
            self.provider = "azure"  # Normalize provider name
        else:
            # OpenAI or OpenAI-compatible (vLLM, Ollama, LM Studio)
            kwargs = {"api_key": config.get("api_key", "dummy"), "timeout": 120.0}
            if config.get("base_url"):
                kwargs["base_url"] = config["base_url"]
            self.client = OpenAI(**kwargs)
        
        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Performance tracking
        self.latencies = []
        self.error_count = 0
        self.timeout_count = 0

        # Lock for stats mutation (concurrent item processing)
        self._stats_lock = threading.Lock()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, _logging.WARNING),
    )
    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate completion with unified interface (with retry logic)
        
        Returns:
            {
                'content': str,
                'tool_calls': List[Dict] or None,
                'usage': {'input_tokens': int, 'output_tokens': int},
                'latency': float,
                'model': str
            }
        """
        start_time = time.time()
        
        # Set defaults
        temperature = temperature if temperature is not None else self.config.get("temperature", 0.0)
        max_tokens = max_tokens if max_tokens is not None else self.config.get("max_tokens", 4096)

        # Global runtime overrides from pipeline should win over per-test defaults
        if self.config.get("force_temperature") is not None:
            temperature = float(self.config["force_temperature"])
        if self.config.get("force_max_tokens") is not None:
            max_tokens = int(self.config["force_max_tokens"])
        
        logger.debug(f"[{self.model_name}] → API call | messages={len(messages)} temperature={temperature} max_tokens={max_tokens}")

        try:
            if self.provider == "mock":
                result = self._generate_mock(messages, tools, **kwargs)
            elif self.provider == "anthropic":
                result = self._generate_anthropic(messages, tools, temperature, max_tokens, **kwargs)
            else:
                result = self._generate_openai(messages, tools, temperature, max_tokens, **kwargs)

            latency = time.time() - start_time
            result['latency'] = latency
            with self._stats_lock:
                self.latencies.append(latency)
                self.total_input_tokens += result['usage']['input_tokens']
                self.total_output_tokens += result['usage']['output_tokens']

            logger.debug(f"[{self.model_name}] ← response | {result['usage']['output_tokens']} tokens in {latency:.2f}s")
            return result

        except Exception as e:
            latency = time.time() - start_time
            is_timeout = "timeout" in str(e).lower() or "timed out" in str(e).lower()
            with self._stats_lock:
                self.latencies.append(latency)
                self.error_count += 1
                if is_timeout:
                    self.timeout_count += 1
            if is_timeout:
                logger.error(f"[{self.model_name}] TIMEOUT after {latency:.2f}s: {type(e).__name__}: {e}")
            else:
                logger.error(f"[{self.model_name}] ERROR after {latency:.2f}s: {type(e).__name__}: {e}")
            return {
                'content': None,
                'tool_calls': None,
                'usage': {'input_tokens': 0, 'output_tokens': 0},
                'latency': latency,
                'model': self.model_name,
                'error': str(e)
            }
    
    def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """OpenAI/Azure/vLLM/Ollama/LM Studio generation"""
        
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add top_p if configured
        if "top_p" in self.config:
            params["top_p"] = self.config["top_p"]
        
        # Add seed for reproducibility
        if "seed" in self.config:
            params["seed"] = self.config["seed"]
        
        # Add tools if supported and provided
        if tools and self.supports_function_calling:
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")

        if "response_format" in kwargs and kwargs["response_format"] and self.supports_response_format:
            params["response_format"] = kwargs["response_format"]
        
        # Build extra_body for vLLM-specific parameters
        extra_body = self.config.get("extra_body", {})
        if extra_body:
            params["extra_body"] = extra_body
        
        response = self.client.chat.completions.create(**params)
        
        # Parse response
        message = response.choices[0].message
        raw = message.content or ""
        content = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        tool_calls = None
        
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": tc.function.arguments}
                tool_calls.append({
                    'id': tc.id,
                    'name': tc.function.name,
                    'arguments': arguments
                })
        
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        result = {
            'content': content,
            'tool_calls': tool_calls,
            'usage': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            },
            'model': self.model_name
        }
        return result
    
    def _generate_anthropic(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Anthropic Claude generation"""
        
        # Convert messages format
        system_message = None
        claude_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                claude_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        params = {
            "model": self.model_name,
            "messages": claude_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if system_message:
            params["system"] = system_message
        
        # Add tools if supported
        if tools and self.supports_function_calling:
            params["tools"] = self._convert_tools_to_anthropic(tools)
        
        response = self.client.messages.create(**params)
        
        # Parse response
        content = ""
        tool_calls = None
        
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    'id': block.id,
                    'name': block.name,
                    'arguments': block.input
                })
        
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
        return {
            'content': content,
            'tool_calls': tool_calls,
            'usage': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            },
            'model': self.model_name
        }
    
    def _generate_mock(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]],
        **kwargs
    ) -> Dict[str, Any]:
        """Deterministic offline responses for demo runs (provider: mock).

        - Judge-style prompts (mention JSON output) get a fixed positive verdict
          covering every judge schema in the pipeline (label / score / result).
        - Tool-enabled calls answer with a call to the first available tool.
        - Everything else gets a short canned answer echoing the question.
        """
        prompt_text = " ".join(str(m.get("content") or "") for m in messages)
        content = None
        tool_calls = None

        if "JSON" in prompt_text or "json" in prompt_text:
            content = json.dumps({
                "label": "TAM_DOGRU",
                "score": 4,
                "reasoning": "Demo mock judge: sabit olumlu karar (gerçek bir değerlendirme değildir).",
                "result": "pass",
            }, ensure_ascii=False)
        elif tools:
            tool = tools[0].get("function", tools[0])
            required = (tool.get("parameters") or {}).get("required", [])
            props = (tool.get("parameters") or {}).get("properties", {})
            arguments = {}
            for param in required:
                param_type = (props.get(param) or {}).get("type", "string")
                arguments[param] = 1 if param_type in ("integer", "number") else "demo"
            tool_calls = [{
                "id": "mock-call-1",
                "name": tool.get("name", "unknown_tool"),
                "arguments": arguments,
            }]
            content = ""
        else:
            last_user = next(
                (m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            content = (
                "Bu bir demo yanıtıdır (mock model). Soru özeti: "
                + last_user[:160]
            )

        input_tokens = max(1, len(prompt_text) // 4)
        output_tokens = max(1, len(content or "") // 4)
        time.sleep(0.01)  # tiny non-zero latency so throughput metrics stay sane

        return {
            'content': content,
            'tool_calls': tool_calls,
            'usage': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            },
            'model': self.model_name
        }

    def _convert_tools_to_anthropic(self, openai_tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI tool format to Anthropic format"""
        anthropic_tools = []
        for tool in openai_tools:
            anthropic_tools.append({
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"]
            })
        return anthropic_tools
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_requests': len(self.latencies),
            'latency_avg': sum(self.latencies) / len(self.latencies) if self.latencies else 0,
            'latency_p50': self._percentile(self.latencies, 0.5),
            'latency_p95': self._percentile(self.latencies, 0.95),
            'latency_p99': self._percentile(self.latencies, 0.99),
            'error_count': self.error_count,
            'timeout_count': self.timeout_count
        }
    
    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def reset_stats(self):
        """Reset tracking statistics"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.latencies = []
        self.error_count = 0
        self.timeout_count = 0
