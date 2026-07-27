import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_unified_adapter_module():
    fake_openai = types.ModuleType("openai")

    class _BaseClient:
        instances = []
        should_fail_times = 0  # class-level: settable by a test before adapter init

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.create_calls = []
            self._calls_made = 0
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create)
            )
            self.__class__.instances.append(self)

        def _create(self, **kwargs):
            self.create_calls.append(kwargs)
            self._calls_made += 1
            if self._calls_made <= self.__class__.should_fail_times:
                raise RuntimeError(f"simulated transient error (call {self._calls_made})")
            message = types.SimpleNamespace(content="ok", tool_calls=None)
            choice = types.SimpleNamespace(message=message)
            usage = types.SimpleNamespace(prompt_tokens=11, completion_tokens=5)
            return types.SimpleNamespace(choices=[choice], usage=usage)

    class OpenAI(_BaseClient):
        instances = []

    class AzureOpenAI(_BaseClient):
        instances = []

    fake_openai.OpenAI = OpenAI
    fake_openai.AzureOpenAI = AzureOpenAI
    sys.modules["openai"] = fake_openai

    fake_anthropic = types.ModuleType("anthropic")

    class Anthropic:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs

    fake_anthropic.Anthropic = Anthropic
    sys.modules["anthropic"] = fake_anthropic

    fake_tiktoken = types.ModuleType("tiktoken")
    sys.modules["tiktoken"] = fake_tiktoken

    module_path = Path(__file__).resolve().parent.parent / "adapters" / "unified_adapter.py"
    spec = importlib.util.spec_from_file_location("isolated_unified_adapter", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, OpenAI, AzureOpenAI


class UnifiedLLMAdapterContractTests(unittest.TestCase):
    def test_openai_generation_forwards_supported_request_params(self):
        module, OpenAI, _ = _load_unified_adapter_module()
        adapter = module.UnifiedLLMAdapter(
            {
                "provider": "openai",
                "model_name": "demo-model",
                "base_url": "http://localhost:8000/v1",
                "api_key": "dummy",
                "supports_function_calling": True,
                "supports_response_format": True,
                "top_p": 0.85,
                "seed": 42,
                "extra_body": {"best_of": 2},
            }
        )

        result = adapter._generate_openai(
            [{"role": "user", "content": "hello"}],
            [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            0.2,
            512,
            tool_choice="required",
            response_format={"type": "json_object"},
        )

        self.assertEqual(len(OpenAI.instances), 1)
        request = OpenAI.instances[0].create_calls[0]
        self.assertEqual(request["model"], "demo-model")
        self.assertEqual(request["temperature"], 0.2)
        self.assertEqual(request["max_tokens"], 512)
        self.assertEqual(request["top_p"], 0.85)
        self.assertEqual(request["seed"], 42)
        self.assertEqual(request["tool_choice"], "required")
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["extra_body"], {"best_of": 2})
        self.assertEqual(result["usage"]["input_tokens"], 11)
        self.assertEqual(result["usage"]["output_tokens"], 5)

    def test_azure_detection_normalizes_provider_name(self):
        module, _, AzureOpenAI = _load_unified_adapter_module()
        adapter = module.UnifiedLLMAdapter(
            {
                "provider": "openai",
                "model_name": "azure-model",
                "base_url": "https://my-resource.openai.azure.com/",
                "api_key": "secret",
                "api_version": "2024-02-15-preview",
            }
        )

        self.assertEqual(adapter.provider, "azure")
        self.assertEqual(len(AzureOpenAI.instances), 1)
        self.assertEqual(AzureOpenAI.instances[0].init_kwargs["azure_endpoint"], "https://my-resource.openai.azure.com")

    def test_generate_retries_transient_errors_and_succeeds(self):
        module, OpenAI, _ = _load_unified_adapter_module()
        OpenAI.should_fail_times = 2  # fails twice, succeeds on the 3rd (last) attempt
        adapter = module.UnifiedLLMAdapter(
            {
                "provider": "openai",
                "model_name": "demo-model",
                "base_url": "http://localhost:8000/v1",
                "api_key": "dummy",
            }
        )

        with patch("time.sleep"):
            result = adapter.generate([{"role": "user", "content": "hello"}])

        self.assertEqual(len(OpenAI.instances[0].create_calls), 3)
        self.assertEqual(result["content"], "ok")
        self.assertNotIn("error", result)

    def test_generate_returns_graceful_error_after_exhausting_retries(self):
        module, OpenAI, _ = _load_unified_adapter_module()
        OpenAI.should_fail_times = 999  # always fails
        adapter = module.UnifiedLLMAdapter(
            {
                "provider": "openai",
                "model_name": "demo-model",
                "base_url": "http://localhost:8000/v1",
                "api_key": "dummy",
            }
        )

        with patch("time.sleep"):
            result = adapter.generate([{"role": "user", "content": "hello"}])

        # stop_after_attempt(3): exactly 3 real calls, no more, no crash.
        self.assertEqual(len(OpenAI.instances[0].create_calls), 3)
        self.assertIsNone(result["content"])
        self.assertIn("simulated transient error", result["error"])
        self.assertEqual(adapter.error_count, 1)


if __name__ == "__main__":
    unittest.main()