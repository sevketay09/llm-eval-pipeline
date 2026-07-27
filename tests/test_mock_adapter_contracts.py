"""Contract tests for the offline mock provider (demo runs, no API keys)."""
import json
import unittest

from adapters.unified_adapter import UnifiedLLMAdapter


def make_adapter():
    return UnifiedLLMAdapter({
        "provider": "mock",
        "model_name": "demo-model",
        "api_key": "none",
        "max_tokens": 512,
    }, model_key="demo-model")


class MockProviderTests(unittest.TestCase):
    def test_init_without_credentials(self):
        adapter = make_adapter()
        self.assertIsNone(adapter.client)
        self.assertEqual(adapter.provider, "mock")

    def test_plain_generation_shape(self):
        adapter = make_adapter()
        result = adapter.generate([{"role": "user", "content": "Türkiye'nin başkenti neresidir?"}])
        self.assertIsInstance(result["content"], str)
        self.assertIn("demo", result["content"].lower())
        self.assertIsNone(result["tool_calls"])
        self.assertGreater(result["usage"]["input_tokens"], 0)
        self.assertGreater(result["usage"]["output_tokens"], 0)
        self.assertGreater(result["latency"], 0)
        self.assertEqual(result["model"], "demo-model")

    def test_judge_prompt_returns_valid_judge_json(self):
        adapter = make_adapter()
        result = adapter.generate([
            {"role": "system", "content": "Sen bir değerlendirme uzmanısın."},
            {"role": "user", "content": "Sadece JSON formatında yanıt ver: {\"score\": <1-5>}"},
        ])
        parsed = json.loads(result["content"])
        # One payload must satisfy every judge schema in the pipeline
        self.assertEqual(parsed["label"], "TAM_DOGRU")   # llm_judge
        self.assertIsInstance(parsed["score"], (int, float))  # quality/agent/groundedness judges
        self.assertEqual(parsed["result"], "pass")
        self.assertTrue(parsed["reasoning"])

    def test_tool_call_generation(self):
        adapter = make_adapter()
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather lookup",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                    "required": ["city", "days"],
                },
            },
        }]
        result = adapter.generate([{"role": "user", "content": "Ankara hava durumu"}], tools=tools)
        self.assertEqual(len(result["tool_calls"]), 1)
        call = result["tool_calls"][0]
        self.assertEqual(call["name"], "get_weather")
        self.assertEqual(call["arguments"], {"city": "demo", "days": 1})

    def test_deterministic_output(self):
        adapter = make_adapter()
        messages = [{"role": "user", "content": "Aynı soru"}]
        a = adapter.generate(messages)
        b = adapter.generate(messages)
        self.assertEqual(a["content"], b["content"])

    def test_stats_tracked(self):
        adapter = make_adapter()
        adapter.generate([{"role": "user", "content": "soru"}])
        stats = adapter.get_stats()
        self.assertEqual(stats["total_requests"], 1)
        self.assertEqual(stats["error_count"], 0)
        self.assertGreater(stats["total_output_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
