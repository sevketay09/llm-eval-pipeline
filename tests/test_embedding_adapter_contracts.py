import importlib.util
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np


def _load_embedding_adapter_class():
    module_path = Path(__file__).resolve().parent.parent / "adapters" / "embedding_adapter.py"
    spec = importlib.util.spec_from_file_location("isolated_embedding_adapter", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.UnifiedEmbeddingAdapter


class EmbeddingAdapterContractTests(unittest.TestCase):
    def test_encode_normalizes_single_string_input_and_tracks_usage(self):
        UnifiedEmbeddingAdapter = _load_embedding_adapter_class()
        config = {
            "provider": "openai",
            "model_name": "text-embedding-3-small",
            "base_url": "http://localhost:8000",
            "api_key": "dummy",
        }

        adapter = UnifiedEmbeddingAdapter(config)

        with patch.object(
            adapter,
            "_encode_api",
            return_value=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        ) as encode_api:
            result = adapter.encode("merhaba dunya", normalize=False)

        encode_api.assert_called_once()
        call_args = encode_api.call_args.args
        self.assertEqual(call_args[0], ["merhaba dunya"])
        self.assertEqual(call_args[1], False)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["model"], "text-embedding-3-small")
        self.assertEqual(adapter.total_embeddings, 1)
        self.assertEqual(result["embeddings"].shape, (1, 3))
        self.assertEqual(len(adapter.latencies), 1)

    def test_unsupported_provider_raises_clear_error(self):
        UnifiedEmbeddingAdapter = _load_embedding_adapter_class()
        with self.assertRaises(ValueError) as context:
            UnifiedEmbeddingAdapter(
                {
                    "provider": "unsupported-provider",
                    "model_name": "demo-embed",
                }
            )

        self.assertIn("Unsupported embedding provider", str(context.exception))


if __name__ == "__main__":
    unittest.main()