from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from app.services import LLMError, LocalLlamaClient, LocalLlamaConfig
from app.services.llm import _parse_json_object


class MockHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


class LocalLlamaClientTests(unittest.TestCase):
    def test_generate_text_calls_local_server(self) -> None:
        client = LocalLlamaClient(
            LocalLlamaConfig(
                base_url="http://127.0.0.1:11434",
                model_name="qwen2.5:7b",
                timeout_seconds=10,
                temperature=0.2,
                max_tokens=128,
            )
        )

        with patch("app.services.llm.urlopen", return_value=MockHTTPResponse({"response": "ok"})) as mocked:
            result = client.generate_text("Привет")

        self.assertEqual(result, "ok")
        self.assertEqual(mocked.call_count, 1)

    def test_parse_json_object_accepts_wrapped_json(self) -> None:
        payload = _parse_json_object('Here is the result:\n{"frames": []}\nThanks.')

        self.assertEqual(payload, {"frames": []})

    def test_parse_json_object_rejects_non_json(self) -> None:
        with self.assertRaises(LLMError):
            _parse_json_object("not json at all")


if __name__ == "__main__":
    unittest.main()
