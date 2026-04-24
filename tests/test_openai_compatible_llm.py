from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from app.services import OpenAICompatibleClient, OpenAICompatibleConfig


class MockHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


class OpenAICompatibleClientTests(unittest.TestCase):
    def test_generate_text_calls_chat_completions_endpoint(self) -> None:
        client = OpenAICompatibleClient(
            OpenAICompatibleConfig(
                base_url="https://api.groq.com/openai/v1",
                api_key="test-key",
                model_name="llama-3.1-8b-instant",
                timeout_seconds=10,
                temperature=0.2,
                max_tokens=128,
            )
        )

        with patch(
            "app.services.llm.urlopen",
            return_value=MockHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "ok",
                            }
                        }
                    ]
                }
            ),
        ) as mocked:
            result = client.generate_text("Привет", system_prompt="Отвечай по-русски")

        self.assertEqual(result, "ok")
        self.assertEqual(mocked.call_count, 1)


if __name__ == "__main__":
    unittest.main()
