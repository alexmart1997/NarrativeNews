from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config.settings import Settings


class LLMError(RuntimeError):
    """Raised when an LLM provider cannot return a valid response."""


class BaseLLMClient(ABC):
    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        raw = self.generate_text(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError("LLM returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise LLMError("LLM returned JSON that is not an object.")
        return payload


@dataclass(frozen=True, slots=True)
class NarrativeLabel:
    title: str
    formulation: str
    explanation: str


@dataclass(frozen=True, slots=True)
class LocalLlamaConfig:
    base_url: str
    model_name: str
    timeout_seconds: float
    temperature: float
    max_tokens: int


class LocalLlamaClient(BaseLLMClient):
    def __init__(self, config: LocalLlamaConfig) -> None:
        self.config = config

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {
                "temperature": self.config.temperature if temperature is None else temperature,
                "num_predict": self.config.max_tokens if max_tokens is None else max_tokens,
            },
        }
        request = Request(
            url=f"{self.config.base_url.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise LLMError("Local Llama server is unavailable.") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError("Local Llama server returned invalid JSON.") from exc

        text = parsed.get("response")
        if not isinstance(text, str) or not text.strip():
            raise LLMError("Local Llama server returned an empty response.")
        return text.strip()


class SimpleExtractiveLLMClient(BaseLLMClient):
    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return prompt.strip()


def create_llm_client(settings: Settings) -> BaseLLMClient | None:
    if settings.llm_provider != "local_llama":
        return None
    return LocalLlamaClient(
        LocalLlamaConfig(
            base_url=settings.llm_base_url,
            model_name=settings.llm_model_name,
            timeout_seconds=settings.llm_timeout_seconds,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    )
