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


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot return a valid vector."""


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
        payload = _parse_json_object(raw)
        if not isinstance(payload, dict):
            raise LLMError("LLM returned JSON that is not an object.")
        return payload


class BaseEmbeddingClient(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


@dataclass(frozen=True, slots=True)
class LocalLlamaConfig:
    base_url: str
    model_name: str
    timeout_seconds: float
    temperature: float
    max_tokens: int


@dataclass(frozen=True, slots=True)
class LocalLlamaEmbeddingConfig:
    base_url: str
    model_name: str
    timeout_seconds: float


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
        body = self._post_json(
            f"{self.config.base_url.rstrip('/')}/api/generate",
            payload,
            timeout_seconds=self.config.timeout_seconds,
        )
        text = body.get("response")
        if not isinstance(text, str) or not text.strip():
            raise LLMError("Local Llama server returned an empty response.")
        return text.strip()

    @staticmethod
    def _post_json(url: str, payload: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = ""
            raise LLMError(f"Local Llama server returned HTTP {exc.code}: {error_body or exc.reason}") from exc
        except (URLError, TimeoutError) as exc:
            raise LLMError("Local Llama server is unavailable.") from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise LLMError("Local Llama server returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise LLMError("Local Llama server returned an invalid payload.")
        return parsed


class LocalLlamaEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, config: LocalLlamaEmbeddingConfig) -> None:
        self.config = config

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def embed_text(self, text: str) -> list[float]:
        payload = {
            "model": self.config.model_name,
            "prompt": text,
        }
        request = Request(
            url=f"{self.config.base_url.rstrip('/')}/api/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = ""
            raise EmbeddingError(
                f"Local embedding server returned HTTP {exc.code}: {error_body or exc.reason}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise EmbeddingError("Local embedding server is unavailable.") from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise EmbeddingError("Local embedding server returned invalid JSON.") from exc

        vector = parsed.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise EmbeddingError("Local embedding server returned an empty embedding.")
        try:
            return [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("Local embedding server returned a malformed embedding.") from exc


def create_llm_client(settings: Settings) -> BaseLLMClient | None:
    return LocalLlamaClient(
        LocalLlamaConfig(
            base_url=settings.llm_base_url,
            model_name=settings.llm_model_name,
            timeout_seconds=settings.llm_timeout_seconds,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    )


def create_embedding_client(settings: Settings) -> BaseEmbeddingClient | None:
    if not settings.embedding_model_name:
        return None
    return LocalLlamaEmbeddingClient(
        LocalLlamaEmbeddingConfig(
            base_url=settings.embedding_base_url,
            model_name=settings.embedding_model_name,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
    )


def _parse_json_object(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return payload

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise LLMError("LLM returned invalid JSON.") from exc
        if isinstance(payload, dict):
            return payload

    raise LLMError("LLM returned invalid JSON.")
