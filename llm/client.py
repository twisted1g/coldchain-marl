from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import httpx

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_JSON_RETRIES = 3

Message = dict[str, str]


class LLMError(RuntimeError):
    """Raised when the backend fails or returns an unusable response."""


@dataclass(frozen=True)
class LLMConfig:
    """Connection settings for any OpenAI-compatible backend.

    Defaults target a local LM Studio server; override ``base_url`` /
    ``api_key`` (or set LLM_BASE_URL / LLM_MODEL / LLM_API_KEY) to point
    at Ollama, vLLM, OpenRouter, etc.
    """

    base_url: str = DEFAULT_BASE_URL
    model: str = ""
    api_key: str = "not-needed"
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_json_retries: int = DEFAULT_MAX_JSON_RETRIES
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides: Any) -> LLMConfig:
        env = {
            "base_url": os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
            "model": os.environ.get("LLM_MODEL", ""),
            "api_key": os.environ.get("LLM_API_KEY", "not-needed"),
        }
        env.update(overrides)
        return cls(**env)


@runtime_checkable
class ChatClient(Protocol):
    """Minimal surface the rest of the project depends on."""

    def complete(self, messages: list[Message], **kwargs: Any) -> str: ...

    def complete_json(
        self,
        messages: list[Message],
        schema: dict[str, Any],
        schema_name: str = "response",
        **kwargs: Any,
    ) -> dict[str, Any]: ...


class OpenAICompatClient:
    """Chat client for any /v1/chat/completions endpoint."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._http = httpx.Client(
            base_url=self.config.base_url,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout=self.config.timeout_s,
        )

    def complete(self, messages: list[Message], **kwargs: Any) -> str:
        payload = self._build_payload(messages, **kwargs)
        return self._request(payload)

    def complete_json(
        self,
        messages: list[Message],
        schema: dict[str, Any],
        schema_name: str = "response",
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = self._build_payload(messages, **kwargs)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }
        last_error: Exception | None = None
        for _ in range(self.config.max_json_retries):
            text = self._request(payload)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(parsed, dict):
                return parsed
            last_error = LLMError(f"expected JSON object, got {type(parsed).__name__}")
        raise LLMError(
            f"no valid JSON after {self.config.max_json_retries} attempts"
        ) from last_error

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OpenAICompatClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _build_payload(self, messages: list[Message], **kwargs: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.config.temperature),
            **self.config.extra_body,
            **kwargs,
        }
        max_tokens = payload.pop("max_tokens", self.config.max_tokens)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    def _request(self, payload: dict[str, Any]) -> str:
        try:
            response = self._http.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"chat completion request failed: {exc}") from exc
        body = response.json()
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"malformed completion response: {body!r}") from exc
        return _extract_content(message)


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _extract_content(message: dict[str, Any]) -> str:
    """Local reasoning models (via LM Studio et al.) may emit their answer inside
    <think> tags or entirely in reasoning_content, leaving content empty."""
    content = _THINK_BLOCK.sub("", message.get("content") or "").strip()
    if content:
        return content
    return (message.get("reasoning_content") or "").strip()
