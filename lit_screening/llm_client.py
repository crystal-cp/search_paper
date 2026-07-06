"""Generic OpenAI-compatible JSON chat client."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class LLMJSONResult:
    """Parsed JSON result from an LLM call."""

    data: dict[str, Any]
    invalid_llm_output: bool = False
    error_type: str = ""
    raw_text: str = ""


def strip_markdown_code_fences(text: str) -> str:
    """Remove common Markdown code fences around model JSON output."""

    cleaned = (text or "").strip()
    fence_match = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return cleaned


def parse_json_safely(text: str, fallback: dict[str, Any] | None = None) -> LLMJSONResult:
    """Parse JSON, returning a safe fallback when model output is invalid."""

    safe_fallback = fallback or {}
    cleaned = strip_markdown_code_fences(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return LLMJSONResult(
            data=safe_fallback,
            invalid_llm_output=True,
            error_type="invalid_json",
            raw_text=text,
        )
    if not isinstance(parsed, dict):
        return LLMJSONResult(
            data=safe_fallback,
            invalid_llm_output=True,
            error_type="json_not_object",
            raw_text=text,
        )
    return LLMJSONResult(data=parsed, raw_text=text)


class GenericLLMClient:
    """Minimal OpenAI-compatible client for JSON chat completions."""

    def __init__(
        self,
        provider_name: str,
        api_key_env_var: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.provider_name = provider_name
        self.api_key_env_var = api_key_env_var
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else os.getenv(api_key_env_var)
        self.timeout = timeout

    @property
    def is_available(self) -> bool:
        """Return whether the client has enough configuration to make a call."""

        return bool(self.api_key and self.base_url and self.model)

    def _chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def chat_json(self, system_prompt: str, user_prompt: str) -> LLMJSONResult:
        """Call a chat-completions API and parse the assistant message as JSON."""

        if not self.is_available:
            return LLMJSONResult(
                data={},
                invalid_llm_output=True,
                error_type="missing_api_key",
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self._chat_url(),
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw = response.json()
            content = (
                raw.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            return LLMJSONResult(
                data={},
                invalid_llm_output=True,
                error_type=exc.__class__.__name__,
            )

        return parse_json_safely(content, fallback={})
