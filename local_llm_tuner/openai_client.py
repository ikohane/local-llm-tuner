"""OpenAI client with the same interface as OllamaClient.

Exists primarily so CascadingLLMClient can fall back from a local model
(Gemma via Ollama) to a hosted model (OpenAI) on per-chunk failure.

Uses the /v1/chat/completions endpoint via urllib — no openai SDK
dependency. API key read from OPENAI_API_KEY unless passed explicitly.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional
from urllib.request import Request, urlopen


DEFAULT_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-5.4-mini"


class OpenAIClient:
    """OpenAI Chat Completions client.

    Interface matches OllamaClient: `.chat(user, *, system, schema,
    temperature, extra_options) -> (content, stats)`.

    Schema handling: if `schema` is provided, the client passes
    `response_format={"type": "json_schema", "json_schema": {...}}`.
    Note that OpenAI's schema support is stricter than Ollama's — some
    schemas that work on Gemma4 via Ollama may be rejected here. When in
    doubt, use schema=None and rely on prompt-engineering for structure.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OPENAI_URL,
        timeout: int = 600,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set (pass api_key= or set env var)."
            )
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def chat(
        self,
        user: str,
        *,
        system: Optional[str] = None,
        schema: Optional[dict] = None,
        temperature: Optional[float] = None,
        extra_options: Optional[dict] = None,
    ) -> tuple[str, dict]:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload: dict = {"model": self.model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if schema is not None:
            # OpenAI's structured output wraps the JSON Schema under
            # response_format.json_schema. Schema must be named and have
            # additionalProperties handling; we wrap minimally.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": False,
                },
            }
        if extra_options:
            payload.update(extra_options)

        req = Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        t0 = time.time()
        with urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        elapsed = time.time() - t0

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        content = message.get("content", "") or ""
        usage = data.get("usage", {}) or {}
        stats = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "seconds": round(elapsed, 2),
            "done_reason": choice.get("finish_reason", ""),
            "content_chars": len(content),
            "thinking_chars": 0,  # OpenAI doesn't return reasoning-mode stats here
            "model": self.model,
        }
        return content, stats
