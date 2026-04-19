"""Ollama client with thinking-mode awareness.

The client implements a `chat(user, *, system, schema, temperature)` method
that returns `(content, stats)`. It speaks Ollama's /api/chat endpoint
directly via urllib — no ollama SDK dependency.

**Gemma4 thinking-mode caveat**: Gemma4:26b on Q4_K_M quantization spends
~2000 tokens of internal thinking before producing structured content.
When used with `format: <json_schema>`, the thinking tokens compete with
content tokens for the `num_predict` budget. If `num_predict` is too low,
the entire budget is consumed by thinking and `content_len=0`. See
docs/gemma4_thinking_mode_case_study.md for full findings.

Default `num_predict=4096` gives thinking ~2000 tokens of headroom while
leaving ~2000 for JSON output. For schemas with many required fields or
long strings, you may need to bump higher or drop the schema.
"""

from __future__ import annotations

import json
import time
from typing import Optional
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:26b"
DEFAULT_TIMEOUT = 1800  # seconds; schema-enforced decoding is 2-3x slower


class OllamaClient:
    """Minimal Ollama /api/chat client.

    Usage:
        client = OllamaClient(base_url="http://localhost:11434",
                              model="gemma4:26b")
        content, stats = client.chat(
            "Summarize this: ...",
            system="You are a summarizer.",
            schema={"type": "object", "properties": {...}},
            temperature=0.15,
        )
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
        default_num_predict_schema: int = 4096,
        default_num_predict_free: Optional[int] = None,
        default_repeat_penalty_schema: float = 1.3,
        default_top_p_schema: float = 0.9,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.default_num_predict_schema = default_num_predict_schema
        self.default_num_predict_free = default_num_predict_free
        self.default_repeat_penalty_schema = default_repeat_penalty_schema
        self.default_top_p_schema = default_top_p_schema

    def chat(
        self,
        user: str,
        *,
        system: Optional[str] = None,
        schema: Optional[dict] = None,
        temperature: Optional[float] = None,
        extra_options: Optional[dict] = None,
    ) -> tuple[str, dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if schema is not None:
            payload["format"] = schema

        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature
        if schema is not None:
            options["num_predict"] = self.default_num_predict_schema
            options["repeat_penalty"] = self.default_repeat_penalty_schema
            options["top_p"] = self.default_top_p_schema
        elif self.default_num_predict_free is not None:
            options["num_predict"] = self.default_num_predict_free
        if extra_options:
            options.update(extra_options)
        if options:
            payload["options"] = options

        req = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        with urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        elapsed = time.time() - t0

        msg = data.get("message", {}) or {}
        content = msg.get("content", "") or ""
        thinking = msg.get("thinking", "") or ""

        stats = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            "seconds": round(elapsed, 2),
            "done_reason": data.get("done_reason", ""),
            "thinking_chars": len(thinking),
            "content_chars": len(content),
            "model": self.model,
        }
        return content, stats
