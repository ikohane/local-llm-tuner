"""Frontier-model client for synthesis + comparison.

Uses the Anthropic Messages API by default. Implement your own client by
subclassing FrontierClient and providing a `.chat(system, user, max_tokens)
-> (content, stats)` method.

Synthesis: given a memory dict (from DocumentHarness.run), produce a
structured final output in whatever format the caller specifies.

Comparison: given a synthesized output and a gold-standard output, produce
a structured gap analysis (also JSON).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional
from urllib.request import Request, urlopen

logger = logging.getLogger("local_llm_tuner.frontier")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5"


class FrontierClient:
    """Protocol: chat(system, user, max_tokens) -> (content, stats)."""

    def chat(self, system: str, user: str, max_tokens: int = 4000
             ) -> tuple[str, dict]:  # pragma: no cover
        raise NotImplementedError


class AnthropicClient(FrontierClient):
    """Minimal Anthropic Messages API client via urllib.

    API key is read from the ANTHROPIC_API_KEY environment variable unless
    passed explicitly.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_CLAUDE_MODEL,
        timeout: int = 300,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set (pass api_key= or set env var)."
            )
        self.model = model
        self.timeout = timeout

    def chat(self, system: str, user: str, max_tokens: int = 4000
             ) -> tuple[str, dict]:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        req = Request(
            ANTHROPIC_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        t0 = time.time()
        with urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        elapsed = time.time() - t0
        blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = data.get("usage", {})
        stats = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "seconds": round(elapsed, 2),
            "model": self.model,
        }
        return text, stats


# ---------- synthesis + comparison helpers ---------- #

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    if not text:
        return {}
    m = _JSON_RE.search(text)
    if not m:
        return {"_error": "no JSON found", "_raw": text[:500]}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[\}\]])", r"\1", m.group(0))
        try:
            return json.loads(cleaned)
        except Exception as e:
            return {"_error": f"parse failed: {e}", "_raw": m.group(0)[:500]}


def synthesize(
    memory: dict,
    *,
    frontier_client: FrontierClient,
    synthesis_system_prompt: str,
    document_id: str = "",
    max_tokens: int = 6000,
) -> tuple[str, dict]:
    """Call a frontier model to synthesize a final output from memory.

    Arguments:
        memory: dict produced by DocumentHarness.run (contains all the
            notes the local LLM accumulated).
        frontier_client: FrontierClient instance.
        synthesis_system_prompt: domain-specific system prompt that tells
            the frontier model the output format to produce.
        document_id: optional identifier included in the user message.
        max_tokens: output token budget.

    Returns (synthesis_text, stats).
    """
    user = (
        f"Document ID: {document_id}\n\n"
        f"Structured notes produced by the local LLM:\n\n"
        f"```json\n{json.dumps(memory, indent=2)}\n```\n\n"
        f"Write the final output now, following the format in the system prompt."
    )
    logger.info(f"synthesizing for {document_id or '(unnamed)'}")
    return frontier_client.chat(synthesis_system_prompt, user, max_tokens=max_tokens)


def compare(
    synthesis_text: str,
    gold_text: str,
    *,
    memory: dict,
    frontier_client: FrontierClient,
    comparison_system_prompt: str,
    document_id: str = "",
    max_tokens: int = 3000,
) -> tuple[dict, dict]:
    """Call a frontier model to diff synthesis against a gold-standard output.

    Arguments:
        synthesis_text: the output produced by synthesize().
        gold_text: the gold-standard output to compare against.
        memory: the notes that fed synthesis (used so the comparator can
            distinguish "synthesis missed something that WAS in notes" vs
            "notes were missing this to begin with").
        comparison_system_prompt: system prompt instructing the format of
            the JSON gap analysis.

    Returns (parsed_json_dict, stats). The dict will include '_error' /
    '_raw' keys if parsing fails.
    """
    user = (
        f"Document ID: {document_id}\n\n"
        f"==================== LOCAL LLM NOTES (input to synthesis) ====================\n"
        f"```json\n{json.dumps(memory, indent=2)}\n```\n\n"
        f"==================== SYNTHESIS ====================\n"
        f"{synthesis_text}\n\n"
        f"==================== GOLD STANDARD ====================\n"
        f"{gold_text}\n\n"
        f"Produce the JSON gap analysis now."
    )
    logger.info(f"comparing vs gold for {document_id or '(unnamed)'}")
    text, stats = frontier_client.chat(comparison_system_prompt, user, max_tokens=max_tokens)
    gaps = _extract_json(text)
    return gaps, stats
