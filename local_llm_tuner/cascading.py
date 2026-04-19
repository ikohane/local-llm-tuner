"""CascadingLLMClient — try multiple LLM backends in order, fall back on failure.

Usage:
    from local_llm_tuner import OllamaClient, OpenAIClient, CascadingLLMClient

    primary = OllamaClient(model="gemma4:26b")
    fallback = OpenAIClient(model="gpt-5.4-mini")

    client = CascadingLLMClient(
        [primary, fallback],
        on_fallback=lambda i, err, backend: print(f"fell back to {backend}: {err}"),
    )

    # Pass to DocumentHarness the same way you'd pass OllamaClient.
    harness = DocumentHarness(llm_client=client, ...)

Each `.chat()` call tries the first client; if it raises, tries the next;
etc. The first client to return a non-empty content wins. If all fail,
the last exception is re-raised.

Empty-content detection: some backends return successfully but with
`content_chars == 0` (Gemma4 thinking-mode budget starvation being the
canonical example). By default, empty content is treated as a failure and
triggers the next client. Disable with `treat_empty_as_failure=False` if
your workflow expects empty content sometimes.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("local_llm_tuner.cascading")


class CascadingLLMClient:
    """Try each client in order; return the first success.

    Arguments:
        clients: list of clients implementing `.chat(user, *, system,
            schema, temperature, extra_options) -> (content, stats)`.
            Must be non-empty.
        on_fallback: optional callback invoked as
            `on_fallback(index, exception_or_reason, backend_name)`
            when client[index] fails and we advance to client[index+1].
            Use for logging.
        treat_empty_as_failure: if True (default), a client that returns
            empty content is treated as a failure and we advance to the
            next one. Rescues Gemma4 thinking-mode empty-content case.
    """

    def __init__(
        self,
        clients: list[Any],
        *,
        on_fallback: Optional[Callable[[int, Any, str], None]] = None,
        treat_empty_as_failure: bool = True,
    ) -> None:
        if not clients:
            raise ValueError("clients list must be non-empty")
        self.clients = list(clients)
        self.on_fallback = on_fallback
        self.treat_empty_as_failure = treat_empty_as_failure

    def chat(
        self,
        user: str,
        *,
        system: Optional[str] = None,
        schema: Optional[dict] = None,
        temperature: Optional[float] = None,
        extra_options: Optional[dict] = None,
    ) -> tuple[str, dict]:
        last_exc: Optional[BaseException] = None
        for i, client in enumerate(self.clients):
            backend = _backend_name(client)
            try:
                content, stats = client.chat(
                    user,
                    system=system,
                    schema=schema,
                    temperature=temperature,
                    extra_options=extra_options,
                )
            except Exception as e:
                last_exc = e
                logger.warning(f"[cascade] client {i} ({backend}) raised: {e}")
                if self.on_fallback:
                    self.on_fallback(i, e, backend)
                continue

            if self.treat_empty_as_failure and not (content and content.strip()):
                reason = (
                    f"empty content (done_reason={stats.get('done_reason','')}, "
                    f"thinking_chars={stats.get('thinking_chars', 0)})"
                )
                logger.warning(f"[cascade] client {i} ({backend}) returned {reason}")
                last_exc = RuntimeError(f"empty content from {backend}")
                if self.on_fallback:
                    self.on_fallback(i, reason, backend)
                continue

            # Success.
            stats = dict(stats)
            stats["cascade_used_index"] = i
            stats["cascade_used_backend"] = backend
            if i > 0:
                logger.info(f"[cascade] succeeded via client {i} ({backend})")
            return content, stats

        # All clients failed.
        assert last_exc is not None
        raise last_exc


def _backend_name(client: Any) -> str:
    """Best-effort human-readable identifier for a backend."""
    cls = client.__class__.__name__
    model = getattr(client, "model", "")
    return f"{cls}({model})" if model else cls
