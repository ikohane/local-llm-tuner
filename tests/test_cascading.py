"""Unit tests for CascadingLLMClient.

Uses fake clients so the tests are deterministic and require no network.
"""

from __future__ import annotations

import pytest

from local_llm_tuner.cascading import CascadingLLMClient


# ---------- fakes ---------- #

class _FakeSuccess:
    """A fake client that always returns a canned string."""

    def __init__(self, content: str, *, model: str = "fake-success"):
        self._content = content
        self.model = model
        self.calls = 0

    def chat(self, user, *, system=None, schema=None, temperature=None,
             extra_options=None):
        self.calls += 1
        return self._content, {
            "prompt_tokens": 1, "output_tokens": 2, "seconds": 0.0,
            "done_reason": "stop", "content_chars": len(self._content),
            "thinking_chars": 0, "model": self.model,
        }


class _FakeEmpty:
    """A fake that returns empty content (simulates Gemma4 thinking-starvation)."""

    def __init__(self, model: str = "fake-empty"):
        self.model = model
        self.calls = 0

    def chat(self, user, *, system=None, schema=None, temperature=None,
             extra_options=None):
        self.calls += 1
        return "", {
            "prompt_tokens": 1, "output_tokens": 2048, "seconds": 0.0,
            "done_reason": "length", "content_chars": 0,
            "thinking_chars": 7500, "model": self.model,
        }


class _FakeRaises:
    """A fake that always raises."""

    def __init__(self, err: str = "boom", *, model: str = "fake-raises"):
        self._err = err
        self.model = model
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError(self._err)


# ---------- tests ---------- #

def test_requires_non_empty_client_list():
    with pytest.raises(ValueError):
        CascadingLLMClient([])


def test_first_client_success_no_fallback():
    primary = _FakeSuccess("hello")
    secondary = _FakeSuccess("nope")
    client = CascadingLLMClient([primary, secondary])
    content, stats = client.chat("test")
    assert content == "hello"
    assert primary.calls == 1
    assert secondary.calls == 0
    assert stats["cascade_used_index"] == 0


def test_first_raises_fall_to_second():
    primary = _FakeRaises("network down")
    secondary = _FakeSuccess("rescued")
    events = []
    client = CascadingLLMClient(
        [primary, secondary],
        on_fallback=lambda i, e, b: events.append((i, str(e), b)),
    )
    content, stats = client.chat("test")
    assert content == "rescued"
    assert primary.calls == 1
    assert secondary.calls == 1
    assert stats["cascade_used_index"] == 1
    assert len(events) == 1
    assert "network down" in events[0][1]


def test_empty_content_treated_as_failure_by_default():
    primary = _FakeEmpty()
    secondary = _FakeSuccess("rescued after empty")
    events = []
    client = CascadingLLMClient(
        [primary, secondary],
        on_fallback=lambda i, e, b: events.append((i, str(e), b)),
    )
    content, _ = client.chat("test")
    assert content == "rescued after empty"
    assert primary.calls == 1
    assert secondary.calls == 1
    assert "empty content" in events[0][1] or "length" in events[0][1]


def test_empty_content_allowed_when_flag_off():
    primary = _FakeEmpty()
    secondary = _FakeSuccess("should not be called")
    client = CascadingLLMClient(
        [primary, secondary],
        treat_empty_as_failure=False,
    )
    content, stats = client.chat("test")
    assert content == ""
    assert primary.calls == 1
    assert secondary.calls == 0
    assert stats["cascade_used_index"] == 0


def test_all_clients_fail_reraises_last():
    primary = _FakeRaises("first err")
    secondary = _FakeRaises("second err")
    client = CascadingLLMClient([primary, secondary])
    with pytest.raises(RuntimeError, match="second err"):
        client.chat("test")


def test_three_client_cascade_skips_middle():
    primary = _FakeRaises("p")
    middle = _FakeRaises("m")
    last = _FakeSuccess("survived")
    client = CascadingLLMClient([primary, middle, last])
    content, stats = client.chat("test")
    assert content == "survived"
    assert stats["cascade_used_index"] == 2
    assert primary.calls == middle.calls == last.calls == 1


def test_cascade_passes_through_kwargs():
    seen_kwargs = {}

    class _Capturing:
        model = "capture"
        def chat(self, user, *, system=None, schema=None, temperature=None,
                 extra_options=None):
            seen_kwargs.update(
                user=user, system=system, schema=schema,
                temperature=temperature, extra_options=extra_options,
            )
            return "ok", {"output_tokens": 1, "content_chars": 2,
                          "done_reason": "stop", "thinking_chars": 0}

    client = CascadingLLMClient([_Capturing()])
    client.chat(
        "U", system="S", schema={"type": "object"}, temperature=0.3,
        extra_options={"num_predict": 100},
    )
    assert seen_kwargs["user"] == "U"
    assert seen_kwargs["system"] == "S"
    assert seen_kwargs["schema"] == {"type": "object"}
    assert seen_kwargs["temperature"] == 0.3
    assert seen_kwargs["extra_options"] == {"num_predict": 100}
