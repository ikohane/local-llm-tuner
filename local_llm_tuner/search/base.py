"""SearchBackend protocol and NullSearch no-op implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SearchBackend(Protocol):
    """Any object with .search(query) -> list[dict] and .format_for_prompt(
    hits) -> str."""

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        """Return a list of hit dicts. Structure is backend-specific but
        should include at minimum: title (str), abstract-or-summary (str),
        authors (str), year (str)."""
        ...

    def format_for_prompt(self, hits: list[dict]) -> str:
        """Render hits into a compact text block suitable for inclusion in
        an LLM prompt."""
        ...


class NullSearch:
    """No-op search backend. Use when you don't want external lookups."""

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        return []

    def format_for_prompt(self, hits: list[dict]) -> str:
        return "(search disabled)"
