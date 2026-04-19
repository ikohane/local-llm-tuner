"""Core agentic loop: chunking, memory merge, per-document orchestration.

The harness is deliberately domain-agnostic. Callers supply:
  - A text extractor (str → str) if the input isn't already text.
  - A local LLM client implementing .chat(system, user, schema, temperature).
  - A chunk-summary prompt template with placeholders {memory}, {chunk_idx},
    {chunk_total}, {chunk_text}.
  - Optionally, a system prompt (recommended for schema-enforced calls).
  - Optionally, a search backend and lit-summary prompt template.
  - A default memory dict (the schema of structured fields to accumulate).

Nothing in this module depends on a specific domain. See examples/ for
concrete setups (biomedical review, etc.).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .search import SearchBackend, NullSearch

logger = logging.getLogger("local_llm_tuner.core")


# ---------- chunking ---------- #

def chunk_text(text: str, chunk_size: int = 8000, overlap: int = 500) -> list[str]:
    """Fixed-size sliding-window chunker that prefers whitespace breaks."""
    text = text.replace("\r", "")
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_size, n)
        if end < n:
            snap = text.rfind("\n", i, end)
            if snap == -1 or snap < i + int(chunk_size * 0.75):
                snap = text.rfind(" ", i, end)
            if snap > i + int(chunk_size * 0.5):
                end = snap
        chunks.append(text[i:end])
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return chunks


# ---------- JSON extraction (tolerant of stray prose/fences) ---------- #

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict:
    """Find the first {...} block and parse it. Returns {} on failure."""
    if not text or not text.strip():
        return {}
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = fence.group(1) if fence else None
    if raw is None:
        m = _JSON_BLOCK_RE.search(text)
        if not m:
            return {}
        raw = m.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[\}\]])", r"\1", raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}. first 200 chars: {raw[:200]!r}")
            return {}


# ---------- memory merge ---------- #

def merge_update(memory: dict, update: dict) -> None:
    """Merge an update dict into memory in place.

    - Strings: replace if update value is non-empty.
    - Lists: extend with de-dup (case-insensitive for strings).
    - Dicts: shallow-merge.
    - Numbers: replace.
    - Unknown keys shorter than 60 chars are accepted (structured updates).
    """
    if not isinstance(update, dict):
        return
    for k, v in update.items():
        if k not in memory:
            if isinstance(v, (str, list, dict)) and len(k) < 60:
                memory[k] = v
            continue
        cur = memory[k]
        if isinstance(cur, list) and isinstance(v, list):
            existing = {
                s.lower().strip() if isinstance(s, str) else json.dumps(s)
                for s in cur
            }
            for item in v:
                key = item.lower().strip() if isinstance(item, str) else json.dumps(item)
                if key and key not in existing:
                    cur.append(item)
                    existing.add(key)
        elif isinstance(cur, dict) and isinstance(v, dict):
            cur.update(v)
        elif isinstance(v, str):
            if v.strip():
                memory[k] = v.strip()
        elif isinstance(v, (int, float)):
            memory[k] = v


# ---------- per-document orchestration ---------- #

@dataclass
class DocumentHarness:
    """Per-document agentic-loop harness.

    Required:
        llm_client: any object with .chat(user: str, *, system: str | None,
            schema: dict | None, temperature: float | None) -> (content, stats).
        chunk_summary_prompt: str with placeholders {memory}, {chunk_idx},
            {chunk_total}, {chunk_text}.
        default_memory: dict describing the initial / schema of accumulated
            notes.

    Optional:
        system_prompt: persistent rules (recommended when using schemas).
        chunk_summary_schema: JSON Schema for token-level constrained decoding
            on chunk-summary calls. Caveat: see docs/gemma4_thinking_mode_case_study.md.
        lit_search_prompt: str with placeholders {memory}, {query}, {abstracts}.
        lit_search_schema: JSON Schema for lit-summary calls.
        search_backend: pluggable (default: no-op NullSearch).
        max_lit_queries: cap on PubMed / search queries per document.
        chunk_size / chunk_overlap: chunking parameters.
        temperature: passed to llm_client on each call.
        logger_fn: optional callable invoked after each LLM call with a
            usage dict {phase, prompt_tokens, output_tokens, seconds, extra}.
    """

    llm_client: Any
    chunk_summary_prompt: str
    default_memory: dict = field(default_factory=dict)
    system_prompt: Optional[str] = None
    chunk_summary_schema: Optional[dict] = None
    lit_search_prompt: Optional[str] = None
    lit_search_schema: Optional[dict] = None
    search_backend: SearchBackend = field(default_factory=NullSearch)
    max_lit_queries: int = 3
    chunk_size: int = 8000
    chunk_overlap: int = 500
    temperature: Optional[float] = None
    logger_fn: Optional[Callable[[dict], None]] = None
    # key in memory that holds search queries Gemma emits (list of strings)
    queries_field: str = "open_questions_for_lit_search"
    # key in memory where search hits are stored
    search_results_field: str = "lit_search_results"

    def run(self, document_id: str, text: str) -> dict:
        """Run the full chunk + search pass over a single document.

        Returns the final memory dict.
        """
        memory = json.loads(json.dumps(self.default_memory))  # deep copy
        memory.setdefault("document_id", document_id)
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        logger.info(f"[{document_id}] {len(chunks)} chunks "
                    f"(size={self.chunk_size}, overlap={self.chunk_overlap})")

        for idx, chunk in enumerate(chunks, 1):
            self._process_chunk(memory, chunk, idx, len(chunks), document_id)

        if self.lit_search_prompt and self.search_backend and not isinstance(
            self.search_backend, NullSearch
        ):
            self._run_search_phase(memory, document_id)

        return memory

    # ---------- internal ---------- #

    def _process_chunk(self, memory: dict, chunk: str, idx: int, total: int,
                       document_id: str) -> None:
        mem_view = {k: v for k, v in memory.items()
                    if k != self.search_results_field}
        user_prompt = (
            self.chunk_summary_prompt
            .replace("{memory}", json.dumps(mem_view, indent=2))
            .replace("{chunk_idx}", str(idx))
            .replace("{chunk_total}", str(total))
            .replace("{chunk_text}", chunk)
        )
        logger.info(f"[{document_id}] chunk {idx}/{total}")
        t0 = time.time()
        try:
            content, stats = self.llm_client.chat(
                user_prompt,
                system=self.system_prompt,
                schema=self.chunk_summary_schema,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.error(f"[{document_id}] chunk {idx} LLM error: {e}")
            if self.logger_fn:
                self.logger_fn({
                    "document_id": document_id,
                    "phase": f"chunk_{idx}",
                    "error": str(e),
                    "seconds": round(time.time() - t0, 2),
                })
            return

        update = extract_json(content)
        merge_update(memory, update)
        memory["chunks_seen"] = idx
        if self.logger_fn:
            self.logger_fn({
                "document_id": document_id,
                "phase": f"chunk_{idx}",
                "prompt_tokens": stats.get("prompt_tokens", 0),
                "output_tokens": stats.get("output_tokens", 0),
                "seconds": stats.get("seconds", round(time.time() - t0, 2)),
                "update_keys": sorted(list(update.keys())) if isinstance(update, dict) else [],
                "chunk_chars": len(chunk),
            })

    def _run_search_phase(self, memory: dict, document_id: str) -> None:
        queries = memory.get(self.queries_field, [])[: self.max_lit_queries]
        if not queries:
            logger.info(f"[{document_id}] no search queries — skipping search phase")
            return
        memory.setdefault(self.search_results_field, {})
        for q in queries:
            try:
                hits = self.search_backend.search(q)
            except Exception as e:
                logger.warning(f"[{document_id}] search failed for {q!r}: {e}")
                hits = []
            memory[self.search_results_field][q] = hits
            if not hits:
                continue

            mem_view = {k: v for k, v in memory.items()
                        if k != self.search_results_field}
            formatted = self.search_backend.format_for_prompt(hits)
            user_prompt = (
                self.lit_search_prompt
                .replace("{memory}", json.dumps(mem_view, indent=2))
                .replace("{query}", q)
                .replace("{abstracts}", formatted)
            )
            logger.info(f"[{document_id}] lit-summary for {q!r}")
            t0 = time.time()
            try:
                content, stats = self.llm_client.chat(
                    user_prompt,
                    system=self.system_prompt,
                    schema=self.lit_search_schema,
                    temperature=self.temperature,
                )
            except Exception as e:
                logger.error(f"[{document_id}] lit-summary LLM error for {q!r}: {e}")
                continue

            update = extract_json(content)
            merge_update(memory, update)
            memory["lit_queries_run"] = memory.get("lit_queries_run", 0) + 1
            if self.logger_fn:
                self.logger_fn({
                    "document_id": document_id,
                    "phase": "lit_summary",
                    "prompt_tokens": stats.get("prompt_tokens", 0),
                    "output_tokens": stats.get("output_tokens", 0),
                    "seconds": stats.get("seconds", round(time.time() - t0, 2)),
                    "query": q,
                    "n_hits": len(hits),
                    "update_keys": sorted(list(update.keys())) if isinstance(update, dict) else [],
                })


# ---------- convenience function ---------- #

def run_document(
    document_id: str,
    text: str,
    *,
    llm_client: Any,
    chunk_summary_prompt: str,
    default_memory: Optional[dict] = None,
    **kwargs: Any,
) -> dict:
    """Single-call convenience wrapper around DocumentHarness."""
    harness = DocumentHarness(
        llm_client=llm_client,
        chunk_summary_prompt=chunk_summary_prompt,
        default_memory=default_memory or {},
        **kwargs,
    )
    return harness.run(document_id, text)
