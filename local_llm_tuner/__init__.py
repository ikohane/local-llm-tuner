"""local-llm-tuner — agentic iterative prompt-tuning harness for local LLMs.

Workflow:
    1. Chunk a document (text or PDF).
    2. Loop chunks through a local LLM (Gemma via Ollama by default),
       accumulating a structured `memory` dict passed by reference.
    3. Optionally call a pluggable search backend (PubMed, web, custom).
    4. Hand memory to a frontier model (Claude by default) to synthesize
       the final output.
    5. Compare synthesis to a gold standard and log structured gaps.
    6. Iterate on prompts; every change logged in prompt_changes.jsonl
       with full diff + rationale.

Public entry points:
    from local_llm_tuner import run_document, synthesize, compare
    from local_llm_tuner.search import PubMedSearch, NullSearch, SearchBackend
    from local_llm_tuner.ollama_client import OllamaClient
    from local_llm_tuner.frontier import AnthropicClient
"""

__version__ = "0.1.0"

from .core import (
    DocumentHarness,
    chunk_text,
    merge_update,
    run_document,
)
from .ollama_client import OllamaClient
from .frontier import AnthropicClient, synthesize, compare
from .search import SearchBackend, NullSearch

__all__ = [
    "DocumentHarness",
    "chunk_text",
    "merge_update",
    "run_document",
    "OllamaClient",
    "AnthropicClient",
    "synthesize",
    "compare",
    "SearchBackend",
    "NullSearch",
]
