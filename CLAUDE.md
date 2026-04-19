# local-llm-tuner — project guide for Claude sessions

If you are a Claude session assisting a user with this repo, orient here first.

## What this repo is

An agentic prompt-tuning harness for local LLMs. The core pattern is:
  (a) small local LLM extracts structured notes chunk-by-chunk,
  (b) frontier LLM synthesizes the final output from those notes,
  (c) a separate frontier call diffs the synthesis against a gold-standard
      output and produces a structured gap report.

The library is deliberately domain-agnostic. Domain-specific behavior lives
in prompts that the caller supplies, plus an optional pluggable search
backend.

## Extracted from

The AutoReview project (private, ~/Dropbox/Coding/AutoReview/). The
`gemma-harness` branch there contains the full 12-commit evolution history
of how this library came to exist — 8 prompt versions, the v8 negative
result on schema-constrained decoding, all the rationale. Reference it when
a design choice looks opinionated and you want the provenance.

## Layout

- `local_llm_tuner/` — the package
  - `core.py` — `DocumentHarness`, `chunk_text`, `merge_update`
  - `ollama_client.py` — local-LLM client, thinking-mode aware
  - `frontier.py` — Anthropic client + `synthesize()` + `compare()`
  - `search/` — pluggable backends: `SearchBackend` protocol, `NullSearch`,
    `PubMedSearch`
  - `logs.py` — usage + prompt-change JSONL helpers
- `examples/biomedical_review/` — reference domain setup (NEJM-AI-style)
- `docs/gemma4_thinking_mode_case_study.md` — **read this before doing any
  schema-constrained work with Gemma4.** Non-obvious findings.
- `tests/test_core.py` — unit tests for chunking + merge + extract_json
- `pyproject.toml`, `LICENSE` (MIT), `.env.example`

## Design principles

1. **Zero required dependencies** for the core library. Everything uses
   Python stdlib (`urllib`, `json`, `re`). This keeps the library portable
   and auditable. Optional extras: `pypdf` for PDF input, pytest for dev.
2. **Prompt = user-supplied.** The library does not ship domain prompts
   in the core. Examples live in `examples/`.
3. **Schema as optional hard guarantee.** When `chunk_summary_schema` is
   passed, the harness routes through Ollama's `format` parameter for
   token-level constrained decoding. Without it, the user relies on prompt
   engineering alone. Both paths coexist.
4. **Every prompt change is loggable.** `logs.log_prompt_change()` writes
   structured JSONL so you can audit performance regressions against
   specific rule changes.

## What a typical modification looks like

**Adding a new search backend:**
Create a class with `.search(query, *, max_results)` returning a list of
dicts and `.format_for_prompt(hits)` returning a string. Stick it in
`local_llm_tuner/search/` and export it from `search/__init__.py`.

**Adding a new frontier model:**
Subclass `FrontierClient` with a `.chat(system, user, max_tokens)` method
that returns `(content_str, stats_dict)`. You can use the existing
`synthesize()` / `compare()` functions by passing your new client.

**Changing chunk size for the biomedical example:**
Pass `chunk_size=N, chunk_overlap=M` when constructing `DocumentHarness`.
See `examples/biomedical_review/run_example.py` for where it's wired.

**Supporting a new local LLM:**
Implement a class with a `.chat(user, *, system, schema, temperature,
extra_options)` method returning `(content, stats)`. Swap it in wherever
`OllamaClient` is used. vLLM, llama.cpp, HF TGI, etc., all map naturally.

## What to watch out for with Gemma4

Read `docs/gemma4_thinking_mode_case_study.md`. Summary: thinking tokens
consume `num_predict` budget, `think=false` breaks `format`, `temperature=0`
causes degenerate loops under constrained decoding. The library defaults
work around these but the workarounds are not bulletproof — if a user
reports "my memory is empty after chunking," it's usually this.

## Versioning convention

This is pre-1.0. Public API can shift. Keep `__version__` in
`local_llm_tuner/__init__.py` in sync with `pyproject.toml` whenever the
package or its top-level API changes.

## When asked to "merge this to the main AutoReview project"

The original project lives at `~/Dropbox/Coding/AutoReview/` on the
`gemma-harness` branch. The intent is that AutoReview eventually imports
`local_llm_tuner` as a dependency and retires its internal `harness/`
subdirectory. But that migration hasn't happened yet — treat the two repos
as independent for now.

## Handoff checklist

Before ending a session that touched this repo:
- [ ] `pytest tests/` passes (if tests were edited)
- [ ] `python -c "import local_llm_tuner"` imports cleanly
- [ ] New public-API additions are exported from `local_llm_tuner/__init__.py`
- [ ] Any bump to `pyproject.toml` matches `__version__` in the package
