# local-llm-tuner

An **agentic harness for iteratively improving a local LLM's structured-output
quality**, scored against a frontier-model gold standard.

You point a local model (e.g. Gemma via Ollama) at long documents in chunks.
It accumulates structured notes in a dict that's passed by reference across
calls (agentic recursion). An optional pluggable search backend (PubMed,
web, custom) can expand the notes with external evidence. A frontier model
(e.g. Claude) then synthesizes a final output from the notes. If you supply a
gold-standard output, a second frontier-model call diffs the synthesis against
it and produces a structured gap report you can use to iterate on prompts.

All prompt changes are logged with full diff + rationale so you can see why
the system performs the way it does.

## Why

Local LLMs are cheap to run and private. They also tend to hedge, over-praise,
and miss critical flags. Rather than tuning a monolithic "please review this"
prompt, this harness decomposes the task: the local LLM **only summarizes**
into a structured schema, and the frontier model **writes the final answer**
using the structured notes. This partition makes prompt iteration
tractable — you tune one end or the other, with measurable feedback each time.

The original use case — editorial triage of manuscripts against a frontier
reviewer — went from **~0% agreement** with the gold reviewer (local-only
monolithic prompt) to **90% direction match** across 20 held-out documents,
using 5 iterations of prompt refinement driven by this harness's gap reports.

## Architecture

```
document  ─►  chunker  ─►  local LLM loop (memory passed by reference)
                                  │
                           open_questions_for_lit_search
                                  │
                                  ▼
                           search backend (optional, pluggable)
                                  │
                                  ▼
                           local LLM summarizes search results into memory
                                  │
                                  ▼
                          memory.json (structured notes)
                                  │
                                  ▼
                         frontier model  ─►  synthesis.md
                                  │
                           (if gold provided)
                                  │
                                  ▼
                         frontier model  ─►  comparison.json  ── feed back into prompt iteration
```

## Install

```bash
pip install -e .

# Optional extras:
pip install -e ".[pdf]"          # pypdf for reading .pdf inputs
pip install -e ".[biomedical]"   # only needed if you run the biomedical example
pip install -e ".[dev]"          # pytest + tooling
```

## Setup: Ollama + Gemma

This harness was developed and validated against **Ollama 0.20.7** running
**`gemma4:26b`** (Q4_K_M quantization, ~18 GB model). Any Ollama-hosted model
with the `format` parameter should work, but the thinking-mode caveats
documented in `docs/gemma4_thinking_mode_case_study.md` are specific to
Gemma4.

**1. Install Ollama** (https://ollama.com/download):

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Start the server (it usually auto-starts as a service; otherwise):

```bash
ollama serve
```

Default endpoint: `http://localhost:11434`.

**2. Pull the model**:

```bash
ollama pull gemma4:26b
```

`gemma4:26b` is ~18 GB. Smaller variants that should work with minor quality
tradeoffs:

```bash
ollama pull gemma4:e4b     # smallest, ~4 GB
ollama pull gemma4:31b     # largest, ~20 GB
ollama pull gemma3         # previous generation; drop-in
```

**3. Verify it's reachable:**

```bash
curl http://localhost:11434/api/tags
# Should list the models you pulled.
```

**4. Set API key for the frontier model:**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

A `.env.example` is included; copy to `.env` and source, or export directly.

**5. Run the biomedical example:**

```bash
cd examples/biomedical_review
python run_example.py sample_manuscript.txt --gold sample_gold_review.md
```

Outputs land in `./out/<doc_id>/` with `memory.json`, `synthesis.md`, and
(if `--gold` provided) `comparison.json` / `comparison.md`.

## Minimal usage (from Python)

```python
from local_llm_tuner import DocumentHarness, OllamaClient, AnthropicClient
from local_llm_tuner.frontier import synthesize, compare
from local_llm_tuner.search import NullSearch

# 1. Your domain-specific prompts (see examples/ for a full set)
CHUNK_PROMPT = """Extract structured notes. Return JSON with fields from this
schema: {your_schema}. memory_so_far: {memory}. chunk {chunk_idx} of
{chunk_total}: {chunk_text}"""
SYNTH_PROMPT = "Write the final output from these notes in format X."
COMPARE_PROMPT = "Return a JSON gap analysis with this shape: {...}"

# 2. Clients
ollama = OllamaClient(model="gemma4:26b")
frontier = AnthropicClient()

# 3. Run harness on a document
harness = DocumentHarness(
    llm_client=ollama,
    chunk_summary_prompt=CHUNK_PROMPT,
    default_memory={"notes": [], "chunks_seen": 0},
    search_backend=NullSearch(),  # or PubMedSearch() or your own
    temperature=0.3,
)
memory = harness.run("doc-001", my_document_text)

# 4. Frontier synthesis
synthesis, _ = synthesize(
    memory,
    frontier_client=frontier,
    synthesis_system_prompt=SYNTH_PROMPT,
    document_id="doc-001",
)

# 5. Compare vs gold (optional)
gaps, _ = compare(
    synthesis, gold_text,
    memory=memory,
    frontier_client=frontier,
    comparison_system_prompt=COMPARE_PROMPT,
    document_id="doc-001",
)
print(gaps["severity"], gaps["section_gaps"])
```

## Failover cascade (v0.2.0+)

If your local LLM sometimes fails — Ollama timeout, attic (the nanme of my 2021 Macbook Pro running Gemma on my home LAN)
unreachable, Gemma4 thinking-mode starves the budget — you can cascade to a hosted model  but your text will **no longer be private**:

```python
from local_llm_tuner import (
    OllamaClient, OpenAIClient, CascadingLLMClient, DocumentHarness,
)

primary = OllamaClient(model="gemma4:26b")
fallback = OpenAIClient(model="gpt-5.4-mini")  # reads OPENAI_API_KEY

client = CascadingLLMClient(
    [primary, fallback],
    on_fallback=lambda i, err, backend: print(f"  ↪ fell back to {backend}: {err}"),
)

harness = DocumentHarness(llm_client=client, ...)  # same as before
```

Each `.chat()` call tries Ollama first. If it raises, or (by default)
returns empty content — the canonical Gemma4-thinking-starvation failure
from `docs/gemma4_thinking_mode_case_study.md` — the cascade advances to
OpenAI for that call only. The returned `stats["cascade_used_backend"]`
tells you which client actually served each chunk.

Flags:
- `treat_empty_as_failure=False` — keep empty content instead of
  advancing (default is True).
- `on_fallback=callable` — invoked on each fallback event for logging
  or metrics.

## Plugging in your own local model

Anything with a `.chat(user, *, system, schema, temperature)` method works.
Implement it for OpenAI-local-server, vLLM, llama.cpp, Hugging Face Inference
Endpoints — pass it wherever `OllamaClient` is used.

```python
class MyLocalClient:
    def chat(self, user, *, system=None, schema=None, temperature=None):
        # ... call your backend ...
        return content_str, {"prompt_tokens": n1, "output_tokens": n2, "seconds": t}
```

## Plugging in your own search backend

Subclass `SearchBackend` (a `typing.Protocol`):

```python
from local_llm_tuner.search import SearchBackend

class MyWebSearch:
    def search(self, query, *, max_results=5):
        # ... return list of dicts with at least title, abstract, authors, year
        ...
    def format_for_prompt(self, hits):
        return "\n".join(f"[{i+1}] {h['title']}" for i, h in enumerate(hits))
```

## Findings worth knowing before you use Gemma4

See `docs/gemma4_thinking_mode_case_study.md` for the full writeup. TL;DR:

- Gemma4 (including `gemma4:26b`) has internal **thinking-mode tokens** that
  are returned in `message.thinking` separately from `message.content`.
- When you pass `format: <json_schema>` to force structured output, the
  thinking-mode tokens eat your `num_predict` budget. With `num_predict=2048`
  (Ollama default), content success rate is ~20% — most calls return empty
  content while thinking_chars=~7500.
- Setting `think=false` silently disables the `format` constraint
  (upstream Ollama bug #15260 as of 0.20.7).
- Workaround: larger `num_predict` gives thinking headroom but Gemma's
  thinking can expand to fill the budget. The library's `OllamaClient`
  defaults to `num_predict=4096` for schema calls and exposes stats
  (`thinking_chars`, `content_chars`, `done_reason`) so you can detect the
  failure mode and retry.
- If you genuinely need rock-solid constrained decoding, try Qwen3 or
  another non-thinking model — community reports indicate better reliability.

## Iterating on prompts

The workflow this library is designed for:

1. Run the harness on a few documents with prompt v1.
2. Read `comparison.json` for each. Note recurring `systematic_signals`.
3. Edit your prompts. Bump a version number in your own prompt-versioning
   scheme.
4. Call `local_llm_tuner.logs.log_prompt_change(...)` to append a structured
   entry to `prompt_changes.jsonl` — before/after diff, triggering docs,
   expected effect.
5. Re-run. Compare. Iterate.

The ability to audit every prompt change and correlate it with
before/after gap reports is the core value proposition.

## License

MIT. See `LICENSE`.

## Status

Version 0.1.0 — extracted from a working private project (AutoReview). Public
API is stable enough to use but may still shift based on feedback.
