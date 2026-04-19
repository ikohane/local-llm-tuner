# Biomedical Review Example

End-to-end demonstration of `local-llm-tuner` doing NEJM-AI-style editorial
triage: chunk a manuscript, have a local LLM (Gemma) accumulate structured
notes, run PubMed searches, then have a frontier model (Claude) synthesize
a review and compare it to a gold-standard review.

This mirrors the `AutoReview` project that originally motivated the library.

## Quick start

```bash
# From the repo root:
pip install -e ".[biomedical,anthropic]"

# Set your API key:
export ANTHROPIC_API_KEY=sk-ant-...

# Make sure Ollama + Gemma are running (see main README).

cd examples/biomedical_review
python run_example.py path/to/manuscript.pdf --gold path/to/gold_review.md
```

Outputs land in `./out/<doc_id>/`:

- `memory.json` — Gemma's accumulated structured notes
- `synthesis.md` — Claude-written review
- `comparison.md` and `comparison.json` — gap analysis vs gold

Per-call usage stats go to `./out/logs/usage.jsonl`.

## What to look at

- `memory.json` — this is the whole point of the library. Gemma's chunk-by-chunk
  updates accumulate into a structured note bundle. The frontier model only
  sees this dict, never the raw PDF.
- `comparison.json` — the feedback loop. Tells you where Gemma's notes lost
  information relative to the gold review.

## Adapting to another domain

1. Copy this directory.
2. Rewrite `prompts.py`:
   - `DEFAULT_MEMORY` = what structured fields you want to accumulate.
   - `CHUNK_SUMMARY_PROMPT` = instructions for how the local LLM updates
     those fields from each chunk.
   - `SYNTHESIS_SYSTEM_PROMPT` = instructions for how the frontier model
     turns memory into your final output format.
   - `COMPARISON_SYSTEM_PROMPT` = instructions for the gap analysis format.
3. Replace `PubMedSearch()` with `NullSearch()` (or your own backend) if your
   domain isn't biomedical.
4. Keep `run_example.py` largely as-is — it's just wiring.
