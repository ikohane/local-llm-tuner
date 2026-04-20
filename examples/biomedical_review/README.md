# Biomedical Review Example

End-to-end demonstration of `local-llm-tuner` doing NEJM-AI-style editorial
triage: chunk a manuscript, have a local LLM (Gemma) accumulate structured
notes, run PubMed searches, then have a frontier model (Claude) synthesize
a review and compare it to a gold-standard review.

## Ships with runnable samples

Two files in this directory let the demo work out of the box:

- `sample_manuscript.txt` — the real CC-BY open-access article
  *Kung et al., "Performance of ChatGPT on USMLE: Potential for AI-assisted
  medical education using large language models,"*
  PLOS Digital Health (2023), reproduced verbatim with full license
  attribution.
- `sample_gold_review.md` — a **synthetic** gold review written for the
  demo (the actual peer reviews of the Kung paper were not published). It
  is clearly labeled as synthetic at the top of the file.

## Prerequisites

1. **Ollama running locally** with a Gemma model pulled. Verified against
   Gemma 4 26B (Q4_K_M quantization) on Ollama 0.20.7. Smaller variants
   (`gemma4:e4b` ~4 GB) should also work with minor quality tradeoffs:

       ollama pull gemma4:26b       # ~18 GB
       ollama serve                 # if not auto-started

   Default endpoint is `http://localhost:11434`. Override via
   `OLLAMA_BASE_URL` if different.

2. **Anthropic API key** for the synthesis + comparison steps:

       export ANTHROPIC_API_KEY=sk-ant-...

3. **Library installed:**

       pip install -e ".[biomedical,anthropic]"

   (from the repo root).

## Run

```bash
cd examples/biomedical_review
python run_example.py sample_manuscript.txt --gold sample_gold_review.md
```

Outputs land in `./out/<doc_id>/`:

- `memory.json` — Gemma's accumulated structured notes
- `synthesis.md` — Claude-written review
- `comparison.md` and `comparison.json` — gap analysis vs gold

Per-call usage stats go to `./out/logs/usage.jsonl`.

## Expected wall-time

For the shipped sample (~30 KB of manuscript text → ~4 chunks at the 8 KB
default), expect ~5–10 minutes end-to-end on a typical local Gemma setup:

- Chunk pass: ~60–90 s per chunk × 4 chunks
- PubMed phase: 2–4 Gemma calls on returned abstracts
- Claude synthesis + comparison: ~10 s each

Longer manuscripts scale roughly linearly in the chunk count.

## What to look at

- `memory.json` — this is the whole point of the library. Gemma's
  chunk-by-chunk updates accumulate into a structured note bundle. The
  frontier model only sees this dict, never the raw PDF.
- `comparison.json` — the feedback loop. Tells you where Gemma's notes
  lost information relative to the gold review.

## Adapting to another domain

1. Copy this directory.
2. Rewrite `prompts.py`:
   - `DEFAULT_MEMORY` = what structured fields you want to accumulate.
   - `CHUNK_SUMMARY_PROMPT` = instructions for how the local LLM updates
     those fields from each chunk.
   - `SYNTHESIS_SYSTEM_PROMPT` = instructions for how the frontier model
     turns memory into your final output format.
   - `COMPARISON_SYSTEM_PROMPT` = instructions for the gap analysis format.
3. Replace `PubMedSearch()` with `NullSearch()` (or your own backend) if
   your domain isn't biomedical.
4. Keep `run_example.py` largely as-is — it's just wiring.

## Sample attribution

The sample manuscript is:

> Kung TH, Cheatham M, Medenilla A, Sillos C, De Leon L, Elepaño C,
> Madriaga M, Aggabao R, Diaz-Candido G, Maningo J, Tseng V.
> "Performance of ChatGPT on USMLE: Potential for AI-assisted medical
> education using large language models."
> *PLOS Digital Health* 2(2): e0000198 (2023).
> https://doi.org/10.1371/journal.pdig.0000198

Distributed under the Creative Commons Attribution License (CC-BY).
