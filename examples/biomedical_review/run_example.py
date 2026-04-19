#!/usr/bin/env python3
"""End-to-end example: review a biomedical manuscript PDF with local-llm-tuner.

Usage:
    # Text file input:
    python run_example.py manuscript.txt --gold gold_review.md

    # PDF input (requires pypdf):
    python run_example.py manuscript.pdf --gold gold_review.md

Environment:
    ANTHROPIC_API_KEY   — required for synthesis + comparison
    OLLAMA_BASE_URL     — defaults to http://localhost:11434
    OLLAMA_MODEL        — defaults to gemma4:26b

Output (under ./out/<doc_id>/):
    memory.json         — accumulated structured notes
    synthesis.md        — frontier-written review from the notes
    comparison.md       — gap analysis vs gold (if --gold provided)
    comparison.json     — structured gap record
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from local_llm_tuner import DocumentHarness, AnthropicClient, OllamaClient
from local_llm_tuner.frontier import synthesize, compare
from local_llm_tuner.search import PubMedSearch
from local_llm_tuner.logs import UsageLogger

from prompts import (
    DEFAULT_MEMORY,
    CHUNK_SUMMARY_PROMPT,
    LIT_SEARCH_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    COMPARISON_SYSTEM_PROMPT,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def extract_text(path: Path) -> str:
    """Read a .txt or .pdf file. PDF requires pypdf (install separately)."""
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            print("Error: PDF input requires `pip install pypdf`", file=sys.stderr)
            sys.exit(1)
        text = ""
        for page in PdfReader(str(path)).pages:
            text += page.extract_text() or ""
            text += "\n"
        return text
    return path.read_text(errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="Path to .txt or .pdf manuscript")
    ap.add_argument("--gold", type=Path, help="Optional gold-standard review (.md/.txt)")
    ap.add_argument("--doc-id", help="Override document ID (default: input filename)")
    ap.add_argument("--out-dir", type=Path, default=Path("out"))
    ap.add_argument("--no-lit-search", action="store_true")
    ap.add_argument("--ollama-url", default=None)
    ap.add_argument("--ollama-model", default=None)
    args = ap.parse_args()

    doc_id = args.doc_id or args.input.stem
    doc_out = args.out_dir / doc_id
    doc_out.mkdir(parents=True, exist_ok=True)
    usage_logger = UsageLogger(args.out_dir / "logs" / "usage.jsonl")

    # 1. Extract text
    text = extract_text(args.input)
    (doc_out / "extracted.txt").write_text(text)
    print(f"[{doc_id}] text length: {len(text)} chars")

    # 2. Build clients
    ollama = OllamaClient(
        base_url=args.ollama_url or __import__("os").environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"),
        model=args.ollama_model or __import__("os").environ.get(
            "OLLAMA_MODEL", "gemma4:26b"),
    )
    frontier = AnthropicClient()

    # 3. Run the agentic loop
    harness = DocumentHarness(
        llm_client=ollama,
        chunk_summary_prompt=CHUNK_SUMMARY_PROMPT,
        lit_search_prompt=LIT_SEARCH_PROMPT,
        default_memory=DEFAULT_MEMORY,
        search_backend=PubMedSearch() if not args.no_lit_search else None,
        max_lit_queries=3,
        chunk_size=8000,
        chunk_overlap=500,
        temperature=0.3,
        logger_fn=usage_logger,
    )
    memory = harness.run(doc_id, text)
    (doc_out / "memory.json").write_text(json.dumps(memory, indent=2))
    print(f"[{doc_id}] memory dumped. chunks_seen={memory.get('chunks_seen')}, "
          f"lit_queries_run={memory.get('lit_queries_run', 0)}")

    # 4. Synthesize final review
    synth_system = (SYNTHESIS_SYSTEM_PROMPT
                    .replace("{document_id}", doc_id)
                    .replace("{title}", memory.get("title", "")))
    synth_text, synth_stats = synthesize(
        memory,
        frontier_client=frontier,
        synthesis_system_prompt=synth_system,
        document_id=doc_id,
    )
    (doc_out / "synthesis.md").write_text(synth_text)
    print(f"[{doc_id}] synthesis written ({synth_stats['output_tokens']} out tokens)")

    # 5. Compare to gold if provided
    if args.gold and args.gold.exists():
        gold_text = args.gold.read_text()
        gaps, compare_stats = compare(
            synth_text, gold_text,
            memory=memory,
            frontier_client=frontier,
            comparison_system_prompt=COMPARISON_SYSTEM_PROMPT,
            document_id=doc_id,
        )
        (doc_out / "comparison.json").write_text(json.dumps(gaps, indent=2))
        md = [f"# Comparison: {doc_id}",
              "",
              f"- Overall direction match: {gaps.get('overall_direction_match')}",
              f"- Synthesis decision: {gaps.get('synthesis_decision')}",
              f"- Gold decision: {gaps.get('gold_decision')}",
              f"- Severity: {gaps.get('severity')}",
              ""]
        for sec, items in (gaps.get("section_gaps") or {}).items():
            if items:
                md.append(f"## {sec}")
                md.extend(f"- {it}" for it in items)
                md.append("")
        for key, label in (("hallucinations", "Hallucinations"),
                           ("systematic_signals", "Systematic signals")):
            items = gaps.get(key) or []
            if items:
                md.append(f"## {label}")
                md.extend(f"- {it}" for it in items)
                md.append("")
        (doc_out / "comparison.md").write_text("\n".join(md))
        print(f"[{doc_id}] comparison written. severity={gaps.get('severity')}")


if __name__ == "__main__":
    main()
