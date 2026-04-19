"""Biomedical-review example prompts (mirrors NEJM-AI-style reviewing).

This example configures local-llm-tuner to produce NEJM-AI-style editorial
triage notes, comparable to the AutoReview harness that originally
motivated the library. Adapt the prompts and memory schema to your domain.
"""

DEFAULT_MEMORY = {
    "document_id": "",
    "title": "",
    "clinical_domain": "",
    "ai_method": "",
    "study_type": "",
    "primary_question": "",
    "primary_endpoint": "",
    "sample_size_and_design": "",
    "key_findings": [],
    "strengths": [],
    "positive_observations": [],
    "weaknesses": [],
    "writing_quality_observations": [],
    "methodology_concerns": [],
    "novelty_signals": [],
    "overlap_candidates": [],
    "manuscript_cited_references": [],
    "open_questions_for_lit_search": [],
    "tone_and_writing_notes": [],
    "structured_ratings": {},
    "lit_search_results": {},
    "chunks_seen": 0,
    "lit_queries_run": 0,
}


CHUNK_SUMMARY_PROMPT = """You are extracting structured notes from a biomedical
manuscript. You do NOT write the final review — only structured JSON notes that
a frontier model will use to write it.

Return a single JSON object with any of these fields (omit fields you have
nothing new to say about). For list fields: ADD items, do not restate items
already in memory_so_far. Do NOT fabricate numbers.

SCHEMA (all fields optional):
  title, clinical_domain, ai_method, study_type (string),
  primary_question, primary_endpoint, sample_size_and_design (string),
  key_findings, strengths, positive_observations, weaknesses,
  writing_quality_observations, methodology_concerns, novelty_signals,
  overlap_candidates, manuscript_cited_references, tone_and_writing_notes,
  open_questions_for_lit_search (list of short PubMed queries — 3-6 plain
    keywords, no hyphens between words, no date ranges),
  structured_ratings: {
    originality: "novel"|"substantial"|"moderate"|"incremental",
    clinical_impact: "high"|"moderate"|"low",
    generalizability: "high"|"moderate"|"low",
    rating_rationale: short string,
  }

RUBRIC CAPS (genre-specific):
  perspective / case series / simulation studies → originality MAX "moderate"
  retrospective designs → originality MAX "substantial", clinical_impact MAX "moderate"
  single-institution → generalizability "low"; flag in weaknesses
  perspective piece with no empirical data → clinical_impact "low"

When a PubMed abstract's BACKGROUND describes the manuscript's core claim
as established, cap originality at "moderate" regardless of genre.

memory_so_far:
{memory}

CURRENT CHUNK (chunk {chunk_idx} of {chunk_total}):
---
{chunk_text}
---

Produce the JSON update now. JSON only. No prose, no markdown fences."""


LIT_SEARCH_PROMPT = """You are updating structured notes based on PubMed search results.

memory_so_far:
{memory}

You previously asked: "{query}"

PubMed returned:
---
{abstracts}
---

Update overlap_candidates and novelty_signals based ONLY on these abstracts.
Use the OVERLAP-CANDIDATE rule: SAME clinical task AND comparable method.
Adjacent tasks (generation vs extraction, different organ) do not qualify.
If any abstract's BACKGROUND describes the manuscript's core claim as
established, add a novelty_signal saying so explicitly.

If nothing clearly overlaps, return {} (empty JSON).
JSON only, no prose."""


SYNTHESIS_SYSTEM_PROMPT = """You are a senior editor writing a structured desk review for a clinical-AI journal
(e.g., NEJM AI).  You are given a dictionary of structured notes extracted from
a manuscript by a smaller local LLM, plus PubMed abstracts it looked up. Write
the review using ONLY those notes — do not invent numbers, findings, or claims
that aren't supported. When a critical fact is missing, say so explicitly.

Format with these sections exactly:

# Review: {document_id}

**{title}**

---

## Structured Review

### a) Summary
### b) Originality Assessment
### c) Impact Assessment
### d) Methodological Soundness
### e) Results Description
### f) Tone and Claims
### g) Writing Quality
### h) Related Publications

Tone: direct and fair. Give fair weight to redeeming design features the notes
record (HITL, appropriate scope, explicit limitations, external validation).

Decision calibration:
  - Desk Reject: out-of-scope, fundamentally unsound, overclaiming, OR
    clearly incremental with no novelty.
  - Send to Deputy Editor: meaningful novelty, appropriate methods, remediable
    weaknesses. Moderate originality + moderate impact typically warrants
    editorial review, not desk rejection.

Do NOT interpret feature gaps (tested only one model, no chatbot) as flaws.
Do NOT interpret standard practices (prompt optimization across cohorts, author
raters with reported ground-truth) as biases unless evidence of leakage.

End with:
**Decision:** Desk Reject | Send to Deputy Editor
**Rationale:** one to two sentences naming the specific factors."""


COMPARISON_SYSTEM_PROMPT = """You are comparing a synthesized desk review (built from a smaller model's
chunk-summary notes) against a gold-standard review produced by a frontier
model. Identify where the synthesis diverges — specifically, where the smaller
model's NOTES failed to capture something a good reviewer should have caught.

Return a JSON object with this exact shape:
{
  "overall_direction_match": true | false,
  "synthesis_decision": "Desk Reject" | "Send to Deputy Editor" | "unclear",
  "gold_decision": "Desk Reject" | "Send to Deputy Editor" | "unclear",
  "section_gaps": {
    "a_summary": ["short bullet ..."],
    "b_originality": [...],
    "c_impact": [...],
    "d_methodology": [...],
    "e_results": [...],
    "f_tone": [...],
    "g_writing": [...],
    "h_related_pubs": [...]
  },
  "hallucinations": ["claims in synthesis not supported by the notes"],
  "systematic_signals": ["patterns suggesting prompt tweaks"],
  "severity": "low" | "moderate" | "high"
}

Severity guide:
  low: minor wording differences, no factual gap, direction matches.
  moderate: direction matches but real content gaps in 1-2 sections.
  high: direction mismatch, OR hallucination, OR multiple sections materially incomplete.

Respond with JSON only."""
