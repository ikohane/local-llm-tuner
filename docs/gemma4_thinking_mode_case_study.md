# Gemma4 thinking-mode and schema-constrained decoding

> Empirical findings from the project that motivated this library, reproduced
> here so you don't have to rediscover them.

## TL;DR

As of **Ollama 0.20.7** with **`gemma4:26b` (Q4_K_M)**, using
`format: <json_schema>` to force structured JSON output is **not reliable**
without careful tuning. Gemma's internal "thinking" tokens consume your
`num_predict` budget, leaving zero tokens for the required JSON content
in roughly 4 of 5 calls at the Ollama-default budget.

Mitigations the library applies by default:
- `num_predict = 4096` on schema-constrained calls (vs the Ollama
  default of 2048). Gives thinking ~2000 tokens of headroom.
- `temperature = 0.15–0.3` (strictly >0). Deterministic temp=0 triggers
  degenerate repetition loops inside constrained decoding.
- `repeat_penalty = 1.3` and `top_p = 0.9`. Breaks "im_im_im_im..."-style
  loops that Gemma falls into in small-output regimes.

The `OllamaClient.chat()` return value includes `thinking_chars`,
`content_chars`, and `done_reason` in the stats dict so you can detect the
failure mode (`content_chars == 0` with `thinking_chars` in the thousands
= thinking starved your content budget).

## Observed failure modes

### 1. Thinking starves content

A 5-run reliability test with `num_predict=2048` on a single biomedical
manuscript chunk:

| Run | eval_count | done_reason | content_chars | thinking_chars |
|-----|-----------|-------------|---------------|----------------|
| 1   | 2048      | length      | 0             | 7539           |
| 2   | 282       | stop        | 943           | 6692           |
| 3   | 2048      | length      | 0             | 7323           |
| 4   | 2048      | length      | 0             | 7359           |
| 5   | 2048      | length      | 0             | 7175           |

Same inputs, same options. Non-deterministic. 1 of 5 (20%) produced valid
JSON content; the other 4 exhausted their budget on internal reasoning
with zero visible output.

Bumping to `num_predict=4096` improves but doesn't solve it — Gemma's
thinking can expand to fill the new budget, especially when the schema is
large or the system prompt is long.

### 2. `think=false` breaks `format`

Ollama/Gemma4 has an upstream bug
([ollama/ollama#15260](https://github.com/ollama/ollama/issues/15260))
where setting `"think": false` on a request also silently disables the
`"format": <schema>` constraint. The response is plain text instead of
schema-conforming JSON. So you cannot simply disable thinking to reclaim
the token budget.

### 3. Degenerate repetition at temperature=0

With `temperature=0` + large schemas, Gemma falls into low-entropy
repetition inside long string fields:

```
"rating_rationale": "The paper proposes a proposed/perspective/simulation/
case series/genre-specific cap on originalityity_max_max_im_im_im_im_im_im
_im_im_im_im_im_im_im_im_im_im_im..."
```

The schema structure is respected (object shape is correct, enums are
satisfied) but string content is unusable. Raising temperature to 0.15+ and
adding `repeat_penalty=1.3` breaks the loop. Going higher than ~0.3 hurts
determinism for scoring / evaluation use cases.

### 4. Long strings get truncated mid-string

If a required field is `"type": "string"` with no `maxLength`, Gemma tends
to generate long strings that hit the `num_predict` cap mid-string. The
response ends with an unterminated string token, JSON parse fails.

**Mitigation:** add `maxLength` to string fields. Around 300–400 chars works
for rationales; 150–200 for short fields. Very small caps (50–100) make it
harder for Gemma to terminate the string at a sensible point and can hurt
content quality.

### 5. The schema itself amplifies the problem

A 20+ property schema with deeply nested required objects triggers more
thinking than a flat schema with a handful of fields. If you find Gemma
reliably timing out or emitting empty content, **simplify the schema**
before adding more `num_predict`. Splitting one complex schema into two
separate calls is often cheaper than one big call.

## Recommendations

For best results with Gemma4 + schema-constrained decoding:

1. Keep the **schema flat and minimal**. If you need deep structure,
   split into multiple calls.
2. **Only require fields you really need.** Every `required` adds pressure
   on the decoder. Mark it optional and instruct in the prompt instead.
3. **Add `maxLength`** to every string field.
4. **Short, focused system prompts.** The longer the system prompt, the
   longer the thinking tends to run.
5. **Budget generously.** `num_predict=4096` is this library's default for
   schema calls; bump higher if `stats["content_chars"] == 0` is recurring.
6. **Detect failure, retry cheaply.** Check `stats["content_chars"] > 0` or
   try `json.loads()` on the content; on empty, retry once or twice with
   a slightly different temperature.
7. **Consider another model.** Qwen3, Llama3, and Mistral variants have
   been reported by the community to handle Ollama's `format` parameter
   more reliably because they don't have Gemma4's thinking mode. If your
   workload is strict-schema-heavy, benchmark before committing to Gemma4.

## Why thinking mode isn't always bad

The thinking tokens aren't wasted in general — they encode real reasoning
and improve output quality for open-ended generation. The problem is
specifically the collision with `format: <json_schema>` + small
`num_predict` + long required strings. Free-form chat with Gemma4 is
perfectly fine without these caveats.

---

_Findings documented at the close of the project that spawned this library
(AutoReview editorial-triage harness, April 2026). Contributions correcting
or extending these observations welcome._
