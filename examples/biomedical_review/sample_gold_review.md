# Gold Review (DEMO / SYNTHETIC)

**Performance of ChatGPT on USMLE: Potential for AI-assisted medical
education using large language models** — Kung et al. 2023,
PLOS Digital Health.

> **Note.** This gold review is **synthetic** — it was written by hand for
> the `local-llm-tuner` demo so that `sample_gold_review.md` exists and the
> `--gold` flag has something to compare against. It is **not** the actual
> peer-review report for the Kung et al. paper (peer reviews for this
> article were not published). Treat the decision and section content as
> an illustrative example only, not as authoritative editorial judgment.
>
> Sample manuscript (`sample_manuscript.txt`) is the real CC-BY article
> text; only this gold-review file is synthetic.

---

## Structured Review

### a) Summary

This report evaluates the performance of ChatGPT (GPT-3.5, no
domain-specific fine-tuning) on 350 publicly available practice questions
drawn from the USMLE Step 1, Step 2CK, and Step 3 exams. The authors
formatted each question in three ways — open-ended, multiple-choice
without justification, and multiple-choice with forced justification —
and two independent physician raters scored ChatGPT's responses along
three dimensions: accuracy, concordance (between answer and explanation),
and insight. ChatGPT scored between approximately 52% and 75% accuracy
across the exams and formats, reached or hovered near the 60% passing
threshold for all three steps, achieved 94.6% concordance between answer
and explanation, and produced at least one "novel, valid, and non-obvious"
insight in 88.9% of responses.

### b) Originality Assessment

Moderate. This was among the first systematic public evaluations of a
non-domain-specific large language model on a comprehensive US medical
licensing benchmark, and the choice to evaluate across all three USMLE
steps (rather than a single step) was a useful contribution. However, the
core finding — that a frontier large language model can pass a
medical-knowledge benchmark — sits within a rapidly populating area; by
the time of this submission, several preprints were already probing LLM
performance on medical examinations (PubMed queries on "LLM medical
licensing examination" return multiple contemporary evaluations). The
specific methodological contributions here are the tri-format prompting
scheme and the operationalization of "insight" as a rater-scored
dimension, both of which are useful but incremental.

### c) Impact Assessment

Low-to-moderate. The findings are widely citable and likely to be
referenced in downstream work on LLM benchmarking, but the study has no
direct clinical impact: no patients were seen, no clinical decisions were
modified, and no deployment in a care pathway was evaluated. The authors
frame the work as a prelude to education and decision-support
applications, which is reasonable, but the paper does not itself provide
evidence for those applications. Potential downstream impact on medical
education is plausible but untested.

### d) Methodological Soundness

Generally sound for the question asked, with several acknowledged
limitations. Strengths include the use of a public, standardized question
bank that the model is unlikely to have been exactly trained on; the use
of two independent physician raters with explicit adjudication procedures
for disagreement; and the parallel evaluation across three prompt formats.
Weaknesses include: (1) the 350-question sample, while adequate for a
point estimate, is small relative to the full USMLE question universe and
no confidence intervals are reported for the headline accuracy figures;
(2) the raters were not blinded to the hypothesis, which could affect the
insight ratings in particular; (3) the authors' "insight" construct is
defined with sufficient latitude that inter-rater agreement on that
specific dimension is difficult to assess from the numbers reported; and
(4) the possibility that some of the USMLE practice questions were
present in the GPT-3.5 training corpus cannot be ruled out and is not
addressed with a contamination analysis.

### e) Results Description

Results are reported clearly. Accuracy by exam is given as a point
estimate for each format, with the overall pattern being that the model
performed best on Step 1 (75% accuracy in the open-ended format) and
worst on Step 2CK (52.4% in multiple-choice-without-justification).
Concordance was 94.6% overall and insight was 88.9% overall. Minor gaps:
no raw numerators are given for some subgroup analyses, and the
interaction between question difficulty and prompt format is described
qualitatively rather than quantitatively.

### f) Tone and Claims

Appropriately cautious. The authors explicitly state that their results
do not demonstrate clinical competence and that the findings are a
prelude to, rather than evidence of, clinical deployment. The language
around "insight" could be read as slightly overreaching in the abstract,
but the Discussion walks this back with appropriate caveats.

### g) Writing Quality

Clearly written and well organized. Figures are useful and the methods
section is sufficiently detailed for a secondary group to replicate the
evaluation with a different question set.

### h) Related Publications

The paper does not deeply situate itself against the already-growing
corpus of LLM-on-medical-exam evaluations. A more thorough related-work
discussion would strengthen the originality claim.

---

**Decision:** Send to Deputy Editor

**Rationale:** The manuscript provides a methodologically sound,
moderately original evaluation of a non-domain-specific LLM on a
comprehensive medical-licensing benchmark, with appropriately scoped
conclusions and clearly acknowledged limitations. The main weaknesses —
the absence of confidence intervals, the contamination question, and
shallow related-work coverage — are remediable during peer review rather
than disqualifying.
