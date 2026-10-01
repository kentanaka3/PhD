# LLM-assisted experiment lifecycle

## Navigation

Use [`../templates/experiment.md`](../templates/experiment.md) for the
run record and [`../05_evaluations/rubric.md`](../05_evaluations/rubric.md) for
scoring. Governance decisions and unresolved questions are kept in
[`../00_governance/`](../00_governance/).

```text
question → scope/data check → context snapshot → prompt version
    → bounded run → structural validation → scientific evaluation
    → human decision → approved output or archived revision
```

1. Write one operational or falsifiable question.
2. Classify inputs: public, internal, unpublished, restricted, or secret.
3. Select minimum context and record exact paths/commit.
4. Copy `templates/experiment.md` into `04_experiments/` with date and slug.
5. Record provider/model, prompt version, settings, and time; use `unknown`
   rather than inventing missing values.
6. Start with a small, representative, redacted sample.
7. Validate format and score using `05_evaluations/rubric.md`.
8. Human-review factual claims, code, scientific interpretation, and data use.
9. Move only reviewed deliverables to `06_outputs/`.
10. Record the decision and archive superseded versions.

Stop if the task requires secrets, restricted data, expensive compute,
destructive actions, autonomous publication, or unsupported interpretation.

## Required handoff information

Another researcher should be able to answer: what question was asked, what
context was supplied, which prompt/model/settings were used, what was checked,
what remains uncertain, and who approved the next action. Missing values are
recorded as `unknown`, never guessed.
