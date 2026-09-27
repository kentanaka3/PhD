# Prompt Library

## Navigation and versioning

Every reusable prompt should declare its purpose, trusted context, input and
output schema, uncertainty behavior, safety boundary, human checkpoint, and
version history. Keep provider-specific settings in an experiment record.

Use [`code_review.md`](code_review.md) as the concrete example. A prompt is a
reviewable instruction, not permission to access data or change the pipeline;
the authority levels in [`../00_governance/scope.md`](../00_governance/scope.md)
still apply.

## Inputs, outputs, and side effects

A prompt takes only the context and inputs explicitly supplied by its caller
and should produce the documented output schema. It does not grant access to
repository files, data, tools, or compute; callers must perform the relevant
human checkpoint and record the run separately.

## Minimal Prompt Skeleton

```text
Role: [bounded task].
Trusted context: [paths, commit, or supplied excerpts].
Input: [schema and constraints].
Task: [precise task].
Output: [schema, units, and citation requirements].
Uncertainty: identify missing evidence; invent no values.
Boundary: propose changes; require human approval before execution.
```
