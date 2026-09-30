# Open questions and decision checkpoints

## Purpose and status

Use this file to track unresolved research, data, provider, reproducibility,
and authority choices. Questions remain open until a human decision is recorded
in [`decisions.md`](decisions.md); repository evidence can narrow a question
without silently answering a scientific or policy choice.

## Navigation

Resolve governance questions before connecting an LLM to source code, data, or
compute. Decisions that become stable belong in
[`decisions.md`](decisions.md); workflow questions belong in
[`../03_workflows/`](../03_workflows/).

These choices should not be guessed once real data or an LLM API is involved.

## Inputs, outputs, and safety

- **Inputs:** unresolved findings from repository review and explicit
  researcher questions.
- **Outputs:** decision-ready questions and evidence boundaries for future
  work.
- **Safety:** do not include credentials, private data, or unredacted provider
  prompts. Listing a proposed action does not approve that action.

## Research goals

1. Which use case comes first: code assistance, literature synthesis, catalog
   QA, experiment design, or thesis writing?
2. What is the first measurable success criterion: time saved, factual
   accuracy, fewer coding errors, better documentation, or scientific recall?
3. Which outputs may remain drafts, and which may enter reports after review?

## Data and privacy

4. May prompts contain unpublished results, or only public/project metadata?
5. Which data must never leave Leonardo storage?
6. Is there an approved provider, local model, or institutional endpoint?
7. What retention, logging, and citation rules apply to model calls?

## Reproducibility

8. Which model identifier, prompt version, context snapshot, and settings must
   be recorded for every run?
9. Should evaluation be manual, scripted, or both?
10. What is the minimum test set for an LLM-assisted OGS change?

## Integration and authority

11. Should future helpers read selected OGS files or a generated source index?
12. May an assistant execute tests and Make dry-runs automatically, or only
    prepare patches?
13. Which actions always require confirmation: data access, job submission,
    dependency installation, publication, or deletion?

14. Is `/leonardo_work/IscrC_AISeism/PhD/OGS` the intended runtime checkout,
    and what rule guarantees that it remains the same checkout as
    `/leonardo/home/userexternal/ktanakah/AISeism/PhD/OGS` on future nodes?

## Environment evidence observed in this checkout

In the current checkout, `pwd -P`, `readlink -f`, and device/inode checks show
that the home path and work path are the same directory. This resolves the
immediate path ambiguity but does not establish a permanent cluster-wide mount
policy; future agents should repeat the check. The current executable-path
configuration is that the Leonardo Makefile defines `PYTHON_BIN` and the
standalone recipes use it
(`OGS/utils/Leonardo/Makefile:95,188-218,408-409`). Runtime readiness still
depends on the configured Conda environment existing on the target system.

## Path identity policy

The home path (`/leonardo/home/userexternal/ktanakah/AISeism/PhD`) and work
path (`/leonardo_work/IscrC_AISeism/PhD`) were verified as the same checkout
on 2026-08-28 using `readlink -f` and device/inode checks.

**This is an observed environment fact, not a portable guarantee.**

Agents must verify path equivalence on each new node before relying on it:

```bash
readlink -f /leonardo/home/userexternal/ktanakah/AISeism/PhD
readlink -f /leonardo_work/IscrC_AISeism/PhD
# Compare output; if different, treat as separate checkouts.
stat -c '%d:%i' /leonardo/home/userexternal/ktanakah/AISeism/PhD
stat -c '%d:%i' /leonardo_work/IscrC_AISeism/PhD
# Compare device:inode pairs; mismatch means different filesystems.
```

Record the verification result and date in
[`decisions.md`](decisions.md) when working on a new node. Question 14 below
is addressed by this policy.

## Provenance and review — addressed

Questions 4–7 (data and privacy) and 8–10 (reproducibility) remain open for
detailed configuration, but the following baseline decisions are now recorded
in [`decisions.md`](decisions.md):

- **Reference catalog:** the 2024 Parquet fixtures under
  `OGS/test/OGSCatalog/` are the approved ML evaluation benchmark.
- **Human reviewer:** the thesis advisor or designated co-author must review
  and approve LLM-generated scientific outputs before use.
- **Operational approvals:** see the approval matrix in
  [`scope.md`](scope.md) for action-level requirements.

## Agent review rule

Each sequential documentation agent must add at least five difficult,
repository-specific questions to its handoff. Questions are not answers: they
should identify what evidence or researcher decision is still missing.

## Contradiction to resolve

“LLM Artificial Intelligence folder” can mean an LLM knowledge base or a place
to develop/train AI models. This scaffold assumes the former because the
request mentions `AGENTS.md`, prompts, and agent instructions, and because
model/data assets would conflict with existing large-file and compute-storage
conventions. If model development is intended, create a separately named
project with its own data, environment, and artifact policy.
