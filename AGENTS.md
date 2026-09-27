# Agent Instructions

Shared rules for people and agents; [`CLAUDE.md`](CLAUDE.md), [`GEMINI.md`](GEMINI.md), and [`.github/copilot-instructions.md`](.github/copilot-instructions.md) may add guidance.

> MUST ALWAYS ASK MANY QUESTIONS
> Do not `git status`

## Authority and Evidence

- Platform and explicit user instructions govern. Executable source, Makefiles, and configuration define behavior; flag narrative conflicts for correction.
- Distinguish observations, analyst labels, predictions, hypotheses, and demonstrated results. Never report LLM output or predictions as established results without traceable sources, dataset/version provenance, and human review. Record reviewed decisions in [`LLM/00_governance/decisions.md`](LLM/00_governance/decisions.md) or [`LLM/06_outputs/`](LLM/06_outputs/).
- Navigation manifests are bounded discovery aids: Tier 1 routes; Tier 2 outlines and grep-derived declarations are hints, not complete call graphs or proof of absence. Verify behavior in owning code, callers, tests, and configuration.
- Use **observed** only for deterministic emitted/scanned facts or source/execution-verified behavior; **inference**/**hypothesis** for interpretations or evidence gaps; **proposal** for future design. Bash navigators do not emit these labels.

## Approval And Preservation

- Inspect targets and guidance; clarify ambiguities and obtain explicit approval before editing. Preserve the worktree and unrelated changes.
- Obtain approval before changing pipeline semantics, moving/deleting files, installing software, or accessing restricted data.
- Mark uncertainty. Reusable prompts and experiments retain purpose, inputs, provenance, settings, assumptions, failure behavior, and reviewer information.

## Workflows

- Before Python or `SBC_RUN_BIN`, activate Conda using canonical variables from [`OGS/utils/Leonardo/Makefile`](OGS/utils/Leonardo/Makefile); derive paths, never hardcode the prefix.
- For navigation/docs, follow [`LLM/README.md`](LLM/README.md) and [`LLM/scripts/README.md`](LLM/scripts/README.md); use [`LLM/scripts/handler.sh`](LLM/scripts/handler.sh), keep docs repository-relative and grounded in executable behavior, and do not recreate removed scripts.
- Keep changes scoped; run the smallest relevant validation; report failures and unresolved decisions.