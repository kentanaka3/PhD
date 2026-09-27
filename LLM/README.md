# LLM and Artificial Intelligence Workspace

Controlled knowledge and experimentation layer for researcher–LLM collaboration. Keep source, scientific data, prompts, and generated outputs separate. Use it for curated context, bounded prompts, experiment records, output evaluation, and reviewed handoffs; it is not a runtime, training workspace, or substitute for executable code/configuration.

## Contract and Side Effects

- **Inputs:** repository-relative context, redacted excerpts, prompt versions, experiment metadata, and human decisions.
- **Outputs:** reviewable Markdown records and explicitly promoted outputs.
- **Assumptions:** missing provenance is recorded as `unknown`; narrative
  context is not automatically a scientific result.
- **Side effects:** the scripts can create missing workspace directories and starter files, but they do not install packages, download data, call an LLM provider, or submit jobs. Inspect script help and paths before running helpers in a new checkout.

## Navigation

```text
00_governance → 01_context → 02_prompts → 03_workflows
  │               │            │            │
  └───────────────┴────────────┴────────→ 04_experiments
                                            │
                                            ▼
                      05_evaluations → 06_outputs → 07_archive
                                            │
                                            ▼
                          human review ← handoffs/final report
```

The repository-level operating rules are in [`../AGENTS.md`](../AGENTS.md). For a fresh crawl, run [`scripts/handler.sh`](scripts/handler.sh) with `navigate --root .`; it is the only supported navigation and validation entrypoint. The command emits a deterministic, token-dense Tier 1 YAML Context Manifest that routes modules, validation targets, and conflicts. To append sorted Make, Bash, and Python symbols, use a scoped Tier 2 manifest: `navigate --module PATH --scripts`.

## Relationship to `make init`

The Leonardo `make init` target delegates to `OGS/utils/Leonardo/init.sh`. The Makefile passes its configured paths to that script. The script first requires the configured workspace, OGS checkout, external repositories, and commands to exist; it verifies or clones the configured external repositories when possible, copies Leonardo launch files, loads cluster modules, conditionally downloads and installs Miniconda, conditionally creates the configured environment, installs `ml_catalog` in editable mode, links [`OGS/conf`](../OGS/conf), [`OGS/data`](../OGS/data), and [`OGS/src`](../OGS/src) into `WORK_PATH`, and submits a dummy launch test. These are external side effects; reading this description does not perform them. See `OGS/utils/Leonardo/Makefile:167-172` and `OGS/utils/Leonardo/init.sh:169-213`.

That is a computational-environment initializer. This directory is a
knowledge-and-review initializer. It does not touch `WORK_PATH`, install
packages, download models, call an LLM provider, or submit jobs.

## Important path distinction

The requested Git path is
`/leonardo/home/userexternal/ktanakah/AISeism/PhD`. In this environment it
resolves to `/leonardo_work/IscrC_AISeism/PhD`; `readlink -f` and device/inode
checks on 2026-08-28 identified the two names as the same checkout. This is an
observed environment fact, not a portable institutional guarantee. On another
node, verify both paths before asking an LLM to inspect source or compare
results. The Makefile defaults are `WORK_PATH=/leonardo_work/IscrC_AISeism/WORK`
and `OGS_PATH=$(WORK_PATH)/../PhD/OGS` (`OGS/utils/Leonardo/Makefile:65-73`).
Use `make -n init` to inspect expansion without running initialization.

## Context manifest and routing

The navigation handler converts raw repository structure into actionable
routing state rather than a prose inventory. Run
`bash LLM/scripts/handler.sh navigate --root "$PWD"` for the global Tier 1
manifest. Use `--module PATH --scripts` for a scoped Tier 2 symbol inventory,
optionally with `--include-assets`; both tiers use stable path ordering.

The optional `03_workflows/handoffs/` directory currently contains only its
`.gitkeep` marker; no final integration record is present in this checkout.
When a handoff is created, link it from this overview and record its evidence
boundaries, checks, and questions for the human researcher.

## Mental Model

```text
                    HUMAN RESEARCHER
                            │
              question ─────┼───── review/decision
                            ▼
┌───────────────────────────────────────────────────────┐
│                     LLM workspace                     │
│   context → prompt → workflow → experiment → review   │
└───────────────────────────────────────────────────────┘
      │                     │                     │
      ▼                     ▼                     ▼
  OGS source       external WORK_PATH      doc/reports
  and tests        waveforms/catalogs      approved writing
```

Arrows are documented references, not automatic data movement. Existing output is not scientific approval.

## Directory Guide

| Path | Function |
|---|---|
| `00_governance/` | scope/decisions |
| `01_context/` | curated context |
| `02_prompts/` | reusable prompts |
| `03_workflows/` | procedures |
| `04_experiments/` | run records |
| `05_evaluations/` | rubrics/tests |
| `06_outputs/` | human-review/promoted outputs |
| `07_archive/` | superseded records |
| `config/` | safe config |
| `scripts/` | executable scripts |
| `templates/` | record templates |
## Lifecycle

```text
question → scope/data check → context snapshot → prompt version → bounded run → format validation → scientific evaluation → human decision → approved output OR revised/archived version
```

## First Use

```bash
cd <repository-root>
bash LLM/scripts/handler.sh init
bash LLM/scripts/handler.sh validate
```

`handler.sh init` is idempotent: it creates missing LLM directories and
starter files but never overwrites existing files. `navigate` writes its
Context Manifest only to standard output; `validate` checks the fixed
documentation scaffold and handler entry point without modifying files.
Neither command runs inference, downloads data, or validates Python
dependencies.

## Example

```text
Question: Can an LLM classify parser failures from existing error messages?
Context: 01_context/pipeline_map.md plus source commit abc123
Prompt: 02_prompts/code_review.md, version 0.1
Input: redacted examples only; no credentials or raw waveform data
Output: structured labels plus uncertainty
Decision: human reviewer accepts labels for triage only; no automatic fixes
```

Read `00_governance/open_questions.md` before connecting this workspace to a
provider, production catalog, or automated cluster workflow.

## Evidence Boundary

The LLM workspace describes how to inspect and review the project; it is not a
scientific source of truth. Executable behavior is defined by the current
Makefile, Python source, tests, and configuration under `OGS/`. Narrative
reports under `doc/reports/` are useful context but must be checked against
their underlying sources before being treated as results.
