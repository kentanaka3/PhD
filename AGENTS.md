# Agent instructions for `PhD`

This file is the repository-level contract for any human, coding agent, or assistant working in this repository.

## Authority and conflict resolution

Higher-priority platform and explicit user instructions take precedence. Within this repository, this file governs shared safety, evidence, and workflow rules; provider-specific files may add format or tool guidance but may not weaken it.
Current Makefiles, executable source, configuration, and tests define implemented behavior. When narrative documentation conflicts with executable evidence, report the discrepancy and seek human review rather than silently reconciling it.

### Provider-specific entry points

| Provider | File |
|---|---|
| Claude Code | [`CLAUDE.md`](CLAUDE.md) |
| Gemini Intelligence | [`GEMINI.md`](GEMINI.md) |
| GitHub Copilot | [`.github/copilot-instructions.md`](.github/copilot-instructions.md) |

## Repository map

```text
PhD/
├── .agents/  # Google Antigravity / Gemini skills, rules, and validation harness
├── .claude/  # Claude Code subagents and skills
├── .github/  # Copilot agent profiles, instructions, and CI workflows
├── OGS/      # Python toolkit, configuration, tests, and project data definitions
├── doc/      # thesis, reports, figures, and supporting documentation
└── LLM/      # LLM knowledge, prompts, workflows, experiments, and reviews
```

## Custom agents

Use the [`.github/agents/latex-docxer.agent.md`](.github/agents/latex-docxer.agent.md) agent for Scientific LaTeX document creation, review, and evidence-bounded builds.

Use the [`.github/agents/bash-scripter.agent.md`](.github/agents/bash-scripter.agent.md) agent for Scientific Bash scripting, validation, and reproducible workflows.

## Scientific context

The OGS project analyzes seismicity. Its pipeline includes waveform acquisition, legacy-catalog parsing, catalog management, ML phase picking, phase association, event location, comparison, clustering, and visualization.

Preserve the distinction between observations, model predictions, analyst labels, hypotheses, and demonstrated results. Never present an LLM-generated statement as a scientific result without a traceable source, command, dataset version, or human review record.

## Before changing files

1. Read this file, the target file, and nearby README/configuration docs.
2. Run `git status --short`; existing changes belong to the user.
3. Map the repository with `bash LLM/scripts/handler.sh navigate --root "$PWD"`.
   Use `--scripts` only with `--module PATH` for scoped Tier 2 symbol extraction.
4. Inspect the controlling implementation, Makefile or Bash entry point, configuration, and focused tests as relevant to the requested behavior or claim.
5. State scope, assumptions, risks, affected files, and focused validation in the work log or commit message.
6. Make small, incremental changes that are easy to review or revert.
7. Prefer repository-relative paths in documentation.

## After changing files

Run the smallest meaningful checks and record the outcome:

```bash
# Documentation and scaffold checks
bash LLM/scripts/handler.sh validate --root "$PWD"
git diff --check
git status --short
```

For OGS code, run focused tests or a dry run:
```bash
make -n TARGET -C OGS/utils/Leonardo
```
Do not run downloads, SLURM submissions, Conda installation, or inference as a
smoke test.

## Python environment

Activate the project Conda environment before any Python or `SBC_RUN_BIN` command. The canonical variables are defined in [`OGS/utils/Leonardo/Makefile`](OGS/utils/Leonardo/Makefile):

```bash
WORK_PATH="${WORK_PATH:-/leonardo_work/IscrC_AISeism/WORK}"
CONDA_ROOT="$(cd "$WORK_PATH/.." && pwd)/.miniconda3"
eval "$("$CONDA_ROOT"/bin/conda shell.bash hook)"
conda activate "$CONDA_ROOT/envs/SBC_3.12"
```

Do not hardcode the absolute Conda prefix; derive it from the Makefile variables so that path changes propagate automatically.

## Data, privacy, and secrets

- Never commit `.env` files, API keys, access tokens, SSH keys, passwords, or private prompts.
- Never commit raw waveforms, large catalogs, model checkpoints, or generated artifacts. Keep them in the external `WORK_PATH` or approved storage.
- Redact credentials, personal data, and internal URLs in examples.
- Treat external LLM services as untrusted data processors until approved.

## LLM quality rules

- Reusable prompts declare purpose, inputs, expected output, assumptions, and failure behavior.
- Experiments record model/provider, date, prompt version, context sources, settings when available, and reviewer.
- Mark uncertain or inferred statements explicitly.
- Validate structured output before using it in a pipeline.
- Require explicit human approval before restricted-data access, external provider use, pipeline-semantic changes, cluster jobs, or publication.
  Ordinary documentation and source edits require stated scope and focused validation.

## Documentation and script navigation

Use the unified Markdown handler before and after documentation work:

```bash
bash LLM/scripts/handler.sh validate --root "$PWD"
```

The handler is the authoritative entry point for documentation maintenance:

- `init` creates only missing LLM directories and starter documents; it does not recreate deleted legacy helper scripts.
- `navigate` produces a deterministic, token-dense YAML Context Manifest that turns repository structure into routing state for an LLM subagent. Tier 1 reports modules, validation targets, and conflicts. Use `--module PATH` for a Tier 2 scoped manifest; only Tier 2 supports `--scripts` to append a sorted Make, Bash, and Python symbol index.
- `validate` checks the documentation scaffold, handler prerequisites, and required executable script/documentation entry points without modifying files.

When reconciling Markdown with implementation, inspect the referenced
Makefile, Bash, and Python files directly. Record repository-relative paths and line ranges in the Markdown, distinguish documented behavior from inferred behavior, and update documentation when a referenced script has been removed.
Do not recreate deleted scripts merely to satisfy stale documentation. Prefer focused checks such as `bash -n`, Python syntax/type checks already provided by the project, and `make -n`; never run downloads, inference, or cluster jobs as documentation validation.

For Bash function documentation, place a short description immediately before the function declaration and include an inline `# N` body line-count comment on the opening brace counting the complete function body (from opening to closing brace):

```bash
# Description of function purpose and behavior
sample_function() { # 5
    local -r arg="$1"
    printf '%s\n' "$arg"
    return 0
}
```

Update that count whenever the function changes.
```bash
`bash LLM/scripts/handler.sh validate --file <path>`
```

verifies that the documented line count strictly matches the actual lines in the body.

## Escalation and definition of done

Ask before changing pipeline semantics, moving/deleting existing files,
installing software, accessing restricted data, submitting jobs, or publishing.
When requirements are ambiguous, make a conservative assumption, record it in [`LLM/00_governance/decisions.md`](LLM/00_governance/decisions.md), and list the unresolved issue in [`LLM/00_governance/open_questions.md`](LLM/00_governance/open_questions.md).

A change is done when the intended files are present, unrelated files are untouched, safety-sensitive paths are ignored, validation passes (or its failure is documented), and the next human decision is clear.
