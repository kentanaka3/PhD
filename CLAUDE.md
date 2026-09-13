# Claude Code project instructions

Read and follow [`AGENTS.md`](AGENTS.md) — it is the repository-wide contract.
This file adds Claude Code-specific guidance; it may not weaken `AGENTS.md`.

## Scientific project overview

The OGS project is a seismicity analysis pipeline (waveform acquisition → catalog parsing → ML phase picking → phase association → event location → clustering →
visualization).

```text
PhD/
├── .agents/  # Google Antigravity / Gemini skills, rules, and validation harness
├── .claude/  # Claude Code subagents and skills
├── .github/  # Copilot agent profiles, instructions, and CI workflows
├── OGS/      # Python toolkit, configuration, tests, and project data definitions
├── doc/      # thesis, reports, figures, and supporting documentation
└── LLM/      # LLM knowledge, prompts, workflows, experiments, and reviews
```

## Key commands

```bash
# Pre-flight — always run before editing
git status --short

# Repository navigation (Tier 1 Context Manifest)
bash LLM/scripts/handler.sh navigate --root "$PWD"

# Scoped navigation with symbol index (Tier 2)
bash LLM/scripts/handler.sh navigate --module PATH --scripts

# Post-edit validation
bash LLM/scripts/handler.sh validate --root "$PWD"
git diff --check
git status --short


# Single-file Bash or Markdown validation
bash LLM/scripts/handler.sh validate --file path/to/script.sh
bash LLM/scripts/handler.sh validate --file path/to/document.md

# Agent/skill schema validation
bash .agents/skills/agent-creator/scripts/validate-agent.sh claude path/to/SKILL.md

# OGS dry-run (never run real downloads, inference, or SLURM jobs)
make -n TARGET -C OGS/utils/Leonardo
```

## Navigation contract

Treat executable files as the source of truth. Begin with `git status --short`, inspect `LLM/README.md`, and use the safe documentation handler:

```bash
bash LLM/scripts/handler.sh navigate --root "$PWD"
```

The navigation command emits the deterministic, token-dense YAML Context Manifest used to route an LLM subagent. The root command produces Tier 1; append the sorted Make, Bash, and Python symbol index only for a scoped Tier 2 manifest:

```bash
bash LLM/scripts/handler.sh navigate --module PATH --scripts
```

## Python environment

Activate the project Conda environment before any Python or `SBC_RUN_BIN` command. Derive the prefix from [`OGS/utils/Leonardo/Makefile`](OGS/utils/Leonardo/Makefile) variables:

```bash
WORK_PATH="${WORK_PATH:-/leonardo_work/IscrC_AISeism/WORK}"
CONDA_ROOT="$(cd "$WORK_PATH/.." && pwd)/.miniconda3"
eval "$("$CONDA_ROOT"/bin/conda shell.bash hook)"
conda activate "$CONDA_ROOT/envs/SBC_3.12"
```

Do not hardcode the absolute Conda prefix — derive it so path changes propagate.

## Custom agent specifications

Claude Code uses hierarchical discovery (`AGENTS.md`, `CLAUDE.md`, `.claude/agents/*.md`, `.github/agents/`) and on-demand skill activation under `.agents/skills/`.

- [`.claude/agents/latex-docxer.md`](.claude/agents/latex-docxer.md) — Scientific LaTeX document creation, review, and evidence-bounded builds.
- [`.github/agents/latex-docxer.agent.md`](.github/agents/latex-docxer.agent.md) — Scientific LaTeX document creation, review, and evidence-bounded builds.
- [`.github/agents/bash-scripter.agent.md`](.github/agents/bash-scripter.agent.md) — Scientific Bash scripting, function line-count tracking (`{ # N`), and reproducible workflows.
- [`.github/agents/agent-creator.agent.md`](.github/agents/agent-creator.agent.md) — Meta-agent architecture and cross-ecosystem agent validation.
- [`.agents/skills/agent-creator/SKILL.md`](.agents/skills/agent-creator/SKILL.md) — Meta-agent specialized in designing, architecting, revising, and validating agents, skills, and rules across Google Antigravity, GitHub Copilot, and Claude Code ecosystems.

## Planning mode, artifacts, and subagents

- **Planning Mode**: Before executing non-trivial architectural refactors, pipeline adjustments, or complex workflows, create an implementation plan artifact and seek user approval before changing code.
- **Documentation validation**: Before completing documentation or script-navigation work, run `bash LLM/scripts/handler.sh validate --root "$PWD"`. Do not run `init` unless workspace creation is explicitly requested.
- **Artifact Hygiene**:
  - Present multi-step summaries, reports, and walkthroughs via markdown artifacts (`implementation_plan.md`, `walkthrough.md`).
  - Store temporary data, ad-hoc test scripts, or ephemeral debug logs in the artifact scratch directory (`<appDataDir>/brain/<conversation-id>/scratch/`), never in the Git repository tree.
- **Subagents**: Offload deep research surveys or isolated tasks to subagents (`research`, `self`) rather than running unbounded tool loops in the primary context.

## Gotchas

- **Preserve user changes**: Check `git status --short` before editing; do not overwrite unstaged work.
- **Repository-relative paths**: Use repository-relative paths in documentation, never machine-specific absolute paths.
- **Epistemic discipline**: Keep raw observations, analyst labels, model predictions (picks, associations, locations), hypotheses, and demonstrated facts strictly distinct. Never present LLM-generated output as scientific evidence without traceable provenance and review.
- **Escalation**: Seek human approval before changing pipeline semantics, moving or deleting files, installing packages, accessing restricted data, submitting cluster jobs, or publishing.