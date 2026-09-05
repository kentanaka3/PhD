# `validate-agent.sh` Reference Manual

The `validate-agent.sh` script is a deterministic, dependency-light CLI utility that provides unified validation for agent definitions across **GitHub Copilot Agent Profiles**, **Google Antigravity / Gemini CLI Skills & Agents**, and **Anthropic Claude Code Subagents & Skills**.

## Location

Repository-relative paths:
- Unified Runner: `file:///.agents/skills/agent-creator/scripts/validate-agent.sh`
- GitHub Validator: `file:///.agents/skills/agent-creator/scripts/validate-agent-github.sh`
- Gemini Validator: `file:///.agents/skills/agent-creator/scripts/validate-agent-gemini.sh`
- Claude Validator: `file:///.agents/skills/agent-creator/scripts/validate-agent-claude.sh`

---

## Capabilities & Architecture

```text
validate-agent.sh
├── github <FILE>   # Validates YAML frontmatter in GitHub Copilot Agent Profiles (.agent.md)
├── gemini <FILE>   # Validates YAML frontmatter in Gemini CLI Agents / Antigravity Skills (SKILL.md)
└── claude <FILE>   # Validates YAML frontmatter in Claude Code Subagents (.md) / Skills (SKILL.md)
```

### Key Design Principles

1. **Read-Only & Side-Effect Free**: The validator never modifies input files or workspace state.
2. **Deterministic Shell Execution**: Built with `set -euo pipefail`, `shopt -s extglob`, and `umask 077`.
3. **Single Source of Truth**: Each platform defines an explicit `ALLOWED_KEYS` array and derived pattern:
   - `ALLOWED_KEYS_GITHUB`
   - `ALLOWED_KEYS_GEMINI`
   - `ALLOWED_KEYS_CLAUDE`
4. **Zero Heavy External Dependencies**: Implemented in standard POSIX/GNU Bash and `awk` (`date`, `basename`, `dirname`, `printf`).
5. **Accurate Code Navigation**: Every function includes a `# N` comment counting the exact body lines per repository standards in `AGENTS.md`.

---

## Subcommands & Usage

### 1. GitHub Copilot Agent Profile Validation (`github`)

Validates GitHub Copilot custom agent profiles (e.g. `.github/agents/*.agent.md`).

```bash
bash .agents/skills/agent-creator/scripts/validate-agent.sh github <path-to-agent-file>
```

#### Validation Rules
- **Delimiter Integrity**: The first line must be `---` and matched by a closing `---`.
- **Required Fields**: `description` is required and must not be empty.
- **Allowed Keys**:
  - `agents`, `argument-hint`, `description`, `disable-model-invocation`, `handoffs`, `infer`, `mcp-servers`, `metadata`, `model`, `name`, `target`, `tools`, `user-invocable`.
- **Target Value**: If `target` is declared, it must be empty, `vscode`, or `github-copilot`.
- **Boolean Fields**: `disable-model-invocation`, `user-invocable`, and `infer` must be `true` or `false`.
- **Tools**: Checks for list format, `[]`, or wildcard `[*]`.

#### Example
```bash
bash .agents/skills/agent-creator/scripts/validate-agent.sh github .github/agents/bash-scripter.agent.md
```

---

### 2. Gemini CLI Agent & Antigravity Skill Validation (`gemini`)

Validates Google Antigravity skills (e.g. `skills/*/SKILL.md`) and Gemini CLI custom agent configurations.

```bash
bash .agents/skills/agent-creator/scripts/validate-agent.sh gemini <path-to-skill-file>
```

#### Validation Rules
- **Delimiter Integrity**: Must be enclosed between valid `---` markers.
- **Required Fields**:
  - `name`: Identifier is required and must not be empty.
  - `description`: Semantic routing description is required and must not be empty.
- **Allowed Keys**:
  - `allowed-tools`, `argument-hint`, `commandExecutionPolicy`, `compatibility`, `description`, `execution-priority`, `license`, `mainAgent`, `metadata`, `model`, `model-engine`, `model-version`, `name`, `permissionMode`, `subagent`, `tags`, `tools`, `usage-limits`, `user-experience`, `user-invocable`.
- **Antigravity Policy Enums**:
  - `permissionMode`: Must be one of `acceptEdits`, `default`, `edit`, or `none`.
  - `commandExecutionPolicy`: Must be one of `auto`, `off`, `on`, `onSuccess`, or `onError`.
- **Role Booleans / Nulls**:
  - `mainAgent`, `subagent`, `user-invocable`: Must be `true`, `false`, `null`, or empty.

#### Example
```bash
bash .agents/skills/agent-creator/scripts/validate-agent.sh gemini .agents/skills/agent-creator/SKILL.md
```

---

### 3. Claude Code Subagent & Skill Validation (`claude`)

Validates Claude Code subagents (`.claude/agents/*.md`) and skills (`.claude/skills/*/SKILL.md`).

```bash
bash .agents/skills/agent-creator/scripts/validate-agent.sh claude <path-to-agent-or-skill-file>
```

#### Validation Rules
- **Delimiter Integrity**: Must be enclosed between valid `---` markers.
- **Required Fields**:
  - `description`: Required and non-empty (used for delegation and routing).
- **Allowed Keys**:
  - `agent`, `allowed-tools`, `argument-hint`, `arguments`, `context`, `description`, `disable-model-invocation`, `disallowedTools`, `effort`, `experimental`, `hooks`, `maxTurns`, `model`, `name`, `paths`, `permissionMode`, `shell`, `tools`, `user-invocable`, `when_to_use`.
- **Enum & Type Constraints**:
  - `permissionMode`: Must be one of `default`, `acceptEdits`, `auto`, `bypassPermissions`, `plan`, `dontAsk`.
  - `effort`: Must be one of `low`, `medium`, `high`, `xhigh`, `max`.
  - `maxTurns`: Must be a positive integer.
  - `shell`: Must be `bash` or `powershell`.
  - `disable-model-invocation`, `user-invocable`: Must be `true` or `false`.

#### Example
```bash
bash .agents/skills/agent-creator/scripts/validate-agent.sh claude .claude/agents/code-reviewer.md
```

---

## Exit Codes

- `0`: Validation passed successfully.
- `1`: Validation failure (schema violation, missing required field, unreadable file, empty frontmatter).
- `2`: Command-line usage error (unknown command, missing argument).
- `127`: Missing required system utility (`awk`, `date`, `printf`).
