# GitHub Copilot Agent Specification (`.github/agents/<name>.agent.md`)

- **Format**: Markdown with YAML frontmatter delimited by `---`.
- **Allowed Keys**:
  - `name`: Human-readable display name string.
  - `description`: Required routing description string.
  - `target`: Execution environment (`vscode`, `github-copilot`, or omitted for both).
  - `tools`: Allowlist array (`[read, search, edit, execute, todo, agent]`, `[*]`, or `[]`).
  - `model`: Specific model pin or default inheritance.
  - `disable-model-invocation`: Boolean (`true` or `false`).
  - `user-invocable`: Boolean (`true` or `false`).
  - `infer`: Retired legacy boolean flag.
  - `argument-hint`: Placeholder text string for chat inputs.
  - `agents`: Allowlist array of subagents invokable via meta-tools. When 'agents' and 'tools' are specified, the 'agent' tool must be included in the 'tools' attribute.
  - `handoffs`: Transition suggestion array (`agent`, `label`, `prompt`, `send`, `model`).
  - `metadata`: String key/value dictionary.
  - `mcp-servers`: Agent-scoped MCP configurations.