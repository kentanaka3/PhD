# Gemini Intelligence Agent Specification

Google Antigravity & Gemini CLI Skills (`skills/<name>/SKILL.md`)

- **Format**: Directory package with main `SKILL.md` starting with `---` frontmatter.
- **Allowed Keys**:
  - `name`: Required lowercase kebab-case identifier matching the directory.
  - `description`: Required semantic routing description.
  - `permissionMode`: Enum (`acceptEdits`, `default`, `edit`, `none`).
  - `commandExecutionPolicy`: Enum (`auto`, `off`, `on`, `onSuccess`, `onError`).
  - `mainAgent`: Boolean or null (`true`, `false`, `null`).
  - `subagent`: Boolean or null (`true`, `false`, `null`).
  - `tools`, `allowed-tools`: List of tool permissions.
  - `argument-hint`, `user-invocable`: UI controls.
  - `model`, `model-engine`, `model-version`: Engine controls.
  - `tags`, `license`, `compatibility`, `metadata`: Operational metadata.