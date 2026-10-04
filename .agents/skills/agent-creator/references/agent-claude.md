# Claude Code Agent Specification

Claude Code Subagents & Skills

- **Format**: Markdown file with YAML frontmatter.
- **Allowed Keys**:
  - `name`: Identifier string.
  - `description`: Required description explaining delegation trigger conditions.
  - `when_to_use`, `argument-hint`, `arguments`: Invocation hints.
  - `tools`, `allowed-tools`, `disallowedTools`: Tool permissions and denylists.
  - `model`: Model override (`sonnet`, `opus`, `haiku`, `inherit`).
  - `effort`: Enum (`low`, `medium`, `high`, `xhigh`, `max`).
  - `permissionMode`: Enum (`default`, `acceptEdits`, `auto`, `bypassPermissions`, `plan`, `dontAsk`).
  - `maxTurns`: Positive integer limit on steps.
  - `context`, `agent`, `hooks`, `paths`, `shell`, `experimental`: Advanced execution policies.
  - `disable-model-invocation`, `user-invocable`: Delegation flags.

## Frontmatter Fields

| Field                      | Required    | Type       | Notes                                                            |
| -------------------------- | ----------- | ---------- | ---------------------------------------------------------------- |
| `name`                     | No          | string     | Lowercase letters, numbers, hyphens; max 64 chars                |
| `description`              | Recommended | string     | What the agent does and when to use it                           |
| `when_to_use`              | No          | string     | Additional invocation context                                    |
| `argument-hint`            | No          | string     | Hint shown in autocomplete                                       |
| `arguments`                | No          | list       | Named positional arguments for `$name` substitution              |
| `disable-model-invocation` | No          | boolean    | Prevents automatic invocation when `true`                        |
| `user-invocable`           | No          | boolean    | Hide from `/` menu when `false`                                  |
| `allowed-tools`            | No          | list       | Tools allowed without permission prompts                         |
| `disallowedTools`          | No          | list       | Tools explicitly denied                                          |
| `tools`                    | No          | list       | Tool names available to the agent                                |
| `model`                    | No          | string     | Model override                                                   |
| `effort`                   | No          | enum       | `low`, `medium`, `high`, `xhigh`, or `max`                      |
| `maxTurns`                 | No          | integer    | Maximum agentic turns (positive integer)                         |
| `context`                  | No          | enum       | `fork` to run in a forked subagent context                       |
| `agent`                    | No          | string     | Subagent type when `context: fork` is used                       |
| `hooks`                    | No          | object     | Skill-scoped lifecycle hooks                                     |
| `paths`                    | No          | list       | Glob patterns controlling automatic activation                   |
| `shell`                    | No          | enum       | `bash` or `powershell` for inline shell commands                 |
| `permissionMode`           | No          | enum       | `default`, `acceptEdits`, `auto`, `bypassPermissions`, `plan`, `dontAsk` |
| `experimental`             | No          | object     | Experimental feature flags                                       |

## Body Structure

After the closing `---` delimiter, write the agent's system prompt with clear sectioning:

1. **Role Statement**: One-paragraph identity and capability summary.
2. **Contract Reference**: Link to `AGENTS.md` and declare that this agent may not weaken it.
3. **Scope and Boundaries**: What the agent operates on, what it must not do.
4. **Workflow / Procedures**: Sequential, deterministic steps for the agent's tasks.
5. **Verification Steps**: Validation commands, checklists, and quality gates.
6. **Response Format**: Expected output structure for task completion.