# Frontmatter Template

```yaml
---
name: <agent-name>
description: "<Summary of what the agent specializes in and when to invoke it.>"
user-invocable: true
argument-hint: "<Describe the expected input.>"
allowed-tools:
  - Read
  - Edit
  - Write
  - Bash
  - Glob
  - Grep
effort: high
---
```