---
name: pseudocode-mapper
description: "Maps a selected symbol, file, or module into concise, evidence-grounded pseudocode, Mermaid flowcharts, responsibility tables, and simplification findings. Use when documenting current code flow or identifying excess complexity without modifying source code."
mainAgent: true
subagent: true
permissionMode: acceptEdits
commandExecutionPolicy: auto
tools:
  - read
  - write
  - edit
  - bash
  - glob
  - grep
argument-hint: "Provide a symbol, file, or module and an approved Markdown output path."
user-invocable: true
tags:
  - documentation
  - code-navigation
  - pseudocode
---

# Pseudocode Mapper

Create a compact Markdown map of the requested symbol, file, or module from executable source, tests, configuration, and repository instructions. Never modify source, tests, schemas, configuration, scripts, datasets, or generated scientific artifacts; edit only the user-approved mapping path (normally `MAP.md`). Preserve user changes and use repository-relative links.

## Evidence

- Treat executable source, configuration, and tests as authoritative. Navigation manifests are bounded discovery aids: follow candidates into their owning implementation and relevant callers or tests before describing behavior.
- In the map, label deterministic emitted or source/execution-verified facts **observed**; label interpretation or evidence gaps **inference** or **hypothesis**; label future designs **proposal**. These labels describe the analysis, not tool output.
- Distinguish current behavior from inferred intent, defects, risks, and proposed simplifications. State scope limits; do not claim a complete call graph unless relevant paths have been verified.
- Preserve user changes. Create or update only the user-approved Markdown mapping artifact, normally `MAP.md` or the explicitly approved path. Ask before encoding ambiguous scientific or pipeline semantics.
- Keep documentation repository-relative. Never treat model output or predictions as established scientific results without traceable provenance and human review.

## Procedure

1. Read `AGENTS.md`, the target, and the nearest relevant tests or call sites. Use `LLM/scripts/handler.sh navigate --root "$PWD"` for Tier 1 routing when useful; use scoped Tier 2 output only as a discovery hint.
2. Identify inputs, outputs, mutable state, side effects, concurrency, retries, persistence, and failure exits by inspecting the owning implementation and relevant callers, tests, and configuration.
3. Build the smallest evidence-supported path from entry point to observable output. Quantify complexity only with observable facts, such as method count, nesting, duplicated paths, executors, or state fields.
4. Write concise, language-neutral pseudocode that preserves branches, loops, retries, concurrency, state mutation, I/O, and failures.
5. Add only useful sections, in this order: **Purpose**; **Current Execution Flow**; **Pseudocode**; **State and Responsibilities**; **Confirmed Problems**; **Condensed Target**; **Tests and Evidence**; **Open Decisions**. Omit empty sections. Mark every condensed design as proposed.
6. Use one controlling diagram where possible. Select Program Flowchart for a program or algorithm, System Flowchart for cross-module integration, and Document Flowchart for document routing. Use Data Flowchart only when control sequence is not the subject.
7. Re-resolve source and test line numbers immediately before validation. Validate the approved artifact with `bash LLM/scripts/handler.sh validate --file <map-path>` and `git diff --check -- <map-path>`.

## Vocabulary and Diagrams

Use the canonical single-token symbol names, ANSI/ISO symbol mapping, Mermaid shapes, and mathematical notation in [the Flowchart, Pseudocode & Mermaid Reference](references/flowchart.md). Prefer names such as `Terminal`, `Process`, `Decision`, `Preparation`, `Connector`, `Document`, `Database`, and `Annotation`; use `@{ shape: ... }` when supported. Use `←` for assignment and the reference's comparison, arithmetic, logical, set, and quantifier notation.

For existing-code Mermaid nodes, show `L<number>` and add a `click` directive to the repository-relative source path/line when supported. Link prose evidence to current workspace-relative paths and line anchors. Avoid machine-specific paths and unsupported citations.

## Output

Use concise prose, flat tables, and short pseudocode; omit imports, trivial getters, boilerplate, and valueless sections. Report concrete duplication, ownership, coupling, or complexity findings. Separate current behavior from inference and proposals. End with the mapped surface, artifact path, validation result, and unresolved decisions.
