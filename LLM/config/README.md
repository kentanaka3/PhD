# Safe configuration

## Purpose and navigation

Keep non-secret defaults, schemas, and path conventions here. Never commit
tokens, passwords, private keys, or unredacted sensitive data. Use environment
variables or the cluster's approved secret mechanism and document only the
variable name and classification.

The values below are a documentation example, not a request to create a
Conda environment or compute workspace. Check the Leonardo Makefile before
using any path; its defaults are at `OGS/utils/Leonardo/Makefile:53-99`.

## Example non-secret configuration

```yaml
workspace: LLM
source_repository: PhD
compute_workspace: /leonardo_work/IscrC_AISeism/WORK
default_data_classification: internal
human_review_required: true
```

## Configuration authority map

```text
LLM/config/README.md        → safe examples and conventions
OGS/utils/Leonardo/Makefile → runtime defaults and target expansion
OGS/conf/                   → stage configuration
approved environment        → secrets and installed dependencies
```

### Interpretation rule

The example above is descriptive only. It does not create `/leonardo_work`,
activate Conda, install a package, or prove that a model is available. For
runtime claims, inspect the current Makefile and stage YAML, then record the
exact command and environment in an experiment record. When this document and
an executable file disagree, the executable file is evidence of current
behavior; the disagreement belongs in
[`../00_governance/open_questions.md`](../00_governance/open_questions.md).
