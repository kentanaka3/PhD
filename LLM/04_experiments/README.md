# Experiment Records

Create one record per meaningful LLM run from [`../templates/experiment.md`](../templates/experiment.md), follow [`../03_workflows/experiment_lifecycle.md`](../03_workflows/experiment_lifecycle.md). Records preserve provenance for review/repetition, but do not authorize code edits, data access, compute, or publication. Classify and bound inputs before each run.

## Record Flow

```text
question
   │
   ├── context snapshot + prompt/model identity
   ├── bounded input + expected behavior
   ├── safe output + structural checks
   └── rubric scores + named human decision
                         │
                         ▼
              06_outputs/ or 07_archive/
```

## Required Provenance

Record question, data classification, safe input ID, repository commit or exact context paths, prompt version, provider/model, settings or `unknown`, run time, validation, reviewer, and allowed use. Use identifiers/checksums for large or sensitive artifacts; template fields are in [`../templates/experiment.md`](../templates/experiment.md).

## Path Examples

```text
04_experiments/2026-08-28-parser-qa.md       # safe run record
06_outputs/REVIEWED__parser-qa.md            # reviewed internal output
07_archive/2026-08-28-old-parser-qa.md       # superseded record
```

