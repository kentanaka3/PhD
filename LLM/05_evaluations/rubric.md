# Evaluation rubric

## Use and interpretation

Score each dimension from 0 to 2 and explain the evidence.

## Inputs and outputs

- **Inputs:** a bounded model output, its context snapshot, validation
  evidence, and the applicable experiment record.
- **Outputs:** dimension scores, evidence notes, and a named human decision.
- **Assumption:** a score is meaningful only for the supplied evidence and
  does not authorize a downstream action.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Factuality | false/unsupported | mixed/incomplete | source-supported |
| Completeness | misses task | major parts covered | all required parts |
| Uncertainty | overconfident | some caveats | calibrated/explicit |
| Reproducibility | no run details | partial details | context/prompt/settings recorded |
| Safety | unsafe action/data use | boundary unclear | checkpoints and limits respected |
| Usefulness | unusable | major rework | useful after normal review |

Suggested acceptance rule: no zero in factuality or safety, plus a named human
reviewer. Use-case-specific thresholds belong in the experiment record.

Scores summarize the supplied evidence; they are not model accuracy, seismic
detection quality, or publication approval. Link the evidence to the relevant
source, command output, and experiment record.

## Evaluation flow

```text
claim or output
      │
      ▼
source + command + context snapshot
      │
      ├── factuality / completeness / uncertainty
      ├── reproducibility / safety / usefulness
      └── domain-specific scientific checks
                         │
                         ▼
            named human reviewer and decision
```

### Domain-specific caution

For OGS work, this rubric evaluates the quality of an LLM-assisted record. It
does not replace catalog comparison metrics, clustering validation, or a
scientific reference-catalog decision. Those require their own experiment
record, data/version provenance, and researcher-approved criteria. In
particular, do not interpret a high rubric score as evidence that a picker,
associator, locator, or clusterer is scientifically accurate.
