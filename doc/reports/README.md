# PhD Progress Reports (`doc/reports/`)

## Overview

This directory preserves bimonthly progress reports documenting milestones in seismic phase detection, machine learning pipeline integration, Leonardo HPC benchmarking, and catalog evaluation on AdriaArray data.

## Purpose and maintenance workflow

Each Markdown report is a period-level narrative record. Use the index to
locate a period, read the report for context, and verify implementation or
scientific claims against the current source, configuration, tests, and
experiment records. Add a new report to the Markdown index and state its
evidence boundary rather than silently presenting a narrative as a result.

## Inputs and outputs

- **Inputs:** reviewed notes and period-specific research summaries.
- **Outputs:** concise historical context for researchers and reviewers; the
  reports do not themselves produce catalogs, figures, or validated metrics.
- **Assumptions:** older reports may refer to infrastructure or model versions
  that are no longer current.

## Report Index

### Historical PDF Reports (2023–2024)

- `2023_Novembre_Dicembre_Ken_Tanaka.pdf`: Initial network setup, data acquisition, and literature review.
- `2024_Gennaio_Febbraio_Ken_Tanaka.pdf`: First ML picker benchmarks on local clusters.
- `2024_Marzo_Aprile_Ken_Tanaka.pdf`: 2024 Italian sequence benchmarking and early associator tests.
- `2024_Maggio_Giugno_Ken_Tanaka.pdf`: GaMMA and PyOcto association parameter tuning.
- `2024_Luglio_Agosto_Ken_Tanaka.pdf`: NonLinLoc 1D/3D location integration.
- `2024_Settembre_Ottobre_Ken_Tanaka.pdf`: Preliminary catalog comparison and clustering results.

## Evidence Boundary Note

The reports in this directory provide narrative context on research progress. As stated in [`AGENTS.md`](../../AGENTS.md), narrative summaries are not executable evidence; verify specific parameters, model names, and dataset versions against the Python source code, tests, and configuration files under `OGS/`.
