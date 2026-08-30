# OGS Package

## Overview

The `OGS/` directory contains the Python codebase, configuration, test suite, utility scripts, and reference velocity models for the OGS (Istituto Nazionale di Oceanografia e di Geofisica Sperimentale) seismic processing toolkit.

## Purpose and workflow

Use this directory to parse and manage OGS catalogs, retrieve or index
waveforms, run configured machine-learning stages, compare catalogs, and
analyze event sequences. A typical workflow is:

```text
legacy catalog or waveform service
        → parser/downloader
        → Parquet catalog or waveform archive
        → catalog comparison, clustering, or ML pipeline
        → tables and plots
```

## Inputs, outputs, and safety

- **Inputs:** legacy catalog files, waveform-service parameters, Parquet
  catalogs, Hydra YAML, and sequence-clustering JSON metadata.
- **Outputs:** date-partitioned catalogs, analysis CSV/plot artifacts, and
  SLURM jobs or logs when the Leonardo helpers are used.
- **Assumptions:** runtime dependencies and external workspaces are prepared
  separately; the links below describe repository interfaces rather than
  guaranteeing that optional services or models are available.
- **Safety:** downloader and cluster targets perform network, filesystem, or
  scheduler actions. Prefer `--help`, `make -n`, and focused tests before a
  real run. Keep raw data and generated artifacts in approved external
  storage.

## Subdirectory Structure

```text
OGS/
├── config/     # Hydra configuration tree (builder, cluster, data, group/joint modules)
├── data/       # Static datasets: OGSCatalog benchmarks and 1D/3D VelocityModel files
├── src/        # 24 Python source modules for downloading, parsing, ML picking, association, location, clustering
├── test/       # Test suite (10 test modules + Makefile)
└── utils/      # Cluster initialization, SLURM templates, and HPC execution scripts (Leonardo)
```

## Subpackage Navigation

- **Source Code**: [`src/README.md`](src/README.md) describes all 24 Python modules in `OGS/src/`, including core components, ML integrations, and analysis drivers.
- **Configuration**: [`config/README.md`](config/README.md) documents the
  hierarchical Hydra/YAML configuration system used by `ml_catalog_run`.
- **Testing**: [`test/README.md`](test/README.md) details the test suite and Makefile test targets.
- **Cluster Execution**: [`utils/README.md`](utils/README.md) explains the Leonardo cluster environment, `init.sh`, `LAUNCHME.sh`, and `Makefile`.

## Primary Entrypoints

- **Waveform Download**: `python OGS/src/ogsdownloader.py`
- **Catalog Parsing & Merging**: `python OGS/src/ogsparser.py`
- **Sequence Clustering**: `python OGS/src/ogssequence.py`
- **Machine Learning Pipeline**: `ml_catalog_run` (via `SBC_RUN_BIN`) driven by `OGS/utils/Leonardo/Makefile`
- **Catalog Comparison**: `OGSCatalog.bgmaEvents()` / `OGSCatalog.bgmaPicks()` in `OGS/src/ogscatalog.py`

## Safe first steps

```bash
python OGS/src/ogsdownloader.py --help
python OGS/src/ogsparser.py --help
python OGS/src/ogssequence.py --help
make -C OGS/test constants
make -C OGS/utils/Leonardo -n help
```

The first three commands only display CLI help. The test and dry-run examples
avoid downloading data or submitting a job; inspect generated commands before
running a target with external side effects.

## Workflow selection

Use the standalone Python entrypoints for local parsing, downloading, station
inventory extraction, or sequence clustering. Use
`OGS/utils/Leonardo/Makefile` for the configured `SBC_RUN_BIN` (= `ml_catalog_run`)
stages and SLURM submission path. The Makefile's bracketed targets must be
quoted, for example:

```bash
make -C OGS/utils/Leonardo -n "PhaseNet[INSTANCE,0.1]"
make -C OGS/utils/Leonardo -n "NLL1D[PyOcto,PhaseNet,INSTANCE,0.1]"
```

`-n` previews the expanded command; it does not prove that the external
environment, model, waveform archive, or scheduler allocation is available.
Before a real run, confirm `WORK_PATH`, date bounds, output directories,
credentials/tokens, and requested resources. Generated catalogs and waveform
files belong in the configured external workspace, not in `OGS/`.
