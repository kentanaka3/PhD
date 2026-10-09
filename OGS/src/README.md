# OGS Source Modules (`OGS/src/`)

## Overview

The `OGS/src/` directory houses the core Python source code for data ingestion, catalog parsing, machine learning picking, phase association, event location, catalog comparison, and sequence clustering.

For runtime setup, catalog storage, comparison semantics, and the sequence
flow diagram, see the [toolkit guide](../README.md). This index documents
module responsibilities rather than a single automatic pipeline.

## Purpose and usage boundary

The modules are the implementation layer behind the repository entrypoints and Leonardo jobs. Use the module index to locate an interface, then inspect the module's CLI help, docstrings, configuration, and tests before relying on an undocumented argument or scientific interpretation.

## Inputs, outputs, and assumptions

- **Inputs:** legacy catalog files, waveform/archive paths, Parquet catalogs, stage configuration, and sequence-clustering JSON metadata.
- **Outputs:** in-memory DataFrames, date-partitioned Parquet data, CSV cluster members, plots, logs, and model/pipeline results depending on the selected module.
- **Assumptions:** optional dependencies and external data services are available for the selected path; file schemas and units follow the constants/configuration used by the current checkout.
- **Side effects:** downloader and persistence helpers write files and may contact external services. Analysis drivers can process large datasets. Prefer focused tests or dry runs and keep raw data outside this source directory.

Reference catalogs contain analyst labels; ML picks, associations, locations, magnitudes, and clustering assignments are derived outputs, not independent ground truth. Module availability and documented algorithms do not establish
scientific validity for a dataset without provenance and review.

## Module Directory Index

The directory contains 25 Python implementation modules and one package initializer. They are organized functionally below:

### 1. Configuration, Types, and Utilities

| Module | Primary Class / Functions | Description | Key Dependencies |
|---|---|---|---|
| [`ogsconstants.py`](ogsconstants.py) | Constants, date formats, zone codes, FDSN clients | Central configuration hub (column names, geographic bounds, tolerances: `PICK_TIME_OFFSET=0.5s`, `EVENT_TIME_OFFSET=2s`, `EVENT_DIST_OFFSET=8km`). | standard library |
| [`ogsutils.py`](ogsutils.py) | `OGSBPGraph`, `OGSBPGraphPicks`, `OGSBPGraphEvents` | Logging, CLI parsers, waveform/metadata discovery, and weighted bipartite matching using NetworkX `max_weight_matching`; pick/event distance metrics and coordinate helpers. `OGSBPGraph` is abstract: subclasses must implement `makeMatch()`. Construction invokes the hook only for two non-empty inputs; initialize hook-dependent subclass state before the parent constructor. | networkx, numpy, scipy |

### 2. Legacy Catalog Ingestion and Parsing

| Module | Primary Class | Extends | Format / Purpose | Key Dependencies |
|---|---|---|---|---|
| [`ogsdatafile.py`](ogsdatafile.py) | `OGSDataFile` | `OGSCatalog`, `ABC` | Shared normalization, diagnostics, and Parquet persistence for legacy readers; abstract `read()` must be implemented before a subclass can be instantiated. Shared static helpers remain callable on the base class. | re, pandas, pyarrow |
| [`ogsdat.py`](ogsdat.py) | `DataFileDAT` | `OGSDataFile` | Parses legacy fixed-width `.dat` phase arrival files (station, onset, polarity, weight, P/S times, zone). | re, pandas |
| [`ogshpl.py`](ogshpl.py) | `DataFileHPL` | `OGSDataFile` | Parses `.hpl` hypocenter files with embedded pick records and event headers. | re, pandas |
| [`ogspun.py`](ogspun.py) | `DataFilePUN` | `OGSDataFile` | Parses `.pun` punch card event records (lat, lon, depth, magnitude, GAP, RMS, ERH, ERZ). | re, pandas |
| [`ogstxt.py`](ogstxt.py) | `DataFileTXT` | `OGSDataFile` | Parses `.txt` summary catalogs containing local and duration magnitudes (ML, MD) and error estimates. | re, pandas |
| [`ogsparser.py`](ogsparser.py) | `DataCatalog` | `OGSDataFile` | Multi-format aggregator and CLI; merges multi-format files into unified date-partitioned Parquet catalogs. | re, pandas, pyarrow |

HPL, PUN, and TXT readers accumulate named event dictionaries using the column constants, with canonical origin times and year-encoded event identifiers.
Each reader encodes its event IDs; the shared builder only coerces them to
integers and does not infer an ID's year. HPL headers use `normalize_index()`: internal spaces become zeros, while blank IDs become `None` and then builder zero, not an inferred year base. HPL pick-ID parsing remains unchanged.
`OGSDataFile._build_events_dataframe()` orders these records by `_EVENT_COLUMNS`, omits extractor-only fields, and supplies defaults for absent fields: zero for P/S/paired pick counts and `None` otherwise. Explicit values, including `None`, are preserved before normalization. HPL notes update the last retained event by field name. Existing DataFrame inputs are aligned by column name, not relabeled by position; positional-list inputs remain supported. The exported event schema
still has 28 columns.

Event construction uses `normalize_groups()` to normalize non-datetime origin timestamps once before coercing reader-supplied event IDs. Existing datetime dtypes, including timezone-aware and non-nanosecond values, are retained. Valid group dates are preserved; missing or invalid groups fall back to the origin time.

All four readers use shared input validation with their declared extension.
HPL depth/RMS and PUN depth/MD remain raw until the shared event builder coerces them to numeric values. Format-specific zero-padding and HPL magnitude-selection rules remain in their readers.

### 3. Core Catalog Management and Comparison

| Module | Primary Class | Description | Key Dependencies |
|---|---|---|---|
| [`ogscatalog.py`](ogscatalog.py) | `OGSCatalog` | Catalog container managing `EVENTS` and `PICKS` DataFrames, lazy CSV/Parquet reads, daily/aggregate caches, geographic event filtering, statistical plotting, and BPGMA comparison (`BPGMAEvents`, `BPGMAPicks`, `bpgma`). Legacy Parquet writing belongs to `OGSDataFile`. | pandas, matplotlib, pyarrow |

### 4. Waveform Acquisition, Storage, and Inventory

| Module | Primary Class / Function | Description | Key Dependencies |
|---|---|---|---|
| [`ogsdownloader.py`](ogsdownloader.py) | `BaseDownloader`, `ObsPyDownloader`, `PyrockoDownloader` | ObsPy FDSN downloads with rectangular, circular, or global domains and EIDA token support; `PyrockoDownloader.download()` is unimplemented. | obspy |
| [`ogsdata.py`](ogsdata.py) | `OGSSquirrelDataSource` | Day-sharded waveform data access using Pyrocko Squirrel; provides lazy per-day database instances for the ML catalog pipeline. | pyrocko, ml_catalog, obspy |
| [`ogsstation.py`](ogsstation.py) | CLI main | Writes waveform/station inventory CSVs and station maps; availability plotting depends on discovered inputs. Its parser retains a required legacy `src_root` argument. | ogsutils, obspy |

### 5. Machine Learning Pipeline Modules (`ml_catalog` Integration)

| Module | Primary Class | Extends / Wraps | Description | Key Dependencies |
|---|---|---|---|---|
| [`ogspicker.py`](ogspicker.py) | `OGSAmplitudeExtractor` | `ml_catalog.modules.AmplitudeExtractor` | Measures component SNR and Wood-Anderson amplitudes around picks; magnitude-stage SNR gating is performed by `OGSLocalMagnitude`. | obspy, ml_catalog |
| [`ogsmagnitude.py`](ogsmagnitude.py) | `OGSLocalMagnitude` | `ml_catalog.modules.LocalMagnitude` | Computes calibrated local magnitude from amplitude geometric means with P-component SNR gating and station corrections. Rejects deviations greater than 5x MAD only with at least three non-NaN station magnitudes and positive MAD. | numpy, pandas, ml_catalog |
| [`ogsqc.py`](ogsqc.py) | `OGSPickStatQC`, `OGSEventStatQC` | `ml_catalog` QC modules | Quality control filters using minimum pick counts (P/S/total) and geographic polygon bounding. | dask, matplotlib, ml_catalog |
| [`real.py`](real.py) | `REALAssociator` | `AbstractAssociator` | Wrapper around the REAL (Rapid Earthquake Association and Location) grid-search phase associator with TauP travel times. | obspy.taup, ml_catalog |
| [`ogsgrid2time.py`](ogsgrid2time.py) | Travel-time grid utilities | NonLinLoc grid files | Generates P/S travel-time grids and station mappings using analytical homogeneous fields or heterogeneous eight-direction sweeps. | numpy |
| [`ogsnonlinloc.py`](ogsnonlinloc.py) | `OGSNonLinLoc` | NonLinLoc integration | Stages environment-selected travel-time tables and performs relocation with adaptive chunks and local process parallelism. | ml_catalog |
| [`ogsbuilderMPI.py`](ogsbuilderMPI.py) | `OGSCatalogBuilderMPI` | `ml_catalog.CatalogBuilder` | MPI-parallel catalog generation driver using `dask_mpi`. | dask_mpi, ml_catalog |
| [`ogstrainer.py`](ogstrainer.py) | `OGSTrainer`, CLI main | PyTorch + SeisBench | Fine-tunes pretrained PhaseNet/EQTransformer using reference picks with weights $\le 2$, waveform conditioning, training/validation splits, and checkpoints. PhaseNet uses probabilistic-label cross-entropy; EQTransformer uses weighted detection/P/S binary cross-entropies with Gaussian phase targets ($\sigma=30$ samples) and a detection interval. | torch, seisbench, obspy, pandas, numpy |

### 6. Clustering and Visualization

| Module | Primary Class | Description | Key Dependencies |
|---|---|---|---|
| [`ogsclustering.py`](ogsclustering.py) | `OGSClusteringZoo`, `BaseClusterer`, `BaseClusteringScores` | Feature-array clustering, plotting, scoring, parameter sweeps, and synthetic manifold generation. Registers 14 algorithms (12 estimator/backend wrappers and two local density-peaks variants) and 11 scores (four unsupervised and seven reference-label). Local density-peaks/PAk routines do not require `dadapy`; algorithm equivalence and speedups are not established by their presence. | scikit-learn, numpy, scipy, matplotlib |
| [`ogssequence.py`](ogssequence.py) | `OGSSequence` | Existing-catalog analysis using standardized horizontal coordinates, depth, and log-transformed integer-second interevent time: $\log_{10}(\max(\Delta t_{sec}, 0.1))$. Searches one algorithm-specific parameter per algorithm/metric pair and generates maps/cross-sections. Map bounds are not applied to catalog loading; reference labels are not passed to optimization scores. | ogsclustering, matplotlib, pandas |
| [`ogsplotter.py`](ogsplotter.py) | `OGSPlotter`, plot builders | Reusable plotting utilities for seismic maps, waveforms, histograms, and diagnostic figures. | matplotlib, cartopy, obspy |

### 7. Thesis Analysis Drivers

| Module | Purpose | Notes |
|---|---|---|
| [`MHPCThesis.py`](MHPCThesis.py) | Compares PhaseNet/EQTransformer with STEAD/Original configurations at threshold 0.3 for March 20-June 20, 2024. | Other scenario loops are disabled; contains workstation-specific paths. |
| [`UNITSThesis.py`](UNITSThesis.py) | Compares 2020/2021 reference catalogs with PhaseNet[INSTANCE]+GaMMA QC and subsequent NLL1D/local-magnitude stages. | Direct plotting is disabled; contains workstation-specific paths. |

## Build and Execution Support

- [`input.json`](input.json): Sequence-clustering parameter example with a
  workstation-specific catalog directory, date ranges, algorithm/score names,
  search ranges, and plotting bounds. Review paths and scientific selections
  before use; this file is not a portable default.
[`real.py`](real.py) first imports its integration dependencies from
`ml_catalog`; its existing relative-import fallback is intended for an external
associator-package layout, not a generic OGS dependency fallback. That fallback
is unchanged. The local parser Makefile that previously lived in this directory
has been removed.
For parser regression tests, use [`OGS/test/Makefile`](../test/Makefile)
(e.g. `make -C OGS/test parser`).

## Public Python API

With the repository root on Python's import path, import public classes through
the lazy registry in [`OGS/__init__.py`](../__init__.py):

```python
from OGS import OGSCatalog, DataCatalog, ObsPyDownloader
```

| Area | Public classes |
|---|---|
| Catalog and parsers | `OGSCatalog`, `OGSDataFile`, `DataCatalog`, `DataFileDAT`, `DataFileHPL`, `DataFilePUN`, `DataFileTXT` |
| Analysis | `OGSClusteringZoo`, `OGSSequence`, `OGSTrainer` |
| Acquisition | `BaseDownloader`, `ObsPyDownloader`, `PyrockoDownloader` |
| Pipeline integration | `OGSSquirrelDataSource`, `OGSAmplitudeExtractor`, `OGSLocalMagnitude`, `OGSPickStatQC`, `OGSEventStatQC`, `OGSNonLinLoc`, `REALAssociator`, `OGSCatalogBuilderMPI` |

`import OGS` and `dir(OGS)` do not import the implementation modules.
Accessing a public class imports its owning module and caches the resolved
class. Unknown names raise `AttributeError`; dependency/import errors propagate
without being masked or cached as successful results. Type-checking imports
make the public classes visible to editors without eager runtime loading.

Dependencies are still required for the selected class: integration modules
require a compatible `ml_catalog` installation, and `OGSCatalogBuilderMPI`
additionally requires `dask_mpi`. Requesting those classes does not run a
pipeline, but their owning modules retain existing import-time behavior.
Avoid `from OGS import *`: it requests every export, including dependency-heavy
classes. Plotting dependencies remain lazy when importing `OGSCatalog`.

Use package-qualified imports:

```python
from OGS.src.ogscatalog import OGSCatalog

# Existing WORK deployment, when its src directory's parent is on the path:
from src.ogscatalog import OGSCatalog

```

```bash
python -m OGS.src.ogsparser --help
python -m OGS.src.ogsdownloader --help
python -m OGS.src.ogsstation --help
python -m OGS.src.ogstrainer --help
```

Run these commands from the repository root or place that root on
`PYTHONPATH`. The Leonardo Makefile exports the repository root and uses
module execution for these entrypoints. Setting `PYTHONPATH` alone does not
give a directly executed Python file a package context; top-level imports
such as `from ogscatalog import OGSCatalog` and file-path execution are not
the supported invocation for modules with relative sibling imports.
Existing Hydra `src.*` targets use the deployed package layout and are unchanged.
Use one import layout consistently within an application: importing the same
source under different namespaces creates distinct modules and class identities.
