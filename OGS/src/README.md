# OGS Source Modules (`OGS/src/`)

## Overview

The `OGS/src/` directory houses the core Python source code for data ingestion, catalog parsing, machine learning picking, phase association, event location, catalog comparison, and sequence clustering.

## Purpose and usage boundary

The modules are the implementation layer behind the repository entrypoints and Leonardo jobs. Use the module index to locate an interface, then inspect the module's CLI help, docstrings, configuration, and tests before relying on an undocumented argument or scientific interpretation.

## Inputs, outputs, and assumptions

- **Inputs:** legacy catalog files, waveform/archive paths, Parquet catalogs, stage configuration, and sequence-clustering JSON metadata.
- **Outputs:** in-memory DataFrames, date-partitioned Parquet data, CSV cluster members, plots, logs, and model/pipeline results depending on the selected module.
- **Assumptions:** optional dependencies and external data services are available for the selected path; file schemas and units follow the constants/configuration used by the current checkout.
- **Side effects:** downloader and persistence helpers write files and may contact external services. Analysis drivers can process large datasets. Prefer focused tests or dry runs and keep raw data outside this source directory.

## Module Directory Index

The 24 Python source modules (+ 1 package initializer) are organized functionally below:

### 1. Configuration, Types, and Utilities

| Module | Primary Class / Functions | Description | Key Dependencies |
|---|---|---|---|
| [`ogsconstants.py`](ogsconstants.py) | Constants, date formats, zone codes, FDSN clients | Central configuration hub (column names, geographic bounds, tolerances: `PICK_TIME_OFFSET=0.5s`, `EVENT_TIME_OFFSET=2s`, `EVENT_DIST_OFFSET=8km`). | standard library |
| [`ogsutils.py`](ogsutils.py) | `OGSBPGraph`, `OGSBPGraphPicks`, `OGSBPGraphEvents` | Bipartite graph matching backend (BGMA) using NetworkX `max_weight_matching`; distance metrics (`dist_pick`, `dist_event`), coordinate projection, date parsing. | networkx, numpy, scipy |

### 2. Legacy Catalog Ingestion and Parsing

| Module | Primary Class | Extends | Format / Purpose |
|---|---|---|---|
| [`ogsdatafile.py`](ogsdatafile.py) | `OGSDataFile` | `OGSCatalog` | Abstract base class for regex-driven parsing of legacy OGS catalog formats; defines `read()`, `log()`, `debug()`. | re, pandas, pyarrow |
| [`ogsdat.py`](ogsdat.py) | `DataFileDAT` | `OGSDataFile` | Parses legacy fixed-width `.dat` phase arrival files (station, onset, polarity, weight, P/S times, zone). | re, pandas |
| [`ogshpl.py`](ogshpl.py) | `DataFileHPL` | `OGSDataFile` | Parses `.hpl` hypocenter files with embedded pick records and event headers. | re, pandas |
| [`ogspun.py`](ogspun.py) | `DataFilePUN` | `OGSDataFile` | Parses `.pun` punch card event records (lat, lon, depth, magnitude, GAP, RMS, ERH, ERZ). | re, pandas |
| [`ogstxt.py`](ogstxt.py) | `DataFileTXT` | `OGSDataFile` | Parses `.txt` summary catalogs containing local and duration magnitudes (ML, MD) and error estimates. | re, pandas |
| [`ogsparser.py`](ogsparser.py) | `DataCatalog` | `OGSDataFile` | Multi-format aggregator and CLI; merges multi-format files into unified date-partitioned Parquet catalogs. | re, pandas, pyarrow |

### 3. Core Catalog Management and Comparison

| Module | Primary Class | Description | Key Dependencies |
|---|---|---|---|
| [`ogscatalog.py`](ogscatalog.py) | `OGSCatalog` | Core catalog container managing `EVENTS` and `PICKS` DataFrames, lazy loading, geographic filtering, Parquet I/O, statistical plotting, and BGMA comparison (`bgmaEvents`, `bgmaPicks`). | pandas, matplotlib, pyarrow |

### 4. Waveform Acquisition, Storage, and Inventory

| Module | Primary Class / Function | Description | Key Dependencies |
|---|---|---|---|
| [`ogsdownloader.py`](ogsdownloader.py) | `OGSDownloader` | Waveform downloader using ObsPy `MassDownloader` with rectangular/circular domain selection and EIDA token support. | obspy |
| [`ogsdata.py`](ogsdata.py) | `OGSSquirrelDataSource` | Day-sharded waveform data access using Pyrocko Squirrel; provides lazy per-day database instances for the ML catalog pipeline. | pyrocko, ml_catalog, obspy |
| [`ogsstation.py`](ogsstation.py) | CLI main | Station metadata and inventory extractor for waveform archives across date ranges. | ogsutils, obspy |

### 5. Machine Learning Pipeline Modules (`ml_catalog` Integration)

| Module | Primary Class | Extends / Wraps | Description | Key Dependencies |
|---|---|---|---|---|
| [`ogspicker.py`](ogspicker.py) | `OGSAmplitudeExtractor` | `ml_catalog.modules.AmplitudeExtractor` | Extracts Wood-Anderson peak amplitudes from horizontal components around SeisBench picks with SNR gating. | obspy, ml_catalog |
| [`ogsmagnitude.py`](ogsmagnitude.py) | `OGSLocalMagnitude` | `ml_catalog.modules.LocalMagnitude` | Computes OGS-calibrated local magnitude ($M_L$) with attenuation curve, station corrections, and 5x MAD outlier rejection. | numpy, pandas, ml_catalog |
| [`ogsqc.py`](ogsqc.py) | `OGSPickStatQC`, `EventStatQC` | `ml_catalog.modules.PickStatQC` | Quality control filters using minimum pick counts (P/S/total) and geographic polygon bounding. | dask, matplotlib, ml_catalog |
| [`real.py`](real.py) | `REALAssociator` | `AbstractAssociator` | Wrapper around the REAL (Rapid Earthquake Association and Location) grid-search phase associator with TauP travel times. | obspy.taup, ml_catalog |
| [`ogsbuilderMPI.py`](ogsbuilderMPI.py) | `OGSCatalogBuilderMPI` | `ml_catalog.CatalogBuilder` | MPI-parallel catalog generation driver using `dask_mpi`. | dask_mpi, ml_catalog |
| [`ogstrainer.py`](ogstrainer.py) | `OGSTrainer`, CLI main | PyTorch + SeisBench | Complete fine-tuning engine for SeisBench pickers (PhaseNet, EQTransformer) on OGS waveform archives: catalog pick filtering (Hypo71 weights $\le 2$), DSP trace conditioning, arrival-centered augmentation, probabilistic Gaussian target labeling ($\sigma=30$), vector cross-entropy loss, Adam optimization, validation, and epoch/best checkpointing. | torch, seisbench, obspy, pandas, numpy |

### 6. Clustering and Visualization

| Module | Primary Class | Description | Key Dependencies |
|---|---|---|---|
| [`ogsclustering.py`](ogsclustering.py) | `OGSClusteringZoo`, `BaseClusterer`, `BaseClusteringScores` | Clustering framework implementing 14 algorithm wrappers (KMeans, HDBSCAN, DBSCAN, ADP, etc.) and 11 evaluation metrics. ADP and PAk are implemented from scratch in pure NumPy/SciPy without external `dadapy` dependency. | scikit-learn, numpy, scipy, matplotlib |
| [`ogssequence.py`](ogssequence.py) | `OGSSequence` | Automated earthquake sequence clustering pipeline; standardizes spatiotemporal features ($X_{km}, Y_{km}, Depth_{km}, \Delta t_{sec}$), optimizes hyperparameters, and generates map/cross-section plots. | ogsclustering, matplotlib, pandas |
| [`ogsplotter.py`](ogsplotter.py) | `OGSPlotter`, plot builders | Reusable plotting utilities for seismic maps, waveforms, histograms, and diagnostic figures. | matplotlib, cartopy, obspy |

### 7. Thesis Analysis Drivers

| Module | Purpose | Notes |
|---|---|---|
| [`MHPCThesis.py`](MHPCThesis.py) | Master in HPC thesis driver evaluating ML pipeline configurations against the 2024 Italian sequence benchmark. | Contains workstation-specific path defaults. |
| [`UNITSThesis.py`](UNITSThesis.py) | Università degli Studi di Trieste thesis driver for catalog comparison across 2020–2021 target datasets. | Contains workstation-specific path defaults. |

## Build and Execution Support

- [`input.json`](input.json): Example sequence-clustering parameter configuration for `ogssequence.py`.
The standalone `real.py` file uses relative imports intended for placement
inside the corresponding `ml_catalog` package hierarchy; importing it directly
as a top-level `OGS/src` module is not established by this checkout. The local
parser Makefile that previously lived in this directory has been removed.
For parser regression tests, use [`OGS/test/Makefile`](../test/Makefile)
(e.g. `make -C OGS/test parser`).
