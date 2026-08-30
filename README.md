# OGS Seismic Toolkit (AI2Seism)

A Python toolkit for parsing, managing, clustering, comparing, and visualizing
seismic catalogs from OGS (Istituto Nazionale di Oceanografia e di Geofisica
Sperimentale), focused on the seismicity of north-eastern Italy and surrounding
regions.

## Purpose and scope

This repository documents and supports the OGS seismic-catalog workflow. It
contains reusable Python modules, Hydra configuration, tests, High-Performance
shared systems (CINECA Leonardo) and local workstation execution helpers, and
research documentation. The repository is not a replacement for the external
waveform/catalog workspace and does not by itself provide a complete runtime
environment.

## Inputs, outputs, and operational boundary

- **Inputs:** OGS catalog files (`.dat`, `.hpl`, `.pun`, `.txt`), date/region
  and client options for FDSN waveform retrieval, Parquet catalogs, and JSON
  metadata for sequence clustering.
- **Outputs:** date-partitioned Parquet catalogs, comparison tables and plots,
  clustering CSV files, and diagnostic figures. Exact paths depend on the
  selected command or Makefile variables.
- **Assumptions:** optional dependencies such as ObsPy, Pyrocko, SeisBench,
  Hydra, and `ml_catalog` are installed and configured before the related
  workflow is run. Source code, tests, configuration, and the Makefile are
  authoritative when this overview differs from implementation.
- **Side effects and safety:** downloading contacts external data services and
  writes waveform files; parsing and analysis commands write output files;
  Leonardo Makefile targets can submit SLURM jobs. Inspect help or a dry run
  first, keep outputs in the configured workspace, and do not use destructive
  targets such as `make clean` as a smoke test.


<code><pre>
                          ###
                   #################
                ########################
             #############################
            ################################
          ###################################
  ........---------------------+++++##########
 ........--------------------+++++++++#########
........--------------------+++++++++++#########
........---------                     ...........+++
 ......--------                       ...........++++
  .....-------                      .............+++++
       ######....................................+++++
       #######...................................+++++
       #########-................................++++
        ################+           +###########
        ################.          .###########
         ##############+           -###########
          #############+           ##########
           ############.          -#########
             ##########           +#######
                ######.          .######
                   ###           -###
</pre></code>

<code><pre><pink>
&nbsp;                         ++++++++++++++++++++    +++++++++++
&nbsp;                     ++++++++++++++++++++++++   +++++++++++
&nbsp;                +++++++++++++++++++++++++++    +++++++++++
&nbsp;               +++++++++++
&nbsp;                 +++
&nbsp;                             +++++    +++++++++    ++++++++++++++
&nbsp;  ++++++++++++++       +++++++++        +++++     +++++++++++++++
&nbsp; ++++++++++++++++++      +++++     +++          +++++++++++++++++
&nbsp;+++++++++++++++++++++             +++++++         +++++++++++++++
&nbsp;+++++++++++++++++++++++      +++   +++++     +         ++++++++++
&nbsp;+++++++++++++++++++++++++     ++++         ++++++            ++++
&nbsp; ++++++++++++++++++++++++++     +++      ++++++++++++
&nbsp;  ++          +++++++++++++++              ++++++++++++++
&nbsp;                ++++++++++++++       +++      +++++++++++++++
&nbsp;                                      +++++     +++++++++++++
&nbsp;                                        +++++     +++++++++++
&nbsp;健                                        +++++      ++++++++
</pink></pre></code>



## Table of Contents

- [Purpose and scope](#purpose-and-scope)
- [Inputs, outputs, and operational boundary](#inputs-outputs-and-operational-boundary)
- [Overview](#overview)
- [Architecture](#architecture)
- [Modules](#modules)
  - [ogsconstants.py](#ogsconstantspy)
  - [ogsdownloader.py](#ogsdownloaderpy)
  - [ogscatalog.py](#ogscatalogpy)
  - [ogsdatafile.py](#ogsdatafilepy)
  - [Format Parsers](#format-parsers-ogsdatpy-ogshplpy-ogspunpy-ogstxtpy)
  - [ogsparser.py](#ogsparserpy)
  - [ogsclustering.py](#ogsclusteringpy)
  - [ogssequence.py](#ogssequencepy)
- [Catalog comparison](#catalog-comparison)
- [Additional Modules](#additional-modules)
- [Data Flow](#data-flow)
- [Supported File Formats](#supported-file-formats)
- [Installation and Dependencies](#installation-and-dependencies)
- [CLI Usage](#cli-usage)
- [Bipartite Graph Matching Algorithm (BGMA)](#bipartite-graph-matching-algorithm-bgma)
- [Clustering Framework](#clustering-framework)
- [Documentation and agent navigation](#documentation-and-agent-navigation)

---

## Overview

The OGS Seismic Toolkit provides an end-to-end pipeline for seismic catalog analysis:

1. **Download** waveform data from FDSN data centers (INGV, GFZ, IRIS, ETH, ORFEUS)
2. **Parse** legacy OGS catalog formats (.dat, .hpl, .pun, .txt) into unified DataFrames
3. **Merge** multi-format catalogs with cross-referenced picks and events
4. **Store** catalogs in Parquet format with date-partitioned directory structure
5. **Compare** reference and ML-pipeline catalogs using bipartite graph matching
6. **Cluster** seismic events using 14 registered algorithms with hyperparameter optimization
7. **Visualize** results with map views, cross-sections, dendrograms, and diagnostic plots

The source registry contains 14 algorithm wrappers and 11 metric wrappers
(4 unsupervised plus 7 supervised). These are registry counts, not a claim
that optional dependencies are installed or that every metric is valid for
every catalog.

The study region covers north-eastern Italy: Friuli, Veneto, Trentino-Alto Adige, Venezia Giulia, Lombardia, Emilia-Romagna, and bordering areas of Austria, Slovenia, and Croatia (approximately 9.5-15.0 E, 44.3-47.5 N).

## Documentation and agent navigation

The repository operating contract is [`AGENTS.md`](AGENTS.md),
and the LLM review workspace is
[`LLM/README.md`](LLM/README.md). To inspect all Markdown files with
current line-numbered headings, run:

```bash
bash LLM/scripts/handler.sh navigate --root .
```

The report excludes Git internals and is the navigation index used by the
sequential documentation cycle. Source code, tests, configuration, and the
Makefile remain the authoritative implementation evidence.

---

## Architecture

```
                        ogsconstants.py
                    (constants and defaults)
                              |
        +---------------------+---------------------+
        |                     |                     |
ogsdownloader.py        ogscatalog.py        ogsclustering.py
(FDSN download)         (core container)     (14 algorithms + metrics)
                              |                     |
                        ogsdatafile.py      OGSClusteringZoo
                        (ABC for parsing)   (factory + optimization)
                              |                     |
              +-------+-------+-------+       ogssequence.py
              |       |       |       |     (sequence pipeline)
            ogsdat  ogshpl  ogspun  ogstxt
            (.dat)  (.hpl)  (.pun)  (.txt)
              |       |       |       |
              +-------+-------+-------+
                              |
                        ogsparser.py
                        (multi-format aggregator)
                              |
                    OGSCatalog.bgmaEvents
                    OGSCatalog.bgmaPicks
                    (catalog comparison)
```

### Class Hierarchy

```
OGSCatalog                          BaseClusterer (ABC)
  |                                   |
  +-- OGSDataFile (ABC)               +-- CentroidClusterer
  |     |                             |     +-- OGSKMeans
  |     +-- DataFileDAT (.dat)        |     +-- OGSMiniBatchKMeans
  |     +-- DataFileHPL (.hpl)        |     +-- OGSBisectingKMeans
  |     +-- DataFilePUN (.pun)        +-- OGSDBSCAN
  |     +-- DataFileTXT (.txt)        +-- OGSHDBSCAN
  |     +-- DataCatalog (aggregator)  +-- OGSOPTICS
  |                                   +-- OGSAdvancedDensityPeaks
  |                                   |     +-- OGSAdvancedDensityPeaksPP
  +-- (uses ogsutils BGMA)            +-- OGSAgglomerative
                                      +-- OGSFeatureAgglomeration
BaseClusteringScores (ABC)            +-- OGSAffinityPropagation
  |                                   +-- OGSMeanShift
  +-- SilhouetteScore                 +-- OGSSpectralClustering
  +-- CalinskiHarabaszScore           +-- OGSBirch
  +-- DaviesBouldinScore
  +-- PAkDensitySeparationScore   OGSClusteringZoo
  +-- AdjustedRandScore             |
  +-- NormalizedMutualInfoScore     +-- OGSSequence (pipeline)
  +-- AdjustedMutualInfoScore
  +-- HomogeneityScore
  +-- CompletenessScore
  +-- VMeasureScore
  +-- FowlkesMallowsScore
```

---

## Modules

### ogsconstants.py

Central configuration hub for the entire project (851 lines in the current
checkout).

**Contains:**
- **String constants**: Column names (`latitude`, `longitude`, `depth`, `time`, `ML`, `station`, `phase`, etc.)
- **Date formats**: `DATE_FMT = "%Y-%m-%d"`, `YYYYMMDD_FMT = "%Y%m%d"`, `YYMMDD_FMT = "%y%m%d"`, and others
- **File extensions**: `.dat`, `.hpl`, `.pun`, `.txt`
- **OGS study region**: `[9.5, 15.0, 44.3, 47.5]` (lonW, lonE, latS, latN)
- **Geographic zone codes**: A (Alto Adige), C (Croatia), E (Emilia), F (Friuli), G (Venezia Giulia), L (Lombardia), O (Austria), R (Romagna), S (Slovenia), T (Trentino), V (Veneto)
- **FDSN clients**: OGS, INGV, GFZ, IRIS, ETH, ORFEUS, Collalto
- **Matching tolerances**: `PICK_TIME_OFFSET = 0.5s`, `EVENT_TIME_OFFSET = 2s`, `EVENT_DIST_OFFSET = 8 km`

---

### ogsutils.py

**Contains:**
- **Distance functions**: `dist_pick()` (weighted by 97% `time`, 2% `phase`, 1% `probability`), `dist_event()` (weighted by 50% `time`, 25% `location`, 25% `magnitude`)
- **Utility functions**: `labels_to_colormap()`, `inventory()`, `waveforms()`, `is_date()`, `is_julian()`, `is_file_path()`, and `is_dir_path()`
- **Bipartite graph matching**: `OGSBPGraph`, `OGSBPGraphPicks`, and `OGSBPGraphEvents` classes are implemented using NetworkX's `max_weight_matching`; `OGSCatalog.bgmaEvents()` and `bgmaPicks()` assemble review tables and plots.

---

### ogsdownloader.py

FDSN waveform data downloader using ObsPy's `MassDownloader`.

**Features:**
- Rectangular or circular geographic domain selection
- Day-by-day download to `YYYY/MM/DD` directory structure
- EIDA token authentication for restricted data access
- Multiple FDSN client support; per-client failures are logged so other
  configured clients can continue
- Optional clip time for event-centered downloads
- PyRocko integration option for multi-threaded downloads

**Usage:**
```bash
# Download waveforms for a region and date range
python ogsdownloader.py -D 20240320 20240620 --rectdomain 9.5 15.0 44.3 47.5

# With EIDA authentication token
python ogsdownloader.py -D 20240320 20240620 -K /path/to/token --client INGV GFZ

# Circular domain around a point
python ogsdownloader.py -D 20240320 20240620 --circdomain 13.0 46.0 0.0 0.5
```

---

### ogscatalog.py

Core catalog container class (2,988 lines in the current checkout). Manages
EVENTS and PICKS DataFrames with lazy loading, geographic filtering, and
Parquet/CSV/JSON I/O.

**Key Features:**
- **Lazy loading**: `preload()` scans file paths, `load()` reads on demand, `get()` aggregates into a single DataFrame
- **Parquet and CSV I/O**: Date-partitioned directory structure (`events/YYYY-MM-DD`, `assignments/YYYY-MM-DD`)
- **Geographic filtering**: Polygon-based containment via `matplotlib.path.Path`
- **Date range filtering**: Temporal subsetting at load time
- **Plotting methods**: `plot_events()`, `plot_cumulative_events()`, `plot_cumulative_picks()`, `plot_erh_histogram()`, `plot_erz_histogram()`, `plot_ert_histogram()`, `plot_magnitude_histogram()`, `plot_depth_histogram()`
- **BGMA comparison**: `bgmaEvents()` and `bgmaPicks()` for catalog matching, with confusion matrices and TP/FN/FP computation
- **Operator overloads**: `+=` (merge catalogs), `-=` (subtract catalogs)

**Programmatic usage:**
```python
from ogscatalog import OGSCatalog
from pathlib import Path
from datetime import datetime

catalog = OGSCatalog(
    input=Path("/path/to/catalog"),
    start=datetime(2022, 1, 1),
    end=datetime(2022, 12, 31),
    verbose=True
)
catalog.get("EVENTS")
catalog.plot_events()
```

---

### ogsdatafile.py

Abstract base class for regex-based parsing of OGS file formats (278 lines in
the current checkout). Extends `OGSCatalog`.

**Key Features:**
- Configurable regex patterns: `RECORD_EXTRACTOR_LIST` (picks) and `EVENT_EXTRACTOR_LIST` (events), defined by subclasses
- Compiled regex matching via `re.compile()`
- `read()` abstract method for format-specific parsing
- `log()` writes parsed picks and events to Parquet in date-based directory structure
- `debug()` utility for diagnosing regex failures by progressively testing truncated patterns

---

### Format Parsers (ogsdat.py, ogshpl.py, ogspun.py, ogstxt.py)

Four format-specific parsers, each extending `OGSDataFile`:

| Module | Class | Extension | Content |
|--------|-------|-----------|---------|
| `ogsdat.py` | `DataFileDAT` | `.dat` | Legacy fixed-width phase picks (P/S arrivals, station, onset, polarity, weight, zone) |
| `ogshpl.py` | `DataFileHPL` | `.hpl` | Hypocenter locations with embedded pick records and event headers |
| `ogspun.py` | `DataFilePUN` | `.pun` | Punch card event locations (lat, lon, depth, magnitude, GAP, RMS, ERH, ERZ) |
| `ogstxt.py` | `DataFileTXT` | `.txt` | Text catalog with magnitudes (ML, MD), error estimates (ERH, ERZ, ERT, GAP), and location names |

Each parser:
1. Defines format-specific regex patterns in `RECORD_EXTRACTOR_LIST` / `EVENT_EXTRACTOR_LIST`
2. Implements `read()` to parse the file line by line
3. Populates `self.PICKS` and `self.EVENTS` DataFrames
4. Uses `log()` (inherited) to write Parquet output

---

### ogsparser.py

Multi-format catalog aggregator (605 lines in the current checkout). Extends
`OGSDataFile` to automatically dispatch parsing to format-specific handlers and
merge results into a unified catalog.

**File Type Registry:**
```
.hpl -> DataFileHPL  (recommended: hypocenter information)
.dat -> DataFileDAT  (recommended: picks information)
.txt -> DataFileTXT  (local magnitude information)
.pun -> DataFilePUN  (punch card event locations)
```

**Merge Logic:**
- **Picks**: Simple concatenation from all input files
- **Events**: Outer join strategy:
  - HPL files provide primary event information
  - TXT files contribute magnitude data (ML, MD) and error estimates (ERH, ERZ, ERT, GAP)
  - PUN files contribute hypocenter locations
  - After merging, pick statistics are computed per event (P-pick count, S-pick count, stations with both P and S)

**Usage:**
```bash
# Parse and merge files from a directory
python ogsparser.py -d /path/to/catalog/ -x .hpl .dat .txt --merge

# Parse specific files
python ogsparser.py -f file1.hpl file2.dat -D 20220101 20221231 --merge

# With geographic filtering and custom output
python ogsparser.py -d /path/to/catalog/ --merge -o /path/to/output/
```

**Output structure:**
```
{output}/.all/assignments/YYYY-MM-DD  (merged picks)
{output}/.all/events/YYYY-MM-DD       (merged events)
```

---

### ogsclustering.py

Clustering framework (6,685 lines) wrapping scikit-learn algorithms with integrated visualization and evaluation.

**14 registered clustering algorithms:**

| Category | Class | Algorithm | Key Parameters |
|----------|-------|-----------|----------------|
| Centroid-based | `OGSKMeans` | K-Means | `n_clusters`, `init`, `n_init` |
| Centroid-based | `OGSMiniBatchKMeans` | Mini-Batch K-Means | `n_clusters`, `batch_size` |
| Centroid-based | `OGSBisectingKMeans` | Bisecting K-Means | `n_clusters` |
| Density-based | `OGSDBSCAN` | DBSCAN | `eps`, `min_samples` |
| Density-based | `OGSHDBSCAN` | HDBSCAN | `min_cluster_size`, `min_samples` |
| Density-based | `OGSOPTICS` | OPTICS | `min_samples`, `max_eps`, `xi` |
| Density-based | `OGSAdvancedDensityPeaks` | ADP (dadapy) | dadapy parameters |
| Density-based | `OGSAdvancedDensityPeaksPP` | ADP post-processing variant | ADP parameters |
| Connectivity | `OGSAgglomerative` | Agglomerative | `n_clusters`, `linkage` |
| Connectivity | `OGSFeatureAgglomeration` | Feature Agglomeration | `n_clusters` |
| Message-passing | `OGSAffinityPropagation` | Affinity Propagation | `damping`, `preference` |
| Message-passing | `OGSMeanShift` | Mean Shift | `bandwidth` |
| Spectral | `OGSSpectralClustering` | Spectral Clustering | `n_clusters`, `affinity` |
| Tree-based | `OGSBirch` | BIRCH | `n_clusters`, `threshold` |

**11 Evaluation Metrics:**

| Type | Metric | Range | Interpretation |
|------|--------|-------|----------------|
| Unsupervised | SilhouetteScore | [-1, 1] | Higher = better separated |
| Unsupervised | CalinskiHarabaszScore | [0, inf) | Higher = better defined |
| Unsupervised | DaviesBouldinScore | [0, inf) | Lower = better separated |
| Unsupervised | PAkDensitySeparationScore | implementation-defined | Density-separation score |
| Supervised | AdjustedRandScore | [-1, 1] | 1 = perfect agreement |
| Supervised | NormalizedMutualInfoScore | [0, 1] | 1 = perfect correlation |
| Supervised | AdjustedMutualInfoScore | [-1, 1] | Higher = better |
| Supervised | HomogeneityScore | [0, 1] | 1 = perfectly homogeneous |
| Supervised | CompletenessScore | [0, 1] | 1 = perfectly complete |
| Supervised | VMeasureScore | [0, 1] | Harmonic mean of above two |
| Supervised | FowlkesMallowsScore | [0, 1] | Higher = better agreement |

**OGSClusteringZoo** - Factory class for algorithm comparison and optimization:
```python
from ogsclustering import OGSClusteringZoo

metadata = {
    "algorithms": ["KMeans", "HDBSCAN", "DBSCAN"],
    "eval_metrics": ["SilhouetteScore"],
    "num_clusters_range": (2, 10, 1),
    "cluster_size_range": (10, 100, 10),
    "eps_range": (0.3, 1.0, 0.1),
}
zoo = OGSClusteringZoo(metadata=metadata, verbose=True)
zoo.run(X)
```

See [Clustering Framework](#clustering-framework) for details.

---

### ogssequence.py

Seismic sequence clustering pipeline (1,126 lines). Extends `OGSClusteringZoo` for automated spatiotemporal analysis of earthquake catalogs.

**Pipeline Stages:**
1. **Load catalog**: Read events for each time window from Parquet
2. **Prepare features**: Convert lat/lon to Cartesian (equirectangular projection), compute inter-event times
3. **Standardize**: Scale features using `StandardScaler` (zero mean, unit variance)
4. **Optimize**: Grid search for best hyperparameters per algorithm/metric
5. **Cluster**: Assign events to clusters with optimized parameters
6. **Save**: Export per-cluster CSV files
7. **Visualize**: Map views and cross-section plots at configurable azimuths

**Feature Set** (standardized via `StandardScaler`):
- `X_KM`: East-West position in kilometers (from longitude via equirectangular projection)
- `Y_KM`: North-South position in kilometers (from latitude)
- `DEPTH`: Hypocenter depth in kilometers
- `INTEREVENT_LOG`: Base-10 logarithm of the non-negative inter-event time
  (seconds, with a 0.1-second floor) used for clustering; the raw
  `INTEREVENT` column is retained for reference and starts at zero for the
  first event.

**Usage:**
```bash
python ogssequence.py -i config.json -v
```

**Configuration (JSON):**
```json
{
  "directory": "/path/to/catalog",
  "ranges": [["2022-01-01", "2022-06-30"], ["2022-07-01", "2022-12-31"]],
  "angles_deg": [0, 45, 90],
  "map_deg": [13.09, 13.46, 42.44, 42.61],
  "cross_km": [-10.0, 10.0, 15.0, 0.0],
  "map_km": [30.0, 50.0],
  "annotations": [[13.2, 42.5, "Norcia"]],
  "algorithms": ["HDBSCAN", "DBSCAN"],
  "eval_metrics": ["SilhouetteScore", "DaviesBouldinScore"],
  "cluster_size_range": [10, 100, 10],
  "eps_range": [0.3, 1.0, 0.1]
}
```

**Output:**
```
Clusters/{algorithm}/{metric}/{start}_{end}/{cluster_id}.csv  (event rows)
Clusters/{algorithm}_{metric}_{angle}.png                     (plot)
```

The CSV directory name is built from each configured date range, while the
plot is written relative to the process working directory. This follows
`OGS/src/ogssequence.py:569-600`, `:706-735`, and `:1047-1084`.

Each plot contains:
- **Top row**: Map view (longitude vs latitude) with cluster colors, high-magnitude stars (M > 3.5), projection line, and cluster labels
- **Bottom row**: Cross-section (along-strike projection vs depth) at the specified azimuth angle

---

### Catalog comparison

Catalog comparison is exposed through `OGSCatalog.bgmaEvents()` and
`OGSCatalog.bgmaPicks()`; their matching backend is in `ogsutils.py`. The
supported in-tree interface verified for this documentation cycle is
`OGSCatalog.bgmaEvents()` for events and `OGSCatalog.bgmaPicks()` for picks;
their matching classes and distance functions are in `OGS/src/ogsutils.py`.

**Inputs:**
- **Base catalog**: Reference catalog (OGS data in .dat/.hpl/.txt/.pun format)
- **Target catalog**: Pipeline outputs (SeisBench, Gamma, PyOcto, NonLinLoc, etc. in Parquet format)

**Compares:**
- **Picks**: Per-station P/S phase classifications
- **Events**: Detected events and attributes (time, location, depth, magnitude)

**Produces:**
- Confusion matrices and review partitions (`MH`, `MS`, `SM`, `PS`, `SP`)
- Recall and false-discovery-rate metrics; matched rows also support RMSE/MAE plots where applicable
- Diagnostic plots (maps, histograms, scatter plots)

See [Bipartite Graph Matching Algorithm](#bipartite-graph-matching-algorithm-bgma) for the matching methodology and `OGS/src/ogscatalog.py:1733-1808` for the event-comparison implementation.

---

### Additional Modules

The following modules are part of the `OGS/src/` codebase but serve
supporting or specialized roles:

| Module | Purpose | Key Dependencies |
|--------|---------|-----------------|
| `ogsdata.py` | Day-sharded Pyrocko/Squirrel waveform data access for the ml_catalog pipeline | pyrocko, ml_catalog |
| `ogsplotter.py` | Reusable Matplotlib/Cartopy figure builders for maps, waveforms, metrics, and histograms | matplotlib, cartopy, obspy |
| `ogsmagnitude.py` | OGS-calibrated local magnitude (M_L) with per-station corrections and MAD outlier rejection | ml_catalog |
| `ogspicker.py` | Wood-Anderson amplitude extraction with SNR gating for magnitude estimation | obspy |
| `ogsqc.py` | Region-aware quality control: pick-count thresholds and geographic polygon filtering | dask, ml_catalog |
| `ogsstation.py` | CLI wrapper for station waveform inventory discovery | ogsutils |
| `ogstrainer.py` | SeisBench model fine-tuning on OGS catalog data | torch, seisbench |
| `ogsbuilderMPI.py` | MPI-parallel catalog builder extending ml_catalog's `CatalogBuilder` | dask_mpi, ml_catalog |
| `real.py` | REAL phase associator wrapper with TauP travel-time integration | obspy.taup, ml_catalog |
| `MHPCThesis.py` | Thesis driver script: BGMA comparison across picker/associator/locator configurations | (analysis script) |
| `UNITSThesis.py` | Thesis driver script: catalog comparison for UNITS thesis | (analysis script) |

Note: `MHPCThesis.py` and `UNITSThesis.py` contain hardcoded local paths and
are intended as reproducible analysis scripts for specific thesis submissions,
not as general-purpose tools.

---

## Data Flow

```
      Raw Files (.dat, .hpl, .pun, .txt)
                      |
          ogsparser.py (parse + merge)
                      |
        Parquet Files (date-partitioned)
                      |
          ogscatalog.py (load + query)
                      |
          +-----------+-----------+
          |                       |
OGSCatalog.bgmaEvents       ogssequence.py
  (compare catalogs)     (cluster + visualize)
```

**Parquet Directory Structure:**
```
{catalog}/
  .dat/
    assignments/YYYY-MM-DD   (picks)
    events/YYYY-MM-DD        (events)
  .hpl/
    assignments/YYYY-MM-DD
    events/YYYY-MM-DD
  .txt/
    events/YYYY-MM-DD
  .pun/
    events/YYYY-MM-DD
  .all/                      (merged catalog)
    assignments/YYYY-MM-DD
    events/YYYY-MM-DD
```

---

## Supported File Formats

### .dat (Phase Picks)
Legacy fixed-width format containing P and S wave arrival times per station. Fields include station code (4 chars), onset indicator, polarity, weight, datetime, P/S times, geographic zone, event type, duration, and event index.

### .hpl (Hypocenter Locations)
Comprehensive format containing both pick records and event headers. Provides hypocenter locations with associated phase arrival data. Recommended primary format.

### .pun (Punch Card Events)
Single-line event records in punch card format. Each line contains: date, origin time seconds, latitude, longitude, depth, magnitude, number of observations, azimuthal gap, minimum distance, RMS residual, ERH, ERZ, and quality marker.

### .txt (Text Catalog)
Text-format event catalog with magnitudes and error estimates. Contains: event index, origin time, ERT (time error), latitude, longitude, ERH, depth, ERZ, GAP, ML (local magnitude), MD (duration magnitude), location name, and event type.

---

## Installation and Dependencies

### Required:
- **numpy**: Numerical computing and array operations
- **pandas**: DataFrame operations, Parquet I/O, catalog manipulation
- **scikit-learn**: Clustering algorithms, evaluation metrics, feature scaling
- **matplotlib**: Plotting, visualization, geographic polygon filtering
- **obspy**: Seismological time handling (UTCDateTime), FDSN data access
- **networkx**: Bipartite graph matching for catalog comparison

### Optional:
- **dadapy**: Advanced Density Peaks clustering algorithm
- **scipy**: Dendrogram visualization for hierarchical clustering
- **pyrocko**: Alternative multi-threaded waveform downloader

---

## CLI Usage

### Download Waveforms
```bash
python ogsdownloader.py \
  -D 20240320 20240620 \
  --rectdomain 9.5 15.0 44.3 47.5 \
  --client INGV GFZ IRIS \
  -v
```

### Parse and Merge Catalogs
```bash
python ogsparser.py \
  -d /path/to/raw/catalog/ \
  -x .hpl .dat .txt .pun \
  -D 20220101 20221231 \
  --merge \
  -o /path/to/output/ \
  -v
```

### Compare Catalogs
```python
from pathlib import Path
from ogscatalog import OGSCatalog

base = OGSCatalog(input=Path("/path/to/base/.all"))
target = OGSCatalog(input=Path("/path/to/target/.all"))
base.bgmaEvents(target, output=Path("/path/to/review"))
# Pick comparison additionally requires a station inventory on `base`.
# See OGS/src/ogscatalog.py:1733-1808 and :2326-2374.
```

### Run Sequence Clustering
```bash
python ogssequence.py -i config.json -v
```

---

## Bipartite Graph Matching Algorithm (BGMA)

The matching engine behind catalog comparisons. It is implemented in
`ogsutils.py` (classes `OGSBPGraph`, `OGSBPGraphPicks`, and
`OGSBPGraphEvents`) and called by the comparison methods in `ogscatalog.py`.

### Purpose

Solves "what matches what?" between two time-indexed collections:
- **TRUE nodes** (reference catalog) vs **PRED nodes** (predictions)
- Two domains: **Picks** (phase arrivals per station) and **Events** (earthquake hypotheses)

### Distance Functions

**Picks** (`dist_pick`): Composite score blending time proximity, phase agreement, and model probability ratio.
```
dist_pick = 0.97 * dist_time + 0.02 * dist_phase + 0.01 * dist_prob
```
- `dist_time`: `1 - |t_true - t_pred| / PICK_TIME_OFFSET` (0.5s window)
- `dist_phase`: 1 if phases match, else 0
- `dist_prob`: bounded probability ratio. The function `dist_prob(B, T)` computes `P_target / P_base` and clamps to `[0, 1]` (`ogsutils.py:181–206`). The call site in `dist_pick` passes arguments in `(T, B)` order (`ogsutils.py:471`), so the effective score is `P_base / P_target`. Division by zero is guarded by `eps=1e-6`.

**Events** (`dist_event`): Weighted combination of temporal and spatial proximity.
```
dist_event = 0.99 * dist_time + 0.01 * dist_space
```
- `dist_time`: `1 - |t_true - t_pred| / EVENT_TIME_OFFSET` (2s window)
- `dist_space`: `1 - surface_distance / EVENT_DIST_OFFSET` (8 km cutoff)

### Algorithm

1. **Gating**: For each TRUE node, find PRED candidates within the time window (and spatial cutoff for events)
2. **Edge creation**: Compute similarity weights for all feasible pairs
3. **Maximum weight matching**: NetworkX `max_weight_matching` finds optimal 1-to-1 assignment
4. **Confusion matrix**: Traverse matched/unmatched nodes to compute TP, FN, FP

### Outputs

- **Confusion matrix**: DataFrame with categories (P-wave/S-wave/None for picks; Event/None for events)
- **TP set**: Matched pairs with rich metadata (index, timestamp, phase, station, location, magnitude, errors)
- **FN list**: Unmatched TRUE entries (missed detections)
- **FP set**: Unmatched PRED entries (false alarms)

### Example

```python
from ogsutils import OGSBPGraphEvents

# TRUE and PRED are DataFrames with timestamp, latitude, longitude, depth columns
graph = OGSBPGraphEvents(TRUE_df, PRED_df)
# The graph exposes one-to-one matches; catalog-level review creates the
# confusion matrix and MH/MS/PS(/SM/SP) partitions.
matched_pairs = graph.matched_pairs_array()
```

---

## Clustering Framework

### Overview

The clustering framework (`ogsclustering.py`) provides a uniform interface for 14 registered algorithms with:
- Integrated 2D and 3D visualization
- Noise point handling (label = -1)
- Colormap-based cluster coloring
- Algorithm-specific plot features (cluster centers, exemplars, core samples, dendrograms, reachability plots)

### Basic Usage

```python
from ogsclustering import OGSHDBSCAN, OGSKMeans, SilhouetteScore

# Density-based clustering (no need to specify k)
hdbscan = OGSHDBSCAN(min_cluster_size=15)
labels = hdbscan.fit_predict(X)
hdbscan.plot(xlabel="X (km)", ylabel="Y (km)")

# Centroid-based clustering
kmeans = OGSKMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X)
kmeans.plot(show_centers=True)

# Evaluate clustering quality
score = SilhouetteScore(X, labels).compute()
```

### Algorithm Comparison with OGSClusteringZoo

```python
from ogsclustering import OGSClusteringZoo

metadata = {
    "algorithms": ["KMeans", "HDBSCAN", "DBSCAN", "Agglomerative"],
    "eval_metrics": ["SilhouetteScore", "DaviesBouldinScore"],
    "num_clusters_range": (2, 10, 1),        # for KMeans, Agglomerative
    "cluster_size_range": (10, 100, 10),      # for HDBSCAN
    "eps_range": (0.3, 1.0, 0.1),             # for DBSCAN
    "metric": "euclidean",
    "n_jobs": -1,
    "random_state": 42,
}

zoo = OGSClusteringZoo(metadata=metadata, verbose=True)
zoo.run(X)  # Optimizes, compares, and plots all algorithms
```

### Parameter Ranges

Each range is specified as `(start, stop, step)` and converted to a list via `np.arange`:

| Range Key | Used By | Parameter |
|-----------|---------|-----------|
| `num_clusters_range` | KMeans, MiniBatchKMeans, BisectingKMeans, Agglomerative, Spectral, Birch | `n_clusters` |
| `cluster_size_range` | HDBSCAN | `min_cluster_size` |
| `eps_range` | DBSCAN | `eps` |
| `damping_range` | AffinityPropagation | `damping` |
| `bandwidth_range` | MeanShift | `bandwidth` |
| `min_samples_range` | OPTICS, DBSCAN, HDBSCAN | `min_samples` |
| `Z_range` | AdvancedDensityPeaks | `Z` |

### Seismic Applications

- **HDBSCAN/DBSCAN**: Earthquake sequence identification without knowing cluster count. Handles noise (isolated events) naturally. No assumption of cluster shape (can find elongated fault structures).
- **K-Means**: Well-separated clusters with known count. Assumes spherical clusters of similar size.
- **MiniBatch K-Means / BIRCH**: Large catalogs (>50k events) where full algorithms are too slow.
- **Agglomerative**: Exploring hierarchical relationships between sequences. Dendrogram reveals sub-sequences within larger swarms.
- **OPTICS**: Multi-scale clustering, identifying nested sequences within larger swarms.
- **Spectral**: Non-convex clusters following complex spatial patterns (e.g., along curved fault traces).
