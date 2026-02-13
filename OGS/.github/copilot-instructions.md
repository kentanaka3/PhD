# OGS Seismic Catalog Pipeline

This codebase is a **seismic catalog processing pipeline** that integrates machine learning pickers (SeisBench), event associators (GaMMA, PyOcto), and manual OGS catalogs for earthquake detection and analysis in the northeastern Italy region.

## Architecture Overview

**Three-layer pipeline structure:**
1. **Data ingestion**: Manual OGS files (.dat/.hpl/.pun/.txt) containing human-reviewed picks and events
2. **ML pipeline**: YAML-configured modules for detection (SeisBench), association (GaMMA/PyOcto), location (NonLinLoc), magnitude estimation
3. **Comparison/validation**: Bipartite graph matching between manual and ML catalogs to compute confusion matrices and metrics

**Key architectural decisions:**
- YAML-driven module composition using Hydra (see `conf/config.yaml` and `conf/group_modules/`)
- Distributed execution via `dask-mpi` on SLURM clusters (see `src/ogsbuilderMPI.py`)
- Catalog-agnostic comparison via normalized DataFrames with date-based grouping

## Critical File Patterns

### Constants and column names (`src/ogsconstants.py`)
**All DataFrame operations depend on exact column name constants** - never hardcode strings like `"time"` or `"station"`. Always use:
- `TIME_STR`, `PHASE_STR`, `STATION_STR` for picks
- `LATITUDE_STR`, `LONGITUDE_STR`, `DEPTH_STR`, `MAGNITUDE_L_STR` for events
- `PICK_TIME_OFFSET`, `EVENT_TIME_OFFSET`, `EVENT_DIST_OFFSET` for matching thresholds

### Manual catalog parsers (`src/ogs*.py`)
Each file format (`.hpl`, `.dat`, `.pun`, `.txt`) has a dedicated parser inheriting from `OGSDataFile`:
- Uses regex extractors defined in `*_EXTRACTOR_LIST` class variables
- Outputs normalized DataFrames with date-keyed dictionaries: `self.picks[date]` and `self.events[date]`
- Run via `ogsparser.py` CLI: `PYTHONPATH=./src python src/ogsparser.py -f data/manual/RSFVG-2024.hpl -D 240101 240101`

### YAML module wiring (`conf/group_modules/`)
Pipeline modules are **instantiated from YAML**, not Python:
```yaml
# Example: conf/group_modules/picker/ogspicker.yaml
_target_: ml_catalog.modules.SeisBenchPicker
model:
  _target_: seisbench.models.PhaseNet.from_pretrained
  name: instance
amplitude_extractor:
  _target_: OGS.src.ogspicker.OGSAmplitudeExtractor
```
To add behavior, **edit YAML files or add new YAML modules**, not just Python code.

## Developer Workflows

### Running tests
```bash
# From repo root - PYTHONPATH is REQUIRED
PYTHONPATH=./src python -m pytest test/test_ogsparser.py -v
```
Tests use `unittest.mock.patch("sys.argv", ...)` to simulate CLI invocations.

### Running the distributed builder
```bash
# Expects SLURM environment variables (SLURM_MEM_PER_CPU, SLURM_NTASKS)
# Typically submitted via SLURM script that sets these
python src/ogsbuilderMPI.py
```
Configuration loaded via Hydra from `conf/config.yaml` (defaults to ktanaka cluster + SeisBenchPicker modules).

**Implementation notes for `src/ogsbuilderMPI.py`:**
- Extends `ml_catalog.CatalogBuilder` - inherits Hydra-based initialization
- Override `run()` method to customize execution (MPI setup, Dask client configuration)
- The base class handles instantiation via Hydra `@hydra.main` decorator (in `ml_catalog`)
- Builder receives pre-configured modules from YAML: `self.group_modules`, `self.merge_module`, `self.joint_modules`
- Key execution phases:
  1. **MPI init**: `dask_mpi.initialize()` sets up distributed workers
  2. **Status setup**: Register all module parameters, call `module.setup(status)` for each
  3. **Group processing**: For each data group (typically daily), run picker → QC → associator → locator → magnitude
  4. **Merge**: Combine group results into final catalog
  5. **Output**: Write to Parquet/CSV as configured in `conf/builder/default.yaml`

### Parsing manual catalogs
```bash
# Single file
PYTHONPATH=./src python src/ogsparser.py -f data/manual/RSFVG-2024.dat -D 240320 240620

# Directory of files
PYTHONPATH=./src python src/ogsparser.py -d data/manual -x .hpl .dat -D 240101 240630 --merge
```
Output: Normalized Parquet files under output path (default: `catalogs/buildCatalog/<ext>/`)

## Project-Specific Conventions

### DataFrame normalization pattern
All loaders (manual and ML) must produce DataFrames with:
- A `groups` column containing date (YYYY-MM-DD) for daily bucketing
- Standard column names from `ogsconstants.py`
- Timestamps as `obspy.UTCDateTime` objects (converted during matching)

### Bipartite graph matching for catalog comparison
Located in `src/ogsconstants.py` (classes: `OGSBPGraphPicks`, `OGSBPGraphEvents`):

**Algorithm overview:**
1. **Build graph**: Create edges between TRUE (base catalog) and PRED (target catalog) nodes
2. **Gate by time/space**: Only create edges for feasible pairs (time window + spatial constraint)
3. **Weight edges**: Score each pair using distance functions that combine multiple factors
4. **Match greedily**: Use NetworkX `max_weight_matching(maxcardinality=False)` - finds high-quality matches but not globally optimal

**Gating parameters** (defined in `ogsconstants.py`):
- **Picks**: `PICK_TIME_OFFSET` = 0.5s (temporal only, must match station)
- **Events**: `EVENT_TIME_OFFSET` = 1.5s AND `EVENT_DIST_OFFSET` = 3 km (both required)

**Weight functions** (higher = better match):
- **Picks** (`dist_pick`): `(97*time_similarity + 2*phase_match + 1*probability_ratio) / 100`
  - Phase agreement (P vs P, S vs S) is critical but only 2% of weight
  - Time proximity dominates (97%)
  - Model confidence adds minor boost (1%)
- **Events** (`dist_event`): `(99*time_similarity + 1*space_similarity) / 100`
  - Time similarity: `1 - (time_diff / EVENT_TIME_OFFSET)`
  - Space similarity: `1 - (surface_distance / EVENT_DIST_OFFSET)`

**Confusion matrix construction:**
After matching, iterate through TRUE nodes:
- **TP (True Positive)**: TRUE matched to PRED with same phase/category → store as tuple `(BASE_value, TARGET_value)` for numeric fields
- **FN (False Negative)**: TRUE unmatched → append BASE row
- **FP (False Positive)**: PRED unmatched AND inside OGS_POLY_REGION → append TARGET row

**Example usage pattern:**
```python
# Picks comparison (classification task)
bpg = OGSBPGraphPicks(base_df, target_df)  # Auto-runs matching
for i, j in bpg.E:  # E is the matched edge set (base_idx, target_idx+offset)
    base_idx, target_idx = sorted((i, j))
    target_idx -= len(base_df)  # Remove offset
    # Access matched pairs for TP analysis

# Events comparison (detection task)
bpg = OGSBPGraphEvents(base_df, target_df)
# Same pattern, but spatial gating applied
```

**Critical limitation**: The greedy matcher can fail in dense aftershock sequences where multiple events occur within gating windows. Consider Hungarian algorithm if global optimality is needed, but beware O(n³) complexity.

### OGS region filtering
Events are filtered using `matplotlib.path.Path.contains_point()` with `OGS_POLY_REGION` polygon (northeastern Italy). Applied to False Positive detection but **not** to base catalog loading.

## Integration Points

### External ML dependencies
- **SeisBench** (`seisbench.models.PhaseNet`): Phase picking
- **GaMMA** (`ml_catalog.modules.GammaAssociator`): Event association via Gaussian mixture
- **PyOcto**: Alternative associator (check `conf/group_modules/associator/ogspyocto.yaml`)
- **ml_catalog**: Base framework providing `CatalogBuilder`, `AbstractModule`, `Status` classes

### SLURM cluster integration
`ogsbuilderMPI.py` reads environment variables:
- `SLURM_MEM_PER_CPU`: Memory limit per worker
- `SLURM_NTASKS`: Total tasks (workers = ntasks - 2 for scheduler/client overhead)

Cluster configs in `conf/cluster/*.yaml` define Dask cluster parameters (local/Leonardo/Ada).

## Performance Optimizations

### Dask-based distributed computing
- **Lazy evaluation**: Dask graph constructed via `@dask.delayed` decorators (see `real.py`)
- **Graph optimization**: Custom `_optimize_dask_graph` in builder for task fusion
- **Performance monitoring**: HTML report generated at `output_path/dask-report.html`
- **Adaptive scaling**: `self.cluster.adapt(minimum=1, maximum=self.adaptive_maximum)`

### DataFrame operations - optimization opportunities
**Implemented optimizations**:
- **Bipartite matching** (`ogsconstants.py`):
  - ✅ **Picks**: Pre-filter by station using `groupby()` dictionary for O(1) lookup (lines 535-556)
  - ✅ **Events**: Vectorized time filtering with NumPy before nested loops (lines 563-580)
  - **Impact**: Reduces O(n×m) to O(n×k) where k is avg candidates per node
  
- **Event-pick aggregation** (`ogsparser.py`):
  - ✅ Replaced `iterrows()` with vectorized `groupby().size().unstack()` (lines 163-195)
  - ✅ Use `map()` + `fillna()` instead of `.at[]` assignment in loops
  - **Impact**: ~100x speedup for large catalogs (10k+ events)

- **UTCDateTime conversions** (`ogsconstants.py`):
  - ✅ List comprehension instead of `apply(lambda)` (lines 520-526, 563-570)
  - **Impact**: 2-3x faster for timestamp conversions

**Remaining opportunities**:
- Convert more `apply()` with lambdas to vectorized string operations
- Cache travel time tables more aggressively in REAL associator
- Parallelize per-day processing in manual catalog parsers

**Best practices for new code**:
1. **Use vectorized operations**: `df['col'].str.method()` instead of `df['col'].apply(lambda x: x.method())`
2. **Avoid `iterrows()`**: Use `itertuples()` if iteration is unavoidable (5-10x faster)
3. **Pre-filter before matching**: Gate by time/space BEFORE building full graph
4. **Use `inplace=True`** sparingly: Often slower than reassignment in modern pandas

### Caching strategies
- **Path-based caching**: `CacheHelper` in `ml_catalog` for intermediate results
- **Travel time tables**: Pre-computed and cached in REAL associator (`real.py:170-175`)
- **Parquet over CSV**: Default output format for faster I/O (binary columnar)

### Memory management
- **Daily grouping**: Process data in date-keyed chunks to limit memory footprint
- **Reset index**: Use `reset_index(drop=True)` after filtering to avoid index bloat
- **Explicit cleanup**: `client.shutdown()` at end of MPI builder to release resources

## Common Pitfalls

1. **Forgetting PYTHONPATH**: All scripts expect `PYTHONPATH=./src` when run standalone
2. **Hardcoding column names**: Use constants from `ogsconstants.py` to avoid "KeyError: 'time'" bugs
3. **Assuming global optimization**: The bipartite matcher is greedy; dense overlaps may yield suboptimal pairings
4. **Mixing date formats**: Manual parsers use `YYMMDD_FMT` ("%y%m%d"); ensure consistency when adding date arguments
5. **Missing YAML updates**: Adding Python modules without corresponding YAML entries in `group_modules/` won't activate them
6. **Inefficient DataFrame iteration**: Avoid `iterrows()` in hot paths - use vectorized operations or `itertuples()` instead

## Quick Reference: Key Files

- `src/ogsconstants.py` — Column names, thresholds, matching logic, catalog classes
- `src/ogsparser.py` — CLI and DataCatalog for merging manual files
- `src/ogsbuilderMPI.py` — Distributed pipeline executor
- `conf/config.yaml` — Hydra entry point (chains all defaults)
- `conf/group_modules/default.yaml` — Module pipeline definition
- `test/test_ogsparser.py` — Example of mocked CLI tests

---

**When adding features:** Start by identifying which layer (ingestion/ML/comparison) is affected, then locate the corresponding module YAML or parser class. Update both Python implementation and YAML wiring if needed.
