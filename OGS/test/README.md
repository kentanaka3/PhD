# OGS Test Suite

Software checks for catalog readers, normalization, comparison, clustering,
data access, downloads, training, imports, and travel-time/location helpers.
These tests are not scientific validation of the complete ML pipeline.

## Prerequisites and fixtures

Activate the configured environment using the
[toolkit guide](../README.md#runtime-and-entrypoints) before invoking Python
or Make targets that run Python. Dependencies are needed for the selected
tests; install or restore them only through an approved setup workflow.

The current parser tests use checked-in files under [`data/`](data/) and
temporary outputs supplied by [`conftest.py`](conftest.py).
They do not require the old `OGS/data/manual/onlyEQ-2024.*` files.
The reader factory supports line-block slicing and explicit date windows.
Parser expectations are expressed in the test assertions rather than a
universal precomputed Parquet oracle.

## Inventory

| Test file | Make target | Scope |
|---|---|---|
| [testogsconstants.py](testogsconstants.py) | `constants` | Constants, bounds, CPU settings, and discovery helpers |
| [testogsdat.py](testogsdat.py) | `.dat` | DAT reader, timing, filtering, and schemas |
| [testogshpl.py](testogshpl.py) | `.hpl` | HPL event/pick blocks, IDs, timing, and notes |
| [testogspun.py](testogspun.py) | `.pun` | PUN event fields and normalization |
| [testogstxt.py](testogstxt.py) | `.txt` | TXT solutions, exclusions, magnitudes, and date bounds |
| [testogsdatafile.py](testogsdatafile.py) | `datafile` | Shared coordinate, group, and DataFrame normalization |
| [testogsparser.py](testogsparser.py) | `parser` | Real-fixture dispatch, merge identity, deduplication, conflicts, and CLI selection |
| [testogscatalog.py](testogscatalog.py) | `catalog` | Lazy reads, matching weights/eligibility, partitions, and rates |
| [testogsdata.py](testogsdata.py) | `data` | Temporary day-shard paths and mocked data-source registration |
| [testogsdownloader.py](testogsdownloader.py) | `downloader` | Mocked service boundaries, restrictions, backend/CLI selection, and failures |
| [testogsclustering.py](testogsclustering.py) | `clustering` | Synthetic clustering, metrics, optimization, and properties |
| [testogstrainer.py](testogstrainer.py) | `trainer` | CLI, losses, conditioning, augmentation, checkpoints, and loops |
| [test_package_imports.py](test_package_imports.py) | None | Isolated imports, lazy exports, dependency failures, plots, and CLI help |
| [testogsgrid2time.py](testogsgrid2time.py) | None | Synthetic travel-time grids, model readers, and serialization |
| [testogsnonlinloc.py](testogsnonlinloc.py) | None | Native binaries, process settings, staging, and sample relocation |

## Focused commands

From the repository root, after environment activation:

```bash
make -C OGS/test parser-tests
make -C OGS/test catalog-support-tests
python -m pytest -q -rs OGS/test/test_package_imports.py
```

[`Makefile`](Makefile) defines:

- `parser-tests`: `.dat .hpl .pun .txt datafile parser`;
- `catalog-support-tests`: `constants catalog data downloader`;
- `all`: `.dat .hpl .pun .txt parser datafile constants clustering downloader catalog data trainer`.

The parser and catalog-support recipes use `PYTEST`, defaulting to
`python -m pytest -q`. The clustering and trainer recipes execute their
unittest scripts directly. `all` does not include import compatibility,
travel-time-grid, or NonLinLoc tests. A `.PHONY` name alone is not a runnable
recipe; use the targets above.

For the smallest affected slice, combine relevant files in one pytest run:

```bash
python -m pytest -q OGS/test/testogsdatafile.py OGS/test/testogsparser.py
```

## Import compatibility

The import suite launches isolated subprocesses without inherited
`PYTHONPATH` or pytest's source-path injection. It checks `OGS.src.*`,
deployed `src.*`, and the lazy public API in
[`OGS/__init__.py`](../__init__.py). Synthetic plots use temporary outputs;
CLI checks request help. Missing `dask_mpi` can produce explicit MPI-import
skips, while dependency-error propagation is checked separately.

## Integration and side effects

Do not assume every test is environment-independent. The NonLinLoc suite
uses fixed external binary paths and can execute native location/grid
programs; its sample relocation case skips if the sample is absent, but
binary checks still require the installation. Run it only in an approved
environment. Scientific packages remain prerequisites for synthetic tests.

Tests may create caches, temporary tables, figures, and checkpoints.
Review selected setup and service mocks before executing a new test slice.
No documented test command submits SLURM jobs; a passing test does not
prove catalog accuracy, backend compatibility for every algorithm, or a
successful production run.

## Benchmark provenance

For new benchmark fixtures, record the source release, stable identifier
or checksum, schema, parser command, and review decision. Keep raw annual
bulletins and restricted inputs in approved external storage. Add only
reviewed compact fixtures, and keep analyst labels distinct from derived
pipeline output.
