# OGS Test Suite (`OGS/test/`)

## Overview

The `OGS/test/` directory contains unit and integration tests for catalog parsers, constants, data structures, clustering modules, and waveform downloading.

## Purpose and prerequisites

Use these tests to check parser behavior and selected catalog, clustering,
data-access, and downloader interfaces after a documented change. They are
not a substitute for scientific validation of a complete ML pipeline.

The parser tests require the expected sample inputs under
`OGS/data/manual/`; the checked-in Parquet files under `OGS/test/OGSCatalog/`
are comparison outputs/fixtures, not a replacement for every manual input.
Install the dependencies required by the selected test before running it.

## Test Inventory

| Test File | Target Module | Makefile Target | Scope / Coverage |
|---|---|---|---|
| [`testogsconstants.py`](testogsconstants.py) | `ogsconstants.py` | `make constants` | Geographic coordinates, date formats, zone codes, FDSN clients, and tolerance constants. |
| [`testogsdat.py`](testogsdat.py) | `ogsdat.py` | `make .dat` | Parsing of legacy `.dat` fixed-width phase arrivals. |
| [`testogshpl.py`](testogshpl.py) | `ogshpl.py` | `make .hpl` | Parsing of `.hpl` hypocenter files with pick headers. |
| [`testogspun.py`](testogspun.py) | `ogspun.py` | `make .pun` | Parsing of `.pun` punch card format event records. |
| [`testogstxt.py`](testogstxt.py) | `ogstxt.py` | `make .txt` | Parsing of `.txt` event summary catalogs. |
| [`testogsparser.py`](testogsparser.py) | `ogsparser.py` | `make parser` | Multi-format aggregator dispatch test. |
| [`testogscatalog.py`](testogscatalog.py) | `ogscatalog.py` | *(manual / pytest)* | Catalog DataFrame operations, filtering, BGMA matching integration. |
| [`testogsclustering.py`](testogsclustering.py) | `ogsclustering.py` | *(manual / pytest)* | 14 clustering algorithm wrappers and evaluation metrics. |
| [`testogsdata.py`](testogsdata.py) | `ogsdata.py` | *(manual / pytest)* | Pyrocko Squirrel data source access and sharding. |
| [`testogsdownloader.py`](testogsdownloader.py) | `ogsdownloader.py` | *(manual / pytest)* | FDSN client parameter validation and domain bounds. |

## Running Tests

### Self-Contained Unit Tests (No External Fixtures Required):
These tests use synthetic in-memory data, mocks, and constant definitions:
```bash
cd OGS/test
python testogsconstants.py    # Validates constant definitions and tolerances
python testogscatalog.py     # Tests event prefiltering and candidate masking with synthetic DataFrames
python testogsclustering.py  # 1,968 lines of algorithmic and property tests on synthetic manifolds
```

### Parser Tests with External Data Fixture Requirements:
The format parser tests (`testogsdat.py`, `testogshpl.py`, `testogspun.py`, `testogstxt.py`) expect sample files in `OGS/data/manual/onlyEQ-2024.*` and compare against `OGS/test/OGSCatalog/onlyEQ-2024.*.parquet`. If `OGS/data/manual/` is not populated, these tests fail with `FileNotFoundError`.

```bash
cd OGS/test
# Requires OGS/data/manual/ fixture files:
make all          # Runs: .dat, .hpl, .pun, .txt, parser, constants
make .dat         # Run .dat parser tests only
make constants    # Run constants tests only (passes without fixtures)
```

## Outputs and safety

Tests may create Python caches, read local fixtures, and exercise code paths
that access configured data services. Review test setup before running a
network-dependent case, and do not treat a passing unit test as proof of
catalog accuracy. The commands above do not submit SLURM jobs.

## Annual benchmark fixture workflow

Keep raw annual bulletins and manual source files outside the repository's
checked-in fixture directory. For a new benchmark year, record the source,
release date, checksum or other stable identifier, schema, and parser command;
run the relevant parser tests; review the resulting Parquet tables; then add
only the approved compact fixtures under `OGS/test/OGSCatalog/` and update the
catalog documentation. This preserves reproducibility without treating a
fixture as an independently validated scientific ground truth.

## Note on Test Makefile Coverage
The `OGS/test/Makefile` target `all` currently runs targets `.dat .hpl .pun .txt parser constants`. The richer, self-contained test suites ([`testogscatalog.py`](testogscatalog.py) and [`testogsclustering.py`](testogsclustering.py)) are not yet wired into the test Makefile and should be executed directly via Python or pytest.
