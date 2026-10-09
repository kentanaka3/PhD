# OGS Utilities and Cluster Infrastructure (`OGS/utils/`)

## Overview

The `OGS/utils/` directory contains local and high-performance computing (HPC)
workstation scripts, environment definitions, and job submission helpers for
running the AISeism pipeline on the HPC CINECA Leonardo supercomputer, as well
as on local Ubuntu and macOS workstations. It includes Makefiles, SLURM
templates, and Python scripts for downloading waveforms, running machine
learning models, and managing the runtime workspace.

## Purpose, inputs, and outputs

These utilities translate Makefile selections into local commands or SLURM
submissions and prepare the external runtime workspace. Inputs include Make
variables, environment variables, stage configuration, and cluster templates.
Outputs include copied/link-managed runtime files, waveform/catalog artifacts,
job submissions, and scheduler logs; exact locations are determined by
`WORK_PATH` and related variables.

The utilities assume access to the intended cluster, approved storage, and
required external repositories. They are operational helpers, not isolated
library functions.

## Leonardo HPC Subdirectory (`OGS/utils/Leonardo/`)

```text
OGS/utils/Leonardo/
├── Makefile            # Main pipeline driver: orchestrates download, picking, association, location, and evaluation
├── init.sh             # Copies configuration, links data/source, prepares dependencies, builds NonLinLoc, submits smoke jobs
├── LAUNCHME.sh         # Dynamic SLURM submission wrapper: selects serial/MPI, sets cores/GPUs, handles template replacement
├── ACTIVATEME.sh       # Environment activator: loads CUDA/NVHPC modules and activates Conda SBC_3.12
├── LEONARDO.yml        # Conda environment definition with full CUDA, PyTorch, SeisBench, PyOcto, GaMMA, and ObsPy pins
├── download.sh         # SLURM batch script for serial waveform downloading
├── dummy.sh            # SLURM batch script for test execution and initialization smoke tests
├── ktanakah.sh         # Generic SLURM template for GPU/MPI jobs
├── common.sh           # Shared logging, failures, and argument checks
├── index.sh            # Index-job SLURM template
├── station.sh          # Station-inventory SLURM template
├── parse.sh            # Catalog-parser SLURM template
├── nonlinloc.sh        # Location-job SLURM template
└── test/
    └── dummy.py        # Smoke test script executed during make init
```

## HPC Execution Workflow

```text
User / make target
       │
       ▼
Makefile (target expansion)
       │
       ▼
LAUNCHME.sh (SLURM template configuration & sbatch submission)
       │
       ▼
Compute Node Execution
   ├── ACTIVATEME.sh (module load + conda activate)
   └── $(SBC_RUN_BIN) [= ml_catalog_run] + Hydra overrides
```

## Key Scripts and Roles

1. **`Makefile`**: Exposes pipeline targets:
   - `make init`: Passes configured paths to `init.sh`, which requires the
     workspace and external repositories before preparing the environment.
   - `make download YEAR=YYYY`: Submits waveform chunk downloads.
   - `make "PhaseNet[INSTANCE,0.1]"`: Runs SeisBench picking on GPU.
   - `make "PyOcto[PhaseNet,INSTANCE,0.1]"`: Runs phase association.
   - `make "NLL1D[PyOcto,PhaseNet,INSTANCE,0.1]"`: Runs NonLinLoc event location.
   - `REAL` has a configuration file and appears in help text, but is not in
     the current `ASSOCS` registry and therefore is not a generated Make
     target.
   - `make MHPC` / `make PhD`: Submit picker combinations, registered
     associators using PhaseNet/INSTANCE inputs, and corresponding NLL1D
     jobs for hard-coded date windows. These are submission sweeps, not
     every stage combination or dependency-managed end-to-end workflows.

Generated stage recipes link upstream output folders but do not submit
upstream stages as prerequisites or pass scheduler dependencies. The
`all` target selects a locator recipe; it does not orchestrate a complete
download-to-location run. Confirm upstream completion and output contents
before submitting downstream work.

2. **`LAUNCHME.sh`**: `LAUNCHME.sh` defaults `CORE_COUNT` to 32 and selects configurations with
   `tasks * cpus-per-task = CORE_COUNT`. When `OVERRIDE_CORES` is set it bypasses
   that equality test; standalone Make recipes pass `CORES` through this
   override. GPU/task and thread allocations still depend on the template.

3. **`init.sh`**: Requires the configured `WORK_PATH`, `OGS_PATH`,
   `NLL_PATH`, and `DATASET_PATH` directories before setup. Copies `conf`
   into the workspace, extracts Leonardo utilities, links `data` and `src`,
   checks external dependencies, loads modules, conditionally installs
   Conda/creates the environment, builds NonLinLoc binaries when needed,
   and submits dummy and `SBC_RUN_BIN --help` smoke jobs. Editable
   `ml_catalog` installation occurs only when creating the environment.
   This is neither read-only validation nor a safe automatic first step.

   Dependency setup is not a reliable fresh-checkout bootstrap:
   `SBC_PATH` has a placeholder Git URL, and `DATASET_PATH` is assigned a
   Zenodo archive endpoint but processed by `git clone`. Existing nonempty
   dependency directories bypass cloning. Prepare approved dependencies
   separately and inspect this source before running initialization.

## Non-HPC workstation utilities (`OGS/utils/.local/`)

The `.local/` tree is the non-HPC boundary for Ubuntu and MacBook workstations.
Start with [`OGS/utils/.local/README.md`](.local/README.md), which records the
layout, platform profiles, exact Makefile paths, and unresolved portability
gaps. The current Makefile is an Ubuntu-oriented local runner; the MacBook
profile is documentation and bridge guidance until date handling is ported:

- `WORK_PATH` defaults to `$HOME/AISeism/WORK` and `OGS_PATH` is derived from
  the `.local` location (`.local/Makefile:19-24`).
- `PYTHON_BIN` resolves `python3` from `PATH` (`.local/Makefile:25`); override
  it with a verified workstation environment.
- `init.sh` links the actual `OGS/conf`, `OGS/data`, and `OGS/src`
  directories without replacing conflicting paths
  (`.local/init.sh:130-161`).
- Direct Python entrypoints are `OGS/src/ogsdownloader.py`,
  `OGS/src/ogsstation.py`, and `OGS/src/ogsparser.py`
  (`.local/Makefile:96-148`).
- `LAUNCHME.sh` is the argv-safe local command runner used by command targets
  (`.local/Makefile:38-40`, `96-148`); no SLURM launcher is used.
- GNU `date -d` is used during Make expansion (`.local/Makefile:35-46`), so
  macOS requires an explicit portability change rather than an undocumented
  assumption about `gdate`.

Use `make -n` and `DRY_RUN=1` for inspection. Do not run local downloads
or processing until the checklist in
[`OGS/utils/.local/PENDING.md`](.local/PENDING.md) is complete.

The `OGS/src/Makefile` (a local parser convenience target) has been removed.
To run parser tests, use `OGS/test/Makefile` instead (e.g. `make -C OGS/test parser`).
`OGS/utils/convert_tabs_to_spaces.sh` is another separate utility: it edits
every discovered Python file in place (excluding Git and Miniconda paths) by
replacing leading tabs with two spaces. It is not a read-only formatter check;
review the selected root and resulting diff before using it.

## Safe usage workflow

1. Change to `OGS/utils/Leonardo/` (or use `make -C`); recipes assume that
   directory for relative paths such as `LAUNCHME.sh`.
2. Read the relevant Makefile target and configuration values.
3. Use `make -n TARGET ...` to inspect expansion without running the target.
4. Confirm `WORK_PATH`, date ranges, resource requests, external paths, and
   output destinations.
5. Run initialization, download, or submission targets only with explicit
   approval and the required cluster access.

For example, these commands are inspection-only:

```bash
make -C OGS/utils/Leonardo -n help
make -C OGS/utils/Leonardo -n "NLL1D[PyOcto,PhaseNet,INSTANCE,0.1]"
```

Bracketed target names must be quoted so the shell does not interpret their
square brackets as filename patterns. A dry run shows Make expansion but does
not validate remote data, installed models, scheduler policy, or available
storage.

`make init`, waveform downloads, and stage targets have filesystem, network,
package-installation, or scheduler side effects. `make clean` removes runtime
links and copied utility files under `WORK_PATH` (the catalog-deletion recipe
is currently commented out); it is destructive and must not be used as a smoke
test.

`OGSCatalogBuilderMPI` currently maps a detected local MPI/SLURM rank modulo
the visible CUDA device count before initializing the Dask client. This basic
binding is deployment-dependent; the repository does not configure
`dask_cuda`, so verify the cluster allocation and worker logs before relying
on multi-GPU behavior.

## Configuration ownership

Initialization copies `conf` but symlinks `data` and `src`. Later edits to
the checkout's YAML are therefore not automatically reflected in an
existing copied workspace configuration. Record and review the configuration
actually supplied to `SBC_RUN_BIN`, not only the repository defaults.

For Hydra group composition and the distinction between the local
`OGSNonLinLoc` wrapper and the external default target, see the
[configuration guide](../conf/README.md). Package CLI invocations use
`python -m OGS.src.<module>`; the Leonardo Makefile exports the repository
root on `PYTHONPATH`.

## Known Caveats and Operational Notes

- **Leonardo Makefile Python executable:** the current Leonardo Makefile
  defines `PYTHON_BIN` and uses it for the standalone `download`, `compress`,
  `decompress`, `station`, and `parse` recipes. The separate local Makefile
  uses `PYTHON_BIN` as a workstation override; see the local section above.
- **Resource allocation:** `CORE_COUNT` defaults to 32; `OVERRIDE_CORES`
  bypasses the normal task/CPU product constraint. Inspect the selected
  template's GPU and thread directives and confirm the actual allocation
  rather than relying on a fixed-32-core description.
- **SLURM template selection:** `SLURM_TEMPLATE` chooses a user-named
  `OGS/utils/Leonardo/<USER>.sh` when present. A user without a matching
  template must provide one or override `SLURM_TEMPLATE` before invoking
  generated stage targets.
- **Template concurrency:** `LAUNCHME.sh` edits the selected template in
  place, submits it, and restores its placeholders. Do not share one template
  between concurrent submissions unless the surrounding workflow provides
  serialization.
- **Reproducibility:** `init.sh` downloads the moving `Miniconda3-latest`
  installer URL when Conda is absent and clones repositories without recording
  commits or checksums. Capture installer/repository revisions in an approved
  experiment record when reproducing an environment.
- **Batch-only activation:** `ACTIVATEME.sh` expects Leonardo's `module`
  command, an existing Conda installation/environment, and
  `SLURM_CPUS_PER_TASK`; it is not a standalone local-environment validator.
