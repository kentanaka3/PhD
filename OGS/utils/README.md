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
├── init.sh             # Workspace initializer: validates paths, verifies Git clones, installs Miniconda and dependencies
├── LAUNCHME.sh         # Dynamic SLURM submission wrapper: selects serial/MPI, sets cores/GPUs, handles template replacement
├── ACTIVATEME.sh       # Environment activator: loads CUDA/NVHPC modules and activates Conda SBC_3.12
├── LEONARDO.yml        # Conda environment definition with full CUDA, PyTorch, SeisBench, PyOcto, GaMMA, and ObsPy pins
├── download.sh         # SLURM batch script for serial waveform downloading
├── dummy.sh            # SLURM batch script for test execution and initialization smoke tests
├── ktanakah.sh         # Generic SLURM template for GPU/MPI jobs
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
   - `make MHPC` / `make PhD`: Batch submission sweeps; the checked-in
     recipes submit registered picker models and `NLL1D` jobs for their
     hard-coded date windows, not every possible stage combination.

2. **`LAUNCHME.sh`**: SLURM template engine. Ensures
   `tasks * cpus-per-task = 32` for the selected node configuration, uses
   template-and-replace to configure `#SBATCH` directives, submits with
   `sbatch`, and restores the template on exit.

3. **`init.sh`**: Validates required paths and commands, verifies the
   configured `ml_catalog_main` checkout (it has no clone URL here), clones
   `NonLinLoc` and the bulletin dataset when absent, installs Conda if
   missing, creates the `SBC_3.12` environment, links OGS source/config
   directories, and submits a smoke test job. `WORK_PATH` must already exist:
   although the script later calls `mkdir -p`, it requires that directory
   before setup begins.

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
- `init.sh` links the actual `OGS/config`, `OGS/data`, and `OGS/src`
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

## Known Caveats and Operational Notes

- **Leonardo Makefile Python executable:** the current Leonardo Makefile
  defines `PYTHON_BIN` and uses it for the standalone `download`, `compress`,
  `decompress`, `station`, and `parse` recipes. The separate local Makefile
  uses `PYTHON_BIN` as a workstation override; see the local section above.
- **Resource allocation:** `LAUNCHME.sh` strictly enforces
  `tasks * cpus-per-task = 32` per selected node configuration. The exported
  OpenMP/Numba thread count is additionally scaled by the requested node count;
  verify the resulting allocation and scheduler policy before relying on
  multi-node behavior.
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
