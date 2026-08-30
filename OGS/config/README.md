# OGS Hydra Configuration (`OGS/config/`)

## Overview

The `OGS/config/` directory contains the Hydra/YAML configuration hierarchy
for the `ml_catalog` seismic data processing pipeline. Hydra composes these
YAML fragments at runtime based on the top-level configuration in
[`config.yaml`](config.yaml) and command-line overrides supplied by the
Makefile or user.

## Purpose and configuration boundary

These files declare stage parameters and environment-specific defaults; they
do not install dependencies, create workspaces, download data, or submit
jobs. Runtime behavior also depends on the selected `ml_catalog` version,
external data services, and values supplied by the launcher.

## Inputs and outputs

- **Inputs:** `config.yaml`, selected defaults groups, Hydra command-line
  overrides, and environment paths resolved by the launcher.
- **Outputs:** a composed runtime configuration consumed by `ml_catalog_run`
  (invoked via the `SBC_RUN_BIN` Makefile variable);
  downstream stages write catalogs, logs, and figures under their configured
  output path.
- **Assumptions:** group names, `_target_` values, paths, and model names must
  match the installed runtime packages and available external resources.
  YAML presence alone does not prove that a dependency or model is usable.

## Configuration Structure

```text
OGS/config/
├── config.yaml                     # Root defaults configuration
├── builder/
│   └── default.yaml                # Catalog builder parameters (Dask / MPI graph execution)
├── cluster/                        # Cluster and machine execution environments
│   ├── Ada.yaml                    # ADA Cloud cluster profile
│   ├── Leonardo.yaml               # CINECA Leonardo SLURM cluster profile
│   ├── LeonardoNode.yaml           # Leonardo single-node profile (default in config.yaml)
│   ├── Udine.yaml                  # University of Udine compute node profile
│   ├── ktanaka.yaml                # Workstation profile
│   └── local.yaml                  # Local workstation testing profile
├── data/
│   └── ogsDB.yaml                  # Data source definition: Squirrel database and waveform archive paths
├── group_modules/                  # Pipeline stage definitions executed on waveform groups
│   ├── Associator.yaml             # Group associator group defaults
│   ├── Locator.yaml                # Group locator defaults
│   ├── OGS.yaml                    # Full OGS standard group module chain
│   ├── Picker.yaml                 # Group picker defaults
│   ├── default.yaml                # Minimal default group chain
│   ├── associator/                 # Phase association algorithms
│   │   ├── ogsgamma.yaml           # GaMMA (Gaussian Mixture Model Association)
│   │   ├── ogspyocto.yaml          # PyOcto fast phase associator
│   │   ├── ogsreal.yaml            # REAL (Rapid Earthquake Association and Location)
│   │   └── real.yaml               # Alternative REAL configuration
│   ├── magnitude/
│   │   └── ogslocalmagnitude.yaml  # OGS-calibrated local magnitude ($M_L$) calculation
│   ├── nonlinloc/                  # Earthquake location (NonLinLoc)
│   │   ├── ogsnonlinloc1d.yaml     # NonLinLoc 1D velocity model location
│   │   └── ogsnonlinloc3d.yaml     # NonLinLoc 3D velocity model location
│   ├── picker/                     # Machine learning phase pickers
│   │   ├── ogsphninstance.yaml     # PhaseNet pretrained on INSTANCE dataset
│   │   └── ogsseisbenchpicker.yaml # SeisBench picker wrapper (PhaseNet, EQTransformer)
│   ├── qcevents/
│   │   └── ogsqcevents.yaml        # Event-level quality control
│   └── qcpicks/
│       ├── default.yaml            # Default pick quality control
│       └── ogsqcpicks.yaml         # OGS pick count and bounding box filtering
├── joint_modules/                  # Modules operating jointly across grouped events
│   ├── default.yaml                # Default joint configuration
│   ├── hypodd/
│   │   └── ogshypodd.yaml          # HypoDD double-difference relocation configuration
│   └── none.yaml                   # Null joint module placeholder (default)
└── merge_module/
    └── default.yaml                # Event and pick merge logic across chunks
```

## How Hydra Composes Stages

When `ml_catalog_run` executes (invoked via `LAUNCHME.sh` or direct CLI),
Hydra loads `config.yaml` and selects the following checked-in defaults:

- data source: `data/ogsDB.yaml`;
- group chain: `group_modules/OGS.yaml`;
- merge module: `merge_module/default.yaml`;
- joint module: `joint_modules/none.yaml`;
- builder: `builder/default.yaml`;
- execution profile: `cluster/LeonardoNode.yaml`.

The root file's default `output_path` is `catalog/OGS`; launcher overrides can
replace it with a project- and stage-specific path.

```bash
# Example override invoked by Makefile:
ml_catalog_run output_path="catalogs/Leonardo_Dev/PhaseNet/INSTANCE/0.1/2024" \
  group_modules/picker=ogsseisbenchpicker.yaml \
  group_modules.picker.model._target_="seisbench.models.PhaseNet.from_pretrained" \
  group_modules.picker.model.name=instance \
  group_modules.picker.classify_args.P_threshold=0.1 \
  group_modules.picker.classify_args.S_threshold=0.1 \
  data.starttime=20240320 data.endtime=20240620
```

## Safe validation workflow

1. Inspect the relevant YAML and the Makefile target that supplies its
   overrides.
2. Use `make -n` or the launcher help to inspect command expansion without
   submitting a job.
3. Run a focused local test or approved pipeline command only after checking
   output paths, date ranges, credentials, and resource requirements.

The example above is an invocation shape, not a guarantee that the referenced
model, data source, or output directory exists. It may write catalog output
when executed through `ml_catalog_run`. Configuration files are inputs to the
runtime; they do not install the referenced packages, download waveform data,
create the external `WORK_PATH`, or submit a scheduler job.

## Optional HypoDD configuration

`joint_modules/hypodd/ogshypodd.yaml` defines an available HypoDD module, but
the root configuration selects `joint_modules/none.yaml` by default and the
Leonardo `PhD` sweep does not include a HypoDD stage. Selecting this file
requires a compatible `ml_catalog` module and approved runtime setup; the
configuration file alone does not compile or provide an external HypoDD
binary.
