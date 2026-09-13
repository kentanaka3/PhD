# Non-HPC local utilities

`OGS/utils/.local/` is the workstation boundary for AISeism utilities. It is
separate from [`../Leonardo/`](../Leonardo/), whose scripts load Leonardo
modules and submit SLURM jobs. The local lifecycle is:

```text
dummy.sh -> init.sh -> LAUNCHME.sh -> OGS Python utility
```

## Ubuntu utilities

| File | Role |
| --- | --- |
| `dummy.sh` | Read-only validation of commands and `OGS/{config,data,src}`. |
| `init.sh` | Idempotently creates `catalogs`, `waveform`, and `logs`, then links `config`, `data`, and `src`. |
| `LAUNCHME.sh` | Executes one argv-safe local command with optional thread variables. |
| `Makefile` | Local `check`, `init`, `base`, `download`, `station`, `index`, `parse`, and `test` targets. |
| `PENDING.md` | Future deployment and validation checklist; not a completion claim. |

The scripts use `set -euo pipefail`, restrictive `umask`, safe quoting,
explicit option validation, and documented inline function line counts. They
do not load modules, install packages, clone repositories, submit jobs, or
store credentials.

### Quick start

From the PhD checkout:

```bash
cd /leonardo_work/IscrC_AISeism/PhD
bash OGS/utils/.local/dummy.sh \
  --ogs-root "$PWD/OGS" \
  --workspace "$HOME/AISeism/WORK"
bash OGS/utils/.local/init.sh \
  --ogs-root "$PWD/OGS" \
  --workspace "$HOME/AISeism/WORK" \
  --dry-run
make -C OGS/utils/.local check \
  OGS_PATH="$PWD/OGS" \
  WORK_PATH="$HOME/AISeism/WORK"
make -C OGS/utils/.local -n test \
  OGS_PATH="$PWD/OGS" \
  WORK_PATH="$HOME/AISeism/WORK"
```

The default workspace is `$WORK_PATH` when set, otherwise
`$HOME/AISeism/WORK`. Override `OGS_PATH`, `WORK_PATH`, and `PYTHON_BIN` for a
different checkout or separately managed local Python environment.

Use `DRY_RUN=1` with side-effecting Make targets:

```bash
make -C OGS/utils/.local init DRY_RUN=1
make -C OGS/utils/.local download YEAR=2025 DRY_RUN=1
```

`init.sh` never removes or replaces an existing path. A conflicting
file or symlink causes an error. `LAUNCHME.sh` requires `--` before the
command, preserves each argument as an array element, and prints a
shell-escaped preview in dry-run mode. `dummy.sh` is read-only.
Only run `init.sh` without `--dry-run`, or run `make test` without
`-n`/`DRY_RUN=1`, after the deployment checklist and an explicit workspace
approval are complete.

The local workflow intentionally does not create or activate Conda
environments. Select and validate an approved environment independently, then
pass its interpreter as `PYTHON_BIN`. The `index` target likewise requires
`ml_catalog_run` to already be on `PATH`.

## Relationship to Leonardo

The Leonardo implementation is intentionally unchanged. Its SLURM templates,
module loading, Conda installation, and template substitution are not used by
the local scripts. Do not run Leonardo's `LAUNCHME.sh`, `init.sh`, or cluster
Makefile as a substitute for these utilities.

## Bash convention

Every Bash function under `.local` follows this form:

```bash
# function_name
# -------------
# Description and side effects.
function_name() { # N
    :
}
```

`N` counts physical lines from the declaration through the closing brace.
Validate an individual script with:

```bash
bash <repository-handler> validate \
  --file OGS/utils/.local/LAUNCHME.sh
```

Future deployment and host-side tests remain in [`PENDING.md`](PENDING.md).
