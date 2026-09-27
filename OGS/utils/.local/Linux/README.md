# Ubuntu workstation profile

This profile describes the non-HPC Ubuntu path implemented by the shared
[`../Makefile`](../Makefile). The local workflow is:

```text
dummy.sh -> init.sh -> LAUNCHME.sh -> OGS Python utility
```

The scripts do not submit jobs, load modules, install packages, clone
repositories, or modify `OGS/utils/Leonardo/`.

## Prerequisites and paths

- Bash, GNU Make, Git, GNU coreutils (`date -d` is used by
  `../Makefile:35-48`), and common tools such as `find`, `ln`, and `nproc`.
- A writable external workspace. The Makefile defaults to
  `$HOME/AISeism/WORK` (`../Makefile:20-24`).
- Python 3, selected through `PYTHON_BIN` (`../Makefile:25`), with the
  dependencies required by the chosen `OGS/src/*.py` entrypoint.
- For `index` only: an already-installed `ml_catalog_run`; this workflow does
  not create or activate Conda environments.

The default OGS root is derived from this checkout
(`../Makefile:19-20`). Override `OGS_PATH`, `WORK_PATH`, `WAVE_PATH`,
`MANUAL_PATH`, and `PYTHON_BIN` when the checkout or environment differs.

## Safe workflow

From the PhD checkout:

```bash
bash OGS/utils/.local/dummy.sh \
  --ogs-root "$PWD/OGS" \
  --workspace "$HOME/AISeism/WORK"
bash OGS/utils/.local/init.sh \
  --ogs-root "$PWD/OGS" \
  --workspace "$HOME/AISeism/WORK" \
  --dry-run
make -C OGS/utils/.local help
make -C OGS/utils/.local -n download \
  OGS_PATH="$PWD/OGS" \
  WORK_PATH="$HOME/AISeism/WORK" \
  PYTHON_BIN="$HOME/.venvs/aisseism/bin/python3"
```

`dummy.sh` is read-only. `init.sh --dry-run` previews directory
and symlink operations; without `--dry-run`, it creates only `catalogs`,
`waveform`, and `logs`, plus `config`, `data`, and `src` links, and refuses
conflicting paths. Use `DRY_RUN=1` with Make targets that may create files or
run commands:

```bash
make -C OGS/utils/.local init DRY_RUN=1
make -C OGS/utils/.local base DRY_RUN=1 YEAR=2025
make -C OGS/utils/.local download DRY_RUN=1 YEAR=2025
```

`LAUNCHME.sh` requires `--` before the command, preserves argv elements, and
sets `OMP_NUM_THREADS` and `NUMBA_NUM_THREADS` only when `--threads` is
provided. `download`, `station`, and `parse` use the repository-relative
entrypoints `OGS/src/ogsdownloader.py`, `OGS/src/ogsstation.py`, and
`OGS/src/ogsparser.py` (`../Makefile:99-158`).

## Acceptance checks

1. Confirm `dummy.sh` reports the intended OGS root, Python executable,
   and workspace.
2. Inspect Make expansion with `make -n`; check date bounds and output paths.
3. Run Python `--help` or focused existing tests without writing into `OGS/`.
4. Use a bounded, approved fixture before any real download or catalog write.
5. Record the Python environment and preserve a rollback path for generated
   workspace links/files.

The shared deployment checklist is
[`../PENDING.md`](../PENDING.md).
