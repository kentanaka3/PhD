# Pending local deployment checklist

Local deployment is **not complete**. This checklist is for a future agent
or operator with an approved Ubuntu workstation, MacBook, and (if needed) a
configured remote bridge. Nothing below authorizes deployment, network
transfers, package installation, or destructive cleanup.

## 1. Baseline and dry-run gate

- [ ] Record the checkout revision and preserve the pre-deployment diff.
- [ ] Run the repository handler's navigation report before changes with
      `--root OGS/utils/.local --scripts`.
- [ ] Run `make -C OGS/utils/.local help` and inspect all intended target
      expansions with `make -n`; do not substitute a real target for a dry run.
- [ ] Capture the exact commands, paths, date bounds, and output destinations
      for the deployment record.

## 2. Prerequisites and implementation blockers

- [ ] Confirm Bash, GNU Make, Git, Python 3.12, and a writable external
      `WORK_PATH` on Ubuntu; verify the local Python executable in
      `PYTHON_BIN`.
- [ ] Confirm macOS version, GNU Make, SSH, and a compatible `rsync` 3.x on
      the MacBook. Record whether Homebrew tools are used.
- [ ] Install or verify `ml_catalog_run`, Hydra configuration, model caches,
      and optional ML dependencies only through the approved environment
      process.
- [ ] Verify that all local Makefile command recipes continue to use
      `LAUNCHME.sh` or an explicitly reviewed, argv-safe local command path;
      no SLURM launcher is appropriate here (`Makefile:96-148`).
- [ ] Verify the initializer links the repository's actual `OGS/config`,
      `OGS/data`, and `OGS/src` paths and refuses conflicting existing paths
      (`init.sh:130-161`).
- [ ] Verify the selected workstation Python through `PYTHON_BIN`
      (`Makefile:25`); no Leonardo Conda path should be inherited accidentally.
- [ ] Keep GNU `date -d` usage limited to Ubuntu (`Makefile:35-46`);
      introduce a date abstraction before attempting to share these Make
      targets with macOS.
- [ ] Confirm that legacy `compress`, `decompress`, and `retrieve` targets
      remain intentionally omitted until reviewed local implementations exist.

## 3. Safety and approval gate

- [ ] Keep raw waveforms, catalogs, environments, and logs outside the Git
      checkout unless a task explicitly requires fixtures.
- [ ] Review every path for accidental `/`, home-directory, shared-storage,
      or unrelated-repository targets; reject empty or inherited defaults.
- [ ] Bound downloads and processing to a small approved date range and
      disposable output fixture for first execution.
- [ ] Do not run `.local/Makefile` `init`, `download`, `station`, `index`, or
      parser/ML targets as validation; use `DRY_RUN=1` first. There is no
      local `clean` target, by design.
- [ ] Keep credentials, private keys, tokens, and host-specific endpoints
      in the operator's SSH configuration or approved secret store, never in
      this tree.
- [ ] Confirm that no step edits `OGS/utils/Leonardo/` and that no deleted
      legacy LLM helper script is recreated.

## 4. Platform-specific testing

### Ubuntu

- [ ] Test Makefile help, variable overrides, and GNU date expansion.
- [ ] Run repository-supported Python `--help` or focused tests against
      `OGS/src/ogsdownloader.py`, `OGS/src/ogsstation.py`, and
      `OGS/src/ogsparser.py` without network writes, using `PYTHON_BIN`.
- [ ] Exercise the eventual local runner in dry-run mode, then perform one
      approved fixture operation and verify idempotence.

### MacBook

- [ ] Run `doctor`; record `uname`, macOS, Python, Make, and `rsync` versions.
- [ ] Verify Python imports and repository-relative source paths.
- [ ] Test the date abstraction on a leap-year boundary and a bounded
      cross-year rejection case.
- [ ] Dry-run a small bridge fixture, inspect itemized changes, interrupt and
      resume safely, then verify remote artifact integrity.
- [ ] Perform one small `--execute` transfer only after the safety gate and
      document its remote rollback procedure.

## 5. Rollback and recovery

- [ ] Save the pre-deployment Git diff, Makefile variable set, environment
      specification, and bridge command as an approved record.
- [ ] Define rollback for generated links/files in `WORK_PATH` and for a
      partially transferred remote fixture before execution.
- [ ] Verify that rollback does not require `make clean`, broad `rm -rf`, or
      remote deletion; remove only explicitly identified test artifacts.
- [ ] If a test changes a tracked file unexpectedly, stop, preserve the diff,
      and restore only with a reviewed, path-specific operation.

## 6. Handler and final acceptance

- [ ] Run the repository handler's navigation report with
      `--root OGS/utils/.local --scripts` after integration and archive it.
- [ ] Run the repository handler's single-file `validate --file` mode for
      `dummy.sh`, `init.sh`, and `LAUNCHME.sh`; all function line counts and
      Bash syntax must pass.
- [ ] Ensure every future local Bash function uses `# function_name`, a
      separator, description/side effects, then `function_name() { # N`.
- [ ] Run the handler's repository-wide validator and record any baseline
      failures separately from local changes; do not weaken the validator to
      obtain a green result.
- [ ] Review changed-file scope: `.local` and directly related documentation
      only; no Leonardo script changes and no unrelated project edits.
- [ ] Mark deployment complete only after all blockers, platform tests,
      rollback checks, and handler validations are evidenced in the approved
      deployment record.
