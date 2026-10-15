#!/usr/bin/env python3
"""
FU.py - High-throughput parallel File Updater (FU) for CINECA Leonardo Lustre
scratch.

Refreshes file access and/or modification times across directory hierarchies
using parallel worker threads or processes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import multiprocessing as mp
import os
import time
from typing import List, Optional, Tuple


def single_fu(
    filepath: str,
    method: str = "both",
    read_bytes: int = 1,
    min_age_seconds: float = 30.0 * 86400.0,
    now: Optional[float] = None,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
  """Single file update (FU) operation.

  Returns (files_scanned=1, files_touched=0/1, byte_count).
  """
  try:
    st = os.stat(filepath, follow_symlinks=False)
    # Check if age threshold applies
    if min_age_seconds > 0.0 and now is not None:
      file_age = now - max(st.st_atime, st.st_mtime)
      if file_age < min_age_seconds:
        return 1, 0, 0

    if dry_run:
      return 1, 1, st.st_size

    if method in ("open_close", "both"):
      with open(filepath, "rb") as f:
        if read_bytes > 0:
          _ = f.read(read_bytes)

    if method in ("utime", "both"):
      os.utime(filepath, None)

    return 1, 1, st.st_size
  except (FileNotFoundError, PermissionError, IsADirectoryError, OSError):
    return 1, 0, 0


def process_directory(
    args_tuple: Tuple[str, str, int, float, float, bool]
) -> Tuple[int, int, int, int]:
  """
  Processes files directly inside a single directory (non-recursive per
  directory).

  Parameters: (dir_path, method, read_bytes, min_age_seconds, now, dry_run)
  Returns: (dir_exists, files_scanned, files_touched, total_bytes)
  """
  dir_path, method, read_bytes, min_age_seconds, now, dry_run = args_tuple
  files_scanned = 0
  files_touched = 0
  total_bytes = 0
  try:
    with os.scandir(dir_path) as it:
      for entry in it:
        if entry.is_file(follow_symlinks=False):
          scanned, touched, size = single_fu(
              entry.path,
              method=method,
              read_bytes=read_bytes,
              min_age_seconds=min_age_seconds,
              now=now,
              dry_run=dry_run,
          )
          files_scanned += scanned
          files_touched += touched
          total_bytes += size
    return 1, files_scanned, files_touched, total_bytes
  except (FileNotFoundError, NotADirectoryError, PermissionError):
    return 0, 0, 0, 0


def generate_day_dirs(
    root_dir: str,
    start_date: datetime.date,
    end_date: Optional[datetime.date] = None,
) -> List[str]:
  """Generates all YYYY/MM/DD directory path strings in memory using date arithmetic."""
  end_date = end_date if end_date is not None else start_date

  num_days = (end_date - start_date).days + 1
  return [
      os.path.join(root_dir, f"{d.year:04d}", f"{d.month:02d}", f"{d.day:02d}")
      for d in (start_date + datetime.timedelta(days=i) for i in range(num_days))
  ]


def run_threads(
    work_items: List[Tuple[str, str, int, float, float, bool]], workers: int
) -> Tuple[float, int, int, int, int]:
  t_start = time.perf_counter()
  existing_dirs = 0
  total_scanned = 0
  total_touched = 0
  total_bytes = 0
  num_workers = min(workers, max(1, len(work_items)))
  with ThreadPoolExecutor(max_workers=num_workers) as executor:
    for d_found, f_scan, f_touch, b_count in executor.map(
        process_directory, work_items
    ):
      existing_dirs += d_found
      total_scanned += f_scan
      total_touched += f_touch
      total_bytes += b_count
  elapsed = time.perf_counter() - t_start
  return elapsed, existing_dirs, total_scanned, total_touched, total_bytes


def run_processes(
    work_items: List[Tuple[str, str, int, float, float, bool]],
    workers: int,
    chunksize: int = 4,
) -> Tuple[float, int, int, int, int]:
  num_workers = min(workers, max(1, len(work_items)))
  t_start = time.perf_counter()
  existing_dirs = 0
  total_scanned = 0
  total_touched = 0
  total_bytes = 0
  with mp.Pool(processes=num_workers) as pool:
    for d_found, f_scan, f_touch, b_count in pool.imap_unordered(
        process_directory, work_items, chunksize=chunksize
    ):
      existing_dirs += d_found
      total_scanned += f_scan
      total_touched += f_touch
      total_bytes += b_count
  elapsed = time.perf_counter() - t_start
  return elapsed, existing_dirs, total_scanned, total_touched, total_bytes


def parse_args():
  parser = argparse.ArgumentParser(
      description="High-throughput parallel file refresher for CINECA Leonardo Lustre scratch."
  )
  parser.add_argument(
      "--path",
      "-p",
      dest="target_path",
      default=os.environ.get(
          "WAVEFORM_PATH", "/leonardo_work/IscrC_AISeism/WORK/waveform"
      ),
      help="Target directory path to touch (default: $TARGET_PATH or $WAVEFORM_PATH).",
  )
  parser.add_argument(
      "--method",
      "-m",
      choices=["both", "open_close", "utime"],
      default=os.environ.get("METHOD", "both").lower(),
      help="Touch method: 'both' (open-read-close + utime), 'open_close', or 'utime' (default: both).",
  )
  parser.add_argument(
      "--min-age-days",
      "-a",
      type=float,
      default=float(os.environ.get("MIN_AGE_DAYS", "30")),
      help="Minimum age in days since last access/modification to touch a file (default: 30, set 0 to touch all).",
  )
  add_threads_arguments(parser)
  parser.add_argument(
      "--mode",
      choices=["processes", "threads", "both"],
      default=os.environ.get("MODE", "processes").lower(),
      help="Concurrency model: 'processes' (default), 'threads', or 'both'.",
  )
  add_date_range_arguments(
      parser,
      default_dates=[
          datetime.datetime(2005, 1, 1),
          datetime.datetime(2025, 12, 31)
      ]
  )
  parser.add_argument(
      "--chunksize",
      "-c",
      type=int,
      default=int(os.environ.get("CHUNKSIZE", "4")),
      help="Task chunksize for worker pool (default: 4).",
  )
  parser.add_argument(
      "--read-bytes",
      type=int,
      default=int(os.environ.get("READ_BYTES", "1")),
      help="Number of bytes to read when using open_close/both (default: 1).",
  )
  parser.add_argument(
      "--dry-run",
      "-n",
      action="store_true",
      default=bool(
          os.environ.get("DRY_RUN", "0").lower() in ("1", "true", "yes")
      ),
      help="Scan and report eligible files without touching them.",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  root_dir = os.path.abspath(args.target_path)

  start_date = args.dates[0].date()
  end_date = args.dates[1].date()

  t_gen_start = time.perf_counter()
  target_dirs = generate_day_dirs(
      root_dir,
      start_date=start_date,
      end_date=end_date,
  )
  scope_desc = (
      f"{start_date.isoformat()}"
      if start_date == end_date
      else f"{start_date.isoformat()} → {end_date.isoformat()}"
  )
  gen_ms = (time.perf_counter() - t_gen_start) * 1000.0

  min_age_seconds = args.min_age_days * 86400.0
  work_items = [
      (d, args.method, args.read_bytes, min_age_seconds, time.time(), args.dry_run)
      for d in target_dirs
  ]
  print(
      f"Discovered {len(target_dirs):,} target directories ({scope_desc}) in {gen_ms:.2f} ms."
  )

  if args.mode in ("both", "processes"):
    dur_proc, dirs_proc, scanned_proc, touched_proc, bytes_proc = run_processes(
        work_items, args.threads, chunksize=args.chunksize
    )
    thr_proc = touched_proc / dur_proc if dur_proc > 0 else 0
    gb_proc = bytes_proc / (1024**3)
    skipped_proc = scanned_proc - touched_proc
    print(
        f"[Processes] Elapsed: {dur_proc:.3f}s | Dirs Found: {dirs_proc:,}/{len(target_dirs):,} | "
        f"Scanned: {scanned_proc:,} | Touched: {touched_proc:,} (Skipped < {args.min_age_days}d: {skipped_proc:,}) | "
        f"Throughput: {thr_proc:,.2f} touched/s ({gb_proc:.2f} GB)"
    )

  if args.mode in ("both", "threads"):
    dur_thrd, dirs_thrd, scanned_thrd, touched_thrd, bytes_thrd = run_threads(
        work_items, args.threads
    )
    thr_thrd = touched_thrd / dur_thrd if dur_thrd > 0 else 0
    gb_thrd = bytes_thrd / (1024**3)
    skipped_thrd = scanned_thrd - touched_thrd
    print(
        f"[Threads]   Elapsed: {dur_thrd:.3f}s | Dirs Found: {dirs_thrd:,}/{len(target_dirs):,} | "
        f"Scanned: {scanned_thrd:,} | Touched: {touched_thrd:,} (Skipped < {args.min_age_days}d: {skipped_thrd:,}) | "
        f"Throughput: {thr_thrd:,.2f} touched/s ({gb_thrd:.2f} GB)"
    )

  if args.mode == "both" and dur_proc > 0 and dur_thrd > 0:
    if thr_proc >= thr_thrd:
      ratio = thr_proc / thr_thrd if thr_thrd > 0 else 0
      print(f"Multiprocessing is {ratio:.2f}x faster than Threading.")
    else:
      ratio = thr_thrd / thr_proc if thr_proc > 0 else 0
      print(f"Threading is {ratio:.2f}x faster than Multiprocessing.")


if __name__ == "__main__":
  main()
