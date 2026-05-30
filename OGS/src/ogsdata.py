"""
=============================================================================
OGS Data Module - Day-Sharded Pyrocko Squirrel Data Access
=============================================================================

OVERVIEW:
This module provides a high-throughput data source for OGS waveform archives.
It extends the ``ml_catalog`` Squirrel data interface with filesystem-aware
index selection, day-partitioned Squirrel databases, lazy read-only database
copies, and ObsPy conversion helpers for downstream seismic ML workflows.

The module implements:

1. OPTIONAL PYROCKO / SQUIRREL INTEGRATION
    - Import-time fallback when Pyrocko is unavailable
    - Runtime validation with clear installation guidance
    - Conversion from Pyrocko traces to ObsPy streams

2. OGS FILESYSTEM DISCOVERY
    - Recognition of OGS day directories: YYYY/MM/DD
    - Station metadata detection for inventory/station sidecar files
    - Date-window filtering to avoid indexing unrelated years or days
    - Directory-signature caching based on modification timestamps

3. FAST SQUIRREL INDEX MAINTENANCE
    - Set-difference updates for newly added and deleted files
    - Restricted removal only under the requested source roots
    - Per-day worker function for parallel shard construction

4. DAY-SHARDED DATA ACCESS
    - One Squirrel environment per active day when shards exist
    - Shard layout mirrors the OGS archive: ``<env>/YYYY/MM/DD``
    - Automatic fallback to a monolithic database when shards are absent
    - Optional ``check=True`` indexing to create/update missing shards
    - Lazy Squirrel instance creation and per-day instance caching

5. ML-CATALOG DATA SOURCE API
    - groups(): discover available waveform groups
    - get_group(): yield station-prioritized ObsPy streams for one group
    - get_segments(): fetch arbitrary time windows from a segment table
    - get_stations() / get_inventory(): expose station metadata

ARCHITECTURE:

    Source paths / OGS archive roots
                    |
                    v
    _get_cached_ogs_day_dirs() -> date-filtered YYYY/MM/DD directories
                    |
                    v
    _select_squirrel_add_paths() -> daily waveform dirs + station metadata
                    |
                    v
    _fast_update_squirrel_db() -> add/remove set difference in Squirrel DB
                    |
                    v
    OGSSquirrelDataSource
        ├── day_envs: {date: env_path} for sharded databases
        ├── get_sq(day): lazy per-day Squirrel instance
        ├── groups(): day/group discovery
        ├── get_group() / get_segments(): waveform retrieval

SEISMIC APPLICATIONS:
    - Efficient indexing of large OGS waveform archives split by calendar day
    - Daily waveform-group generation for ML catalog pipelines
    - Station-prioritized trace selection across HH/BH/EH channel families
    - Read-only execution on shared filesystems by copying SQLite environments
      to local temporary directories

USAGE:
    from ogsdata import OGSSquirrelDataSource

    data = OGSSquirrelDataSource(
            env="/path/to/ogsDB",
            paths=["/path/to/waveform/archive"],
            starttime="2024-01-01",
            endtime="2024-01-31",
            check=False,
    )

    for group in data.groups():
            for stream in data.get_group(group):
                process(stream)

DEPENDENCIES:
    - numpy: Time-grid construction for group generation
    - pandas: Station metadata tables and segment input
    - obspy: UTCDateTime, Inventory, and Stream objects
    - pyrocko: Squirrel waveform database and trace representation
    - ml_catalog: DataSource API, path/time aliases, logging helpers

AUTHOR: AI2Seism Project
=============================================================================
"""

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import os                         # CPU-count discovery for parallel indexing
import copy                       # Defensive copies of cached group lists
import datetime                   # Date arithmetic for OGS archive windows
import shutil                     # Temporary Squirrel directory cleanup/copying
import tempfile                   # Isolated read-only Squirrel workspaces
from abc import ABC, abstractmethod  # Kept for compatibility with older imports
from pathlib import Path          # Filesystem path normalization and traversal
from typing import Iterable, Optional  # Type hints for public helper signatures

# =============================================================================
# THIRD-PARTY LIBRARY IMPORTS
# =============================================================================
import numpy as np                # Group time-grid generation
import obspy                      # ObsPy Stream/Inventory objects
import pandas as pd               # Station and segment DataFrame handling
from obspy import UTCDateTime     # Robust seismic timestamp conversion

# =============================================================================
# ML-CATALOG INTEGRATION
# =============================================================================
from ml_catalog.data import SquirrelDataSource as BaseSquirrelDataSource
from ml_catalog.types import DataInterface, PyrockoTime, pathlike
from ml_catalog.util import logger, normalize_pyrocko_time

# =============================================================================
# OPTIONAL PYROCKO / SQUIRREL IMPORTS
# =============================================================================
# Pyrocko is an optional runtime dependency in some development environments.
# The module can still be imported without it, but methods that need Squirrel
# call _require_squirrel_runtime() or _verify_squirrel() before doing work.

try:
    import pyrocko
    from pyrocko import obspy_compat
    from pyrocko.squirrel import Squirrel, init_environment
    from pyrocko.squirrel.error import SquirrelError
    from pyrocko.trace import NoData
except ImportError:
    pyrocko = None
    Squirrel = None
    init_environment = None
    obspy_compat = None
    NoData = None


# =============================================================================
# MODULE CONSTANTS AND CACHES
# =============================================================================
# Cache filesystem scans for OGS day directories. The cache key includes the
# archive root and the requested date window; the value stores a lightweight
# directory signature and the sorted list of discovered day directories.
_SQUIRREL_OGS_DAY_DIR_CACHE: dict[
    tuple[Path, Optional[datetime.date], Optional[datetime.date]],
    tuple[tuple[tuple[str, Optional[int]], ...], list[tuple[datetime.date, Path]]],
] = {}

# Metadata names accepted as station/inventory sidecar paths. These are added
# together with daily waveform directories so Squirrel can answer station and
# channel queries without scanning the full archive tree.
_STATION_METADATA_NAMES = {
    "inventory",
    "inventory.xml",
    "station",
    "station.xml",
    "stations",
    "stations.xml",
}
_STATION_METADATA_TOKENS = ("inventory", "station")


# =============================================================================
# PATH AND TIME NORMALIZATION HELPERS
# =============================================================================


def _deduplicate_paths(paths: Iterable[pathlike]) -> list[pathlike]:
    """
    Return paths in input order while removing duplicates after expansion.

    Parameters
    ----------
    paths : Iterable[pathlike]
        Candidate files or directories passed to Squirrel indexing.

    Returns
    -------
    list[pathlike]
        The first occurrence of each expanded path string. The original object
        is preserved so callers keep their input path type where possible.
    """
    deduplicated = []
    seen = set()
    for path in paths:
        key = str(Path(path).expanduser())
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(path)

    return deduplicated


def _squirrel_add_time_to_date(time) -> Optional[datetime.date]:
    """
    Convert a broad Squirrel/ml_catalog time value to ``datetime.date``.

    Accepts ``None``, ObsPy ``UTCDateTime``, standard datetime/date objects,
    and values accepted by ``UTCDateTime``. Returning ``None`` tells callers
    that date-window pruning is not safe and the original paths should be kept.
    """
    if time is None:
        return None
    if isinstance(time, UTCDateTime):
        return time.date
    if isinstance(time, datetime.datetime):
        return time.date()
    if isinstance(time, datetime.date):
        return time

    try:
        return UTCDateTime(time).date
    except Exception:
        return None


def _is_station_metadata_path(path: Path) -> bool:
    """
    Return True if a path looks like station or inventory metadata.

    OGS archives may place metadata as either files (``inventory.xml``) or
    directories (``stations``). Matching both exact names and stem tokens keeps
    metadata attached when date filtering trims waveform directories.
    """
    name = path.name.lower()
    stem = path.stem.lower()
    if name in _STATION_METADATA_NAMES or stem in _STATION_METADATA_NAMES:
        return True

    return any(token in stem for token in _STATION_METADATA_TOKENS)


def _station_metadata_children(root: Path) -> list[Path]:
    """
    List station/inventory children directly below an archive root.

    Only existing children are returned. This keeps Squirrel indexing calls
    compact while preserving metadata required for later station lookups.
    """
    return [
        root / name
        for name in (
            "station",
            "stations",
            "inventory",
            "inventory.xml",
            "station.xml",
            "stations.xml",
        )
        if (root / name).exists()
    ]


def _parse_ogs_day(year: str, month: str, day: str) -> Optional[datetime.date]:
    """
    Parse an OGS ``YYYY/MM/DD`` directory triplet into a date.

    Returns ``None`` for malformed names and impossible calendar dates. The
    helper is intentionally strict because its output controls date-window
    pruning during indexing.
    """
    if not (
        len(year) == 4
        and len(month) == 2
        and len(day) == 2
        and year.isdigit()
        and month.isdigit()
        and day.isdigit()
    ):
        return None

    try:
        return datetime.date(int(year), int(month), int(day))
    except ValueError:
        return None


def _stat_mtime_ns(path: Path) -> Optional[int]:
    """
    Return a path modification timestamp for cache invalidation.

    ``None`` is used when the path cannot be stat'ed, allowing the caller to
    build a signature without failing on transient filesystem errors.
    """
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _day_relative_path(day: datetime.date) -> str:
    """
    Return the ``YYYY/MM/DD`` relative path used for day-sharded Squirrel envs.
    """
    return f"{day.year:04d}/{day.month:02d}/{day.day:02d}"


def _squirrel_day_env_candidates(env_path: Path, day: datetime.date) -> tuple[Path, Path]:
    """
    Return supported filesystem layouts for a daily Squirrel environment.

    Two layouts are supported for backward compatibility with existing OGS
    workspaces:
    - ``env_parent/env_name/YYYY/MM/DD``
    - ``env_path/env_name/YYYY/MM/DD``
    """
    relative = _day_relative_path(day)
    return (
        env_path.parent / env_path.name / relative,
        env_path / env_path.name / relative,
    )


def _existing_day_env(env_path: Path, day: datetime.date) -> Optional[Path]:
    """
    Find an initialized daily Squirrel shard for one calendar day, if present.

    A candidate is considered initialized when it contains a ``.squirrel``
    directory. The first matching layout is returned.
    """
    for candidate in _squirrel_day_env_candidates(env_path, day):
        if (candidate / ".squirrel").is_dir():
            return candidate
    return None


def _target_day_env(env_path: Path, day: datetime.date) -> Path:
    """
    Choose the destination directory for creating/updating a daily shard.

    Existing initialized shards are reused in place. New shards default to the
    sibling layout ``env_parent/env_name/YYYY/MM/DD`` so they remain separate
    from a possible monolithic base environment.
    """
    existing = _existing_day_env(env_path, day)
    if existing is not None:
        return existing
    return env_path.parent / env_path.name / _day_relative_path(day)


def _is_under_resolved_roots(file_path: Path, resolved_roots: set[str]) -> bool:
    """
    Return True when a database file belongs to one requested source root.

    This protects unrelated files in the same Squirrel database from removal
    during fast set-difference updates.
    """
    if str(file_path) in resolved_roots:
        return True
    return any(str(parent) in resolved_roots for parent in file_path.parents)


def _require_squirrel_runtime():
    """
    Return Pyrocko Squirrel runtime objects or raise a helpful ImportError.

    Importing this module should be cheap even without Pyrocko installed, but
    indexing and waveform retrieval cannot proceed without Squirrel.
    """
    if Squirrel is None or init_environment is None:
        raise ImportError(
            "Missing dependency Squirrel/Pyrocko. "
            "Installation instructions at https://pyrocko.org/docs/current/install/"
        )
    return Squirrel, init_environment


# =============================================================================
# FAST SQUIRREL INDEX MAINTENANCE
# =============================================================================


def _fast_update_squirrel_db(sq, squirrel_add_paths: list[pathlike]) -> None:
    """
    Synchronize a Squirrel database with selected files by set difference.

    The standard Squirrel ``add(check=True)`` path can be expensive for large
    archives because it revisits many files. This helper computes the current
    filesystem target set and the database's known paths, then performs only
    the required additions and deletions.

    Parameters
    ----------
    sq : pyrocko.squirrel.Squirrel
        Open Squirrel instance to update.
    squirrel_add_paths : list[pathlike]
        Files or directories selected for indexing, usually after date-window
        pruning by ``_select_squirrel_add_paths``.

    Notes
    -----
    Deletions are restricted to files under the requested roots. This avoids
    removing paths from a shared database simply because they were not included
    in the current indexing request.
    """
    if Squirrel is None:
        return

    # 1. Resolve all selected daily/metadata directories to concrete files.
    target_files = set()
    resolved_roots = set()
    for path in squirrel_add_paths:
        p = Path(path).expanduser().resolve()
        resolved_roots.add(str(p))
        if p.is_file():
            target_files.add(str(p))
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    target_files.add(str(f.resolve()))

    # 2. Read the files currently registered in this Squirrel database.
    db_files = set()
    try:
        paths_in_db = sq.get_paths()
    except Exception as e:
        logger.warning(f"Failed to get paths from Squirrel DB: {e}")
        paths_in_db = []

    for f in paths_in_db:
        try:
            db_files.add(str(Path(f).resolve()))
        except Exception:
            db_files.add(str(f))

    # 3. Determine newly added files and deleted files under selected roots.
    to_add = target_files - db_files

    to_remove = [
        f
        for f in db_files
        if f not in target_files and _is_under_resolved_roots(Path(f), resolved_roots)
    ]

    # 4. Apply only the minimal add/remove operations.
    if to_remove:
        logger.info(f"Fast-removing {len(to_remove)} deleted files from Squirrel DB")
        try:
            sq.remove(to_remove)
        except Exception as e:
            logger.warning(f"Failed to remove files from DB: {e}")

    if to_add:
        logger.info(f"Fast-adding {len(to_add)} new files to Squirrel DB")
        try:
            sq.add(list(to_add), check=False)
        except Exception as e:
            logger.warning(f"Failed to add files to DB: {e}")


def _index_day_worker(
    env: str,
    day_iso: str,
    paths: list[str],
    persistent: str,
    check: bool,
) -> str:
    """
    Worker entry point for building/updating one day-sharded database.

    Parameters
    ----------
    env : str
        Base Squirrel environment path supplied by ``OGSSquirrelDataSource``.
    day_iso : str
        ISO-formatted calendar day (``YYYY-MM-DD``) to index. Strings keep
        the worker payload picklable and unambiguous across processes.
    paths : list[str]
        Source archive roots or files.
    persistent : str
        Pyrocko Squirrel persistence namespace.
    check : bool
        Preserved for worker signature compatibility. The worker always uses
        the fast update path because it already receives the requested day.

    Returns
    -------
    str
        Human-readable status message logged by the parent process.
    """
    try:
        import datetime
        from pathlib import Path
        from pyrocko.squirrel import Squirrel, init_environment

        day = datetime.date.fromisoformat(day_iso)

        env_path = Path(env).expanduser().resolve()
        db_dir = _target_day_env(env_path, day)

        db_dir.mkdir(parents=True, exist_ok=True)
        if not (db_dir / ".squirrel").is_dir():
            init_environment(str(db_dir))

        sq = Squirrel(env=str(db_dir), persistent=persistent)

        squirrel_add_paths = _select_squirrel_add_paths(paths, day, day)

        _fast_update_squirrel_db(sq, squirrel_add_paths)

        return f"Day {day_iso} indexed successfully."
    except Exception as e:
        return f"Error indexing day {day_iso}: {e}"


# =============================================================================
# OGS DAY-DIRECTORY DISCOVERY AND CACHING
# =============================================================================


def _ogs_day_dir_cache_signature(
    root: Path,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
) -> Optional[tuple[tuple[str, Optional[int]], ...]]:
    """
    Build a lightweight modification-time signature for an OGS archive tree.

    The signature is used to decide whether cached ``YYYY/MM/DD`` discovery
    results are still valid. When a date window is supplied, only years and
    months that can contain requested days are stat'ed; this keeps group and
    indexing setup fast on very large archives.

    Parameters
    ----------
    root : Path
        Candidate OGS archive root.
    start_date, end_date : datetime.date or None
        Optional inclusive date range used to reduce the scan surface.

    Returns
    -------
    tuple or None
        Signature entries ``(relative_path, mtime_ns)``. ``None`` means the
        root is not a directory or could not be inspected.
    """
    if not root.is_dir():
        return None

    signature = [(".", _stat_mtime_ns(root))]

    # Date-bounded signatures avoid touching years/months outside the request.
    if start_date is not None and end_date is not None:
        years = list(range(start_date.year, end_date.year + 1))
        for yr in years:
            year_path = root / str(yr)
            if year_path.is_dir():
                signature.append((str(yr), _stat_mtime_ns(year_path)))

                start_month = start_date.month if yr == start_date.year else 1
                end_month = end_date.month if yr == end_date.year else 12
                for mo in range(start_month, end_month + 1):
                    month_path = year_path / f"{mo:02d}"
                    if month_path.is_dir():
                        signature.append((f"{yr}/{mo:02d}", _stat_mtime_ns(month_path)))
        return tuple(signature)

    try:
        year_paths = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError:
        return None

    for year_path in year_paths:
        if not (
            year_path.is_dir()
            and len(year_path.name) == 4
            and year_path.name.isdigit()
        ):
            continue
        signature.append((year_path.name, _stat_mtime_ns(year_path)))

        try:
            month_paths = sorted(year_path.iterdir(), key=lambda path: path.name)
        except OSError:
            continue

        for month_path in month_paths:
            if not (
                month_path.is_dir()
                and len(month_path.name) == 2
                and month_path.name.isdigit()
            ):
                continue
            signature.append(
                (f"{year_path.name}/{month_path.name}", _stat_mtime_ns(month_path))
            )

    return tuple(signature)


def _get_cached_ogs_day_dirs(
    root: Path,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
) -> list[tuple[datetime.date, Path]]:
    """
    Return cached OGS ``YYYY/MM/DD`` directories for a root and date window.

    The result is sorted by date then path. Cache invalidation is based on the
    signature returned by ``_ogs_day_dir_cache_signature`` so repeated group
    discovery and indexing calls do not repeatedly traverse the same tree.
    """
    root = root.expanduser()
    signature = _ogs_day_dir_cache_signature(root, start_date=start_date, end_date=end_date)
    if signature is None:
        return []

    cache_key = (root.resolve(strict=False), start_date, end_date)
    cached = _SQUIRREL_OGS_DAY_DIR_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        return cached[1]

    day_dirs = []

    if start_date is not None and end_date is not None:
        years = list(range(start_date.year, end_date.year + 1))
        for yr in years:
            year_path = root / str(yr)
            if not year_path.is_dir():
                continue

            start_month = start_date.month if yr == start_date.year else 1
            end_month = end_date.month if yr == end_date.year else 12
            for mo in range(start_month, end_month + 1):
                month_path = year_path / f"{mo:02d}"
                if not month_path.is_dir():
                    continue

                try:
                    day_paths = sorted(month_path.iterdir(), key=lambda path: path.name)
                except OSError:
                    continue

                for day_path in day_paths:
                    if not (day_path.is_dir() and day_path.name.isdigit()):
                        continue
                    day = _parse_ogs_day(str(yr), f"{mo:02d}", day_path.name)
                    if day is None:
                        continue
                    if start_date <= day <= end_date:
                        day_dirs.append((day, day_path))
    else:
        try:
            year_paths = sorted(root.iterdir(), key=lambda path: path.name)
        except OSError:
            return []

        for year_path in year_paths:
            if not (year_path.is_dir() and year_path.name.isdigit()):
                continue
            try:
                month_paths = sorted(year_path.iterdir(), key=lambda path: path.name)
            except OSError:
                continue

            for month_path in month_paths:
                if not (month_path.is_dir() and month_path.name.isdigit()):
                    continue
                try:
                    day_paths = sorted(month_path.iterdir(), key=lambda path: path.name)
                except OSError:
                    continue

                for day_path in day_paths:
                    if not (day_path.is_dir() and day_path.name.isdigit()):
                        continue
                    day = _parse_ogs_day(year_path.name, month_path.name, day_path.name)
                    if day is None:
                        continue
                    day_dirs.append((day, day_path))

    day_dirs.sort(key=lambda item: (item[0], str(item[1])))
    _SQUIRREL_OGS_DAY_DIR_CACHE[cache_key] = (signature, day_dirs)
    return day_dirs


def _select_squirrel_add_paths(
    paths: Iterable[pathlike], starttime, endtime
) -> list[pathlike]:
    """
    Select the minimal set of paths that Squirrel should index for a time span.

    For OGS archive roots with a ``YYYY/MM/DD`` hierarchy, this returns only
    day directories inside the inclusive date range plus station metadata
    children. For non-OGS paths, metadata paths, or unparseable time windows,
    the original input paths are preserved.
    """
    paths = list(paths)
    start_date = _squirrel_add_time_to_date(starttime)
    end_date = _squirrel_add_time_to_date(endtime)
    if start_date is None or end_date is None:
        return _deduplicate_paths(paths)

    selected_paths = []
    for original_path in paths:
        path = Path(original_path).expanduser()
        day_dirs = _get_cached_ogs_day_dirs(path, start_date=start_date, end_date=end_date)
        if day_dirs:
            selected_paths.extend(_station_metadata_children(path))
            selected_paths.extend(
                day_path
                for day, day_path in day_dirs
                if start_date <= day <= end_date
            )
            continue

        if _is_station_metadata_path(path):
            selected_paths.append(original_path)
            continue

        selected_paths.append(original_path)

    return _deduplicate_paths(selected_paths)


def _available_ogs_days(
    paths: Iterable[pathlike],
    start_date: Optional[datetime.date],
    end_date: Optional[datetime.date],
) -> list[datetime.date]:
    """
    Return days with OGS day directories available in the requested window.

    Used during initialization to decide which daily shards should be indexed
    when ``check=True`` and which shards are relevant for querying.
    """
    days = set()
    for original_path in paths:
        path = Path(original_path).expanduser()
        for day, _ in _get_cached_ogs_day_dirs(path, start_date, end_date):
            days.add(day)
    return sorted(days)


def _available_ogs_groups(
    paths: Iterable[pathlike],
    start_date: Optional[datetime.date],
    end_date: Optional[datetime.date],
) -> list[str]:
    """
    Return available daily group names directly from the OGS filesystem.

    The returned group strings are ISO dates (``YYYY-MM-DD``). This path avoids
    querying Squirrel for every day when the archive layout already identifies
    which days have files.
    """
    if start_date is None or end_date is None:
        return []

    groups = set()
    for original_path in paths:
        path = Path(original_path).expanduser()
        for day, day_path in _get_cached_ogs_day_dirs(path, start_date, end_date):
            try:
                has_children = any(day_path.iterdir())
            except OSError:
                has_children = False
            if has_children:
                groups.add(day.isoformat())
    return sorted(groups)


# =============================================================================
# DAY-SHARDED SQUIRREL DATA SOURCE
# =============================================================================


class OGSSquirrelDataSource(BaseSquirrelDataSource):
    """
    Fast OGS waveform data source backed by Pyrocko Squirrel.

    This class implements the ``ml_catalog`` data-source interface while adding
    OGS-specific optimizations for large archives. It can query prebuilt yearly
    Squirrel databases, build/update those databases in parallel, or fall back
    to a single monolithic Squirrel environment.

    Design Goals
    ------------
    - Keep database locks and SQLite files small by sharding per calendar day.
    - Avoid full-archive scans when a start/end time window is known.
    - Support read-only shared filesystems by copying Squirrel environments to
      temporary local directories before opening them.
    - Preserve the public behavior expected by ``ml_catalog`` data pipelines.

    Parameters
    ----------
    env : str
        Base Squirrel environment path. Day shards are discovered relative to
        this path using ``_squirrel_day_env_candidates``.
    paths : list[pathlike], optional
        Source waveform/archive paths to index when ``check=True`` and to use
        for filesystem-based group discovery.
    persistent : str, default="v0"
        Pyrocko Squirrel persistence namespace.
    group_duration_days : int, default=1
        Number of days represented by each group string.
    channel_priorities : list[str], optional
        Channel patterns tried in order for each station, defaulting to
        ``["HH?", "BH?", "EH?"]``.
    starttime, endtime : PyrockoTime, optional
        Optional query/indexing window. When both are available, filesystem
        discovery and indexing are pruned to the relevant days.
    check : bool, default=False
        If True, create/update Squirrel databases before querying.
    codes, codes_exclude : list[str], optional
        Station/channel code include and exclude filters passed to Squirrel.
    accessor_buffer : int, default=100
        Number of segment requests to keep in one Squirrel accessor before
        clearing it in ``get_segments``.
    index_workers : int, optional
        Worker process count for year-sharded indexing. Defaults to the number
        of years, capped by CPU count.
    copy_to_tmp : bool, default=True
        Copy Squirrel environments to temporary directories before opening.
        This is useful for read-only or shared network filesystems.
    filesystem_groups : bool, default=True
        Prefer daily group discovery from the OGS directory tree when possible.

    Attributes
    ----------
    active_days : list[datetime.date]
        Days covered by the requested time window.
    index_days : list[datetime.date]
        Days that should be indexed when ``check=True``.
    day_envs : dict[datetime.date, Path]
        Mapping from day to initialized or planned Squirrel environment.
    """

    def __init__(
        self,
        env: str,
        paths: Optional[list[pathlike]] = None,
        persistent: str = "v0",
        group_duration_days: int = 1,
        channel_priorities: Optional[list[str]] = None,
        starttime: Optional[PyrockoTime] = None,
        endtime: Optional[PyrockoTime] = None,
        check: bool = False,
        codes: Optional[list[str]] = None,
        codes_exclude: Optional[list[str]] = None,
        accessor_buffer: int = 100,
        index_workers: Optional[int] = None,
        copy_to_tmp: bool = True,
        filesystem_groups: bool = True,
        **kwargs,
    ):
        """
        Initialize the data source and optionally update Squirrel databases.

        Initialization is intentionally lazy for read operations: Squirrel
        instances are not opened until a query asks for a specific day. When
        ``check=True``, the constructor performs the requested database update
        so later calls can use pre-indexed environments.
        """
        super(BaseSquirrelDataSource, self).__init__(**kwargs)
        self.env = env
        self.persistent = persistent
        self.group_duration_days = group_duration_days
        self.channel_priorities = channel_priorities or ["HH?", "BH?", "EH?"]
        self.codes = ["*.*.*"] if codes is None else list(codes)
        self.codes_exclude = None if codes_exclude is None else list(codes_exclude)
        self.accessor_buffer = accessor_buffer
        self.starttime = normalize_pyrocko_time(starttime)
        self.endtime = normalize_pyrocko_time(endtime)
        self.paths = paths or []
        self.check = check
        self.index_workers = index_workers
        self.copy_to_tmp = copy_to_tmp
        self.filesystem_groups = filesystem_groups
        self._filesystem_groups_cache: Optional[list[str]] = None

        # Normalize the base environment and precompute date-window helpers.
        env_path = Path(self.env).expanduser().resolve()
        self._env_path = env_path
        self._start_date = _squirrel_add_time_to_date(self.starttime)
        self._end_date = _squirrel_add_time_to_date(self.endtime)

        # Discover days available on the OGS filesystem in the requested window.
        available_days = _available_ogs_days(
            self.paths, self._start_date, self._end_date
        )

        # Determine the inclusive day range that may be queried. When a full
        # window is given, the active range is materialized day-by-day; without
        # a window, we rely on filesystem discovery to avoid an explosion of
        # days that may not exist on disk.
        if self._start_date is not None and self._end_date is not None:
            span = (self._end_date - self._start_date).days + 1
            self.active_days = [
                self._start_date + datetime.timedelta(days=i) for i in range(span)
            ]
        else:
            self.active_days = list(available_days)

        active_set = set(self.active_days)
        if available_days:
            self.index_days = [d for d in available_days if d in active_set] \
                if active_set else list(available_days)
        else:
            self.index_days = list(self.active_days)

        # Discover already initialized daily Squirrel databases.
        self.day_envs: dict[datetime.date, Path] = {}
        for d in self.active_days:
            existing = _existing_day_env(env_path, d)
            if existing is not None:
                self.day_envs[d] = existing

        if self.check and available_days:
            for d in self.index_days:
                self.day_envs.setdefault(d, _target_day_env(env_path, d))

        # Runtime caches: Squirrel objects are opened lazily and are not
        # preserved when the object is pickled for multiprocessing.
        self._tmp_squirrel_paths = {}
        self._sq_instances = {}
        self._base_sq = None
        self._tmp_squirrel_path = None
        self._sq = None  # Compatibility field for superclass checks.
        self._all_station_codes: Optional[list[str]] = None

        # Choose sharded mode if any relevant daily database exists or is
        # planned for creation under check=True. Otherwise use one base DB.
        if self.day_envs:
            existing_count = sum(
                1 for path in self.day_envs.values() if (path / ".squirrel").is_dir()
            )
            logger.info(
                "Day-sharded Squirrel mode active. Existing shards: "
                f"{existing_count}, planned shards: {len(self.day_envs) - existing_count}, "
                f"days: {len(self.day_envs)}"
            )
        else:
            logger.warning(
                f"No day-sharded databases found matching {env_path.name}/YYYY/MM/DD. "
                "Falling back to single monolithic database."
            )
            # Ensure the primary monolithic DB exists before fallback queries.
            if not (env_path / ".squirrel").is_dir():
                _, init_env = _require_squirrel_runtime()
                logger.warning("Base Squirrel environment not found. Initialising new environment.")
                env_path.mkdir(exist_ok=True, parents=True)
                init_env(str(env_path))

        # In write/check mode, update the selected sharded or monolithic DBs.
        if self.check:
            if self.day_envs:
                logger.info("Updating sharded day-level databases in parallel...")
                self.index_parallel(self.index_workers)
            else:
                logger.info("Adding/checking paths in monolithic Squirrel (fast set-difference updates)")
                squirrel_add_paths = _select_squirrel_add_paths(
                    self.paths, self.starttime, self.endtime
                )
                _fast_update_squirrel_db(self.base_sq, squirrel_add_paths)
            logger.debug("Finished building/updating Squirrel DB")
        else:
            logger.debug("Skipping writable master setup (using pre-indexed read-only sharded/base DBs)")

        self._groups = None
        logger.debug("Station codes will be loaded lazily")

    def index_parallel(self, n_workers: Optional[int] = None) -> None:
        """
        Build or update daily Squirrel shards in parallel worker threads.

        Parameters
        ----------
        n_workers : int, optional
            Number of worker threads. If omitted, uses one worker per target
            day capped by ``os.cpu_count()``.

        Notes
        -----
        Each worker opens its own per-day Squirrel instance, so threads do not
        share SQLite connections. Threading is preferred over multiprocessing
        because daily indexing is dominated by filesystem I/O and SQLite work
        that releases the GIL, and it avoids the pickling/process-startup
        overhead of ``ProcessPoolExecutor``. After workers finish, ``day_envs``
        is refreshed so newly created shards are available for queries.
        """
        import concurrent.futures

        days = self.index_days or sorted(self.day_envs.keys()) or self.active_days
        if not days:
            logger.warning("No active days available for sharded Squirrel indexing")
            return

        max_workers = n_workers or min(len(days), os.cpu_count() or 1)
        logger.info(
            f"Starting parallel sharded indexing for {len(days)} day(s) "
            f"({days[0].isoformat()} .. {days[-1].isoformat()}) "
            f"with {max_workers} thread(s)"
        )

        paths_str = [str(p) for p in self.paths]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _index_day_worker,
                    str(self.env),
                    d.isoformat(),
                    paths_str,
                    self.persistent,
                    self.check,
                ): d
                for d in days
            }

            for future in concurrent.futures.as_completed(futures):
                d = futures[future]
                try:
                    res = future.result()
                    logger.info(f"Shard {d.isoformat()}: {res}")
                except Exception as exc:
                    logger.error(f"Shard {d.isoformat()} generated an exception: {exc}")

        # Refresh day_envs to include updated or newly created environments.
        env_path = self._env_path
        self.day_envs = {}
        for d in self.active_days:
            existing = _existing_day_env(env_path, d)
            if existing is not None:
                self.day_envs[d] = existing

    def get_sq(self, day: datetime.date):
        """
        Return a cached Squirrel instance for one calendar day.

        If day-sharded mode is active and the day has a shard, that shard is
        used. Otherwise the monolithic base Squirrel instance is returned. When
        ``copy_to_tmp`` is enabled, the shard directory is copied once to a
        temporary location before opening to avoid writing into shared DBs.
        """
        if day in self._sq_instances:
            return self._sq_instances[day]

        if day not in self.day_envs:
            return self.base_sq

        env_path = self.day_envs[day]
        squirrel_cls, _ = _require_squirrel_runtime()
        if not self.copy_to_tmp:
            self._sq_instances[day] = squirrel_cls(
                env=str(env_path), persistent=self.persistent
            )
            return self._sq_instances[day]

        # Read-only copy-to-temp logic: day shards are small enough that the
        # copy cost is usually lower than fighting shared SQLite locks.
        if day not in self._tmp_squirrel_paths:
            tmp_path = Path(tempfile.mkdtemp()) / f"squirrel_env_{day.isoformat()}"
            shutil.copytree(env_path, tmp_path)
            self._tmp_squirrel_paths[day] = tmp_path

        self._sq_instances[day] = squirrel_cls(
            env=self._tmp_squirrel_paths[day], persistent=self.persistent
        )
        return self._sq_instances[day]

    @property
    def base_sq(self):
        """
        Lazily open the monolithic fallback Squirrel environment.

        The base environment is used when no day shard exists for a query, or
        when sharded mode is not active. With ``copy_to_tmp=True``, the base
        environment is copied into a temporary directory before opening.
        """
        if self._base_sq is None:
            squirrel_cls, init_env = _require_squirrel_runtime()
            if not self.copy_to_tmp:
                env_path = Path(self.env).expanduser()
                if not (env_path / ".squirrel").is_dir():
                    env_path.mkdir(parents=True, exist_ok=True)
                    init_env(str(env_path))
                self._base_sq = squirrel_cls(env=str(env_path), persistent=self.persistent)
                return self._base_sq
            if self._tmp_squirrel_path is None:
                tmp_squirrel_path = Path(tempfile.mkdtemp()) / "squirrel_env"
                if Path(self.env).exists():
                    shutil.copytree(self.env, tmp_squirrel_path)
                else:
                    tmp_squirrel_path.mkdir(parents=True, exist_ok=True)
                    init_env(str(tmp_squirrel_path))
                self._tmp_squirrel_path = tmp_squirrel_path
            self._base_sq = squirrel_cls(env=self._tmp_squirrel_path, persistent=self.persistent)
        return self._base_sq

    @property
    def sq(self):
        """
        Representative Squirrel accessor expected by the base data source.

        In sharded mode this returns the first active day's Squirrel instance;
        query methods should still use ``get_sq(day)`` for day-specific data.
        """
        # Override base-class accessor to dynamically choose an active shard.
        if self.day_envs:
            first_day = min(self.day_envs.keys())
            return self.get_sq(first_day)
        return self.base_sq

    @property
    def tmp_squirrel_path(self):
        """
        Return the temporary Squirrel path used by the representative DB.

        This preserves compatibility with the base ``SquirrelDataSource`` API,
        which expects one temporary environment path even though this subclass
        may maintain one temporary copy per day.
        """
        # Override base-class accessor for sharded temporary environments.
        if self.day_envs:
            first_day = min(self.day_envs.keys())
            if first_day in self._tmp_squirrel_paths:
                return self._tmp_squirrel_paths[first_day]
        if self._tmp_squirrel_path is None:
            _ = self.base_sq
        return self._tmp_squirrel_path

    def _query_days(self) -> list[datetime.date]:
        """
        Return days that should be queried for metadata and waveform checks.

        Sharded mode queries only initialized shards. Monolithic mode queries
        the active day range and falls back internally to ``base_sq``.
        """
        return sorted(self.day_envs.keys()) if self.day_envs else list(self.active_days)

    def _filesystem_day_groups(self) -> list[str]:
        """
        Return daily groups from the OGS filesystem when that shortcut applies.

        Filesystem group discovery is valid only for one-day groups because the
        OGS directory hierarchy is daily. Results are cached on the instance.
        """
        if not self.filesystem_groups or self.group_duration_days != 1:
            return []
        if self._filesystem_groups_cache is None:
            self._filesystem_groups_cache = _available_ogs_groups(
                self.paths, self._start_date, self._end_date
            )
            if self._filesystem_groups_cache:
                logger.info(
                    "Using %d filesystem-indexed OGS day groups",
                    len(self._filesystem_groups_cache),
                )
        return self._filesystem_groups_cache

    def _get_all_station_codes(self) -> list:
        """
        Load all station code triplets from available Squirrel environments.

        The result is used by ``get_group`` to try station/channel combinations
        in priority order. If sharded metadata lookup fails or returns empty,
        the method falls back to the monolithic base database.
        """
        codes_set = set()
        if self.day_envs:
            for d in self._query_days():
                try:
                    sq = self.get_sq(d)
                    for station in sq.get_stations(codes=self.codes):
                        codes_set.add(station.codes)
                except Exception as e:
                    logger.warning(f"Failed to get station codes for day {d.isoformat()}: {e}")

        # Fallback if sharded metadata is empty or sharded mode is inactive.
        if not codes_set:
            stations = self.base_sq.get_stations(codes=self.codes)
            codes_set.update(station.codes for station in stations)

        return list(sorted(codes_set))

    def _station_codes(self) -> list[str]:
        """
        Return cached station codes, loading them lazily on first use.
        """
        if self._all_station_codes is None:
            logger.debug("Getting all station codes")
            self._all_station_codes = self._get_all_station_codes()
        return self._all_station_codes

    def groups(self) -> list[str]:
        """
        Return available group identifiers for the configured time window.

        Groups are date strings accepted by ``_parse_group_times``. The method
        first tries filesystem-derived daily groups, then falls back to the
        Squirrel waveform time span and verifies each candidate with
        ``_check_group``.
        """
        if self._groups is None:
            filesystem_groups = self._filesystem_day_groups()
            if filesystem_groups:
                self._groups = filesystem_groups
                return copy.deepcopy(self._groups)

            if self.starttime is not None and self.endtime is not None:
                tmin = self.starttime
                tmax = self.endtime
            else:
                tmin, tmax = self.base_sq.get_time_span(kinds="waveform")
                if self.starttime is not None:
                    tmin = max(tmin, self.starttime)
                if self.endtime is not None:
                    tmax = min(tmax, self.endtime)

            group_duration_s = 24 * 60 * 60 * self.group_duration_days
            starts = np.arange(tmin, tmax + group_duration_s, group_duration_s)

            # Group names are the calendar day when each group starts.
            groups = [str(UTCDateTime(t))[:10] for t in starts]
            self._groups = [group for group in groups if self._check_group(group)]

        return copy.deepcopy(self._groups)

    def _check_group(self, group: str) -> bool:
        """
        Return True if a group has at least one waveform nut in Squirrel.

        This is used by the fallback group discovery path to remove empty days
        from the generated time grid. Each day shard intersecting the group is
        consulted until one reports waveform data.
        """
        logger.debug(f"Squirrel checking group {group}")
        t0, t1 = self._parse_group_times(group)
        for d in self._group_days(t0, t1):
            sq = self.get_sq(d)
            try:
                nuts = sq.get_waveform_nuts(
                    tmin=t0,
                    tmax=t1,
                    codes=self.all_codes,
                    codes_exclude=self.codes_exclude,
                )
            except Exception as e:
                logger.warning(
                    f"Error checking group waveforms for {group} on day {d.isoformat()}: {e}"
                )
                continue
            if len(nuts) > 0:
                return True
        return False

    def get_group(self, group: str) -> DataInterface:
        """
        Yield ObsPy streams for one group using station/channel priorities.

        For each station code, channels are tried in ``channel_priorities``
        order. The first channel pattern that returns waveform data is yielded
        as an ObsPy stream, then the method moves to the next station. When a
        group spans more than one day, traces from every intersecting day
        shard are concatenated before being yielded.
        """
        t0, t1 = self._parse_group_times(group)
        days = self._group_days(t0, t1)

        accessor_ids = {d: f"data_get_group_{d.isoformat()}" for d in days}

        for station_code in self._station_codes():
            for channel in self.channel_priorities:
                codes = f"{station_code}.{channel}"
                waveforms = []
                hard_error = False
                for d in days:
                    sq = self.get_sq(d)
                    try:
                        traces = sq.get_waveforms(
                            codes=codes,
                            codes_exclude=self.codes_exclude,
                            tmin=t0,
                            tmax=t1,
                            accessor_id=accessor_ids[d],
                        )
                    except SquirrelError:
                        logger.warning(
                            f"Squirrel failed getting {codes} {t0} {t1} "
                            f"(day {d.isoformat()}). Skipping."
                        )
                        hard_error = True
                        continue
                    except NoData:
                        continue
                    waveforms.extend(traces)

                if hard_error and not waveforms:
                    continue
                if waveforms:
                    yield self.pyrocko_traces_to_obspy_stream(waveforms)
                    break

            for d in days:
                try:
                    self.get_sq(d).clear_accessor(accessor_ids[d])
                except Exception:
                    pass

    def get_stations(self) -> pd.DataFrame:
        """
        Return station coordinates as a de-duplicated pandas DataFrame.

        The DataFrame contains ``id``, ``longitude``, ``latitude``, and
        ``elevation`` columns derived from Squirrel channel metadata. Sharded
        metadata is preferred, with monolithic fallback when no channels are
        found.
        """
        channels = []
        for d in self._query_days():
            try:
                sq = self.get_sq(d)
                channels.extend(
                    sq.get_channels(
                        tmin=self.starttime, tmax=self.endtime, codes=self.all_codes
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Failed to query stations for day {d.isoformat()}: {e}"
                )

        if not channels:
            channels = self.base_sq.get_channels(
                tmin=self.starttime, tmax=self.endtime, codes=self.all_codes
            )

        station_df = [
            {
                "id": str(channel.codes)[:-4],
                "longitude": channel.lon,
                "latitude": channel.lat,
                "elevation": channel.elevation,
            }
            for channel in channels
        ]
        return pd.DataFrame(station_df).drop_duplicates()

    def get_inventory(self) -> Optional[obspy.Inventory]:
        """
        Return an ObsPy inventory assembled from station metadata files.

        Squirrel station nuts provide the inventory file paths. Each unique
        path is read with ``obspy.read_inventory`` and added to a single
        ``obspy.Inventory`` object.
        """
        inventory_paths = set()
        for d in self._query_days():
            try:
                sq = self.get_sq(d)
                paths = [
                    nut.file_path
                    for nut in sq.get_nuts(
                        kind="station", codes=self.codes, codes_exclude=self.codes_exclude
                    )
                ]
                inventory_paths.update(paths)
            except Exception as e:
                logger.warning(
                    f"Failed to query inventory paths for day {d.isoformat()}: {e}"
                )

        if not inventory_paths:
            inventory_paths = set(
                [
                    nut.file_path
                    for nut in self.base_sq.get_nuts(
                        kind="station", codes=self.codes, codes_exclude=self.codes_exclude
                    )
                ]
            )

        inv = obspy.Inventory()
        for path in sorted(inventory_paths):
            inv += obspy.read_inventory(path)
        return inv

    def get_segments(self, segments: pd.DataFrame) -> list[obspy.Stream]:
        """
        Fetch waveform streams for arbitrary segment time windows.

        Parameters
        ----------
        segments : pandas.DataFrame
            Table with ``starttime``, ``endtime``, and ``station`` columns.
            Station values may be full Pyrocko/ObsPy code patterns or bare
            station codes, which are expanded to ``*.STA.*``.

        Returns
        -------
        list[obspy.Stream]
            One stream per input segment. Missing data or Squirrel errors are
            represented by empty ObsPy streams so output order stays aligned
            with the input table.
        """
        output = []
        for seg_idx, (t0, t1, station) in enumerate(
            zip(segments["starttime"], segments["endtime"], segments["station"])
        ):
            t0_norm = normalize_pyrocko_time(t0)
            t1_norm = normalize_pyrocko_time(t1)
            seg_days = self._group_days(t0_norm, t1_norm)

            if "." not in station:
                station = f"*.{station}.*"

            codes = [f"{station}.{channel}" for channel in self.channel_priorities]

            waveforms = []
            had_error = False
            had_data = False
            for d in seg_days:
                sq = self.get_sq(d)
                accessor_id = f"data_get_segments_{d.isoformat()}"
                try:
                    traces = sq.get_waveforms(
                        tmin=t0_norm,
                        tmax=t1_norm,
                        codes=codes,
                        codes_exclude=self.codes_exclude,
                        accessor_id=accessor_id,
                    )
                    waveforms.extend(traces)
                    had_data = True
                    if (seg_idx + 1) % self.accessor_buffer == 0:
                        sq.clear_accessor(accessor_id)
                except SquirrelError:
                    logger.warning(
                        f"Squirrel failed getting {codes} {t0_norm} {t1_norm} "
                        f"(day {d.isoformat()}). Skipping."
                    )
                    had_error = True
                except NoData:
                    logger.warning(
                        f"Squirrel returned no data for {codes} {t0_norm} {t1_norm} "
                        f"(day {d.isoformat()}). Skipping."
                    )

            if waveforms:
                output.append(self.pyrocko_traces_to_obspy_stream(waveforms))
            elif had_error or not had_data:
                output.append(obspy.Stream())
            else:
                output.append(obspy.Stream())

        # Clean accessors for all opened daily Squirrel instances.
        for d, sq in list(self._sq_instances.items()):
            try:
                sq.clear_accessor(f"data_get_segments_{d.isoformat()}")
            except Exception:
                pass

        return output

    def __del__(self):
        """
        Remove temporary Squirrel copies created for read-only access.
        """
        for path in getattr(self, "_tmp_squirrel_paths", {}).values():
            try:
                shutil.rmtree(path.parent)
            except Exception:
                pass
        tmp_squirrel_path = getattr(self, "_tmp_squirrel_path", None)
        if tmp_squirrel_path is not None:
            try:
                shutil.rmtree(tmp_squirrel_path.parent)
            except Exception:
                pass

    def __getstate__(self):
        """
        Return pickle state without live Squirrel objects or temp paths.

        This makes the data source safe to serialize for multiprocessing. Each
        worker reopens Squirrel environments lazily after unpickling.
        """
        state = self.__dict__.copy()
        state["_sq"] = None
        state["_base_sq"] = None
        state["_sq_instances"] = {}
        state["_tmp_squirrel_paths"] = {}
        state["_tmp_squirrel_path"] = None
        return state


    @property
    def all_codes(self):
        """
        Expand station code patterns across channel priorities.

        Returns patterns of the form ``NET.STA.LOC.CHAN`` suitable for Squirrel
        channel and waveform queries.
        """
        codes = []
        for code in self.codes:
            for channel in self.channel_priorities:
                codes.append(f"{code}.{channel}")
        return codes

    def _parse_group_times(self, group: str) -> tuple[float, float]:
        """
        Convert a group identifier into a Pyrocko-compatible time interval.

        ``group`` is usually an ISO date string. The returned interval covers
        ``group_duration_days`` full days starting at that time.
        """
        t0 = normalize_pyrocko_time(group)
        t1 = t0 + self.group_duration_days * 24 * 60 * 60
        return t0, t1

    @staticmethod
    def _group_days(t0: float, t1: float) -> list[datetime.date]:
        """
        Enumerate calendar days intersecting the half-open interval ``[t0, t1)``.

        Day-sharded data sources need this enumeration to route waveform
        queries to the correct per-day Squirrel environments. The end timestamp
        is treated as exclusive to avoid double-counting the boundary day when
        a request stops exactly at midnight.
        """
        start_day = UTCDateTime(t0).date
        end_utc = UTCDateTime(t1 - 1e-6) if t1 > t0 else UTCDateTime(t0)
        end_day = end_utc.date
        days = []
        d = start_day
        while d <= end_day:
            days.append(d)
            d = d + datetime.timedelta(days=1)
        return days

    @staticmethod
    def pyrocko_traces_to_obspy_stream(traces: list) -> obspy.Stream:
        """
        Convert Pyrocko traces returned by Squirrel into an ObsPy Stream.
        """
        return obspy.Stream([obspy_compat.to_obspy_trace(trace) for trace in traces])

    def version(self) -> dict[str, str]:
        """
        Return dependency versions reported by this data source.
        """
        return {"pyrocko": pyrocko.__version__}

    def citations(self) -> list[str]:
        """
        Return bibliographic citations required for this data source.
        """
        return [
            "@article{heimann2017pyrocko,\n"
            "  title={Pyrocko-An open-source seismology toolbox and library},\n"
            "  author={Heimann, Sebastian and Kriegerowski, Marius and Isken, Marius and Cesca, Simone and "
            "Daout, Simon and Grigoli, Francesco and Juretzek, Carina and "
            "Megies, Tobias and Nooshiri, Nima and Steinberg, Andreas and others},\n"
            "  year={2017},\n"
            "  publisher={GFZ Data Services}\n"
            "}"
        ]

    @staticmethod
    def _verify_squirrel():
        """
        Raise ImportError when Pyrocko Squirrel is unavailable.

        This mirrors the base class verification hook while keeping the module
        importable in environments where Pyrocko is not installed.
        """
        if Squirrel is None:
            raise ImportError(
                "Missing dependency Squirrel/Pyrocko. "
                "Installation instructions at https://pyrocko.org/docs/current/install/"
            )


# Backward-compatible alias used by existing configuration files.
SquirrelDataSource = OGSSquirrelDataSource
