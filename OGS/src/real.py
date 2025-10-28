import datetime
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional

import dask
import numpy as np
import pandas as pd
from obspy.clients.fdsn.mass_downloader.utils import EARTH_RADIUS
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model

from ..base import CacheHelper, Status
from ..types import pathlike
from ..util import StationMapper, set_cwd
from .associator_base import AbstractAssociator

FAKE_NETWORK_CODE = "XX"


class REALAssociator(AbstractAssociator):
    """
    This module is a wrapper around the `REAL phase associator
    <https://github.com/Dal-mzhang/REAL>`_. REAL uses a grid search approach for performing phase association.
    To use this tool, REAL needs to be installed and available in your ``$PATH`` or at the provided ``real_prefix``.
    Most parameters are passed directly to REAL. The names used are either descriptive, or follow verbatim the parameter
    naming for REAL. See the documentation for REAL for details on how to set it up. Below, we document additional
    parameters.

    :param use_model1d: If true, uses the 1D model from the Status. Otherwise, reverts to the homogeneous velocity
                        model. When using a 1D model, the wrapper assumes a radius of 6,378 km for the planet. So in
                        case you want to use this module for a celestial body other than Earth (cool!) open an issue
                        on Github.
    :param tt_range_horizontal_deg: Horizontal range of the travel time grid for the 1D model in degrees
    :param tt_grid_size_horizontal_deg: Horizontal grid spacing for travel time grid in degrees
    :param tt_grid_size_vertical_km: Vertical grid spacing for travel time grid in kilometers
    :param real_prefix: Prefix for the REAL binary if not available in ``$PATH``
    """

    def __init__(
        self,
        *,
        search_range_horizontal_deg: float,
        search_range_vertical_km: float,
        grid_size_horizontal_deg: float,
        grid_size_vertical_km: float,
        event_separation_sec: float,
        vp: float,
        vs: float,
        p_picks: int,
        s_picks: int,
        total_picks: int,
        p_and_s_picks: int,
        max_residual_std: float,
        min_p_to_s_separation: float,
        nrt: float = 1.5,  # Not sure what this does.
        drt: float = 0.5,  # Not sure what this does
        nxd: float = 1.0,  # Distance criterion, 1.0 deactivates it
        tolerance_multiplier: float = 4.0,
        shallow_vp: float = np.nan,
        shallow_vs: float = np.nan,
        elevation_correction: bool = False,
        max_azimuthal_gap: float = 360,
        max_distance_pick_deg: float = 180,
        latref0: Optional[float] = None,
        lonref0: Optional[float] = None,
        use_model1d: bool = True,
        tt_range_horizontal_deg: Optional[float] = None,
        tt_grid_size_horizontal_deg: Optional[float] = None,
        tt_grid_size_vertical_km: Optional[float] = None,
        real_prefix: str = "",
        **kwargs,
    ) -> None:
        self.search_range_horizontal_deg = search_range_horizontal_deg
        self.search_range_vertical_km = search_range_vertical_km
        self.grid_size_horizontal_deg = grid_size_horizontal_deg
        self.grid_size_vertical_km = grid_size_vertical_km
        self.event_separation_sec = event_separation_sec
        self.vp = vp
        self.vs = vs
        self.p_picks = p_picks
        self.s_picks = s_picks
        self.total_picks = total_picks
        self.p_and_s_picks = p_and_s_picks
        self.max_residual_std = max_residual_std
        self.min_p_to_s_separation = min_p_to_s_separation
        self.nrt = nrt
        self.drt = drt
        self.nxd = nxd
        self.tolerance_multiplier = tolerance_multiplier
        self.shallow_vp = shallow_vp
        self.shallow_vs = shallow_vs
        self.elevation_correction = elevation_correction
        self.max_azimuthal_gap = max_azimuthal_gap
        self.max_distance_pick_deg = max_distance_pick_deg
        self.latref0 = latref0
        self.lonref0 = lonref0
        self.use_model1d = use_model1d
        self.tt_range_horizontal_deg = tt_range_horizontal_deg
        self.tt_range_vertical_km = self.search_range_vertical_km
        self.tt_grid_size_horizontal_deg = tt_grid_size_horizontal_deg
        self.tt_grid_size_vertical_km = tt_grid_size_vertical_km
        self._real_prefix = real_prefix

        self.latitude_center = None
        self._station_mapper = None

        self._verify_real()
        super().__init__(**kwargs)

    def citations(self) -> list[str]:
        return [
            "@article{zhang2019rapid,\n"
            "  title={Rapid earthquake association and location},\n"
            "  author={Zhang, Miao and Ellsworth, William L and Beroza, Gregory C},\n"
            "  journal={Seismological Research Letters},\n"
            "  volume={90},\n"
            "  number={6},\n"
            "  pages={2276--2284},\n"
            "  year={2019},\n"
            "  publisher={GeoScienceWorld}\n"
            "}"
        ]

    def version(self) -> dict[str, str]:
        p = subprocess.run(
            [self._real_prefix + "REAL"],
            capture_output=True,
            text=True,
        )
        real_help = p.stderr
        p0 = real_help.find("(")
        p1 = real_help.find(")")
        real_version = real_help[p0 + 7 : p1]
        return {"REAL": real_version}

    def _verify_real(self):
        p = subprocess.run(
            [self._real_prefix + "REAL"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if p.returncode != 255:
            raise ImportError(
                "Missing or incorrectly installed dependency REAL. "
                "Installation instructions at https://github.com/Dal-mzhang/REAL/ . "
                "Run REAL command in your shell for details."
            )

    def setup(self, status: Status) -> None:
        stations = status.get_param("stations")
        self._station_mapper = StationMapper(stations)
        with status.get_cache_path(
            "internal", self.name, drop_group=True
        ) as working_directory:
            # Create working directory
            working_directory.mkdir(exist_ok=True, parents=True)

        self.latitude_center = (
            stations["latitude"].min() + stations["latitude"].max()
        ) / 2

        if self.use_model1d:  # Use constant velocity model
            self._setup_token = self._calc_tt_tables(
                status.cache_path_helper, status.get_param("velocity_model")
            )

    @dask.delayed
    def _calc_tt_tables(
        self, path_helper: CacheHelper, velocity_model: pd.DataFrame
    ) -> bool:
        # Code inspired by https://github.com/Dal-mzhang/REAL/blob/master/demo_syn/tt_db/taup_tt.py
        with path_helper.get_cache_path(
            "internal", self.name, drop_group=True
        ) as working_directory:
            working_directory.mkdir(exist_ok=True, parents=True)
            velocity_model = velocity_model.copy()

            # Add a layer reaching to the Earth center. This is necessary because taup assumes the deepest layer as
            # the radius of the planet.
            PLANET_RADIUS = 6371  # Average radius of Earth
            if (
                velocity_model["depth"].values[-1] < PLANET_RADIUS
            ):  # Otherwise the model is already deep enough
                velocity_model = pd.concat(
                    [
                        velocity_model,
                        pd.DataFrame(
                            {
                                "depth": [PLANET_RADIUS],
                                "vp": [velocity_model["vp"].values[-1]],
                                "vs": [velocity_model["vs"].values[-1]],
                            }
                        ),
                    ]
                )

            velocity_model_nd_path = working_directory / "model.nd"
            velocity_model_npz_path = working_directory / "model.npz"

            with open(velocity_model_nd_path, "w") as f:
                first = True
                for _, row in velocity_model.iterrows():
                    if row["depth"] < 0:
                        # Layers above 0 are not supported by taup
                        continue

                    # make sure the first layer starts at 0
                    if first:
                        depth = min(0, row["depth"])
                        first = False
                    else:
                        depth = row["depth"]

                    f.write(
                        f"{depth:.2f} {row['vp']:.2f} {row['vs']:.2f} 1.00 1.00 1.00\n"
                    )

            build_taup_model(
                str(velocity_model_nd_path), working_directory, verbose=False
            )

            model = TauPyModel(str(velocity_model_npz_path))

        table = ""
        for dep in np.arange(
            0,
            self.tt_range_vertical_km + self.tt_grid_size_vertical_km,
            self.tt_grid_size_vertical_km,
        ):
            for dist in np.arange(
                0,
                self.tt_range_horizontal_deg + self.tt_grid_size_horizontal_deg,
                self.tt_grid_size_horizontal_deg,
            ):
                arrivals = model.get_travel_times(
                    source_depth_in_km=dep,
                    distance_in_degree=dist,
                    phase_list=["P", "p", "S", "s"],
                )

                phases = {}
                for arr in arrivals:
                    if arr.name.lower() not in phases:
                        ray_param = arr.ray_param * 2 * np.pi / 360
                        slowness = -(ray_param / 111.19) / np.tan(
                            arr.takeoff_angle * np.pi / 180
                        )
                        phases[arr.name.lower()] = (
                            arr.time,
                            ray_param,
                            slowness,
                            arr.name,
                        )

                table += f"{dist} {dep}"
                for vp, vs in zip(phases["p"], phases["s"]):
                    table += f" {vp} {vs}"
                table += "\n"

        with path_helper.get_cache_path("tt_db.txt", self.name, mode="w") as path:
            with open(path, "w") as f:
                f.write(table)

        return True

    def get_events(
        self,
        picks: pd.DataFrame,
        stations: pd.DataFrame,
        group: str,
        path_helper: CacheHelper,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # [REAL can process seismic picks recorded in one day (or a few days but only up to 31 days, e.g.,
        # 2016/10/01 – 2016/10/31 but not eligible for 2016/10/02 – 2016/11/01). All picks are relative to
        # ZERO of the day (e.g., 60.00 corresponds to 2016/10/14 00:01:00.00 and 86460 corresponds to
        # 2016/10/15 00:01:00.00)]
        if len(picks) == 0:
            return pd.DataFrame(), pd.DataFrame()

        picks["segment"] = picks["time"].apply(lambda x: x.strftime("%Y/%m"))

        catalog = []
        assignments = []

        event_idx_start = 0
        for i, (_, segment_picks) in enumerate(picks.groupby("segment")):
            t0 = (
                segment_picks["time"]
                .min()
                .replace(hour=0, minute=0, second=0, microsecond=0)
            )

            with path_helper.get_cache_path(
                "internal", self.name, drop_group=True
            ) as working_directory:
                tmp_dir = working_directory / group / str(i)
                tmp_dir.mkdir(exist_ok=True, parents=True)

                self.write_stations(tmp_dir, stations)
                self.write_picks(tmp_dir, segment_picks, t0)

                with path_helper.get_cache_path(
                    "tt_db.txt", self.name, drop_group=True, mode="r"
                ) as tt_path:
                    shutil.copy(tt_path, tmp_dir)

                with set_cwd(
                    tmp_dir
                ):  # some paths are hard coded, so we need to set them manually
                    p = subprocess.run(
                        [self._real_prefix + "REAL"]
                        + self.get_real_args(t0, tt_path="tt_db.txt"),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    if p.returncode != 0:
                        raise ValueError(f"REAL exited with code {p}")

                seg_catalog, seg_assignments = self.parse_outputs(
                    tmp_dir, t0, event_idx_start
                )
                event_idx_start += len(seg_catalog)
                catalog.append(seg_catalog)
                assignments.append(seg_assignments)

        catalog = pd.concat(catalog)
        assignments = pd.concat(assignments)

        if len(assignments) == 0:
            return pd.DataFrame(), pd.DataFrame()

        drop_key = "__drop_key__"
        merge_key = "__merge_key__"

        assignments.reset_index(inplace=True)
        assignments["pick_idx"] = assignments.index
        pick_lookup = picks.copy()
        pick_lookup[merge_key] = pick_lookup["time"].apply(
            lambda x: np.round(x.timestamp(), 4)
        )
        assignments[merge_key] = assignments["time"].apply(
            lambda x: np.round(x.timestamp(), 4)
        )

        # For duplicate keys, keep the ones from pick_lookup as they don't suffer any data loss from format conversions
        assignments = pd.merge(
            assignments,
            pick_lookup,
            on=["station", "phase", "__merge_key__"],
            suffixes=(drop_key, ""),
        )
        columns_to_drop = ["index", "segment", merge_key] + [
            col for col in assignments.columns if col.endswith(drop_key)
        ]
        assignments.drop(columns=columns_to_drop, inplace=True)

        catalog["group"] = group
        assignments["group"] = group

        return catalog, assignments

    def get_real_args(
        self, t0: datetime.datetime, tt_path: Optional[pathlike] = None
    ) -> list[str]:
        # Variable names identical to REAL flag names
        D = t0.strftime("%Y/%m/%d") + f"/{self.latitude_center:.2f}"

        R = (
            f"{self.search_range_horizontal_deg:.2f}/{self.search_range_vertical_km:.1f}/"
            f"{self.grid_size_horizontal_deg:.3f}/{self.grid_size_vertical_km:.2f}/"
            f"{self.event_separation_sec:.1f}/{self.max_azimuthal_gap:.1f}/"
            f"{self.max_distance_pick_deg:.2f}"
        )

        if self.latref0 is not None and self.lonref0 is not None:
            R += f"/{self.latref0:.2f}/{self.lonref0:.2f}"

        V = f"{self.vp}/{self.vs}/{self.shallow_vp}/{self.shallow_vs}/{int(self.elevation_correction)}"

        S = (
            f"{self.p_picks}/{self.s_picks}/{self.total_picks}/{self.p_and_s_picks}/{self.max_residual_std}/"
            f"{self.min_p_to_s_separation}/{self.nrt}/{self.drt}/{self.nxd}/{self.tolerance_multiplier}/0"
        )

        flags = ["-D" + D, "-R" + R, "-V" + V, "-S" + S]
        args = ["stations.dat", "picks"]
        if self.use_model1d:
            G = (
                f"{self.tt_range_horizontal_deg}/{self.tt_range_vertical_km}/"
                f"{self.tt_grid_size_horizontal_deg}/{self.tt_grid_size_vertical_km}"
            )
            flags.extend(["-G" + G])
            args.append(str(tt_path))

        return flags + args

    def parse_outputs(
        self,
        tmp_dir: Path,
        t0: datetime.datetime,
        event_idx_start: int,  # Index for the first event
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        catalog = []
        assignments = []
        with open(tmp_dir / "phase_sel.txt", "r") as f:
            event_idx = event_idx_start - 1
            for line in f:
                if not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) == 17:
                    # Event line
                    # num, year, mon, day, time (hh:mm:ss), origin time (relative to ZERO, sec), residual (sec), lat.,
                    # lon., dep., mag., mag var (uncertainty), number of P picks, number of S picks, total number of
                    # picks, number of stations with both P and S, station gap
                    event_idx += 1
                    catalog.append(
                        {
                            "idx": event_idx,
                            "time": t0 + datetime.timedelta(seconds=float(parts[5])),
                            "real_residual": float(parts[6]),
                            "latitude": float(parts[7]),
                            "longitude": float(parts[8]),
                            "depth": float(parts[9]),
                            "number_p_picks": int(parts[12]),
                            "number_s_picks": int(parts[13]),
                            "number_picks": int(parts[14]),
                            "number_p_and_s_picks": int(parts[15]),
                            "real_station_gap": float(parts[16]),
                        }
                    )

                    pass
                else:
                    # Pick line
                    # network, station, phase name, absolute travetime (relative to ZERO, sec), traveltime
                    # relative to event origin time (sec), phase amplitude in millimeter,
                    # individual phase residual (sec), weight, azimuth
                    assignments.append(
                        {
                            "event_idx": event_idx,
                            "station": self._station_mapper.translate_station(
                                parts[1], inv=True
                            ),
                            "phase": parts[2],
                            "time": t0 + datetime.timedelta(seconds=float(parts[3])),
                            "real_residual": float(parts[6]),
                            "real_weight": float(parts[7]),
                        }
                    )

        return pd.DataFrame(catalog), pd.DataFrame(assignments)

    def write_stations(self, tmp_dir: Path, stations: pd.DataFrame) -> None:
        with open(tmp_dir / "stations.dat", "w") as f:
            for _, station in stations.iterrows():
                # lon., lat., network, station, component, elevation (km)
                # Fake network and channel to provide fixed station id
                mapped_station_code = self._station_mapper.translate_station(
                    station["id"]
                )
                f.write(
                    f"{station['longitude']} {station['latitude']} {FAKE_NETWORK_CODE} "
                    f"{mapped_station_code} HHZ {station['elevation'] / 1e3}\n"
                )

    def write_picks(
        self, tmp_dir: Path, picks: pd.DataFrame, t0: datetime.datetime
    ) -> None:
        pick_dir = tmp_dir / "picks"
        pick_dir.mkdir(exist_ok=True)

        groups = defaultdict(list)
        t0_timestamp = t0.timestamp()

        for _, pick in picks.iterrows():
            mapped_station_code = self._station_mapper.translate_station(
                pick["station"]
            )
            group = (
                f"{FAKE_NETWORK_CODE}.{mapped_station_code}.{pick['phase'].upper()}.txt"
            )
            # arrivaltime (sec), stalta_ratio or phase_probability, amplitude_in_millimeter
            groups[group].append(
                f"{pick['time'].timestamp() - t0_timestamp} {pick['probability']} 0.0\n"
            )

        for group, group_picks in groups.items():
            with open(pick_dir / group, "w") as f:
                f.write("".join(group_picks))
