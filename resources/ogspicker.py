from ml_catalog.modules import AmplitudeExtractor
from collections import defaultdict
import seisbench.util as sbu
import re
import obspy
import numpy as np
from datetime import timedelta

OGS_WOOD_ANDERSON = {
    "poles": [-5.49779 - 5.60886j, -5.49779 + 5.60886j],
    "zeros": [0 + 0j],
    "gain": 1.0,
    "sensitivity": 2080,
}

WATER_LEVEL = 60

# SNR parameters
FREQ_RANGE = [1, 40]
SNR_THRESHOLD = 1.3
EPSILON_TIMEDELTA = timedelta(seconds=0.1)

TIME_BEFORE = 2.0  # seconds before the pick
TIME_AFTER = 10.0  # seconds after the pick
TIME_SLACK = 10.0  # seconds slack around the pick


class OGSAmplitudeExtractor(AmplitudeExtractor):
  def __init__(self, **kwargs):
    super().__init__(time_before=TIME_BEFORE, time_after=TIME_AFTER,
                     components="NE",
                     slack=TIME_SLACK, response_removal_args={
                         "water_level": WATER_LEVEL, "pre_filt": FREQ_RANGE},
                     **kwargs)

  def _extract_single_amplitude(self, large_window: obspy.Stream,
                                pick: sbu.Pick,
                                sub_inv: obspy.Inventory) -> dict[str, float]:
    for component in self.components:
      if any([len(large_window.select(component=f"??{component}")) != 1
              for component in self.components]):
        return np.nan
    # Normalize window
    large_window.detrend("demean")
    large_window.detrend("linear")
    try:
      large_window.remove_response(sub_inv, **self.response_removal_args)
    except ValueError:  # No response information
      return np.nan

    WINDOW = dict()
    WINDOW["NOISE"] = {
        component:
            large_window.slice(pick.peak_time - self.slack,
                               pick.peak_time - EPSILON_TIMEDELTA).select(
                component=f"??{component}")[0].data
        for component in self.components}
    WINDOW["SIGNAL"] = {
        component:
            large_window.slice(pick.peak_time,
                               pick.peak_time + self.slack).select(
                component=f"??{component}")[0].data
        for component in self.components}
    print(WINDOW)
    # Check SNR
    if (all([len(WINDOW[key][component]) for key in WINDOW.keys()
             for component in self.components])):
      return np.nan
    WINDOW["SNR"] = {
        component:
            np.linalg.norm(WINDOW["SIGNAL"][component]) /
            np.linalg.norm(WINDOW["NOISE"][component])
        for component in self.components}



    # Simulate Wood-Anderson response
    large_window.simulate(paz_simulate=OGS_WOOD_ANDERSON)

    WINDOW["AMPLITUDE"] = {
        component: np.nanmax(np.abs(
            large_window.slice(pick.peak_time - self.time_before,
                               pick.peak_time + self.time_after).select(
                component=f"??{component}")[0].data))
        for component in self.components}

    return WINDOW # TODO: Flatten
