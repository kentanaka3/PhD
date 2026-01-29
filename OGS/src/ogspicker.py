import obspy
import numpy as np
import seisbench.util as sbu
from datetime import datetime, timedelta as td
from ml_catalog.modules import AmplitudeExtractor

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
EPSILON_TIMEDELTA = td(seconds=0.1)

TIME_BEFORE = 2.0  # seconds before the pick
TIME_AFTER = 10.0  # seconds after the pick
TIME_SLACK = 10.0  # seconds slack around the pick

class OGSAmplitudeExtractor(AmplitudeExtractor):
  def __init__(self, **kwargs):
    super().__init__(time_before=TIME_BEFORE, time_after=TIME_AFTER,
                     components="NE", slack=TIME_SLACK, response_removal_args={
                         "water_level": WATER_LEVEL}, **kwargs)

  def _extract_single_amplitude(self, large_window: obspy.Stream,
                                pick: sbu.Pick,
                                sub_inv: obspy.Inventory) -> dict[str, float]:
    output = {"amplitude" : np.nan}
    if pick.peak_time is None:
      print(f"No peak time found in {pick}, skipping amplitude extraction.")
      return output
    if any([len(large_window.select(component=component)) != 1
            for component in self.components]):
      return output
    # Normalize window
    large_window.detrend("demean")
    large_window.detrend("linear")
    print("Removing response...")
    try:
      large_window.remove_response(sub_inv, **self.response_removal_args)
    except ValueError:  # No response information
      return output

    # Apply bandpass filter
    large_window.filter("bandpass", freqmin=FREQ_RANGE[0],
                        freqmax=FREQ_RANGE[1], corners=2)

    tmp_windows = dict()
    tmp_windows["noise"] = {
        component:
            large_window.slice(pick.peak_time - self.slack,
                               pick.peak_time - EPSILON_TIMEDELTA).select(
                component=component)
        for component in self.components}
    tmp_windows["signal"] = {
        component:
            large_window.slice(pick.peak_time,
                               pick.peak_time + self.slack).select(
                component=component)
        for component in self.components}
    # Check SNR
    if not (all([len(tmp_windows[key][component]) for key in tmp_windows.keys()
                 for component in self.components])):
      return output
    for key in tmp_windows.keys():
      for component in self.components:
        tmp_windows[key][component] = tmp_windows[key][component][0].data.copy()

    # Simulate Wood-Anderson response
    large_window.simulate(paz_simulate=OGS_WOOD_ANDERSON)

    for component in self.components:
      output["snr_" + component] = (
        np.linalg.norm(tmp_windows["signal"][component]) /
        np.linalg.norm(tmp_windows["noise"][component]))
      output["amplitude_" + component] = np.max(np.abs(
            large_window.slice(pick.peak_time - self.time_before,
                               pick.peak_time + self.time_after).select(
                component=component)[0].data)) * 1000  # Convert to mm
    return output