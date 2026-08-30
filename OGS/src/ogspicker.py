"""
=============================================================================
OGS Amplitude Extractor - Wood-Anderson Simulation with SNR Gating
=============================================================================

OVERVIEW:
Provides :class:`OGSAmplitudeExtractor`, a subclass of
``ml_catalog.modules.AmplitudeExtractor`` that extracts peak amplitudes from
horizontal components around SeisBench picks for use by the OGS local
magnitude module.

Processing pipeline (per pick):
  1. Slice a wide window around the pick peak time.
  2. Detrend (demean + linear) and remove instrument response with a fixed
     water-level deconvolution.
  3. Apply a 1-40 Hz bandpass filter.
  4. Split the window into a noise segment (before pick) and a signal segment
     (after pick) of width ``TIME_SLACK``.
  5. Reject picks whose SNR is below ``SNR_THRESHOLD`` on either component.
  6. Simulate the OGS Wood-Anderson instrument response.
  7. Report the maximum absolute amplitude (in mm) on each component over the
     [pick - TIME_BEFORE, pick + TIME_AFTER] window.

MODULE CONSTANTS:
    OGS_WOOD_ANDERSON : dict
        Poles/zeros/gain/sensitivity describing the OGS Wood-Anderson sim.
    WATER_LEVEL : int
        Water-level (dB) used in deconvolution.
    FREQ_RANGE : list[float]
        Bandpass corner frequencies [Hz].
    SNR_THRESHOLD : float
        Minimum noise/signal ratio gate.
    TIME_BEFORE / TIME_AFTER / TIME_SLACK : float
        Window geometry in seconds.

USAGE:
    from ogspicker import OGSAmplitudeExtractor

    extractor = OGSAmplitudeExtractor()
    builder.add_module(extractor)

DEPENDENCIES:
    - obspy: Stream / Inventory and response handling
    - numpy: vectorized norm / max computations
    - seisbench.util: ``Pick`` object
    - ml_catalog.modules.AmplitudeExtractor: base class

AUTHOR: AI2Seism Project
=============================================================================
"""

import obspy
import numpy as np
import seisbench.util as sbu
from datetime import timedelta as td
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
  """
  OGSAmplitudeExtractor is a subclass of ``ml_catalog.modules.AmplitudeExtractor``
  that extracts peak amplitudes from horizontal components around SeisBench picks
  for use by the OGS local magnitude module.
  """

  def __init__(self, **kwargs):
    super().__init__(time_before=TIME_BEFORE, time_after=TIME_AFTER,
                     components="NE", slack=TIME_SLACK, response_removal_args={
                         "water_level": WATER_LEVEL}, **kwargs)

  def _extract_single_amplitude(self, large_window: obspy.Stream,
                                pick: sbu.Pick,
                                sub_inv: obspy.Inventory) -> dict[str, float]:
    output = {"amplitude": np.nan}
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
    except ValueError as e:  # No response information
      print(f"Failed to remove response for {pick}: {e}")
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
