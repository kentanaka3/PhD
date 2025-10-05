import pandas as pd
from ml_catalog.base import CacheHelper
from matplotlib.path import Path as mplPath
from ml_catalog.modules import GammaAssociator

OGS_STUDY_REGION = [
  (9.5, 47.5),  (44.3, 47.5),
  (9.5, 15.0),  (44.3, 15.0),
]

class OGSGammaAssociator(GammaAssociator):
  """
  OGS-specific implementation of the ML Catalog Gamma Associator.

  We define an enclosed study region for seismic event association and discard
  those events that fall outside this region.
  """
  def __init__(self,
    config: dict,
    region: mplPath = mplPath(OGS_STUDY_REGION, closed=True),
    **kwargs,
  ) -> None:
    self.region = region
    super().__init__(config, **kwargs)

  def get_events(
    self,
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    group: str,
    path_helper: CacheHelper,
  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog, merged = super().get_events(picks, stations, group, path_helper)
    return catalog, merged