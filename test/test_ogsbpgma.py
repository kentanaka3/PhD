import os
import sys
import unittest
from pathlib import Path

import pandas as pd


# Ensure we can import modules from OGS/src
PRJ_PATH = Path(__file__).resolve().parents[1]
OGS_SRC = PRJ_PATH / "OGS" / "src"
if str(OGS_SRC) not in sys.path:
  sys.path.append(str(OGS_SRC))

import ogsbpgma as BPG  # type: ignore
import ogsconstants as OGS_C  # type: ignore


class TestOGSBPGMA(unittest.TestCase):

  def test_picks_basic_matching(self):
    """
    Scenario:
      TRUE picks: P at t=10.0, S at t=20.0 (same station)
      PRED picks: P at t=10.2, P at t=19.6 (same station)

    Expectations:
      - Confusion matrix increments at (P,P) and (S,P)
      - Exactly 1 TP (the P/P correct match)
      - No FN and no FP
    """
    T = pd.DataFrame({
      OGS_C.TIMESTAMP_STR: [10.0, 20.0],
      OGS_C.PHASE_STR: [OGS_C.PWAVE, OGS_C.SWAVE],
      OGS_C.STATION_STR: ["OX.BOO", "OX.BOO"],
      OGS_C.INDEX_STR: [100, 101],
      OGS_C.ERT_STR: [0.3, 0.5],
    })
    P = pd.DataFrame({
      OGS_C.TIMESTAMP_STR: [10.2, 19.6],
      OGS_C.PHASE_STR: [OGS_C.PWAVE, OGS_C.PWAVE],
      OGS_C.STATION_STR: ["OX.BOO", "OX.BOO"],
      OGS_C.INDEX_STR: [900, 901],
      OGS_C.PROBABILITY_STR: [0.90, 0.55],
    })

    CFN, TP, FN, FP = BPG.conf_mtx(
      T, P, model_name=".dat", dataset_name="SeisBenchPicker",
      method=OGS_C.CLSSFD_STR,
    )

    # Confusion entries
    self.assertEqual(int(CFN.loc[OGS_C.PWAVE, OGS_C.PWAVE]), 1)
    self.assertEqual(int(CFN.loc[OGS_C.SWAVE, OGS_C.PWAVE]), 1)
    # NONE row/column should be zeros in this scenario
    self.assertEqual(int(CFN.loc[OGS_C.NONE_STR, OGS_C.PWAVE]), 0)
    self.assertEqual(int(CFN.loc[OGS_C.PWAVE, OGS_C.NONE_STR]), 0)

    # TP/FN/FP counts
    self.assertEqual(len(TP), 1)
    self.assertEqual(len(FN), 0)
    self.assertEqual(len(FP), 0)

  @unittest.skipUnless(getattr(BPG, "_HAS_SCIPY", False),
                       "SciPy required for Hungarian matching test")
  def test_picks_hungarian_vs_greedy_conflict(self):
    """
    Construct a 2x2 conflict where both TRUE rows prefer the same PRED column
    due to higher probability, leading greedy to keep only one match while
    Hungarian returns a 1-1 assignment of two matches.

    Setup:
      - All timestamps equal (within gate), same phase/station.
      - P0 prob = 0.90, P1 prob = 0.80
      - For each TRUE, W(T, P0) > W(T, P1)

    Expectations:
      - Greedy result: 1 matched pair (both TRUE choose P0; only one kept)
      - Hungarian result: 2 matched pairs (T0->P0, T1->P1)
    """
    T = pd.DataFrame({
      OGS_C.TIMESTAMP_STR: [10.0, 10.0],
      OGS_C.PHASE_STR: [OGS_C.PWAVE, OGS_C.PWAVE],
      OGS_C.STATION_STR: ["OX.BOO", "OX.BOO"],
      OGS_C.INDEX_STR: [100, 101],
      OGS_C.ERT_STR: [0.1, 0.1],
    })
    P = pd.DataFrame({
      OGS_C.TIMESTAMP_STR: [10.0, 10.0],
      OGS_C.PHASE_STR: [OGS_C.PWAVE, OGS_C.PWAVE],
      OGS_C.STATION_STR: ["OX.BOO", "OX.BOO"],
      OGS_C.INDEX_STR: [900, 901],
      OGS_C.PROBABILITY_STR: [0.90, 0.80],
    })

    # Greedy
    bpg_g = BPG.myBPGraph(T, P, config=OGS_C.MATCH_CNFG[OGS_C.CLSSFD_STR])
    bpg_g.makeMatch(use_hungarian=False)
    greedy_matched_true = sum(1 for t in bpg_g.G[0] if bpg_g.G[0][t])
    self.assertEqual(greedy_matched_true, 1)

    # Hungarian
    bpg_h = BPG.myBPGraph(T, P, config=OGS_C.MATCH_CNFG[OGS_C.CLSSFD_STR])
    bpg_h.makeMatch(use_hungarian=True)
    hung_matched_true = sum(1 for t in bpg_h.G[0] if bpg_h.G[0][t])
    self.assertEqual(hung_matched_true, 2)
    # Ensure both PRED columns are used
    used_preds = {next(iter(d.keys())) for d in bpg_h.G[0].values() if d}
    self.assertEqual(used_preds, {0, 1})

  def test_events_basic_matching(self):
    """
    Scenario:
      TRUE event near (46.05, 13.15) at t=100.0
      PRED event near (46.06, 13.16) at t=102.0

    Expectations:
      - Confusion matrix increments at (EVENT, EVENT)
      - One TP containing paired fields
      - No FN and no FP
    """
    # Use identical time and location to avoid gating issues and ensure a match
    T = pd.DataFrame({
      OGS_C.TIMESTAMP_STR: [100.0],
      OGS_C.LATITUDE_STR: [46.05],
      OGS_C.LONGITUDE_STR: [13.15],
      OGS_C.DEPTH_STR: [10.0],
      OGS_C.MAGNITUDE_L_STR: [2.1],
      OGS_C.ERZ_STR: [0.5],
      OGS_C.ERH_STR: [0.7],
      OGS_C.INDEX_STR: [1],
    })

    P = pd.DataFrame({
      OGS_C.TIMESTAMP_STR: [100.0],
      OGS_C.LATITUDE_STR: [46.05],
      OGS_C.LONGITUDE_STR: [13.15],
      OGS_C.DEPTH_STR: [10.0],
      OGS_C.MAGNITUDE_L_STR: [2.0],
      OGS_C.ERZ_STR: [1.0],
      OGS_C.ERH_STR: [1.5],
      OGS_C.ID_STR: ["E0001"],
      OGS_C.NOTES_STR: ["SBC"],
    })

    CFN, TP, FN, FP = BPG.conf_mtx(
      T, P, model_name=".hpl", dataset_name="GammaAssociator",
      method=OGS_C.SOURCE_STR,
    )

    self.assertEqual(int(CFN.loc[OGS_C.EVENT_STR, OGS_C.EVENT_STR]), 1)
    # NONE row/column zeros
    self.assertEqual(int(CFN.loc[OGS_C.NONE_STR, OGS_C.EVENT_STR]), 0)
    self.assertEqual(int(CFN.loc[OGS_C.EVENT_STR, OGS_C.NONE_STR]), 0)

    self.assertEqual(len(TP), 1)
    self.assertEqual(len(FN), 0)
    self.assertEqual(len(FP), 0)


if __name__ == "__main__":
  unittest.main()
