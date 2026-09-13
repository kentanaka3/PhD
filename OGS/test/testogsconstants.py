"""
=============================================================================
OGS Constants Test Suite - Unit Tests for Inventory & Waveform Lookups
=============================================================================

OVERVIEW:
Unit test suite for ``ogsconstants.py`` and associated data lookups. Verifies
station inventory parsing, miniSEED waveform file discovery, and standard
catalog column header definitions.

TEST CASES & INVARIANTS:
  1. test_inventory: Validates station inventory DataFrame schema, coordinates,
     and network/station code extraction.
  2. test_waveforms: Validates waveform path indexing, date parsing, and
     channel discovery across archive directories.
  3. test_headers: Ensures consistency of canonical column names across event
     and pick tabular schemas.

USAGE:
python -m unittest OGS/test/testogsconstants.py

DEPENDENCIES:
- unittest: standard library testing framework
  - pandas: DataFrame verification
  - ogsconstants / ogsutils: constants and inventory utilities under test

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
    Centro di Ricerche Sismologiche (CRS)
  - Università degli Studi di Trieste (UniTS)
    Dipartimento di Matematica, Informatica e Geoscienze (MIGe)
    Applied Data Science and Artificial Intelligence (ADSAI)
  - Terabit Network for Research and Academic Big Data in Italy (TeRABIT)
    Consorzio Interuniversitario del Nord-Est per il Calcolo Automatico (CINECA)
=============================================================================
"""

import ogsutils as OGS_U
import ogsconstants as OGS_C
import os
import sys
import pandas as pd
import unittest
from pathlib import Path
from datetime import datetime
THIS_DIR = os.path.dirname(__file__)
sys.path.append(THIS_DIR + "/../src")


class TestOGSConstants(unittest.TestCase):
  def test_inventory(self):
    stations = Path(os.path.abspath(THIS_DIR + "/../data/station"))
    inventory = OGS_U.inventory(stations)
    expected = pd.read_parquet(
        Path(os.path.abspath(THIS_DIR + "/../data/OGSCatalog/station.parquet"))
    )
    pd.testing.assert_frame_equal(
        inventory.reset_index(drop=True),
        expected.reset_index(drop=True)
    )

  def test_waveforms(self):
    start = datetime.strptime("240101", OGS_C.YYMMDD_FMT)
    end = datetime.strptime("241231", OGS_C.YYMMDD_FMT)
    waveforms_path = Path(os.path.abspath(THIS_DIR + "/../data/waveforms"))
    stations_path = Path(os.path.abspath(THIS_DIR + "/../data/station"))
    waveforms, _ = OGS_U.waveforms(waveforms_path, stations_path, start, end)
    expected = pd.read_parquet(
        Path(os.path.abspath(THIS_DIR + "/../data/OGSCatalog/waveforms.parquet"))
    )
    pd.testing.assert_frame_equal(
        waveforms.reset_index(drop=True),
        expected.reset_index(drop=True)
    )

  def test_headers(self):
    self.assertEqual(len(OGS_C.HEADER_MANL), len(set(OGS_C.HEADER_MANL)))
    self.assertEqual(
        OGS_C.HEADER_MANL, [
            OGS_C.INDEX_STR, OGS_C.TIME_STR, OGS_C.PHASE_STR, OGS_C.STATION_STR,
            OGS_C.GROUPS_STR
        ]
    )
    self.assertEqual(OGS_C.HEADER_PRED, OGS_C.HEADER_MODL + OGS_C.HEADER_MANL)


if __name__ == "__main__":
  unittest.main()
