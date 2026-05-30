"""
=============================================================================
MHPC Thesis Driver - BGMA Comparison for the 2024 Italian Sequence
=============================================================================

OVERVIEW:
Reproducible driver script used for the MHPC thesis. It evaluates several
picker / associator / locator configurations against the OGS NLL 1D reference
catalog over the 2024-03-20 to 2024-06-20 sequence (including the M_L 4.6
shock of 2024-03-27 and the 2024-05-22 IT marker).

For each scenario the script:
  1. Loads the reference OGS catalog window.
  2. For each target configuration / model, loads the candidate ML catalog.
  3. Calls :meth:`OGSCatalog.plot` to render comparative figures with the
     configured ``special_days`` vertical markers.
  4. Calls :meth:`OGSCatalog.bpgma` to run the bipartite-graph matching
     review and write per-stage reports.

Configuration families covered:
    - Config1 / Config2 / Config3 (PhaseNet[INSTANCE] + PyOcto + NLL 1D)
    - PhaseNet / EQTransformer model and dataset sweeps
    - GaMMA0.1 / GaMMA0.2 / GaMMA0.3 association thresholds
    - PyOcto0.1 / PyOcto0.2 / PyOcto0.3 association thresholds

USAGE:
    python MHPCThesis.py     # runs the hard-coded configurations

NOTE:
Paths and waveform mount points are hard-coded for the author's workstation;
edit them before reuse on another machine.

DEPENDENCIES:
    - ogsconstants: date format strings and color palette
    - ogscatalog.OGSCatalog: catalog loading, plotting and BPGMA matching

AUTHOR: AI2Seism Project
=============================================================================
"""

from pathlib import Path
from datetime import datetime

import ogsconstants as OGS_C
from ogscatalog import OGSCatalog

def main():
  start = datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT)
  end = datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT)
  waveforms = Path("/Volumes/Expansion/KEN/waveforms")
  stations = Path("/Users/admin/Desktop/OGS_Catalog/station")
  special_days: list[tuple[datetime, str, str]] = [
    (datetime.strptime("20240327", OGS_C.YYYYMMDD_FMT), "$M_L$ 4.6", "r"),
    (datetime.strptime("20240522", OGS_C.YYYYMMDD_FMT), "IT", OGS_C.ALN_GREEN),
  ]
  pre = Path("/Users/admin/Desktop/OGS_Catalog/catalogs/")
  for target in [
    "Config1", "Config2", "Config3",
  ]:
    print(f"Processing target: {target}")
    BaseCatalog = OGSCatalog(
      pre / "OGSCatalog" / ".all",
      start=start,
      end=end,
      name="OGS NLL 1D",
      output=pre / target,
    )
    for name, path in [
      ("PhaseNet[INSTANCE, 0.1]", pre / target / "SeisBenchPicker"),
      ("PhaseNet[INSTANCE, 0.1] | PyOcto", pre / target / "OGSPickStatQC"),
      ("PhaseNet[INSTANCE, 0.1] | PyOcto | NLL 1D", pre / target / "OGSLocalMagnitude"),
    ]:
      print(f"Processing catalog: {name}")
      TargetCatalog = OGSCatalog(
        path,
        start=start,
        end=end,
        name=name,
        output= pre / target.replace("Config", "OGS")
      )
      BaseCatalog.plot([TargetCatalog], vlines=special_days)
      BaseCatalog.bpgma(
        TargetCatalog,
        stations=stations,
        waveforms=waveforms,
        vlines=special_days,
      )
      #TargetCatalog.bpgma(
      #  BaseCatalog,
      #  stations=stations,
      #  waveforms=waveforms,
      #  vlines=special_days,
      #)
  for model, target in [
    ("PhaseNet[SCEDC, 0.1]", "PhaseNet[SCEDC,0.1]"),
    ("PhaseNet[STEAD, 0.1]", "PhaseNet[STEAD,0.1]"),
    ("PhaseNet[Original, 0.1]", "PhaseNet[Original,0.1]"),
    ("EQTransformer[SCEDC, 0.1]", "EQTransformer[SCEDC,0.1]"),
    ("EQTransformer[STEAD, 0.1]", "EQTransformer[STEAD,0.1]"),
    ("EQTransformer[Original, 0.1]", "EQTransformer[Original,0.1]"),
    ("EQTransformer[INSTANCE, 0.1]", "EQTransformer[INSTANCE,0.1]"),
    ("EQTransformer[INSTANCE, 0.2]", "EQTransformer[INSTANCE,0.2]"),
    ("EQTransformer[INSTANCE, 0.3]", "EQTransformer[INSTANCE,0.3]"),
  ]:
    print(f"Processing target: {target}")
    print(f"Processing catalog: SeisBench Picker")
    BaseCatalog = OGSCatalog(
      pre / "OGSCatalog" / ".all",
      start=start,
      end=end,
      name="OGS NLL 1D",
      output=pre / target,
    )
    TargetCatalog = OGSCatalog(
      pre / target / "SeisBenchPicker",
      start=start,
      end=end,
      name=model,
    )
    BaseCatalog.bpgma(
      TargetCatalog,
      stations=stations,
      waveforms=waveforms,
      vlines=special_days,
    )
  for target in [
    "GaMMA0.1", "GaMMA0.2", "GaMMA0.3",
  ]:
    print(f"Processing target: {target}")
    BaseCatalog = OGSCatalog(
      pre / "OGSCatalog" / ".all",
      start=start,
      end=end,
      name="OGS NLL 1D",
      output=pre / target,
    )
    for name, path in [
      ("PhaseNet[INSTANCE, 0.1]", pre / target / "SeisBenchPicker"),
      ("PhaseNet[INSTANCE, 0.1] | GaMMA", pre / target / "OGSPickStatQC"),
      ("PhaseNet[INSTANCE, 0.1] | GaMMA | NLL 1D", pre / target / "OGSLocalMagnitude"),
    ]:
      print(f"Processing catalog: {name}")
      TargetCatalog = OGSCatalog(
        path,
        start=start,
        end=end,
        name=name,
      )
      BaseCatalog.plot([TargetCatalog], vlines=special_days)
      BaseCatalog.bpgma(
        TargetCatalog,
        stations=stations,
        waveforms=waveforms,
        vlines=special_days,
      )
  for target in [
    "PyOcto0.1", "PyOcto0.2", "PyOcto0.3",
  ]:
    print(f"Processing target: {target}")
    BaseCatalog = OGSCatalog(
      pre / "OGSCatalog" / ".all",
      start=start,
      end=end,
      name="OGS NLL 1D",
      output=pre / target,
      verbose=False,
    )
    for name, path in [
      ("PhaseNet[INSTANCE, 0.1]", pre / target / "SeisBenchPicker"),
      ("PhaseNet[INSTANCE, 0.1] | PyOcto", pre / target / "OGSPickStatQC"),
      ("PhaseNet[INSTANCE, 0.1] | PyOcto | NLL 1D", pre / target / "OGSLocalMagnitude"),
    ]:
      print(f"Processing catalog: {name}")
      TargetCatalog = OGSCatalog(
        path,
        start=start,
        end=end,
        name=name,
        output= pre / target,
        verbose=False,
      )
      BaseCatalog.plot([TargetCatalog], vlines=special_days)
      BaseCatalog.bpgma(
        TargetCatalog,
        stations=stations,
        waveforms=waveforms,
        vlines=special_days,
      )
      #TargetCatalog.bpgma(
      #  BaseCatalog,
      #  stations=stations,
      #  waveforms=waveforms,
      #  vlines=special_days,
      #)


if __name__ == "__main__":
  main()
