import os
import shutil
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
  # Parse the files
  for model, target in [
    ("PhaseNet[Original]", "OGSPhaseNet_ORIGINAL"),
    ("PhaseNet[SCEDC]", "OGSPhaseNet_SCEDC"),
    ("PhaseNet[STEAD]", "OGSPhaseNet_STEAD"),
    ("EQTransformer[Original]", "OGSEQTransformer_ORIGINAL"),
    ("EQTransformer[SCEDC]", "OGSEQTransformer_SCEDC"),
    ("EQTransformer[STEAD]", "OGSEQTransformer_STEAD"),
    ("EQTransformer[INSTANCE]", "OGSEQTransformer_INSTANCE"),
    ("EQTransformer[INSTANCE]", "OGSEQTransformer_INSTANCE_TP0.2S0.2"),
    ("EQTransformer[INSTANCE]", "OGSEQTransformer_INSTANCE_TP0.3S0.3"),
  ]:
    print(f"Processing target: {target}")
    print(f"Processing catalog: SeisBench Picker")
    BaseCatalog = OGSCatalog(
      Path(f"/Users/admin/Desktop/Monica/PhD/catalog/OGSCatalog/.all"),
      start=start,
      end=end,
      name="OGS Catalog",
      output=Path(f"/Users/admin/Desktop/MHPCThesis/imgs/OGSCatalog/{target}"),
      verbose=True,
    )
    TargetCatalog = OGSCatalog(
      Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/SeisBenchPicker"),
      start=start,
      end=end,
      name=model,
      verbose=True,
    )
    BaseCatalog.bpgma(
      TargetCatalog,
      stations=stations,
      waveforms=waveforms,
      vlines=special_days,
    )
  for target in [
    #"OGSBackup", "TP0.2S0.2", "TP0.3S0.3",
  ]:
    print(f"Processing target: {target}")
    for name, path in [
      ("PhaseNet[INSTANCE]", Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/SeisBenchPicker")),
      ("PhaseNet[INSTANCE] | GaMMA", Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/OGSPickStatQC")),
      ("PhaseNet[INSTANCE] | GaMMA | NLL 1D", Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/OGSLocalMagnitude")),
    ]:
      print(f"Processing catalog: {name}")
      BaseCatalog = OGSCatalog(
        Path(f"/Users/admin/Desktop/Monica/PhD/catalog/OGSCatalog/.all"),
        start=start,
        end=end,
        name="OGS Catalog",
        output=Path(f"/Users/admin/Desktop/MHPCThesis/imgs/OGSCatalog/{target}"),
        verbose=True,
      )
      TargetCatalog = OGSCatalog(
        path,
        start=start,
        end=end,
        name=name,
        verbose=True,
      )
      BaseCatalog.plot([TargetCatalog], vlines=special_days)
      BaseCatalog.bpgma(
        TargetCatalog,
        stations=stations,
        waveforms=waveforms,
        vlines=special_days,
      )
  for target in [
    "OGSPyOcto", "OGSPyOcto_TP0.2S0.2", "OGSPyOcto_TP0.3S0.3",
  ]:
    for name, path in [
      ("PhaseNet[INSTANCE]", Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/SeisBenchPicker")),
      ("PhaseNet[INSTANCE] | PyOcto", Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/OGSPickStatQC")),
      ("PhaseNet[INSTANCE] | PyOcto | NLL 1D", Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/OGSLocalMagnitude")),
    ]:
      print(f"Processing target: {target}")
      print(f"Processing catalog: {name}")
      BaseCatalog = OGSCatalog(
        Path(f"/Users/admin/Desktop/Monica/PhD/catalog/OGSCatalog/.all"),
        start=start,
        end=end,
        name="OGS Catalog",
        output=Path(f"/Users/admin/Desktop/MHPCThesis/imgs/OGSCatalog/{target}"),
        verbose=True,
      )
      TargetCatalog = OGSCatalog(
        path,
        start=start,
        end=end,
        name=name,
        verbose=True,
      )
      BaseCatalog.plot([TargetCatalog], vlines=special_days)
      BaseCatalog.bpgma(
        TargetCatalog,
        stations=stations,
        waveforms=waveforms,
        vlines=special_days,
      )

if __name__ == "__main__":
  main()
