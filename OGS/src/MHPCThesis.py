import os
import shutil
from pathlib import Path
from datetime import datetime

import ogsconstants as OGS_C
from ogscatalog import OGSCatalog

def main():
  start = datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT)
  end = datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT)
  special_days = [
    (datetime(2024, 3, 27), "$M_L$ 4.6", "r"),
    (datetime(2024, 5, 22), "IT", OGS_C.ALN_GREEN)]
  waveforms = Path("/Volumes/Expansion/KEN/waveforms")
  stations = Path("/Users/admin/Desktop/OGS_Catalog/station")
  # Parse the files
  for targetname, target in [
    ("PhaseNet[Original]", "OGSPhaseNet_ORIGINAL"),
    ("PhaseNet[SCEDC]", "OGSPhaseNet_SCEDC"),
    ("PhaseNet[STEAD]", "OGSPhaseNet_STEAD"),
    ("EQTransformer[Original]", "OGSEQTransformer_ORIGINAL"),
    ("EQTransformer[SCEDC]", "OGSEQTransformer_SCEDC"),
    ("EQTransformer[STEAD]", "OGSEQTransformer_STEAD"),
  ]:
    print(f"Processing target: {target}")
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
      name=targetname,
      verbose=True,
    )
    BaseCatalog.bpgma(
      TargetCatalog,
      stations=stations,
      waveforms=waveforms,
      vlines=special_days,
    )

  for targetname, target in [
    ("PhaseNet[INSTANCE]", "OGSBackup"),
    ("PhaseNet[INSTANCE]", "TP0.2S0.2"),
    ("PhaseNet[INSTANCE]", "TP0.3S0.3"),
    ("EQTransformer[INSTANCE]", "OGSEQTransformer_INSTANCE"),
    ("EQTransformer[INSTANCE]", "OGSEQTransformer_INSTANCE_TP0.2S0.2"),
    ("EQTransformer[INSTANCE]", "OGSEQTransformer_INSTANCE_TP0.3S0.3"),
  ]:
    print(f"Processing target: {target}")
    for base, name, path in [
      (".all", "", Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/SeisBenchPicker")),
      (".all", " — GaMMA", Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/OGSPickStatQC")),
      (".all", " — GaMMA — NLL1D", Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/OGSLocalMagnitude")),
    ]:
      print(f"Processing catalog: {name}")
      BaseCatalog = OGSCatalog(
        Path(f"/Users/admin/Desktop/Monica/PhD/catalog/OGSCatalog/{base}"),
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
        name=targetname + name,
        verbose=True,
      )
      BaseCatalog.plot(others=[TargetCatalog], vlines=special_days)
      BaseCatalog.bpgma(
        TargetCatalog,
        stations=stations,
        waveforms=waveforms,
        vlines=special_days,
      )

  for target in [
    "OGSPyOcto", "OGSPyOcto_TP0.2S0.2", "OGSPyOcto_TP0.3S0.3",
  ]:
    print(f"Processing target: {target}")
    for base, name, path in [
      (".all", "PhaseNet[INSTANCE] — PyOcto", Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/OGSPickStatQC")),
      (".all", "PhaseNet[INSTANCE] — PyOcto — NLL1D", Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/OGSLocalMagnitude")),
    ]:
      print(f"Processing catalog: {name}")
      BaseCatalog = OGSCatalog(
        Path(f"/Users/admin/Desktop/Monica/PhD/catalog/OGSCatalog/{base}"),
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
      BaseCatalog.plot(others=[TargetCatalog], vlines=special_days)
      BaseCatalog.bpgma(
        TargetCatalog,
        stations=stations,
        waveforms=waveforms,
        vlines=special_days,
      )

  for targetname, target in [
    ("PhaseNet[INSTANCE]  — {} — NLL1D", "OGSLocalMagnitude"),
    ("PhaseNet[INSTANCE]  — {}", "OGSPickStatQC"),
  ]:
    print(f"Processing target: {target}")
    BaseCatalog = OGSCatalog(
      Path(f"/Users/admin/Desktop/Monica/PhD/catalog/OGSBackup/{target}"),
      start=start,
      end=end,
      name=targetname.format("GaMMA"),
      output=Path(f"/Users/admin/Desktop/MHPCThesis/imgs/OGSBackup/OGSPyOcto"),
      verbose=True,
    )
    TargetCatalog = OGSCatalog(
      Path(f"/Users/admin/Desktop/Monica/PhD/catalog/OGSPyOcto/{target}"),
      start=start,
      end=end,
      name=targetname.format("PyOcto"),
      verbose=True,
    )
    BaseCatalog.plot(others=[TargetCatalog], vlines=special_days)
    BaseCatalog.bpgma(
      TargetCatalog,
      stations=stations,
      waveforms=waveforms,
      vlines=special_days,
    )

if __name__ == "__main__":
  main()
