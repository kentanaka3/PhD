import os
import shutil
from pathlib import Path
from datetime import datetime

import ogsconstants as OGS_C
from ogscatalog import OGSCatalog

def main():
  start = datetime.strptime("20240320", OGS_C.YYYYMMDD_FMT)
  end = datetime.strptime("20240620", OGS_C.YYYYMMDD_FMT)
  #waveforms = Path("/Users/admin/Desktop/OGS_Catalog/waveforms")
  waveforms = Path("/Volumes/Expansion/KEN/waveforms")
  stations = Path("/Users/admin/Desktop/OGS_Catalog/station")
  #stations = Path("/Volumes/Expansion/KEN/station")
  # Parse the files
  for target in [
    "OGSBackup", "TP0.2S0.2", "TP0.3S0.3",
    "OGSPyOcto", "OGSPyOcto_TP0.2S0.2", "OGSPyOcto_TP0.3S0.3",
    "OGSPhaseNet_ORIGINAL", "OGSPhaseNet_SCEDC", "OGSPhaseNet_STEAD",
    "OGSEQTransformer_ORIGINAL", "OGSEQTransformer_SCEDC", "OGSEQTransformer_STEAD",
    "OGSEQTransformer_INSTANCE", "OGSEQTransformer_INSTANCE_TP0.2S0.2", "OGSEQTransformer_INSTANCE_TP0.3S0.3",
  ]:
    print(f"Processing target: {target}")
    for base, name, path in [
      (".all", "SeisBench Picker", Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/SeisBenchPicker")),
      (".all", "GaMMA Catalog", Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/OGSPickStatQC")),
      (".all", "NLL Catalog", Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/OGSLocalMagnitude")),
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
      #BaseCatalog.plot(
      #  TargetCatalog,
      #  waveforms=OGS_C.waveforms(waveforms, start, end))
      BaseCatalog.bpgma(
        TargetCatalog,
        stations=OGS_C.inventory(
          stations,
          output=BaseCatalog.output / "img"
        )
      )

if __name__ == "__main__":
  main()
