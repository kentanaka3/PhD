import os
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

import ogsconstants as OGS_C

def main():
  start = datetime.strptime("240327", OGS_C.YYMMDD_FMT)
  end = datetime.strptime("240327", OGS_C.YYMMDD_FMT)
  start = datetime.strptime("240320", OGS_C.YYMMDD_FMT)
  end = datetime.strptime("240620", OGS_C.YYMMDD_FMT)
  #waveforms = Path("/Users/admin/Desktop/OGS_Catalog/waveforms")
  waveforms = Path("/Volumes/Expansion/KEN/waveforms")
  stations = Path("/Users/admin/Desktop/OGS_Catalog/station")
  #stations = Path("/Volumes/Expansion/KEN/station")
  # Parse the files
  for target in [
    #"OGSBackup", "TP0.2S0.2", "TP0.3S0.3",
    #"OGSPyOcto", "OGSPyOcto_TP0.2S0.2", "OGSPyOcto_TP0.3S0.3",
    #"OGSPhaseNet_ORIGINAL", "OGSPhaseNet_SCEDC", "OGSPhaseNet_STEAD",
    #"OGSEQTransformer_ORIGINAL", #"OGSEQTransformer_SCEDC", "OGSEQTransformer_STEAD",
    "OGSEQTransformer_INSTANCE" #"OGSEQTransformer_TP0.2S0.2", "OGSEQTransformer_TP0.3S0.3",
  ]:
    print(f"Processing target: {target}")
    for base, name, path in [
      (".all", "SeisBench Picker", Path(f"/Users/admin/Desktop/OGS_Catalog/catalogs/{target}/SeisBenchPicker")),
      (".all", "GaMMA Catalog", Path(f"/Users/admin/Desktop/OGS_Catalog/catalogs/{target}/OGSPickStatQC")),
      (".all", "NLL Catalog", Path(f"/Users/admin/Desktop/OGS_Catalog/catalogs/{target}/OGSLocalMagnitude")),
    ]:
      print(f"Processing catalog: {name}")
      BaseCatalog = OGS_C.OGSCatalog(
        Path(f"/Users/admin/Desktop/OGS_Catalog/catalogs/OGSCatalog/{base}"),
        start=start,
        end=end,
        name="OGS Catalog"
      )
      TargetCatalog = OGS_C.OGSCatalog(
        path,
        start=start,
        end=end,
        name=name
      )
      BaseCatalog.bpgma(TargetCatalog, stations=OGS_C.inventory(stations))
      #BaseCatalog.plot(
      #  TargetCatalog,
      #  waveforms=OGS_C.waveforms(waveforms, start, end))
    shutil.copytree(Path(__file__).parent.parent / "img",
                    Path(__file__).parent.parent / target, dirs_exist_ok=True)
    shutil.rmtree(Path(__file__).parent.parent / "img")
    os.makedirs(Path(__file__).parent.parent / "img")

if __name__ == "__main__":
  main()
