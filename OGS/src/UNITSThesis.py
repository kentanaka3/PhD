import os
import shutil
from pathlib import Path
from datetime import datetime

import ogsconstants as OGS_C
from ogscatalog import OGSCatalog

def main():
  year = 21
  start = datetime.strptime(f"20{year}0101", OGS_C.YYYYMMDD_FMT)
  end = datetime.strptime(f"20{year}1231", OGS_C.YYYYMMDD_FMT)
  stations = Path("/Users/admin/Desktop/OGS_Catalog/station")
  # Parse the files
  for targetname, target in [
    ("PhaseNet[INSTANCE]", f"OGS{year}"),
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
        output=Path(f"/Users/admin/Desktop/UNITSThesis/imgs/OGSCatalog/{target}"),
        verbose=True,
      )
      TargetCatalog = OGSCatalog(
        path,
        start=start,
        end=end,
        name=targetname + name,
        verbose=True,
      )
      BaseCatalog.plot(others=[TargetCatalog])
      BaseCatalog.bpgma(
        TargetCatalog,
        stations=OGS_C.inventory(
          stations,
        )
      )

if __name__ == "__main__":
  main()
