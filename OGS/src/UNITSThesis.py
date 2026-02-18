import os
import shutil
from pathlib import Path
from datetime import datetime

import ogsconstants as OGS_C
from ogscatalog import OGSCatalog

def main():
  stations = Path("/Users/admin/Desktop/OGS_Catalog/station")
  # Parse the files
  for target, start, end in [
    ("OGS20", datetime.strptime("20200101", OGS_C.YYYYMMDD_FMT), datetime.strptime("20201231", OGS_C.YYYYMMDD_FMT)),
    ("OGS21", datetime.strptime("20210101", OGS_C.YYYYMMDD_FMT), datetime.strptime("20211231", OGS_C.YYYYMMDD_FMT)),
  ]:
    print(f"Processing target: {target}")
    for base, name, path in [
      #(".all", "PhaseNet[INSTANCE]", Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/SeisBenchPicker")),
      (".all", "PhaseNet[INSTANCE] | GaMMA", Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/OGSPickStatQC")),
      (".all", "PhaseNet[INSTANCE] | GaMMA | NLL 1D", Path(f"/Users/admin/Desktop/Monica/PhD/catalog/{target}/OGSLocalMagnitude")),
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
        name=name,
        verbose=True,
      )
      #BaseCatalog.plot([TargetCatalog])
      BaseCatalog.bpgma(
        TargetCatalog,
        stations=stations,
      )

if __name__ == "__main__":
  main()
