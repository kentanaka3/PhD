from pathlib import Path
from datetime import datetime

import ogsconstants as OGS_C

def main():
  start = datetime.strptime("240320", OGS_C.YYMMDD_FMT)
  end = datetime.strptime("240620", OGS_C.YYMMDD_FMT)
  # Parse the files
  BaseCatalog = OGS_C.OGSCatalog(
    Path("/Users/admin/Desktop/OGS_Catalog/catalogs/OGSCatalog/.txt"),
    start=start,
    end=end,
    name="OGS Catalog"
  )
  TargetCatalog = OGS_C.OGSCatalog(
    #Path("/Users/admin/Desktop/OGS_Catalog/catalogs/OGS/OGSGammaAssociator"),
    Path("/Users/admin/Desktop/OGS_Catalog/catalogs/OGS/OGSLocalMagnitude"),
    start=start,
    end=end,
    name="OGS Local Magnitude Catalog"
  )
  stations = OGS_C.inventory(Path("/Users/admin/Desktop/OGS_Catalog/stations"))
  BaseCatalog.bpgma(TargetCatalog)
  BaseCatalog.plot(TargetCatalog)

if __name__ == "__main__":
  main()
