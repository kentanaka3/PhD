"""
=============================================================================
UniTS Thesis Driver - Multi-Year Catalog Evaluation & BPGMA Benchmarks
=============================================================================

OVERVIEW:
Reproducible driver script used for the UniTS PhD thesis. It benchmarks
machine-learning seismic catalog pipelines against the official OGS reference
catalogs across multi-year observation windows (2020-2021).

The pipeline evaluates:
  1. Base reference OGS catalogs for target years (OGS20, OGS21).
  2. Candidate deep-learning and automated processing stages:
     - PhaseNet[INSTANCE] + GaMMA associator + QC (OGSPickStatQC)
     - PhaseNet[INSTANCE] + GaMMA + NonLinLoc 1D + Local Magnitude (OGSLocalMagnitude)
  3. Bipartite graph matching assessment (BPGMA) comparing reference events
     and picks with automated ML outputs to quantify precision, recall, and
     location residuals.

NOTE:
Paths and station inventory directories are configured for the analysis
workstation; adjust paths before executing on cluster environments.

USAGE:
python UNITSThesis.py

DEPENDENCIES:
- ogsconstants: date formats and styling definitions
  - ogscatalog.OGSCatalog: catalog loading, BPGMA matching, and reporting

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
