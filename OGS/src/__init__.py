"""
=============================================================================
OGS Seismic Analysis Toolkit - Package Initialization
=============================================================================

OVERVIEW:
Top-level package initialization for the OGS seismological data processing and
research toolkit. Integrates end-to-end workflows for:
  - FDSN continuous seismic waveform acquisition (ObsPy & Pyrocko backends).
  - Legacy bulletin and catalog parsing (.dat, .hpl, .pun, .txt formats).
  - High-performance Pyrocko Squirrel day-sharded waveform access.
  - Deep-learning phase picking (PhaseNet, EQTransformer via SeisBench).
  - Phase association and hypocenter location (GaMMA, PyOcto, REAL, NonLinLoc).
  - BPGMA bipartite graph matching for catalog benchmarking.
  - Advanced seismic sequence clustering (DBSCAN, K-Means, ADP, PAk).
  - Scientific publication-grade visualization and mapping.

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
