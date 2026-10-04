"""Public OGS classes, loaded on demand with their runtime dependencies."""

from importlib import import_module as _import_module
from typing import Any, TYPE_CHECKING


_EXPORTS = {
    "OGSCatalog": "ogscatalog",
    "OGSDataFile": "ogsdatafile",
    "DataCatalog": "ogsparser",
    "DataFileDAT": "ogsdat",
    "DataFileHPL": "ogshpl",
    "DataFilePUN": "ogspun",
    "DataFileTXT": "ogstxt",
    "OGSClusteringZoo": "ogsclustering",
    "OGSSequence": "ogssequence",
    "OGSTrainer": "ogstrainer",
    "BaseDownloader": "ogsdownloader",
    "ObsPyDownloader": "ogsdownloader",
    "PyrockoDownloader": "ogsdownloader",
    "OGSSquirrelDataSource": "ogsdata",
    "OGSAmplitudeExtractor": "ogspicker",
    "OGSLocalMagnitude": "ogsmagnitude",
    "OGSPickStatQC": "ogsqc",
    "OGSEventStatQC": "ogsqc",
    "OGSNonLinLoc": "ogsnonlinloc",
    "REALAssociator": "real",
    "OGSCatalogBuilderMPI": "ogsbuilderMPI",
}

__all__ = list(_EXPORTS)

if TYPE_CHECKING:
  from .src.ogscatalog import OGSCatalog
  from .src.ogsdatafile import OGSDataFile
  from .src.ogsparser import DataCatalog
  from .src.ogsdat import DataFileDAT
  from .src.ogshpl import DataFileHPL
  from .src.ogspun import DataFilePUN
  from .src.ogstxt import DataFileTXT
  from .src.ogsclustering import OGSClusteringZoo
  from .src.ogssequence import OGSSequence
  from .src.ogstrainer import OGSTrainer
  from .src.ogsdownloader import BaseDownloader, ObsPyDownloader, PyrockoDownloader
  from .src.ogsdata import OGSSquirrelDataSource
  from .src.ogspicker import OGSAmplitudeExtractor
  from .src.ogsmagnitude import OGSLocalMagnitude
  from .src.ogsqc import OGSPickStatQC, OGSEventStatQC
  from .src.ogsnonlinloc import OGSNonLinLoc
  from .src.real import REALAssociator
  from .src.ogsbuilderMPI import OGSCatalogBuilderMPI


def __getattr__(name: str) -> Any:
  module_name = _EXPORTS.get(name)
  if module_name is None:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  value = getattr(_import_module(f".src.{module_name}", __name__), name)
  globals()[name] = value
  return value


def __dir__() -> list[str]:
  return sorted(set(globals()) | set(__all__))
