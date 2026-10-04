"""Isolated import regressions; no catalog data or pipeline stages are run."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "OGS" / "src"
PUBLIC_CLASSES = {
    "OGSCatalog", "OGSDataFile", "DataCatalog", "DataFileDAT",
    "DataFileHPL", "DataFilePUN", "DataFileTXT", "OGSClusteringZoo",
    "OGSSequence", "OGSTrainer", "BaseDownloader", "ObsPyDownloader",
    "PyrockoDownloader", "OGSSquirrelDataSource", "OGSAmplitudeExtractor",
    "OGSLocalMagnitude", "OGSPickStatQC", "OGSEventStatQC", "OGSNonLinLoc",
    "REALAssociator", "OGSCatalogBuilderMPI",
}
LAYOUTS = (
    ("OGS.src.", ROOT),
    ("src.", ROOT / "OGS"),
    ("", SRC),
)


class TestPackageImports(unittest.TestCase):
  def _run(self, code: str, path: Path = ROOT) -> str:
    bootstrap = f"""
import ctypes
from pathlib import Path
import sys

library = Path(sys.executable).resolve().parents[1] / "lib/libstdc++.so.6"
if library.is_file():
    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
sys.path.insert(0, {str(path)!r})
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(bootstrap)
         + textwrap.dedent(code)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=180,
    )
    self.assertEqual(
        result.returncode, 0,
        f"Isolated Python failed:\n{result.stdout}\n{result.stderr}",
    )
    return result.stdout

  def test_all_modules_in_each_layout(self):
    modules = sorted(
        p.stem for p in SRC.glob("*.py") if p.name != "__init__.py"
    )
    for prefix, path in LAYOUTS:
      with self.subTest(layout=prefix or "standalone"):
        output = self._run(f"""
import importlib
import importlib.util
import json

results = []
for name in {modules!r}:
    if name == "ogsbuilderMPI" and importlib.util.find_spec("dask_mpi") is None:
        results.append([name, "dask_mpi is not installed"])
        continue
    module = importlib.import_module({prefix!r} + name)
    assert module.__package__ == {prefix.rstrip('.')!r}, module.__name__
    for alias, sibling in (("OGS_C", "ogsconstants"), ("OGS_U", "ogsutils")):
        if hasattr(module, alias):
            assert getattr(module, alias).__name__ == {prefix!r} + sibling
    results.append([name, None])
print("IMPORT_RESULTS=" + json.dumps(results))
""", path)
        results = json.loads(next(
            line.removeprefix("IMPORT_RESULTS=")
            for line in output.splitlines()
            if line.startswith("IMPORT_RESULTS=")
        ))
        self.assertEqual([name for name, _ in results], modules)
        for name, reason in results:
          with self.subTest(module=name):
            if reason:
              self.skipTest(reason)

  def test_public_exports_are_lazy_and_cached(self):
    self._run(f"""
import importlib
import importlib.util
import OGS

assert set(OGS.__all__) == {PUBLIC_CLASSES!r}
assert set(OGS.__all__) <= set(dir(OGS))
assert not any(name.startswith("OGS.src") for name in sys.modules)
for name in OGS.__all__:
    if name == "OGSCatalogBuilderMPI" and importlib.util.find_spec("dask_mpi") is None:
        try:
            getattr(OGS, name)
        except ModuleNotFoundError as exc:
            assert exc.name == "dask_mpi", exc
        else:
            raise AssertionError("Missing MPI dependency was hidden")
        assert name not in vars(OGS)
        continue
    value = getattr(OGS, name)
    assert isinstance(value, type), name
    assert value.__name__ == name, name
    owner = importlib.import_module("OGS.src." + OGS._EXPORTS[name])
    assert value is getattr(owner, name), name
    assert vars(OGS)[name] is value
    assert getattr(OGS, name) is value
""")

  def test_unknown_attributes_and_dependency_errors(self):
    self._run("""
from unittest.mock import patch
import OGS

try:
    OGS.not_a_public_class
except AttributeError as exc:
    assert "not_a_public_class" in str(exc)
else:
    raise AssertionError("Unknown attributes must raise AttributeError")

failure = ModuleNotFoundError("Synthetic dependency failure", name="dependency")
with patch("OGS._import_module", side_effect=failure):
    try:
        OGS.OGSCatalog
    except ModuleNotFoundError as exc:
        assert exc is failure
    else:
        raise AssertionError("Dependency errors must propagate")
assert "OGSCatalog" not in vars(OGS)
from OGS import OGSCatalog
assert OGSCatalog.__module__ == "OGS.src.ogscatalog"
""")

  def test_lazy_catalog_plotting_in_each_layout(self):
    for prefix, path in LAYOUTS:
      with self.subTest(layout=prefix or "standalone"):
        self._run(f"""
from importlib import import_module
from tempfile import TemporaryDirectory
import pandas as pd

module = import_module({prefix!r} + "ogscatalog")
assert {prefix!r} + "ogsplotter" not in sys.modules
catalog = module.OGSCatalog.__new__(module.OGSCatalog)
catalog.name = "Synthetic catalog"
events = pd.DataFrame({{module.OGS_C.DEPTH_STR: [1.0, 2.0, 3.0]}})
catalog.get = lambda kind: events
with TemporaryDirectory() as directory:
    output = Path(directory) / "depth.png"
    catalog.plot_depth_histogram(bins=3, output=output)
    assert output.read_bytes().startswith(b"\\x89PNG\\r\\n\\x1a\\n")
assert {prefix!r} + "ogsplotter" in sys.modules
""", path)

  def test_safe_cli_help_in_package_and_script_modes(self):
    modules = ("ogsparser", "ogsdownloader", "ogstrainer", "ogsstation",
               "ogssequence", "ogsdat", "ogshpl", "ogspun", "ogstxt",
               "ogsgrid2time")
    for name in modules:
      with self.subTest(module=name):
        self._run(f"""
import runpy
from contextlib import redirect_stdout
from io import StringIO

outputs = []
for package_mode in (True, False):
    buffer = StringIO()
    sys.argv = [{name!r}, "--help"]
    with redirect_stdout(buffer):
        try:
            if package_mode:
                runpy.run_module("OGS.src." + {name!r}, run_name="__main__")
            else:
                sys.path.insert(0, {str(SRC)!r})
                runpy.run_path({str(SRC / (name + '.py'))!r}, run_name="__main__")
        except SystemExit as exc:
            assert exc.code == 0, exc
        else:
            raise AssertionError("--help must exit before application work")
    outputs.append(buffer.getvalue())
assert all("usage:" in output.lower() for output in outputs)
assert outputs[0] == outputs[1]
""")


if __name__ == "__main__":
  unittest.main()
