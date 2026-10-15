"""
=============================================================================
OGS Config Test Suite - Unit Tests for Native Config Engine & Instantiation
=============================================================================

OVERVIEW:
Unit tests verifying the native YAML configuration engine (ogsconfig.py)
replacing Hydra:
  1. OGSConfig attribute access, mutation, deletion, and list wrapping.
  2. Type coercion for booleans, null/none, numbers, and lists.
  3. Hierarchical defaults resolution without clobbering and slash overrides.
  4. Dynamic class and class-method instantiation (PhaseNet.from_pretrained,
     OGSMerge).

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
=============================================================================
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml
from OGS.src import ogsconstants as OGS_C
from OGS.src.ogsbase import OGSMerge
from OGS.src.ogsconfig import (
    OGSConfig,
    _coerce_value,
    _deep_merge,
    _set_dot_path,
    instantiate,
    load_config,
    load_yaml,
    resolve_defaults,
)


class TestOGSConfig(unittest.TestCase):
  """Tests for OGSConfig dictionary subclass with dot-attribute access."""

  def test_attribute_access_and_mutation(self) -> None:
    cfg = OGSConfig({"param_a": 10, "nested": {"sub": "value"}})
    self.assertEqual(cfg.param_a, 10)
    self.assertEqual(cfg["param_a"], 10)

    # Mutation
    cfg.param_b = "test"
    self.assertEqual(cfg["param_b"], "test")
    self.assertEqual(cfg.param_b, "test")

    # Deletion
    del cfg.param_b
    self.assertNotIn("param_b", cfg)
    with self.assertRaises(AttributeError):
      _ = cfg.param_b

    # Missing attribute raises AttributeError
    with self.assertRaises(AttributeError):
      _ = cfg.nonexistent_key

  def test_nested_dict_wrapping(self) -> None:
    cfg = OGSConfig({"level1": {"level2": {"leaf": 42}}})
    # Dot access wraps intermediate dict into OGSConfig
    self.assertIsInstance(cfg.level1, OGSConfig)
    self.assertIsInstance(cfg.level1.level2, OGSConfig)
    self.assertEqual(cfg.level1.level2.leaf, 42)

  def test_list_wrapping(self) -> None:
    cfg = OGSConfig({
        "elements": [
            {"name": "first", "val": 1},
            {"name": "second", "val": 2},
            "scalar_item",
        ]
    })
    elements = cfg.elements
    self.assertIsInstance(elements[0], OGSConfig)
    self.assertEqual(elements[0].name, "first")
    self.assertEqual(elements[0].val, 1)
    self.assertIsInstance(elements[1], OGSConfig)
    self.assertEqual(elements[1].name, "second")
    self.assertEqual(elements[2], "scalar_item")


class TestTypeCoercion(unittest.TestCase):
  """Tests for _coerce_value parsing strings into typed Python objects."""

  def test_boolean_coercion(self) -> None:
    for truthy in ("true", "True", "TRUE", "yes", "on"):
      self.assertIs(_coerce_value(truthy), True)
    for falsy in ("false", "False", "FALSE", "no", "off"):
      self.assertIs(_coerce_value(falsy), False)

  def test_none_coercion(self) -> None:
    for null_val in ("none", "None", "null", "NULL"):
      self.assertIsNone(_coerce_value(null_val))

  def test_numeric_coercion(self) -> None:
    self.assertEqual(_coerce_value("42"), 42)
    self.assertIsInstance(_coerce_value("42"), int)
    self.assertEqual(_coerce_value("-100"), -100)
    self.assertAlmostEqual(_coerce_value("3.1415"), 3.1415)
    self.assertIsInstance(_coerce_value("3.1415"), float)
    self.assertAlmostEqual(_coerce_value("1e-4"), 1e-4)

  def test_list_coercion(self) -> None:
    self.assertEqual(_coerce_value("[1, 2, 3]"), [1, 2, 3])
    self.assertEqual(_coerce_value("['picks', 'events']"), ["picks", "events"])
    self.assertEqual(_coerce_value("[picks, events]"), ["picks", "events"])
    self.assertEqual(_coerce_value("[]"), [])

  def test_string_fallback(self) -> None:
    self.assertEqual(_coerce_value("SeisBenchPicker"), "SeisBenchPicker")
    self.assertEqual(_coerce_value("some/path/to/file"), "some/path/to/file")


class TestDefaultsResolutionAndOverrides(unittest.TestCase):
  """Tests for resolve_defaults, _deep_merge, and load_config overrides."""

  def test_deep_merge_preserves_unrelated(self) -> None:
    target = {"a": 1, "nested": {"x": 10, "y": 20}}
    source = {"b": 2, "nested": {"y": 30, "z": 40}}
    merged = _deep_merge(target, source)
    self.assertEqual(
        merged,
        {"a": 1, "b": 2, "nested": {"x": 10, "y": 30, "z": 40}},
    )

  def test_set_dot_path(self) -> None:
    d: dict[str, Any] = {}
    _set_dot_path(d, "data.starttime", "2020-01-01")
    _set_dot_path(d, "data.options.verbose", True)
    self.assertEqual(
        d,
        {
            "data": {
                "starttime": "2020-01-01",
                "options": {"verbose": True},
            }
        },
    )

  def test_resolve_defaults_hierarchical_and_slash_overrides(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_path = Path(tmpdir)
      (tmp_path / "group_modules").mkdir(parents=True)
      (tmp_path / "data").mkdir(parents=True)

      # Create default configs
      with open(tmp_path / "data" / "default_data.yaml", "w") as f:
        yaml.dump({"_target_": "MockData", "buffer": 60, "keep": True}, f)

      with open(tmp_path / "group_modules" / "picker.yaml", "w") as f:
        yaml.dump({"_target_": "MockPicker", "threshold": 0.5}, f)

      with open(tmp_path / "group_modules" / "custom_picker.yaml", "w") as f:
        yaml.dump({"_target_": "CustomPicker", "threshold": 0.9}, f)

      # Main config
      main_config = {
          "defaults": [
              {"data": "default_data"},
              {"group_modules": "picker"},
              "_self_",
          ],
          "output_path": "/tmp/out",
          "data": {"buffer": 120},  # overrides default_data buffer
      }
      with open(tmp_path / "config.yaml", "w") as f:
        yaml.dump(main_config, f)

      # Load without overrides: verifies defaults loaded without clobbering
      cfg = load_config(tmp_path, config_name="config.yaml")
      self.assertEqual(cfg.output_path, "/tmp/out")
      self.assertEqual(cfg.data._target_, "MockData")
      self.assertEqual(cfg.data.buffer, 120)  # overrode 60
      self.assertEqual(cfg.data.keep, True)   # preserved from default_data
      self.assertEqual(cfg.group_modules._target_, "MockPicker")
      self.assertEqual(cfg.group_modules.threshold, 0.5)

      # Load with slash override and dot override
      overrides = [
          "group_modules/picker=custom_picker.yaml",
          "data.buffer=300",
          "+extra_flag=true",
      ]
      cfg_overridden = load_config(
          tmp_path, config_name="config.yaml", overrides=overrides
      )
      self.assertEqual(cfg_overridden.group_modules._target_, "CustomPicker")
      self.assertEqual(cfg_overridden.group_modules.threshold, 0.9)
      self.assertEqual(cfg_overridden.data.buffer, 300)
      self.assertEqual(cfg_overridden.data.keep, True)
      self.assertEqual(cfg_overridden.extra_flag, True)


class TestInstantiate(unittest.TestCase):
  """Tests for instantiate resolving classes, callables, and class methods."""

  def test_instantiate_standard_class(self) -> None:
    config = {
        "_target_": "OGS.src.ogsbase.OGSMerge",
    }
    obj = instantiate(config)
    self.assertIsInstance(obj, OGSMerge)
    self.assertEqual(obj.name, "OGSMerge")

  def test_instantiate_class_method_with_mock(self) -> None:
    config = {
        "_target_": "seisbench.models.PhaseNet.from_pretrained",
        "pretrained": "original",
    }
    with patch("seisbench.models.PhaseNet.from_pretrained") as mock_from_pretrained:
      mock_from_pretrained.return_value = MagicMock(name="PhaseNetModel")
      res = instantiate(config)
      mock_from_pretrained.assert_called_once_with(pretrained="original")
      self.assertEqual(res, mock_from_pretrained.return_value)

  def test_instantiate_nested_and_lists(self) -> None:
    config = {
        "merger": {"_target_": "OGS.src.ogsbase.OGSMerge"},
        "modules": [
            {"_target_": "OGS.src.ogsbase.OGSMerge"},
        ],
        "plain_val": 42,
    }
    res = instantiate(config)
    self.assertIsInstance(res.merger, OGSMerge)
    self.assertIsInstance(res.modules, list)
    self.assertIsInstance(res.modules[0], OGSMerge)
    self.assertEqual(res.plain_val, 42)

  def test_instantiate_missing_target_raises(self) -> None:
    config = {"_target_": "non_existent_module.FakeClass"}
    with self.assertRaises(Exception):
      instantiate(config)


if __name__ == "__main__":
  unittest.main()
