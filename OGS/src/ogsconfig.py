"""
===============================================================================
OGS Config Engine - Native YAML Configuration and Class Instantiation
===============================================================================

OVERVIEW:
Replaces Hydra with an in-house YAML configuration resolver tailored for OGS:
  1. DEFAULTS HIERARCHY: Recursively parses and merges `defaults:` lists across
     nested configuration directories (conf/data, conf/group_modules, etc.).
  2. CLI OVERRIDES: Applies dot-notation overrides from command-line arguments
     (e.g., `group_modules=Picker`, `data.starttime=2005-01-01`, `builder.outputs=[picks]`).
  3. DYNAMIC INSTANTIATION: Dynamically resolves and instantiates classes/callables
     specified by `_target_` strings (e.g., `_target_: OGS.src.ogspicker.OGSSeisBenchPicker`).
  4. TYPE COERCION: Coerces strings to int, float, bool, None, and lists.

POLYMORPHIC UNIFICATION:
Anchored in Tier-0 modules: uses `OGS_U.setup_logger` and aligns with `OGS_C` constants.

AUTHORS:
  - 健
  - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
    Centro di Ricerche Sismologiche (CRS)
  - Università degli Studi di Trieste (UniTS)
    Dipartimento di Matematica, Informatica e Geoscienze (MIGe)
    Applied Data Science and Artificial Intelligence (ADSAI)
  - Terabit Network for Research and Academic Big Data in Italy (TeRABIT)
    Consorzio Interuniversitario del Nord-Est per il Calcolo Automatico (CINECA)
    
===============================================================================
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any, Optional
import yaml

from . import ogsconstants as OGS_C, ogsutils as OGS_U

logger = OGS_U.setup_logger("OGSConfig")


class OGSConfig(dict):
  """Dictionary subclass allowing dot-attribute access (e.g. cfg.output_path)."""

  def __getattr__(self, key: str) -> Any:
    try:
      val = self[key]
      if isinstance(val, dict) and not isinstance(val, OGSConfig):
        val = OGSConfig(val)
        self[key] = val
      elif isinstance(val, list):
        val = [
            OGSConfig(item)
            if isinstance(item, dict) and not isinstance(item, OGSConfig)
            else item
            for item in val
        ]
        self[key] = val
      return val
    except KeyError:
      raise AttributeError(f"OGSConfig has no attribute '{key}'")

  def __setattr__(self, key: str, value: Any) -> None:
    self[key] = value

  def __delattr__(self, key: str) -> None:
    try:
      del self[key]
    except KeyError:
      raise AttributeError(f"OGSConfig has no attribute '{key}'")


def _coerce_value(val_str: str) -> Any:
  val_str = val_str.strip()
  if val_str.lower() in ("true", "yes", "on"):
    return True
  if val_str.lower() in ("false", "no", "off"):
    return False
  if val_str.lower() in ("none", "null"):
    return None
  if (val_str.startswith("[") and val_str.endswith("]")) or (
      val_str.startswith("{") and val_str.endswith("}")
  ):
    try:
      return ast.literal_eval(val_str)
    except Exception:
      inner = val_str[1:-1].strip()
      if not inner:
        return []
      return [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
  try:
    if "." in val_str or "e" in val_str.lower():
      return float(val_str)
    return int(val_str)
  except ValueError:
    return val_str


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
  for k, v in source.items():
    if k in target and isinstance(target[k], dict) and isinstance(v, dict):
      _deep_merge(target[k], v)
    else:
      target[k] = v
  return target


def _set_dot_path(d: dict[str, Any], path: str, value: Any) -> None:
  parts = path.split(".")
  curr = d
  for part in parts[:-1]:
    if part not in curr or not isinstance(curr[part], dict):
      curr[part] = {}
    curr = curr[part]
  curr[parts[-1]] = value


def load_yaml(file_path: Path) -> dict[str, Any]:
  if not file_path.is_file():
    raise FileNotFoundError(f"Configuration file not found: {file_path}")
  with open(file_path, "r", encoding="utf-8") as f:
    content = yaml.safe_load(f)
  return content if isinstance(content, dict) else {}


def resolve_defaults(
    config_dir: Path,
    config_data: dict[str, Any],
    top_overrides: dict[str, str],
    prefix: str = "",
) -> dict[str, Any]:
  defaults = config_data.pop("defaults", [])
  merged: dict[str, Any] = {}
  self_merged = False

  for entry in defaults:
    if entry == "_self_":
      _deep_merge(merged, config_data)
      self_merged = True
      continue

    if isinstance(entry, dict):
      for group_key, target_file in entry.items():
        lookup_key = f"{prefix}{group_key}" if prefix else group_key
        slash_key = f"{lookup_key}/{target_file}"
        effective_file = top_overrides.get(
            slash_key,
            top_overrides.get(
                f"{group_key}/{target_file}",
                top_overrides.get(
                    target_file,
                    top_overrides.get(
                        lookup_key, top_overrides.get(group_key, target_file)
                    ),
                ),
            ),
        )
        if not effective_file.endswith(".yaml"):
          effective_file += ".yaml"
        sub_path = config_dir / group_key / effective_file
        sub_data = load_yaml(sub_path)
        sub_resolved = resolve_defaults(
            config_dir / group_key,
            sub_data,
            top_overrides,
            prefix=f"{lookup_key}/",
        )
        if group_key in merged and isinstance(merged[group_key], dict):
          _deep_merge(merged[group_key], sub_resolved)
        else:
          merged[group_key] = sub_resolved

    elif isinstance(entry, str):
      lookup_key = f"{prefix}{entry}" if prefix else entry
      effective_name = top_overrides.get(
          lookup_key, top_overrides.get(entry, entry))
      sub_file = (
          effective_name if effective_name.endswith(
              ".yaml") else f"{effective_name}.yaml"
      )
      sub_path = config_dir / sub_file
      sub_data = load_yaml(sub_path)
      sub_resolved = resolve_defaults(
          config_dir, sub_data, top_overrides, prefix=prefix
      )
      _deep_merge(merged, sub_resolved)

  if not self_merged:
    _deep_merge(merged, config_data)

  return merged


def load_config(
    config_dir: str | Path,
    config_name: str = "config.yaml",
    overrides: Optional[list[str]] = None,
) -> OGSConfig:
  config_dir = Path(config_dir).resolve()
  main_config_path = config_dir / (
      config_name if config_name.endswith(".yaml") else f"{config_name}.yaml"
  )

  overrides = overrides or []
  top_overrides: dict[str, str] = {}
  dot_overrides: list[tuple[str, Any]] = []

  root_data = load_yaml(main_config_path)
  defaults_groups = set()
  for item in root_data.get("defaults", []):
    if isinstance(item, dict):
      defaults_groups.update(item.keys())
    elif isinstance(item, str) and item != "_self_":
      defaults_groups.add(item)

  for arg in overrides:
    has_plus = arg.startswith("+")
    arg = arg.lstrip("+")
    if "=" not in arg:
      continue
    key, val = arg.split("=", 1)
    key, val = key.strip(), val.strip()

    if "/" in key:
      # Slash notation for nested defaults override (e.g. group_modules/picker=file.yaml)
      top_overrides[key] = val
      parts = key.split("/")
      top_overrides[parts[-1]] = val
    elif "." in key or has_plus:
      # Dot notation or explicit new parameter (+param=val)
      dot_overrides.append((key, _coerce_value(val)))
    else:
      # Top-level group selection override or top-level parameter
      top_overrides[key] = val
      if key not in defaults_groups:
        dot_overrides.append((key, _coerce_value(val)))

  resolved_data = resolve_defaults(config_dir, root_data, top_overrides)

  for path, val in dot_overrides:
    _set_dot_path(resolved_data, path, val)

  return OGSConfig(resolved_data)


def instantiate(config: Any, **kwargs: Any) -> Any:
  if isinstance(config, (list, tuple)):
    return [instantiate(item, **kwargs) for item in config]

  if not isinstance(config, dict):
    return config

  params = dict(config)
  target_str = params.pop("_target_", None)

  instantiated_params = {}
  for k, v in params.items():
    instantiated_params[k] = instantiate(v)

  instantiated_params.update(kwargs)

  if target_str is None:
    return OGSConfig(instantiated_params)

  try:
    if ":" in target_str:
      mod_name, obj_name = target_str.split(":", 1)
      mod = importlib.import_module(mod_name)
      obj = getattr(mod, obj_name)
    else:
      # Iteratively find the deepest valid importable module from right to left
      parts = target_str.split(".")
      obj = None
      for i in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:i])
        try:
          obj = importlib.import_module(mod_name)
          for attr in parts[i:]:
            obj = getattr(obj, attr)
          break
        except (ModuleNotFoundError, AttributeError):
          if i == 1:
            raise

    if obj is None:
      raise ValueError(f"Could not resolve target '{target_str}'")

    return obj(**instantiated_params)
  except Exception as e:
    logger.error(f"Failed to instantiate target '{target_str}': {e}")
    raise
