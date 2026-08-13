"""Load config.yaml into a plain, attribute-accessible object.

Keeping this tiny and dependency-light: a recursive namespace wrapper so callers
can write ``cfg.preprocessing.deskew`` instead of ``cfg["preprocessing"]["deskew"]``
while still allowing ``cfg.to_dict()`` for serialisation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


class Config:
    """Recursive attribute/dict access wrapper around parsed YAML."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data
        for key, value in data.items():
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({list(self._data.keys())})"


def load_config(path: str | Path | None = None) -> Config:
    """Read the YAML config from ``path`` (defaults to the bundled config.yaml)."""
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return Config(data)
