from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Eye",
    "Detector",
    "Hand",
    "UnifiedOperation",
    "Visualizer",
]

_OPTIONAL_IMPORTS = {
    "Eye": ".Eye",
    "Detector": ".Detector",
    "Hand": ".Hand",
    "UnifiedOperation": ".UnifiedOperation",
    "Visualizer": ".visualizer",
}


def __getattr__(name: str) -> Any:
    module_name = _OPTIONAL_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
