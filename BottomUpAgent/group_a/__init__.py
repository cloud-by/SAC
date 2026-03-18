from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    "Eye",
    "Detector",
    "Hand",
    "UnifiedOperation",
    "visualizer",
]

_LAZY_EXPORTS: Dict[str, Tuple[str, str | None]] = {
    "Eye": (".Eye", "Eye"),
    "Detector": (".Detector", "Detector"),
    "Hand": (".Hand", "Hand"),
    "UnifiedOperation": (".UnifiedOperation", "UnifiedOperation"),
    "Visualizer": (".visualizer", "Visualizer"),
    "visualizer": (".visualizer", None),
}


def __getattr__(name: str) -> Any:
    export = _LAZY_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = export
    module = import_module(module_name, __name__)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value