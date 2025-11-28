"""Package entry for MemoryOS playground."""

from importlib import import_module
from typing import Any

__all__ = ["Memoryos"]

def __getattr__(name: str) -> Any:
    """Lazily load heavy modules such as memoryos."""
    if name == "Memoryos":
        module = import_module(".memoryos", __name__)
        return module.Memoryos
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")