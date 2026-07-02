"""Plugin loading helpers.

Entry point discovery is enabled for installed packages. Explicit plugin paths are
reported but not auto-executed; importing arbitrary files from a working tree is
intentionally outside the safe core behavior.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any


def load_entry_points(group: str) -> list[Any]:
    entry_points = metadata.entry_points()
    selected = entry_points.select(group=group)
    return [entry_point.load() for entry_point in selected]


def describe_explicit_plugin_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)
