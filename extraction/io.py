from __future__ import annotations

import json
import os
from typing import Any, Optional


def read_json(path: str, default: Optional[Any] = None) -> Any:
    """Read JSON file; return default if missing or invalid."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def write_json(path: str, data: Any) -> None:
    """Write JSON to file, creating parent dirs as needed."""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_text(path: str) -> str:
    """Read a text file as a string (strips trailing whitespace)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

