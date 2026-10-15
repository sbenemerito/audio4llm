"""Compact, readable JSON.

``json.dumps(indent=2)`` puts every array element on its own line, which turns
a 160-beat grid into 800 lines. This formatter indents containers only when
they don't fit on one line, and always keeps arrays of plain numbers/strings on
a single line. Small objects (one beat, one hit) end up one per line.

It also converts numpy scalars/arrays, Paths and tuples, and turns NaN/inf into
``null`` so the output is always strict JSON.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

WIDTH = 120


def to_builtin(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): to_builtin(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_builtin(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, bool) or o is None or isinstance(o, str):
        return o
    if isinstance(o, int):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    # numpy without importing it at module load
    if hasattr(o, "tolist"):
        return to_builtin(o.tolist())
    if hasattr(o, "item"):
        return to_builtin(o.item())
    return o


def _one_line(o: Any) -> str:
    return json.dumps(o, ensure_ascii=False, separators=(", ", ": "))


def _fmt(o: Any, level: int, indent: int, prefix_len: int) -> str:
    one = _one_line(o)
    if not isinstance(o, (dict, list)) or not o:
        return one
    if level * indent + prefix_len + len(one) <= WIDTH:
        return one
    if isinstance(o, list) and all(not isinstance(x, (dict, list)) for x in o):
        return one  # scalar arrays stay on one line however long
    pad = " " * ((level + 1) * indent)
    end = " " * (level * indent)
    if isinstance(o, dict):
        parts = []
        for k, v in o.items():
            key = json.dumps(k, ensure_ascii=False) + ": "
            parts.append(pad + key + _fmt(v, level + 1, indent, len(key)))
        return "{\n" + ",\n".join(parts) + "\n" + end + "}"
    parts = [pad + _fmt(v, level + 1, indent, 0) for v in o]
    return "[\n" + ",\n".join(parts) + "\n" + end + "]"


def dumps(obj: Any, indent: int = 2) -> str:
    return _fmt(to_builtin(obj), 0, indent, 0) + "\n"


def r3(x: float | None) -> float | None:
    """Round a time in seconds to milliseconds (the precision used in all outputs)."""
    return None if x is None else round(float(x), 3)


def r2(x: float | None) -> float | None:
    return None if x is None else round(float(x), 2)
