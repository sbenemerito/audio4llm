#!/usr/bin/env python
"""Refresh the trimmed real examples in docs/OUTPUT_SCHEMA.md from a run folder.

Usage:  python scripts/update_schema_example.py [runs/test_clip]

Long arrays are cut to their first few items followed by a "… N more" string, so
the example stays short; everything else is copied verbatim from the run.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from audio4llm.jsonfmt import dumps

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "OUTPUT_SCHEMA.md"
KEEP = {  # path suffix -> items to keep
    "grid.beats": 6, "grid.bars": 3, "sections": 5, "moments": 9,
    "hits.kick": 4, "hits.snare": 3, "hits.hat": 3, "hits.patterns": 3,
    "notes.bass": 4, "notes.vocals": 2, "notes.per_bar": 3,
    "lyrics.lines": 2, "words": 3, "active": 2, "curves.keys": 20,
}
DEFAULT_KEEP = 8


def trim(o, path=""):
    if isinstance(o, dict):
        return {k: trim(v, f"{path}.{k}" if path else k) for k, v in o.items()}
    if isinstance(o, list):
        keep = next((n for suffix, n in KEEP.items() if path.endswith(suffix)), DEFAULT_KEEP)
        items = [trim(v, path + "[]") for v in o[:keep]]  # rows inside a table are never cut
        if len(o) > keep:
            items.append(f"… {len(o) - keep} more")
        return items
    return o


def splice(text: str, name: str, body: str) -> str:
    start, end = f"<!-- example:{name}:start -->", f"<!-- example:{name}:end -->"
    block = f"{start}\n```json\n{body.rstrip()}\n```\n{end}"
    pattern = re.compile(re.escape(start) + ".*?" + re.escape(end), re.S)
    if not pattern.search(text):
        raise SystemExit(f"marker {start} not found in {DOC}")
    return pattern.sub(lambda _: block, text)


def main() -> None:
    run = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs" / "test_clip")
    analysis = json.loads((run / "analysis.json").read_text())
    curves = json.loads((run / "curves.json").read_text())
    # machine-specific absolute paths would only add noise to the docs
    analysis["source"]["file"] = "test_data/test_clip.wav"
    analysis["source"]["input_wav"] = "runs/test_clip/input.wav"
    curves["curves"] = {k: v[640:648] + [f"… {len(v) - 8} more"] for k, v in curves["curves"].items()}
    text = DOC.read_text()
    text = splice(text, "analysis", dumps(trim(analysis)))
    text = splice(text, "curves", dumps(curves))
    DOC.write_text(text)
    print(f"updated {DOC} from {run}")


if __name__ == "__main__":
    main()
