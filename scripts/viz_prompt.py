#!/usr/bin/env python
"""Copy a visualizer prompt, filled in for one run folder, to the clipboard.

Usage:  python scripts/viz_prompt.py [--creative] [--print] runs/<clip>

The prompt is the ````text block in docs/VISUALIZER_TEST.md (or docs/VISUALIZER_CREATIVE.md with
--creative), so editing the doc changes what gets copied. It ends with a "Special instructions" heading:
paste it into Claude Code and keep typing, e.g. "only 2D, flat vector, for a lyric video".
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = {False: ROOT / "docs" / "VISUALIZER_TEST.md", True: ROOT / "docs" / "VISUALIZER_CREATIVE.md"}
BLOCK = re.compile(r"^````text\n(.*?)^````$", re.S | re.M)
SPECIAL = "\n\n## Special instructions\nThese come from me and take priority over anything above where they conflict:\n"


def run_path(arg: str) -> str:
    """The run folder as the agent should see it: relative to the repo root if inside it."""
    path = Path(arg).expanduser().resolve()
    if not (path / "analysis.json").is_file():
        sys.exit(f"error: {arg} has no analysis.json; pass an audio4llm run folder like runs/<clip>")
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build(doc: Path, run: str) -> str:
    match = BLOCK.search(doc.read_text())
    if not match:
        sys.exit(f"error: no ````text block found in {doc}")
    clip = Path(run).name
    prompt = match.group(1).rstrip()
    prompt = prompt.replace("viz-runs/<clip>", f"viz-runs/{clip}")
    prompt = prompt.replace("runs/<clip>", run)
    prompt = prompt.replace("<clip>", clip)
    return prompt + SPECIAL


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", help="audio4llm output folder, e.g. runs/test_clip")
    ap.add_argument("--creative", action="store_true", help="use the artistic-freedom prompt instead of the test")
    ap.add_argument("--print", action="store_true", help="print the prompt instead of copying it")
    args = ap.parse_args()

    doc = DOCS[args.creative]
    prompt = build(doc, run_path(args.run))
    if args.print:
        print(prompt)
        return
    subprocess.run(["pbcopy"], input=prompt, text=True, check=True)
    print(f"Copied {doc.name} prompt for {run_path(args.run)} ({len(prompt.splitlines())} lines).", file=sys.stderr)


if __name__ == "__main__":
    main()
