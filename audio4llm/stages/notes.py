"""Stage 6 - notes (Basic Pitch on the bass and vocal stems) and a key estimate.

Contract: writes cache/notes.json
{
  "backend": "coreml"|"onnx",
  "stems": {"bass":   {"range_hz": [lo, hi], "notes": [{"start": s, "end": e, "midi": 45, "name": "A2", "velocity": 88}]},
            "vocals": {...}},
  "key": {"label": "A minor", "tonic": "A", "mode": "minor", "confidence": 0..1, "runner_up": "C major",
          "method": "<one line, says it is an estimate>"}
}
plus cache/bass.mid and cache/vocals.mid as extras.
TODO: placeholder until implemented.
"""

from __future__ import annotations

from ..context import RunContext
from ..pipeline import Stage


def run(ctx: RunContext) -> None:
    raise NotImplementedError("notes stage not implemented yet")


STAGE = Stage(
    name="notes",
    run=run,
    outputs=lambda ctx: [ctx.cache_path("notes.json")],
    deps=("stems",),
    required=False,
)
