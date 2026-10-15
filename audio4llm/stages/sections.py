"""Stage 4 - sections and moments.

Contract: writes cache/sections.json
{
  "method": "<one line>",
  "sections": [   # contiguous, cover the clip, boundaries snapped to bar starts (first starts at 0.0, last ends at duration)
    {"start": 0.0, "end": 16.0, "character": "sparse"|"build"|"full"|"breakdown"|"drop"|"outro"|"intro"|"groove"|...,
     "energy": 0..1, "trend": "rising"|"falling"|"flat", "active_stems": ["drums", ...],
     "hits_per_beat": 1.5, "description": "<short phrase>"}
  ],
  "moments": [    # sorted by t
    {"t": 32.0, "type": "drop"|"build_peak"|"vocal_entry"|"drums_in"|"bass_in"|"stop"|"silence"|"loudest"|"breakdown"|...,
     "description": "<short phrase>", "strength": 0..1}
  ]
}
TODO: placeholder until implemented.
"""

from __future__ import annotations

from ..context import RunContext
from ..pipeline import Stage


def run(ctx: RunContext) -> None:
    raise NotImplementedError("sections stage not implemented yet")


STAGE = Stage(
    name="sections",
    run=run,
    outputs=lambda ctx: [ctx.cache_path("sections.json")],
    deps=("rhythm", "energy"),
    required=False,
)
