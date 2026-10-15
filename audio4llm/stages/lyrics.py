"""Stage 5 - lyrics with mlx-whisper on the vocal stem.

Contract: writes cache/lyrics.json
{
  "status": "ok"|"skipped"|"empty", "reason": "<why, when not ok>",
  "model": "<hf repo>", "language": "en", "language_source": "detected"|"user", "language_probability": 0.98|null,
  "regions": [[start, end], ...],   # vocal-active regions that were transcribed, clip time
  "lines": [{"start": s, "end": e, "text": "...", "avg_logprob": x, "no_speech_prob": x,
             "words": [{"w": "hold", "start": s, "end": e, "p": 0.97}]}],
  "dropped": [{"start": s, "end": e, "text": "...", "reason": "..."}]
}
TODO: placeholder until implemented.
"""

from __future__ import annotations

from ..context import RunContext
from ..pipeline import Stage


def run(ctx: RunContext) -> None:
    raise NotImplementedError("lyrics stage not implemented yet")


STAGE = Stage(
    name="lyrics",
    run=run,
    outputs=lambda ctx: [ctx.cache_path("lyrics.json")],
    deps=("stems", "energy"),
    config_keys=("whisper_model", "language"),
    required=False,
)
