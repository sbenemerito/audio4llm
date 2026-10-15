"""Stage 7 - assemble: merge cached stage outputs into the final files. TODO."""

from __future__ import annotations

from ..context import RunContext
from ..pipeline import Stage


def run(ctx: RunContext) -> None:
    pass


STAGE = Stage(
    name="assemble",
    run=run,
    outputs=lambda ctx: [ctx.out_path("analysis.json")],
    deps=("ingest", "stems", "rhythm", "energy", "sections", "lyrics", "notes"),
    config_keys=("curves_hz", "plot"),
    always_run=True,
)
