"""Stage 7 - assemble: merge cached stage outputs into the final files.

Reads only cache/*.json and cache/energy.npz (no audio models), so it runs in
about a second and re-runs on every invocation. Writes, in the run folder:

  analysis.json  the LLM-facing analysis (see docs/OUTPUT_SCHEMA.md)
  curves.json    dense fixed-rate envelopes for code (--curves-hz)
  summary.md     templated plain-language summary
  overview.png   diagnostic plot (skipped with --no-plot)

Optional stages that failed or were skipped show up as null plus a note in
meta.warnings, so a partial analysis is still usable.
"""

from __future__ import annotations

import logging

from ..context import RunContext, write_text_atomic
from ..jsonfmt import dumps
from ..pipeline import Stage
from ..report import analysis as analysis_mod
from ..report import summary

log = logging.getLogger("audio4llm")


def run(ctx: RunContext) -> None:
    analysis, curves = analysis_mod.build(ctx)
    write_text_atomic(ctx.out_path("analysis.json"), dumps(analysis))
    write_text_atomic(ctx.out_path("curves.json"), dumps(curves))
    write_text_atomic(ctx.out_path("summary.md"), summary.render(analysis, curves, ctx.name))
    if ctx.config.plot:
        from ..report import overview

        overview.render(analysis, curves, ctx.input_wav, ctx.out_path("overview.png"), f"audio4llm · {ctx.name}")
    for w in analysis["meta"]["warnings"]:
        log.warning("assemble: %s", w)
    size_kb = ctx.out_path("analysis.json").stat().st_size / 1024
    log.info("assemble: analysis.json %.0f KB, curves.json %d curves @ %g Hz", size_kb, len(curves["curves"]), curves["hz"])


STAGE = Stage(
    name="assemble",
    run=run,
    outputs=lambda ctx: [ctx.out_path(f) for f in ("analysis.json", "curves.json", "summary.md")],
    deps=("ingest", "stems", "rhythm", "energy", "sections", "lyrics", "notes"),
    config_keys=("curves_hz", "plot"),
    always_run=True,
)
