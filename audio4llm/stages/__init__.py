"""Pipeline stages, in run order. Each module exposes ``STAGE`` (see pipeline.Stage)."""

from . import assemble, energy, ingest, lyrics, notes, rhythm, sections, stems

STAGES = [
    ingest.STAGE,
    stems.STAGE,
    rhythm.STAGE,
    energy.STAGE,
    sections.STAGE,
    lyrics.STAGE,
    notes.STAGE,
    assemble.STAGE,
]
