"""Stage 6 - notes: Basic Pitch transcription of the bass and vocal stems, plus a key estimate.

Basic Pitch (Spotify, ICASSP 2022 model) runs through its CoreML model on macOS and falls back
to the bundled ONNX model when CoreML is unavailable, fails to load or fails to run. TensorFlow
is never used. basic-pitch hard-codes CoreML to CPU_ONLY (no GPU/ANE); that is fine, a clip
takes about a second per stem either way, and both backends give the same notes (their
posteriorgrams agree to ~1e-6).

Per stem:
  bass    30-400 Hz. Notes whose onset is outside the bass stem's active regions (energy.json)
          are dropped as separation bleed, then harmonic ghosts (see drop_ghost_notes).
  vocals  80-1100 Hz. Skipped with a reason when energy.json says has_vocals is false;
          otherwise notes outside the vocal active regions are dropped. Spoken or rapped
          vocals still produce "notes" that follow the speech intonation.

The key estimate (audio4llm/key.py) correlates Krumhansl-Kessler profiles with the mean CQT
chroma of bass + other + vocals (the mix without drums). One key for the whole clip.

Contract: writes cache/notes.json
{
  "backend": "coreml"|"onnx",            # the backend that actually produced the notes
  "backend_note": "<CPU_ONLY note, or why CoreML was not used>",
  "model": "basic-pitch icassp_2022",
  "stems": {
    "bass":   {"status": "ok"|"empty", "reason": "<only when not ok>", "range_hz": [30.0, 400.0],
               "midi_range": [lowest, highest kept MIDI note]|null,
               "dropped": {"inactive": n, "ghost": n},          # counts removed by the gate / ghost filter
               "midi_file": "cache/bass.mid",
               "notes": [{"start": s, "end": e, "midi": 45, "name": "A2", "velocity": 92}]},   # sorted by start
    "vocals": {"status": "ok"|"empty"|"skipped", same keys, range_hz [80.0, 1100.0], "dropped": {"inactive": n};
               when skipped: "dropped": {}, "midi_range": null, "midi_file": null, "notes": []}
  },
  "key": {"label": "A minor", "tonic": "A", "mode": "minor", "confidence": 0..1, "correlation": r,
          "runner_up": "C major", "runner_up_correlation": r, "relative": "C major",
          "method": "estimate: ..."}            # label/tonic/mode are null when there is no tonal content
}
plus cache/bass.mid and cache/vocals.mid (the kept notes) as extras; a skipped stem has no .mid.
Note names use ASCII sharps with C4 = MIDI 60. velocity is Basic Pitch's note amplitude * 127.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
from dataclasses import dataclass

import numpy as np

from .. import audio
from ..context import RunContext
from ..jsonfmt import r3
from ..key import PITCH_CLASSES, estimate_key, harmonic_chroma
from ..pipeline import Stage

log = logging.getLogger("audio4llm")

MODEL_NAME = "basic-pitch icassp_2022"
BP_SR = 22050  # Basic Pitch's own input rate; the key chroma uses it too


@dataclass(frozen=True)
class StemSettings:
    range_hz: tuple[float, float]
    min_note_ms: float
    # Basic Pitch's defaults (0.5 / 0.3), tuned by its authors. Bleed is handled by the
    # activity gate below instead of raising thresholds, which would also drop quiet real notes.
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    ghost_filter: bool = False


SETTINGS = {
    # 30 Hz keeps B0 (30.9 Hz, low string of a 5-string bass; Basic Pitch rounds bounds to the
    # nearest semitone and its floor is A0). 400 Hz covers a bass neck up to ~F#4.
    # 70 ms minimum (Basic Pitch default 128 ms) keeps staccato 16th notes up to ~140 BPM;
    # the extra short fragments this lets through are mostly harmonics, removed by the ghost filter.
    "bass": StemSettings((30.0, 400.0), min_note_ms=70.0, ghost_filter=True),
    # 80-1100 Hz spans bass voice E2 to soprano C6. 100 ms keeps 16th-note syllables at 120-140 BPM
    # but drops the shorter fragments Basic Pitch makes from vibrato and pitch glides.
    "vocals": StemSettings((80.0, 1100.0), min_note_ms=100.0),
}

# Harmonic ghosts: Basic Pitch often adds a weaker note at a harmonic of a real bass note
# (harmonic 2..6 = +12, +19, +24, +28, +31 semitones). In synthetic tests ghosts reached 25-75 %
# of the real note's amplitude and sat almost entirely inside it, while a played octave jump
# starts where the previous note ends. So: drop a note when lower kept notes a harmonic below it,
# at least as strong, cover at least half of its duration.
HARMONIC_INTERVALS = (12, 19, 24, 28, 31)
GHOST_MIN_OVERLAP = 0.5

# Activity gate: energy.py's active regions are smoothed over 0.5 s, so they already start a
# little before a stem's first attack. The pad only absorbs rounding at the region edges.
GATE_PAD_S = 0.1

MIDI_PROGRAMS = {"bass": 33, "vocals": 53}  # General MIDI (0-based): Electric Bass (finger), Voice Oohs


# ---- pure helpers (unit-tested) -----------------------------------------------------

def note_name(midi: int) -> str:
    """MIDI number to scientific pitch with ASCII sharps: 60 -> "C4", 61 -> "C#4", 21 -> "A0"."""
    midi = int(midi)
    return f"{PITCH_CLASSES[midi % 12]}{midi // 12 - 1}"


def to_velocity(amplitude: float) -> int:
    return int(np.clip(round(float(amplitude) * 127), 1, 127))


def _covered(start: float, end: float, intervals: list[tuple[float, float]]) -> float:
    """Seconds of [start, end] covered by the union of intervals."""
    clipped = sorted((max(a, start), min(b, end)) for a, b in intervals if b > start and a < end)
    total, cur_a, cur_b = 0.0, None, None
    for a, b in clipped:
        if cur_b is None or a > cur_b:
            total += 0.0 if cur_b is None else cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    return total + (0.0 if cur_b is None else cur_b - cur_a)


def drop_ghost_notes(notes: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split notes ({"start", "end", "midi", "velocity", ...}) into (kept, ghosts).

    Notes are visited from the lowest pitch up, so a ghost is only ever judged against notes
    that were themselves kept. Both lists keep the input's order.
    """
    kept: list[dict] = []
    ghost_ids: set[int] = set()
    for n in sorted(notes, key=lambda n: (n["midi"], n["start"])):
        dur = n["end"] - n["start"]
        support = [(m["start"], m["end"]) for m in kept
                   if n["midi"] - m["midi"] in HARMONIC_INTERVALS and m["velocity"] >= n["velocity"]]
        if dur > 0 and support and _covered(n["start"], n["end"], support) >= GHOST_MIN_OVERLAP * dur:
            ghost_ids.add(id(n))
        else:
            kept.append(n)
    return [n for n in notes if id(n) not in ghost_ids], [n for n in notes if id(n) in ghost_ids]


def gate_to_regions(notes: list[dict], regions: list[list[float]] | None, pad: float = GATE_PAD_S) -> tuple[list[dict], int]:
    """Keep notes whose onset lies inside an active region (None = no gating). Returns (kept, n_dropped)."""
    if regions is None:
        return notes, 0
    kept = [n for n in notes if any(a - pad <= n["start"] <= b + pad for a, b in regions)]
    return kept, len(notes) - len(kept)


# ---- Basic Pitch ----------------------------------------------------------------------

class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def _quiet():
    """Silence basic_pitch/coremltools chatter and re-log it at DEBUG.

    Importing basic_pitch logs a root-logger warning for every backend it cannot find (TF,
    tflite), coremltools warns about untested torch/scikit-learn versions, and predict() prints
    to stdout. A temporary root handler also stops ``logging.warning`` from calling basicConfig.
    """
    root = logging.getLogger()
    saved = root.handlers[:]
    collect = _Collect()
    root.handlers = [collect]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            yield
    finally:
        root.handlers = saved
        for rec in collect.records:
            if rec.name != log.name:
                log.debug("notes: [%s] %s", rec.name, rec.getMessage())
        if out.getvalue().strip():
            log.debug("notes: basic_pitch stdout: %s", out.getvalue().strip())


def _load_model(backend: str):
    with _quiet():
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
        from basic_pitch.inference import Model

        model = Model(build_icassp_2022_model_path(getattr(FilenameSuffix, backend)))
    want = {"coreml": Model.MODEL_TYPES.COREML, "onnx": Model.MODEL_TYPES.ONNX}[backend]
    if model.model_type != want:  # Model() silently tries other runtimes when one fails
        raise RuntimeError(f"basic-pitch loaded a {model.model_type.name} model instead of {backend}")
    return model


def _predict(model, path, s: StemSettings) -> list[dict]:
    with _quiet():
        from basic_pitch.inference import predict

        output, _, events = predict(
            str(path), model,
            onset_threshold=s.onset_threshold, frame_threshold=s.frame_threshold,
            minimum_note_length=s.min_note_ms,
            minimum_frequency=s.range_hz[0], maximum_frequency=s.range_hz[1],
        )
    if not all(np.isfinite(v).all() for v in output.values()):
        raise RuntimeError("model output contains NaN/inf")
    notes = [{"start": float(t0), "end": float(t1), "midi": int(p), "velocity": to_velocity(a)}
             for t0, t1, p, a, _bends in events]
    return sorted(notes, key=lambda n: (n["start"], n["midi"]))


def transcribe(paths: dict[str, os.PathLike]) -> tuple[dict[str, list[dict]], str, str]:
    """Raw Basic Pitch notes for each stem path. Returns (notes per stem, backend used, backend note)."""
    backends = ["coreml", "onnx"] if sys.platform == "darwin" else ["onnx"]
    errors: list[str] = []
    for backend in backends:
        try:
            model = _load_model(backend)
            raw = {stem: _predict(model, p, SETTINGS[stem]) for stem, p in paths.items()}
        except Exception as exc:  # any CoreML problem (load, predict, bad output) -> try ONNX
            errors.append(f"{backend} failed: {exc}")
            log.warning("notes: Basic Pitch %s backend failed (%s)%s", backend, exc,
                        "; falling back to ONNX" if backend != backends[-1] else "")
            continue
        if backend == "coreml":
            note = "CoreML model, CPU_ONLY (hard-coded in basic-pitch)"
        elif errors:
            note = "ONNX fallback; " + "; ".join(errors)
        else:
            note = "ONNX model (CoreML is macOS-only)"
        return raw, backend, note
    raise RuntimeError("Basic Pitch failed on every backend: " + "; ".join(errors))


# ---- output -----------------------------------------------------------------------------

def write_midi(path: os.PathLike, notes: list[dict], program: int) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()  # 120 BPM header; note times are absolute seconds either way
    inst = pretty_midi.Instrument(program=program)
    inst.notes = [pretty_midi.Note(velocity=n["velocity"], pitch=n["midi"], start=n["start"], end=n["end"]) for n in notes]
    pm.instruments.append(inst)
    tmp = f"{path}.tmp"
    pm.write(tmp)
    os.replace(tmp, path)


def _stem_entry(ctx: RunContext, stem: str, notes: list[dict], dropped: dict, reason_if_empty: str) -> dict:
    midi_file = ctx.cache_path(f"{stem}.mid")
    write_midi(midi_file, notes, MIDI_PROGRAMS[stem])
    entry = {"status": "ok" if notes else "empty"}
    if not notes:
        entry["reason"] = reason_if_empty
    entry |= {
        "range_hz": list(SETTINGS[stem].range_hz),
        "midi_range": [min(n["midi"] for n in notes), max(n["midi"] for n in notes)] if notes else None,
        "dropped": dropped,
        "midi_file": str(midi_file.relative_to(ctx.run_dir)),
        "notes": [{"start": r3(n["start"]), "end": r3(n["end"]), "midi": n["midi"], "name": note_name(n["midi"]),
                   "velocity": n["velocity"]} for n in notes],
    }
    return entry


def _skipped_entry(ctx: RunContext, stem: str, reason: str) -> dict:
    stale = ctx.cache_path(f"{stem}.mid")
    if stale.exists():
        stale.unlink()
    return {"status": "skipped", "reason": reason, "range_hz": list(SETTINGS[stem].range_hz),
            "midi_range": None, "dropped": {}, "midi_file": None, "notes": []}


def key_from_stems(ctx: RunContext) -> dict:
    ys = [audio.load_mono(ctx.stem_path(s), sr=BP_SR)[0] for s in ("bass", "other", "vocals")]
    n = min(len(y) for y in ys)
    return estimate_key(harmonic_chroma(sum(y[:n] for y in ys), BP_SR))


def run(ctx: RunContext) -> None:
    energy = ctx.maybe_read_json("energy.json")
    if energy is None:
        log.warning("notes: energy.json missing; transcribing both stems without activity gating")
    active = (energy or {}).get("active", {})
    has_vocals = True if energy is None else bool(energy.get("has_vocals"))

    stems = ["bass"] + (["vocals"] if has_vocals else [])
    raw, backend, backend_note = transcribe({s: ctx.stem_path(s) for s in stems})

    out_stems: dict[str, dict] = {}
    for stem in stems:
        # no energy.json -> no gating; a stem missing from "active" is also left ungated
        kept, n_inactive = gate_to_regions(raw[stem], active.get(stem))
        dropped = {"inactive": n_inactive}
        if SETTINGS[stem].ghost_filter:
            kept, ghosts = drop_ghost_notes(kept)
            dropped["ghost"] = len(ghosts)
        out_stems[stem] = _stem_entry(ctx, stem, kept, dropped, f"no notes found inside the {stem} stem's active regions")
    if not has_vocals:
        out_stems["vocals"] = _skipped_entry(ctx, "vocals", f"energy.json: has_vocals is false "
                                             f"({energy.get('vocal_seconds', 0)} s of vocal activity)")

    key = key_from_stems(ctx)
    ctx.write_json("notes.json", {"backend": backend, "backend_note": backend_note, "model": MODEL_NAME,
                                  "stems": out_stems, "key": key})
    summary = ", ".join(f"{s} {len(e['notes'])} notes (dropped {e['dropped']})" if e["status"] != "skipped" else f"{s} skipped"
                        for s, e in out_stems.items())
    log.info("notes: [%s] %s; key %s (confidence %.2f)", backend, summary, key["label"] or "unknown", key["confidence"])


STAGE = Stage(
    name="notes",
    run=run,
    outputs=lambda ctx: [ctx.cache_path("notes.json")],
    deps=("stems", "energy"),
    required=False,
)
