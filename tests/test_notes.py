"""Notes stage helpers and the stage itself with Basic Pitch stubbed out (no model is loaded)."""

import json
import logging
import sys

import numpy as np
import pytest

from audio4llm import audio
from audio4llm.config import Config
from audio4llm.context import RunContext
from audio4llm.stages import notes


def n(start, end, midi, velocity):
    return {"start": start, "end": end, "midi": midi, "velocity": velocity}


# ---- naming / velocity -------------------------------------------------------------

@pytest.mark.parametrize("midi,name", [(21, "A0"), (45, "A2"), (59, "B3"), (60, "C4"), (61, "C#4"),
                                       (70, "A#4"), (69, "A4"), (108, "C8"), (0, "C-1")])
def test_note_name_ascii_sharps(midi, name):
    assert notes.note_name(midi) == name
    assert notes.note_name(np.int64(midi)) == name


def test_velocity_scaling():
    assert notes.to_velocity(1.0) == 127
    assert notes.to_velocity(np.float32(0.5)) == 64
    assert notes.to_velocity(0.0) == 1  # MIDI velocity 0 means note-off


# ---- ghost-note filter ------------------------------------------------------------

@pytest.mark.parametrize("interval", [12, 19, 24, 28])
def test_harmonic_ghost_inside_stronger_lower_note_is_dropped(interval):
    root = n(1.0, 1.5, 33, 90)
    ghost = n(1.01, 1.3, 33 + interval, 40)
    kept, ghosts = notes.drop_ghost_notes([root, ghost])
    assert kept == [root] and ghosts == [ghost]


def test_non_harmonic_interval_is_kept():
    notes_in = [n(1.0, 1.5, 33, 90), n(1.0, 1.5, 40, 40)]  # +7: a fifth, not a harmonic of the root
    assert notes.drop_ghost_notes(notes_in)[0] == notes_in


def test_upper_note_stronger_than_lower_is_kept():
    notes_in = [n(1.0, 1.5, 33, 40), n(1.0, 1.5, 45, 90)]
    assert notes.drop_ghost_notes(notes_in)[0] == notes_in


def test_sequential_octave_jump_is_kept():
    # Basic Pitch often lets a note ring into the next onset; that small overlap is not a ghost
    notes_in = [n(0.0, 0.56, 33, 90), n(0.5, 1.0, 45, 60)]
    assert notes.drop_ghost_notes(notes_in)[0] == notes_in


def test_ghost_spanning_two_lower_notes_uses_their_union():
    a, b = n(0.0, 0.5, 33, 80), n(0.5, 1.0, 33, 80)
    ghost = n(0.3, 0.75, 45, 40)  # 40 % inside a, 56 % inside b, 100 % inside both together
    kept, ghosts = notes.drop_ghost_notes([a, ghost, b])
    assert kept == [a, b] and ghosts == [ghost]


def test_ghost_of_a_ghost_and_order_preserved():
    root, g12, g24 = n(2.0, 2.5, 33, 90), n(2.0, 2.4, 45, 50), n(2.0, 2.3, 57, 45)
    other = n(3.0, 3.5, 36, 80)
    kept, ghosts = notes.drop_ghost_notes([root, g24, other, g12])
    assert kept == [root, other] and ghosts == [g24, g12]


def test_ghost_filter_handles_empty_and_zero_length():
    assert notes.drop_ghost_notes([]) == ([], [])
    z = [n(1.0, 1.5, 33, 90), n(1.2, 1.2, 45, 10)]
    assert notes.drop_ghost_notes(z)[0] == z


def test_covered_merges_overlaps():
    assert notes._covered(0.0, 1.0, [(0.2, 0.5), (0.4, 0.6), (0.9, 2.0), (-1.0, 0.1)]) == pytest.approx(0.6)
    assert notes._covered(0.0, 1.0, []) == 0.0


# ---- activity gate ----------------------------------------------------------------

def test_gate_keeps_onsets_inside_regions():
    ev = [n(1.0, 1.4, 45, 90), n(10.0, 10.4, 45, 90), n(4.95, 5.4, 45, 90)]
    kept, dropped = notes.gate_to_regions(ev, [[0.5, 2.0], [5.0, 6.0]])
    assert kept == [ev[0], ev[2]] and dropped == 1  # 4.95 is within the 0.1 s edge pad
    assert notes.gate_to_regions(ev, None) == (ev, 0)
    assert notes.gate_to_regions(ev, []) == ([], 3)


# ---- quieting basic_pitch ---------------------------------------------------------

def test_quiet_captures_root_warnings_and_stdout(capsys):
    root = logging.getLogger()
    before = root.handlers[:]
    with notes._quiet():
        logging.warning("tflite-runtime is not installed")
        print("Predicting MIDI for x.wav...")
    assert root.handlers == before
    out = capsys.readouterr()
    assert "Predicting" not in out.out and "tflite" not in out.err


# ---- backend fallback -------------------------------------------------------------

def test_coreml_failure_falls_back_to_onnx(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    loaded = []

    def fake_load(backend):
        loaded.append(backend)
        if backend == "coreml":
            raise RuntimeError("no CoreML here")
        return object()

    monkeypatch.setattr(notes, "_load_model", fake_load)
    monkeypatch.setattr(notes, "_predict", lambda model, path, s: [n(1.0, 1.5, 45, 90)])
    raw, backend, note = notes.transcribe({"bass": "bass.wav"})
    assert loaded == ["coreml", "onnx"] and backend == "onnx"
    assert "coreml failed: no CoreML here" in note and raw["bass"][0]["midi"] == 45


def test_every_backend_failing_raises(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(notes, "_load_model", lambda b: (_ for _ in ()).throw(RuntimeError(f"{b} broken")))
    with pytest.raises(RuntimeError, match="every backend"):
        notes.transcribe({"bass": "bass.wav"})


# ---- the stage, with Basic Pitch stubbed --------------------------------------------

def _make_run(tmp_path, has_vocals: bool) -> RunContext:
    ctx = RunContext(run_dir=tmp_path / "clip")
    sr = 44100
    t = np.arange(4 * sr) / sr
    tone = lambda m: 0.2 * np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * t)
    stems = {"bass": tone(45), "other": tone(57) + tone(60) + tone(64), "vocals": tone(69) * has_vocals, "drums": 0 * t}
    for name, y in stems.items():
        audio.save(ctx.stem_path(name), np.vstack([y, y]).astype(np.float32), sr)
    ctx.write_json("energy.json", {"active": {"bass": [[0.9, 3.0]], "vocals": [[1.0, 2.0]] if has_vocals else []},
                                   "has_vocals": has_vocals, "vocal_seconds": 1.0 if has_vocals else 0.0})
    ctx.cache_path("vocals.mid").write_bytes(b"stale")
    return ctx


def _fake_transcribe(paths):
    raw = {"bass": [n(0.2, 0.6, 45, 70),                                     # before the bass is active: bleed
                    n(1.0, 1.5, 45, 90), n(1.01, 1.3, 57, 40),                # A2 + its octave ghost
                    n(2.0, 2.5, 48, 88)],
           "vocals": [n(1.2, 1.6, 69, 80), n(3.5, 3.8, 72, 60)]}
    return {s: raw[s] for s in paths}, "coreml", "CoreML model, CPU_ONLY (hard-coded in basic-pitch)"


def test_stage_writes_contract_and_skips_vocals(tmp_path, monkeypatch):
    ctx = _make_run(tmp_path, has_vocals=False)
    seen = []

    def fake(paths):
        seen.append(list(paths))
        return _fake_transcribe(paths)

    monkeypatch.setattr(notes, "transcribe", fake)
    notes.run(ctx)
    d = json.loads(ctx.cache_path("notes.json").read_text())

    assert seen == [["bass"]]  # vocals never reach Basic Pitch
    assert d["backend"] == "coreml" and "CPU_ONLY" in d["backend_note"]
    bass = d["stems"]["bass"]
    assert bass["status"] == "ok" and bass["range_hz"] == [30.0, 400.0]
    assert bass["notes"] == [{"start": 1.0, "end": 1.5, "midi": 45, "name": "A2", "velocity": 90},
                             {"start": 2.0, "end": 2.5, "midi": 48, "name": "C3", "velocity": 88}]
    assert bass["dropped"] == {"inactive": 1, "ghost": 1} and bass["midi_range"] == [45, 48]
    assert bass["midi_file"] == "cache/bass.mid" and ctx.cache_path("bass.mid").stat().st_size > 0

    vox = d["stems"]["vocals"]
    assert vox["status"] == "skipped" and "has_vocals is false" in vox["reason"] and vox["notes"] == []
    assert vox["midi_file"] is None and not ctx.cache_path("vocals.mid").exists()

    key = d["key"]
    assert set(key) >= {"label", "tonic", "mode", "confidence", "runner_up", "method"}
    assert key["label"] in ("A minor", "C major") and key["method"].startswith("estimate")


def test_no_vocals_flag_skips_vocal_transcription(tmp_path, monkeypatch):
    ctx = _make_run(tmp_path, has_vocals=True)  # vocals detected, but the user says instrumental
    ctx.config = Config(vocals=False)
    seen = []
    monkeypatch.setattr(notes, "transcribe", lambda paths: (seen.append(list(paths)), _fake_transcribe(paths))[1])
    notes.run(ctx)
    vox = json.loads(ctx.cache_path("notes.json").read_text())["stems"]["vocals"]
    assert seen == [["bass"]]
    assert vox["status"] == "skipped" and "--no-vocals" in vox["reason"] and vox["notes"] == []


def test_stage_transcribes_vocals_inside_active_regions(tmp_path, monkeypatch):
    import pretty_midi

    ctx = _make_run(tmp_path, has_vocals=True)
    monkeypatch.setattr(notes, "transcribe", _fake_transcribe)
    notes.run(ctx)
    vox = json.loads(ctx.cache_path("notes.json").read_text())["stems"]["vocals"]
    assert vox["status"] == "ok" and vox["dropped"] == {"inactive": 1}
    assert [(x["name"], x["velocity"]) for x in vox["notes"]] == [("A4", 80)]
    midi = pretty_midi.PrettyMIDI(str(ctx.cache_path("vocals.mid")))
    assert [(m.pitch, round(m.start, 3)) for m in midi.instruments[0].notes] == [(69, 1.2)]
