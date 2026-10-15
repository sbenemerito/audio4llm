"""assemble on hand-made cache files: edge cases without running any model."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from audio4llm.config import STEM_NAMES, Config
from audio4llm.context import RunContext
from audio4llm.grid import BeatGrid
from audio4llm.report.analysis import drum_patterns
from audio4llm.stages import assemble
from audio4llm.stages.energy import FRAME_HZ

DUR = 10.0


def fake_run(tmp_path, *, beats: bool = True, hits: bool = True, lyrics: str | None = "skipped",
             notes: bool = True, sections: bool = True, plot: bool = False) -> RunContext:
    ctx = RunContext(run_dir=tmp_path / "run", config=Config(plot=plot))
    sr = 8000
    t = np.arange(int(DUR * sr)) / sr
    sf.write(str(ctx.input_wav), np.stack([np.sin(2 * np.pi * 220 * t)] * 2, axis=1) * 0.3, sr)
    ctx.write_json("ingest.json", {"source": "fake.wav", "duration": DUR, "sample_rate": 44100, "channels": 2, "start": 0.0})
    ctx.write_json("stems.json", {"model": "htdemucs", "device": "cpu"})
    if beats:
        grid = BeatGrid.from_tracker(np.arange(0.0, DUR, 0.5), np.arange(0.0, DUR, 2.0), DUR)
        b, d, bpm = [round(float(x), 3) for x in grid.beats], [int(i) for i in grid.downbeat_idx], 120.0
    else:
        b, d, bpm = [], [], None
    ctx.write_json("rhythm.json", {"beats": b, "downbeat_idx": d, "beats_per_bar": 4, "time_signature": "4/4",
                                   "tempo_bpm": bpm, "tempo_std_bpm": None, "duration": DUR, "model": "x", "device": "cpu"})
    n = int(DUR * FRAME_HZ) + 1
    curves = {"times": np.arange(n) / FRAME_HZ, "lufs": np.full(n, -20.0), "brightness": np.full(n, 0.5),
              "onset": np.zeros(n), "drums_low": np.zeros(n), "drums_mid": np.zeros(n), "drums_high": np.zeros(n)}
    levels = {}
    for name in ("mix", *STEM_NAMES):
        curves[f"db_{name}"] = np.full(n, -20.0)
        levels[name] = {"ref_db": -20.0, "p99_db": -20.0, "mean_db": -20.0}
    np.savez_compressed(ctx.cache_path("energy.npz"), **{k: v.astype(np.float32) for k, v in curves.items()})
    ctx.write_json("energy.json", {
        "frame_hz": FRAME_HZ, "range_db": 30.0, "duration": DUR, "levels": levels, "has_vocals": False,
        "loudness": {"integrated_lufs": -20.0, "loudness_range_lu": 1.0, "peak_dbfs": -10.0, "rms_dbfs": -20.0},
        "active": {"vocals": [], "drums": [[0.0, DUR]] if hits else [], "bass": [], "other": [[0.0, DUR]]},
    })
    ctx.write_json("hits.json", {"method": "estimate: test", "onsets": [2.0, 2.5] if hits else [],
                                 "hits": [{"t": 2.0, "type": "kick", "strength": 1.0}, {"t": 2.5, "type": "hat", "strength": 0.5}] if hits else [],
                                 "counts": {"kick": int(hits), "snare": 0, "hat": int(hits)}})
    if sections:
        ctx.write_json("sections.json", {"sections": [{"start": 0.0, "end": DUR, "character": "sparse", "energy": 0.5}],
                                         "moments": [{"t": 2.0, "type": "drums_in", "description": "x", "strength": 0.5}]})
    if lyrics == "skipped":
        ctx.write_json("lyrics.json", {"status": "skipped", "reason": "no vocal activity", "model": "m", "language": None, "lines": []})
    if notes:
        ctx.write_json("notes.json", {"backend": "onnx", "stems": {
            "bass": {"status": "ok", "notes": [{"start": 2.0, "end": 2.4, "midi": 45, "name": "A2", "velocity": 90}]},
            "vocals": {"status": "skipped", "reason": "no vocals", "notes": []}},
            "key": {"label": "A minor", "confidence": 0.5, "runner_up": "C major"}})
    return ctx


def assembled(ctx) -> dict:
    assemble.run(ctx)
    return json.loads(ctx.out_path("analysis.json").read_text())


def test_normal_case(tmp_path):
    a = assembled(fake_run(tmp_path))
    assert a["global"]["bar_count"] == 5 and len(a["stems"]["mix"]["per_beat"]) == 20
    assert a["hits"]["kick"] == [[2.0, 2, 1.0, 1.0]]
    assert a["hits"]["onsets"] == [[2.0, 2, 1.0], [2.5, 2, 2.0]]
    assert a["notes"]["per_bar"] == [{"bar": 2, "bass": "A2", "bass_root": "A"}]
    assert a["notes"]["vocals_note"] == "no vocals"
    assert a["sections"][0]["bars"] == [1, 5]
    assert a["lyrics"] == {"status": "skipped", "reason": "no vocal activity", "language": None, "model": "m", "lines": []}
    assert a["meta"]["warnings"] == []


def test_no_beat_grid(tmp_path):
    """A drone or silence: no beats at all must still produce every output."""
    ctx = fake_run(tmp_path, beats=False, plot=True)
    a = assembled(ctx)
    assert a["grid"] == {"beats": [], "bars": []} and a["global"]["tempo_bpm"] is None
    assert a["sections"][0]["bars"] is None
    assert a["moments"][0]["bar"] == 1
    for f in ("curves.json", "summary.md", "overview.png"):
        assert ctx.out_path(f).exists()


def test_no_hits_is_an_empty_object_not_null(tmp_path):
    a = assembled(fake_run(tmp_path, hits=False))
    assert a["hits"]["counts"] == {"kick": 0, "snare": 0, "hat": 0, "onsets": 0}
    assert a["hits"]["patterns"] == []


def test_failed_optional_stages_are_null_with_warnings(tmp_path):
    a = assembled(fake_run(tmp_path, lyrics=None, notes=False, sections=False))
    assert a["sections"] is None and a["moments"] is None and a["notes"] is None and a["lyrics"] is None
    assert len(a["meta"]["warnings"]) == 3


def test_no_plot_removes_stale_overview(tmp_path):
    ctx = fake_run(tmp_path, plot=True)
    assemble.run(ctx)
    assert ctx.out_path("overview.png").exists()
    ctx.config.plot = False
    assemble.run(ctx)
    assert not ctx.out_path("overview.png").exists()


@pytest.mark.parametrize("beat,expected_bar,step", [(4.9, 2, 0), (4.75, 1, 15), (1.0, 1, 0)])
def test_drum_pattern_steps(beat, expected_bar, step):
    grid = BeatGrid.from_tracker(np.arange(0.0, 8.0, 0.5), np.arange(0.0, 8.0, 2.0), 8.0)
    pats = drum_patterns({"kick": [[0.0, 1, beat, 1.0]]}, grid)
    assert pats[0]["bars"][0] == expected_bar and pats[0]["kick"][step] == "x"


def test_vocals_disabled_is_reported(tmp_path):
    ctx = fake_run(tmp_path)
    energy = ctx.read_json("energy.json")
    ctx.write_json("energy.json", {**energy, "vocals_disabled": True})
    a = assembled(ctx)
    assert a["global"]["vocals_disabled"] is True and a["global"]["has_vocals"] is False
    text = ctx.out_path("summary.md").read_text()
    assert "vocals: off (--no-vocals)" in text and "not analysed (--no-vocals)" in text
