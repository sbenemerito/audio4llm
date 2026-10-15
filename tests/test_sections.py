"""Fast tests for the sections stage: segmentation, labelling rules, moment detectors, and the
output contract of run() on a small synthetic run folder (no audio, no models)."""

from __future__ import annotations

import numpy as np
import pytest

from audio4llm.config import STEM_NAMES
from audio4llm.context import RunContext
from audio4llm.stages import sections as S
from audio4llm.stages.energy import FRAME_HZ


# ---- bars ---------------------------------------------------------------------------------

def test_bar_edges_cover_the_clip():
    starts = np.arange(0.04, 20.0, 2.0)
    e = S.bar_edges(starts, 20.0)
    assert e[0] == 0.0 and e[-1] == 20.0
    assert np.allclose(e[1:-1], starts[1:])  # a short lead-in joins bar 1


def test_bar_edges_long_lead_in_and_early_grid_end():
    e = S.bar_edges(np.arange(5.0, 13.0, 2.0), 20.0)  # grid covers 5..13 s only
    assert e[0] == 0.0 and e[-1] == 20.0
    assert np.all(np.diff(e) > 0.9) and np.all(np.diff(e) <= 3.0)
    assert {5.0, 7.0, 9.0, 11.0} <= set(np.round(e, 6))


def test_bar_edges_without_grid():
    e = S.bar_edges(np.zeros(0), 9.0)
    assert e[0] == 0.0 and e[-1] == 9.0 and len(e) == 5  # 2 s blocks, last one absorbs the tail


# ---- segmentation ---------------------------------------------------------------------------

def _features(levels: list[tuple[int, float]], noise: float = 0.0, seed: int = 0) -> dict[str, np.ndarray]:
    """Bar features where only loudness (and drums level) change between parts."""
    rng = np.random.default_rng(seed)
    loud = np.concatenate([np.full(n, v) for n, v in levels]) + noise * rng.standard_normal(sum(n for n, _ in levels))
    nb = len(loud)
    f = {k: np.zeros(nb) for k in S.SCALES}
    f["loudness"] = loud
    f["drums"] = np.clip(loud + 10, -40, 0)
    return f


def test_segment_finds_steps():
    segs = S.segment(S.feature_matrix(_features([(8, -30), (8, -15), (8, -25)], noise=0.5)))
    assert [a for a, _ in segs] == [0, 8, 16]


def test_segment_keeps_a_ramp_in_one_piece():
    """A build ramps several features at once (level, snare density, brightness) and stays one section."""
    f = _features([(8, -30), (8, -30), (8, -12)])
    f["loudness"][8:16] = np.linspace(-30, -14, 8)
    f["drums"] = np.clip(f["loudness"] + 10, -40, 0)
    f["snare"][8:16], f["snare"][16:] = np.linspace(0.5, 4.0, 8), 0.5
    f["brightness"][:8], f["brightness"][8:16], f["brightness"][16:] = 0.5, np.linspace(0.5, 1.0, 8), 0.55
    assert [a for a, _ in S.segment(S.feature_matrix(f))] == [0, 8, 16]


def test_segment_ignores_jitter_and_one_bar_outliers():
    f = _features([(32, -15)], noise=1.0)
    f["snare"][3::4] = 2.0  # a fill every 4th bar
    assert len(S.segment(S.feature_matrix(f))) == 1


def test_segment_respects_max_sections_and_min_length():
    levels = [(2, -40 + 20 * (k % 2)) for k in range(20)]  # 2-bar alternation, every change dramatic
    segs = S.segment(S.feature_matrix(_features(levels)))
    assert len(segs) <= S.MAX_SECTIONS
    assert all(b - a >= S.HARD_MIN_BARS for a, b in segs)
    assert segs[0][0] == 0 and segs[-1][1] == 40
    assert all(segs[k][1] == segs[k + 1][0] for k in range(len(segs) - 1))


# ---- labelling ------------------------------------------------------------------------------

def _stat(lufs, change=0.0, beat=0.0, active=("other",), dens=(0.0, 0.0)):
    return {"lufs": lufs, "change": change, "beat": beat, "active": list(active), "dens_start": dens[0], "dens_end": dens[1]}


def test_labels_for_the_test_clip_shape():
    stats = [
        _stat(-33, 0.0, 0.0, ("drums", "other"), (2, 2)),  # hats + pad
        _stat(-21, 16.0, 1.9, ("drums", "other"), (1, 5)),  # build
        _stat(-15, -0.8, 1.5, ("drums", "bass", "other", "vocals"), (3.5, 3.5)),  # full
        _stat(-26, 0.0, 0.0, ("other", "vocals")),  # breakdown
        _stat(-18, -9.0, 1.5, ("drums", "bass", "other"), (3.5, 3.5)),  # fading outro
    ]
    assert S.label_sections(stats, loudest_bar=-13.5) == ["sparse", "build", "full", "breakdown", "outro"]


def test_labels_groove_silence_and_cut_off_ending():
    stats = [
        _stat(-80, 0.0),  # leading silence
        _stat(-19, 0.0, 1.0, ("drums", "bass")),  # steady beat, well below the loudest: groove
        _stat(-14, 0.0, 1.5, ("drums", "bass", "other")),  # full, and the clip just ends (not falling)
    ]
    assert S.label_sections(stats, loudest_bar=-13.0) == ["silence", "groove", "full"]


def test_build_by_density_and_breakdown_needs_the_beat_to_go():
    stats = [
        _stat(-15, 0.0, 1.5, ("drums", "bass", "other")),
        _stat(-21, 0.0, 1.5, ("drums", "other")),  # quieter but the beat continues: groove, not breakdown
        _stat(-20, 3.5, 1.0, ("drums", "other"), (1.0, 3.0)),  # +3.5 LU with density tripling: build
    ]
    assert S.label_sections(stats, loudest_bar=-14.0) == ["full", "groove", "build"]


# ---- moment detectors ----------------------------------------------------------------------

def test_find_stops_and_silences():
    level = np.full(10 * FRAME_HZ, -15.0)
    level[400:450] = -40.0  # 0.5 s stop while loud
    level[700:850] = -95.0  # 1.5 s of silence
    stops = S.find_stops(level, loud_ref=-15.0)
    assert [(s["start"], s["end"]) for s in stops if not s["near_silent"]] == [(400, 450)]
    assert S.find_silences(level, ref_db=-12.0) == [(700, 850)]


def test_quiet_passage_dips_are_not_stops():
    level = np.full(10 * FRAME_HZ, -40.0)  # 25 dB below the clip's loud level
    level[400:450] = -60.0
    assert S.find_stops(level, loud_ref=-15.0) == []


def test_stem_events_need_a_gap():
    regions = [[0.0, 10.0], [10.5, 20.0], [25.0, 30.0]]
    assert S.stem_events(regions, duration=30.0, gap_s=2.0) == [("out", 20.0), ("in", 25.0)]
    assert S.stem_events([[3.0, 8.0]], duration=30.0, gap_s=2.0) == [("in", 3.0), ("out", 8.0)]


# ---- run(): output contract on a synthetic run folder ---------------------------------------

def _write_run(tmp_path) -> RunContext:
    """40 s at 120 BPM: 8 quiet bars, 8 loud bars with drums+bass, 4 quiet bars."""
    ctx = RunContext(run_dir=tmp_path / "run")
    dur, n = 40.0, 40 * FRAME_HZ + 1
    t = np.arange(n) / FRAME_HZ
    loud = (t >= 16) & (t < 32)
    beats = np.arange(0.0, dur, 0.5)
    rng = np.random.default_rng(0)
    db = {"mix": np.where(loud, -12.0, -30.0), "drums": np.where(loud, -14.0, -90.0),
          "bass": np.where(loud, -16.0, -90.0), "other": np.full(n, -25.0), "vocals": np.full(n, -90.0)}
    curves = {"times": t, "lufs": db["mix"] - 2 + rng.normal(0, 0.3, n), "brightness": np.full(n, 0.5),
              "onset": np.zeros(n)}
    for k, v in db.items():
        curves[f"db_{k}"] = v + rng.normal(0, 0.3, n)
        curves[f"energy_{k}"] = np.clip((v + 40) / 30, 0, 1)
    np.savez_compressed(ctx.cache_path("energy.npz"), **{k: v.astype(np.float32) for k, v in curves.items()})
    ctx.write_json("rhythm.json", {"beats": beats.tolist(), "downbeat_idx": list(range(0, len(beats), 4)),
                                   "beats_per_bar": 4, "duration": dur})
    ctx.write_json("energy.json", {
        "duration": dur,
        "levels": {k: {"ref_db": -12.0 if k == "mix" else -14.0} for k in ("mix", *STEM_NAMES)},
        "active": {"drums": [[16.0, 32.0]], "bass": [[16.0, 32.0]], "other": [[0.0, dur]], "vocals": []},
    })
    kicks = [{"t": float(b), "type": "kick", "strength": 1.0} for b in beats if 16 <= b < 32]
    ctx.write_json("hits.json", {"method": "test", "onsets": [h["t"] for h in kicks], "hits": kicks,
                                 "counts": {"kick": len(kicks), "snare": 0, "hat": 0}})
    return ctx


def test_run_writes_the_contract(tmp_path):
    ctx = _write_run(tmp_path)
    S.run(ctx)
    out = ctx.read_json("sections.json")
    assert set(out) == {"method", "sections", "moments"}
    secs = out["sections"]
    assert secs[0]["start"] == 0.0 and secs[-1]["end"] == 40.0
    assert all(a["end"] == b["start"] for a, b in zip(secs, secs[1:]))
    assert [s["start"] for s in secs] == [0.0, 16.0, 32.0]
    assert [s["character"] for s in secs] == ["sparse", "full", "breakdown"]
    assert secs[1]["bars"] == [9, 16] and secs[1]["active_stems"] == ["drums", "bass", "other"]
    for s in secs:
        assert s["character"] in ("silence", "sparse", "build", "groove", "full", "breakdown", "outro")
        assert 0 <= s["energy"] <= 1 and s["trend"] in ("rising", "falling", "flat")
        assert set(s["hits_per_beat"]) == {"kick", "snare", "hat", "all"}
    ms = out["moments"]
    assert [m["t"] for m in ms] == sorted(m["t"] for m in ms)
    types = {m["type"] for m in ms}
    assert {"drop", "breakdown", "loudest"} <= types
    drop = next(m for m in ms if m["type"] == "drop")
    assert drop["t"] == pytest.approx(16.0, abs=0.05) and "kick" in drop["description"]
    assert all(0 <= m["strength"] <= 1 for m in ms)


def test_run_without_a_beat_grid(tmp_path):
    """Beatless audio (empty rhythm): 2 s blocks stand in for bars, the contract still holds."""
    ctx = _write_run(tmp_path)
    ctx.write_json("rhythm.json", {"beats": [], "downbeat_idx": [], "beats_per_bar": 4, "duration": 40.0})
    S.run(ctx)
    secs = ctx.read_json("sections.json")["sections"]
    assert secs[0]["start"] == 0.0 and secs[-1]["end"] == 40.0
    assert all(a["end"] == b["start"] for a, b in zip(secs, secs[1:]))
    assert all("bars" not in s for s in secs)
    assert [s["start"] for s in secs] == [0.0, 16.0, 32.0]
