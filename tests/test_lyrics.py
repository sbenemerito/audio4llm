"""Fast tests for the lyrics stage: no model, fake whisper output only."""

from __future__ import annotations

import json

import numpy as np
import pytest

from audio4llm.config import Config
from audio4llm.context import RunContext
from audio4llm.stages import lyrics


def word(text, start, end, p=0.9):
    return {"word": text, "start": start, "end": end, "probability": p}


def segment(words, *, avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.2, text=None):
    return {
        "start": words[0]["start"] if words else 0.0,
        "end": words[-1]["end"] if words else 0.0,
        "text": text if text is not None else "".join(w["word"] for w in words),
        "avg_logprob": avg_logprob,
        "no_speech_prob": no_speech_prob,
        "compression_ratio": compression_ratio,
        "words": words,
    }


def spoken(text, start, step=0.25, p=0.9):
    """Whisper-style words (leading spaces) spaced `step` apart."""
    return [word(" " + t, start + i * step, start + (i + 1) * step - 0.02, p) for i, t in enumerate(text.split())]


# ---- regions ---------------------------------------------------------------------

def test_plan_regions_pads_clips_and_merges():
    active = [[0.1, 1.0], [1.5, 2.0], [10.0, 12.0], [79.8, 80.0]]
    assert lyrics.plan_regions(active, 80.0, pad=0.4, merge_gap=2.0) == [[0.0, 2.4], [9.6, 12.4], [79.4, 80.0]]


def test_plan_regions_merges_fragments_and_sorts():
    active = [[5.0, 5.5], [3.0, 3.5], [4.0, 4.6]]
    assert lyrics.plan_regions(active, 60.0, pad=0.25, merge_gap=1.0) == [[2.75, 5.75]]


def test_voiced_mask_is_relative_to_each_region():
    sr = lyrics.SR
    y = np.full(10 * sr, 1e-5, np.float32)  # -100 dB floor
    y[2 * sr:3 * sr] = 0.1  # loud phrase
    y[6 * sr:7 * sr] = 0.003  # quiet phrase, ~30 dB lower, own region
    mask = lyrics.voiced_mask(y, [[1.5, 3.5], [5.5, 7.5]])
    hz = lyrics.VOICE_HZ
    assert mask[int(2.5 * hz)] and mask[int(6.5 * hz)]
    assert not mask[int(1.8 * hz)] and not mask[int(5.8 * hz)] and not mask[int(4.5 * hz)]


def test_trim_to_voice_moves_edges_off_silence_only():
    hz = lyrics.VOICE_HZ
    voiced = np.zeros(10 * hz, bool)
    voiced[int(2.0 * hz):int(3.0 * hz)] = True
    ws = [{"raw": " Hold", "start": 1.4, "end": 2.3, "p": 0.9},  # swallowed 0.6 s of silence
          {"raw": " on", "start": 2.3, "end": 2.5, "p": 0.9},
          {"raw": " to", "start": 2.92, "end": 3.06, "p": 0.9},  # 60 ms past the voice: too short to trim
          {"raw": " me", "start": 2.9, "end": 3.5, "p": 0.9},
          {"raw": " ghost", "start": 5.0, "end": 5.5, "p": 0.9}]  # no voice at all: untouched
    lyrics.trim_to_voice(ws, voiced, min_gap=0.15)
    assert [(w["start"], w["end"]) for w in ws] == [(2.0, 2.3), (2.3, 2.5), (2.92, 3.06), (2.9, 3.0), (5.0, 5.5)]


def test_overlaps_active_uses_slack():
    act = [[10.0, 12.0]]
    assert lyrics.overlaps_active(11.0, 11.2, act)
    assert lyrics.overlaps_active(12.1, 12.4, act, slack=0.25)
    assert not lyrics.overlaps_active(12.5, 12.8, act, slack=0.25)
    assert not lyrics.overlaps_active(9.0, 9.5, act, slack=0.25)


# ---- segment filters -------------------------------------------------------------

@pytest.mark.parametrize("text", [
    " Thank you.", "Thanks for watching!", " Thank you. Thank you.", " you", "[Music]", "(upbeat music)",
    " ♪ ♪", "...", " Subtitles by the Amara.org community", "ご視聴ありがとうございました",
])
def test_known_hallucinations(text):
    assert lyrics.hallucination_reason(text) is not None


@pytest.mark.parametrize("text", [" Thank you for loving me", " Hold on to the night", " You and I", " So far away"])
def test_real_lines_are_not_hallucinations(text):
    assert lyrics.hallucination_reason(text) is None


def test_segment_drop_reasons():
    ws = spoken("hold on to the night", 0.0)
    assert lyrics.segment_drop_reason(segment(ws)) is None
    assert "compression" in lyrics.segment_drop_reason(segment(ws, compression_ratio=3.1))
    assert "no speech" in lyrics.segment_drop_reason(segment(ws, no_speech_prob=0.8, avg_logprob=-0.7))
    assert lyrics.segment_drop_reason(segment(ws, no_speech_prob=0.8, avg_logprob=-0.2)) is None  # confident text
    assert "low confidence" in lyrics.segment_drop_reason(segment(ws, avg_logprob=-1.3))


# ---- loops -----------------------------------------------------------------------

def test_find_repeats():
    toks = "a b c b c b c b c d".split()
    assert lyrics.find_repeats(toks, min_repeats=4) == [(1, 9, 2)]
    assert lyrics.find_repeats("x x x y".split(), min_repeats=4) == []
    assert lyrics.find_repeats("la la la la la la".split(), min_repeats=4) == [(0, 6, 1)]


def _internal(ws):
    return [{"raw": w["word"], "start": w["start"], "end": w["end"], "p": w["probability"], "seg": 0} for w in ws]


def test_drop_loops_removes_crammed_repeats_keeps_first():
    ws = spoken("go", 1.0) + [word(" I'm", 1.3 + 0.04 * i, 1.33 + 0.04 * i) if i % 2 == 0 else
                              word(" gonna", 1.3 + 0.04 * i, 1.33 + 0.04 * i) for i in range(12)]
    kept, dropped = lyrics.drop_loops(_internal(ws))
    assert [w["raw"].strip() for w in kept] == ["go", "I'm", "gonna"]
    assert len(dropped) == 1 and "repetition loop" in dropped[0]["reason"] and "x6" in dropped[0]["reason"]


def test_drop_loops_keeps_sung_repetition():
    ws = spoken("no no no no", 5.0, step=0.4)
    kept, dropped = lyrics.drop_loops(_internal(ws))
    assert len(kept) == 4 and dropped == []


def test_drop_loops_drops_improbable_repetition():
    ws = spoken("yeah yeah yeah yeah yeah", 5.0, step=0.4, p=0.2)
    kept, dropped = lyrics.drop_loops(_internal(ws))
    assert len(kept) == 1 and len(dropped) == 1


# ---- lines -----------------------------------------------------------------------

def test_split_lines_at_pause():
    ws = _internal(spoken("never let me go", 0.0) + spoken("the music", 2.0))
    lines = lyrics.split_lines(ws)
    assert [len(ln) for ln in lines] == [4, 2]


def test_split_long_line_at_punctuation():
    ws = spoken("Hold on to the night, we are dancing in the light.", 36.0, step=0.22)
    lines = lyrics.split_lines(_internal(ws))
    assert ["".join(w["raw"] for w in ln).strip() for ln in lines] == [
        "Hold on to the night,", "we are dancing in the light."]


def test_split_long_line_without_punctuation_prefers_gap():
    ws = spoken("one two three four five", 0.0) + spoken("six seven eight nine ten", 1.6)  # 0.37 s gap
    lines = lyrics.split_lines(_internal(ws))
    assert [len(ln) for ln in lines] == [5, 5]


def test_short_line_is_not_split():
    lines = lyrics.split_lines(_internal(spoken("Oh. Yeah, baby.", 0.0)))
    assert len(lines) == 1


# ---- whole result ----------------------------------------------------------------

def test_process_result_offsets_filters_and_splits():
    result = {"segments": [
        segment([word(" Thank", 0.1, 0.3), word(" you.", 0.3, 0.5)]),  # filler phrase
        segment(spoken("Never let me go, the music sets us free.", 1.0, step=0.25)),
        segment([word(" hey", 6.0, 6.3)]),  # outside activity once offset
        segment(spoken("garbage words here", 7.0), avg_logprob=-1.4),
        segment([], text=" "),  # empty segment is ignored silently
    ]}
    active = [[61.0, 63.5]]
    lines, dropped = lyrics.process_result(result, 60.0, active)
    assert [ln["text"] for ln in lines] == ["Never let me go,", "the music sets us free."]
    assert lines[0]["start"] == 61.0 and lines[1]["end"] == pytest.approx(63.23)
    assert lines[0]["words"][0] == {"w": "Never", "start": 61.0, "end": 61.23, "p": 0.9}
    reasons = [d["reason"] for d in dropped]
    assert reasons[0].startswith("filler phrase") and dropped[0]["start"] == 60.1
    assert "outside vocal activity" in reasons
    assert any(r.startswith("low confidence") for r in reasons)
    assert all(d["start"] <= d["end"] for d in dropped)


def test_process_result_trims_leading_silence_before_first_word():
    hz = lyrics.VOICE_HZ
    voiced = np.zeros(80 * hz, bool)
    voiced[int(36.0 * hz):int(37.0 * hz)] = True
    seg = segment([word(" Hold", 0.0, 0.74), word(" on", 0.74, 0.9)])  # region starts at 35.5
    lines, _ = lyrics.process_result({"segments": [seg]}, 35.5, [[35.9, 37.0]], voiced)
    assert lines[0]["start"] == 36.0 and lines[0]["words"][0]["start"] == 36.0


def test_process_result_drops_only_the_outside_words_of_a_segment():
    ws = spoken("la", 0.0) + spoken("hold on to me", 2.0)
    lines, dropped = lyrics.process_result({"segments": [segment(ws)]}, 0.0, [[1.9, 3.2]])
    assert [ln["text"] for ln in lines] == ["hold on to me"]
    assert dropped == [{"start": 0.0, "end": 0.23, "text": "la", "reason": "outside vocal activity"}]


# ---- stage -----------------------------------------------------------------------

def _run_dir(tmp_path, energy):
    ctx = RunContext(run_dir=tmp_path / "run")
    ctx.write_json("energy.json", energy)
    return ctx


def _no_model(*a, **k):
    raise AssertionError("the model must not be loaded")


@pytest.mark.parametrize("energy", [
    {"active": {"vocals": [[10.0, 10.5]]}, "has_vocals": False, "vocal_seconds": 0.5},
    {"active": {"vocals": []}, "has_vocals": True},
    {"active": {}, "has_vocals": False},
])
def test_instrumental_skips_without_loading_model(tmp_path, monkeypatch, energy):
    monkeypatch.setattr(lyrics, "load_model", _no_model)
    monkeypatch.setattr(lyrics, "transcribe", _no_model)
    ctx = _run_dir(tmp_path, energy)
    lyrics.run(ctx)
    out = ctx.read_json("lyrics.json")
    assert out["status"] == "skipped" and out["reason"]
    assert out["lines"] == [] and out["dropped"] == [] and out["regions"] == []
    assert out["model"] == Config().whisper_model


class FakeModel:
    is_multilingual = True


def _fake_pipeline(monkeypatch, calls):
    monkeypatch.setattr(lyrics.audio, "load_mono", lambda path, sr: (np.zeros(80 * sr, np.float32), sr))
    monkeypatch.setattr(lyrics, "load_model", lambda repo: FakeModel())
    monkeypatch.setattr(lyrics, "release_model", lambda: calls.append("released"))
    monkeypatch.setattr(lyrics, "detect_language", lambda model, y: ("en", 0.93))

    def fake_transcribe(y, repo, language):
        calls.append((len(y) / lyrics.SR, language))
        return {"language": language, "segments": [segment(spoken("hold on", 0.5))]}

    monkeypatch.setattr(lyrics, "transcribe", fake_transcribe)


def test_run_detects_language_and_offsets_regions(tmp_path, monkeypatch):
    calls = []
    _fake_pipeline(monkeypatch, calls)
    ctx = _run_dir(tmp_path, {"active": {"vocals": [[36.0, 38.5], [60.0, 62.5]]}, "has_vocals": True})
    lyrics.run(ctx)
    out = ctx.read_json("lyrics.json")
    assert out["status"] == "ok" and out["reason"] is None
    assert (out["language"], out["language_source"], out["language_probability"]) == ("en", "detected", 0.93)
    assert out["regions"] == [[35.6, 38.9], [59.6, 62.9]]
    assert [ln["start"] for ln in out["lines"]] == [36.1, 60.1]
    assert calls[0] == (pytest.approx(3.3), "en") and calls[-1] == "released"
    assert list(out) == ["status", "reason", "model", "language", "language_source", "language_probability",
                         "regions", "lines", "dropped"]
    json.dumps(out)


def test_run_language_override(tmp_path, monkeypatch):
    calls = []
    _fake_pipeline(monkeypatch, calls)
    monkeypatch.setattr(lyrics, "detect_language", _no_model)
    ctx = _run_dir(tmp_path, {"active": {"vocals": [[36.0, 38.5]]}, "has_vocals": True})
    ctx.config = Config(language="Spanish")
    lyrics.run(ctx)
    out = ctx.read_json("lyrics.json")
    assert (out["language"], out["language_source"], out["language_probability"]) == ("es", "user", None)
    assert calls[0][1] == "es"


def test_run_everything_filtered_is_empty(tmp_path, monkeypatch):
    calls = []
    _fake_pipeline(monkeypatch, calls)
    monkeypatch.setattr(lyrics, "transcribe", lambda y, repo, language: {
        "segments": [segment([word(" Thank", 0.5, 0.7), word(" you.", 0.7, 0.9)])]})
    ctx = _run_dir(tmp_path, {"active": {"vocals": [[36.0, 38.5]]}, "has_vocals": True})
    lyrics.run(ctx)
    out = ctx.read_json("lyrics.json")
    assert out["status"] == "empty" and out["lines"] == [] and len(out["dropped"]) == 1
