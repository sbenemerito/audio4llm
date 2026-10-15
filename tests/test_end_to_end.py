"""Full pipeline on the synthetic test clip, checked against ground_truth.json.

Tolerances are set for what the models can actually do on this clip; where a
check is looser than you might expect, the comment says why.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def match(est, ref, tol):
    """Greedy one-to-one matching of event times. Returns (matched, precision, recall)."""
    est, ref = sorted(est), sorted(ref)
    used, hits = set(), 0
    for r in ref:
        cands = [(abs(e - r), i) for i, e in enumerate(est) if i not in used and abs(e - r) <= tol]
        if cands:
            used.add(min(cands)[1])
            hits += 1
    precision = hits / len(est) if est else 0.0
    recall = hits / len(ref) if ref else 1.0
    return hits, precision, recall


def section_of(analysis, t):
    return next(s for s in analysis["sections"] if s["start"] <= t < s["end"])


def beats_in_bars(analysis, first, last):
    return [i for i, b in enumerate(analysis["grid"]["beats"]) if first <= b["bar"] <= last]


# ---- pipeline / files ----------------------------------------------------------------

def test_every_stage_succeeded(pipeline_run):
    _, results = pipeline_run
    assert {r.status for r in results} <= {"ran", "cached"}, [(r.name, r.status, r.reason) for r in results]


def test_output_files(run_dir, analysis, curves):
    for f in ("input.wav", "analysis.json", "curves.json", "summary.md", "overview.png",
              "stems/vocals.wav", "stems/drums.wav", "stems/bass.wav", "stems/other.wav"):
        assert (run_dir / f).exists(), f
    assert analysis["schema_version"] == "1.0"
    for key in ("source", "global", "grid", "sections", "moments", "hits", "stems", "notes", "lyrics", "curves"):
        assert analysis[key] is not None, key
    n_beats = len(analysis["grid"]["beats"])
    for name in ("mix", "vocals", "drums", "bass", "other"):
        assert len(analysis["stems"][name]["per_beat"]) == n_beats
    assert curves["hz"] == 20
    assert all(len(v) == curves["n"] for v in curves["curves"].values())
    assert curves["n"] == pytest.approx(analysis["source"]["duration"] * 20, abs=1)


def test_rerun_only_assembles(pipeline_run):
    from audio4llm.pipeline import run

    ctx, _ = pipeline_run
    _, results = run(ctx.run_dir)  # re-run on the run folder
    status = {r.name: r.status for r in results}
    assert status.pop("assemble") == "ran"
    assert set(status.values()) == {"cached"}


def test_summary_mentions_basics(run_dir):
    text = (run_dir / "summary.md").read_text()
    assert "120.0 BPM" in text and "4/4" in text
    assert "## Timeline" in text and "## Lyrics" in text


# ---- rhythm --------------------------------------------------------------------------

def test_tempo_and_meter(analysis):
    g = analysis["global"]
    assert abs(g["tempo_bpm"] - 120.0) <= 1.0
    assert g["time_signature"] == "4/4"


def test_beats_within_70ms(analysis, ground_truth):
    est = [b["t"] for b in analysis["grid"]["beats"]]
    _, precision, recall = match(est, ground_truth["beats"], 0.07)
    assert recall >= 0.9 and precision >= 0.9, (precision, recall)


def test_downbeats_on_the_right_beat(analysis, ground_truth):
    est = [b["t"] for b in analysis["grid"]["beats"] if b["is_downbeat"]]
    _, precision, recall = match(est, ground_truth["downbeats"], 0.07)
    assert recall >= 0.9 and precision >= 0.9, (precision, recall)


# ---- drums ---------------------------------------------------------------------------

def test_kicks_detected_in_full_section(analysis, ground_truth):
    lo, hi = 32.0, 56.0
    ref = [t for t in ground_truth["hits"]["kick"] if lo <= t < hi]
    est = [row[0] for row in analysis["hits"]["kick"] if lo - 0.05 <= row[0] < hi]
    _, precision, recall = match(est, ref, 0.05)
    assert recall >= 0.8, recall
    assert precision >= 0.7, precision


def test_no_kicks_in_breakdown(analysis):
    assert not [row for row in analysis["hits"]["kick"] if 57.0 <= row[0] < 71.0]


# ---- energy / stems ------------------------------------------------------------------

def test_breakdown_is_quieter_than_full(analysis):
    mix = np.asarray(analysis["stems"]["mix"]["per_beat"])
    drums = np.asarray(analysis["stems"]["drums"]["per_beat"])
    full = beats_in_bars(analysis, 17, 28)
    breakdown = beats_in_bars(analysis, 29, 36)
    assert mix[full].mean() - mix[breakdown].mean() >= 0.2, (mix[full].mean(), mix[breakdown].mean())
    assert drums[breakdown].mean() < 0.1


def test_stem_activity(analysis, ground_truth):
    def active_at(stem, t):
        return any(a["start"] <= t < a["end"] for a in analysis["stems"][stem]["active"])

    for v in ground_truth["vocals"]:
        assert active_at("vocals", (v["start"] + v["end"]) / 2)
    assert not active_at("vocals", 10.0) and not active_at("vocals", 50.0)
    assert active_at("drums", 44.0) and not active_at("drums", 64.0)
    assert active_at("bass", 44.0) and not active_at("bass", 8.0)


# ---- sections / moments --------------------------------------------------------------

def test_section_boundaries(analysis, ground_truth):
    starts = [s["start"] for s in analysis["sections"]]
    truth = [s["start"] for s in ground_truth["sections"][1:]]
    found = sum(any(abs(s - t) <= 2.0 for s in starts) for t in truth)  # within one bar
    assert found >= len(truth) - 1, (starts, truth)
    assert len(analysis["sections"]) <= 8


def test_section_characters(analysis):
    assert section_of(analysis, 44.0)["character"] in ("full", "drop")
    assert section_of(analysis, 64.0)["character"] == "breakdown"
    assert section_of(analysis, 4.0)["character"] in ("sparse", "intro")


def test_moments(analysis):
    moments = analysis["moments"]
    drops = [m["t"] for m in moments if m["type"] == "drop"]
    assert any(abs(t - 32.0) <= 1.0 for t in drops), moments
    entries = [m["t"] for m in moments if m["type"] == "vocal_entry"]
    assert entries and abs(entries[0] - 36.0) <= 0.5, moments


# ---- lyrics --------------------------------------------------------------------------

def words(text):
    return re.findall(r"[a-z']+", text.lower())


def test_lyrics_recovered(analysis, ground_truth):
    lyr = analysis["lyrics"]
    assert lyr["status"] == "ok"
    assert analysis["global"]["language"] == "en"
    got = [w for ln in lyr["lines"] for w in words(" ".join(x["w"] for x in ln["words"]))]
    truth = [w for v in ground_truth["vocals"] for w in v["words"]]
    recovered = sum(1 for w in truth if w in got)
    # spoken TTS over music is easy for Whisper; demand most words, not all (stem bleed + short phrases)
    assert recovered / len(truth) >= 0.6, (got, truth)


def test_lyric_timestamps_inside_phrases(analysis, ground_truth):
    windows = [(v["start"] - 0.5, v["end"] + 0.5) for v in ground_truth["vocals"]]
    for ln in analysis["lyrics"]["lines"]:
        for w in ln["words"]:
            assert any(a <= w["start"] <= b for a, b in windows), w


# ---- notes / key ---------------------------------------------------------------------

ROOT_PC = {"Am": "A", "F": "F", "C": "C", "G": "G"}


def test_bass_roots_follow_chords(analysis, ground_truth):
    roots = {b["bar"]: b["bass_root"] for b in analysis["notes"]["per_bar"] if b.get("bass_root")}
    bars_with_bass = sorted({int(n["start"] // 2.0) + 1 for n in ground_truth["bass_notes"]})
    chord_of_bar = {int(c["start"] // 2.0) + 1: ROOT_PC[c["name"]] for c in ground_truth["chords"]}
    hits = sum(roots.get(b) == chord_of_bar[b] for b in bars_with_bass)
    assert hits / len(bars_with_bass) >= 0.75, roots


def test_key_estimate(analysis, ground_truth):
    assert analysis["global"]["key"]["label"] in ground_truth["key_equivalents"]
