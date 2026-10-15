"""Fast tests for drums.analyze on short synthetic drum signals (no models, no files)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from audio4llm import drums

SR = 44100
FRAME_HZ = 100
BEAT = 0.5  # 120 BPM


def _env(n: int, tau: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (tau * SR))


def _kick(rng) -> np.ndarray:
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 48 + 110 * np.exp(-t / 0.03)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.16)
    hp = signal.butter(2, 2000, btype="highpass", fs=SR, output="sos")
    return body + 0.3 * signal.sosfilt(hp, rng.standard_normal(n)) * _env(n, 0.002)


def _snare(rng) -> np.ndarray:
    n = int(0.25 * SR)
    bp = signal.butter(2, [1500, 9000], btype="bandpass", fs=SR, output="sos")
    noise = signal.sosfilt(bp, rng.standard_normal(n)) * _env(n, 0.07)
    return 0.8 * noise + 0.5 * np.sin(2 * np.pi * 190 * np.arange(n) / SR) * _env(n, 0.05)


def _hat(rng) -> np.ndarray:
    n = int(0.07 * SR)
    hp = signal.butter(4, 7500, btype="highpass", fs=SR, output="sos")
    return signal.sosfilt(hp, rng.standard_normal(n)) * _env(n, 0.018)


def _render(events: list[tuple[float, str, float]], seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sounds = {"kick": _kick(rng), "snare": _snare(rng), "hat": _hat(rng)}
    y = np.zeros(int(seconds * SR))
    for t, kind, gain in events:
        i = int(round(t * SR))
        x = sounds[kind][: len(y) - i]
        y[i:i + len(x)] += gain * x
    return (y + 1e-4 * rng.standard_normal(len(y))).astype(np.float32)


def _analyze(y: np.ndarray):
    n_frames = 1 + len(y) // (SR // FRAME_HZ)
    return drums.analyze(y, SR, n_frames, FRAME_HZ), n_frames


def _times(hits: dict, kind: str) -> np.ndarray:
    return np.array([h["t"] for h in hits["hits"] if h["type"] == kind])


def _recall(found: np.ndarray, truth: list[float], tol: float = 0.05) -> float:
    return float(np.mean([np.any(np.abs(found - t) <= tol) for t in truth])) if truth else 1.0


def _false(found: np.ndarray, truth: list[float], tol: float = 0.05) -> int:
    return int(sum(not np.any(np.abs(np.asarray(truth) - f) <= tol) for f in found)) if len(truth) else len(found)


def test_four_on_the_floor_with_coincident_hats():
    """Kick every beat, snare on 2 and 4, 8th hats: kick + hat on the same beat are both found."""
    kicks = [b * BEAT for b in range(16)]
    snares = [b * BEAT for b in range(16) if b % 4 in (1, 3)]
    hats = [k * BEAT / 2 for k in range(32)]
    y = _render([(t, "kick", 0.9) for t in kicks] + [(t, "snare", 0.45) for t in snares]
                + [(t, "hat", 0.16) for t in hats], 8.5)
    (hits, bands), _ = _analyze(y)
    for kind, truth in (("kick", kicks), ("snare", snares), ("hat", hats)):
        found = _times(hits, kind)
        assert _recall(found, truth) >= 0.9, kind
        assert _false(found, truth) <= 1, kind


def test_no_phantom_hats_from_snare_noise_or_kick_click():
    """Offbeat-only hats: snares and kicks on the beat must not spawn hats."""
    kicks = [b * BEAT for b in range(16) if b % 2 == 0]
    snares = [b * BEAT for b in range(16) if b % 2 == 1]
    hats = [b * BEAT + BEAT / 2 for b in range(16)]
    y = _render([(t, "kick", 0.9) for t in kicks] + [(t, "snare", 0.5) for t in snares]
                + [(t, "hat", 0.15) for t in hats], 8.5)
    (hits, _), _ = _analyze(y)
    found = _times(hits, "hat")
    assert _recall(found, hats) >= 0.9
    assert _false(found, hats) == 0
    assert _false(_times(hits, "snare"), snares) == 0  # kick clicks are not snares


def test_snare_roll_without_hats_gives_no_hats():
    roll = [2.0 + k * BEAT / 4 for k in range(16)]
    y = _render([(t, "snare", 0.2 + 0.02 * k) for k, t in enumerate(roll)], 6.0)
    (hits, _), _ = _analyze(y)
    assert _recall(_times(hits, "snare"), roll) >= 0.9
    assert len(_times(hits, "hat")) == 0


def test_bass_bleed_without_transients_is_not_a_kick():
    """Kicks for 4 s, then 4 s of soft sustained low notes (bleed) with no click: no kicks there."""
    kicks = [b * BEAT for b in range(8)]
    y = _render([(t, "kick", 0.9) for t in kicks], 8.5).astype(np.float64)
    t = np.arange(len(y)) / SR
    for k, start in enumerate(np.arange(4.25, 8.0, 0.75)):
        on = (t >= start) & (t < start + 0.6)
        y[on] += 0.1 * np.sin(2 * np.pi * (55 + 10 * (k % 3)) * t[on]) * np.minimum(1, (t[on] - start) / 0.01)
    (hits, _), _ = _analyze(y.astype(np.float32))
    found = _times(hits, "kick")
    assert _recall(found, kicks) == 1.0
    assert not np.any(found > 4.1)


def test_hit_at_time_zero_is_found():
    (hits, _), _ = _analyze(_render([(0.0, "kick", 0.9), (1.0, "kick", 0.9)], 2.0))
    assert _recall(_times(hits, "kick"), [0.0, 1.0]) == 1.0


def test_output_contract():
    y = _render([(0.5, "kick", 0.9), (1.0, "snare", 0.5), (1.25, "hat", 0.2)], 2.0)
    (hits, bands), n_frames = _analyze(y)
    assert set(hits) == {"method", "onsets", "hits", "counts"}
    assert "estimate" in hits["method"]
    assert set(bands) == {"low", "mid", "high"}
    for v in bands.values():
        assert v.shape == (n_frames,) and v.dtype == np.float32
        assert v.min() >= 0.0 and v.max() <= 1.0
    ts = [h["t"] for h in hits["hits"]]
    assert ts == sorted(ts)
    assert all(set(h) == {"t", "type", "strength"} and 0 <= h["strength"] <= 1 for h in hits["hits"])
    assert hits["counts"] == {k: sum(h["type"] == k for h in hits["hits"]) for k in ("kick", "snare", "hat")}
    assert all(any(abs(o - t) <= 0.04 for o in hits["onsets"]) for t in ts)  # every hit is an onset
    assert bands["low"][48:56].max() > 0.9  # the kick at 0.5 s pumps the low band


@pytest.mark.parametrize("seconds", [0.0, 0.01, 2.0])
def test_silence_and_tiny_inputs(seconds):
    y = np.zeros(int(seconds * SR), dtype=np.float32)
    n_frames = 1 + len(y) // (SR // FRAME_HZ)
    hits, bands = drums.analyze(y, SR, n_frames, FRAME_HZ)
    assert hits["hits"] == [] and hits["onsets"] == []
    assert hits["counts"] == {"kick": 0, "snare": 0, "hat": 0}
    assert all(v.shape == (n_frames,) and not v.any() for v in bands.values())
