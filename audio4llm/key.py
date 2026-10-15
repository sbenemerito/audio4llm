"""Global key estimate: Krumhansl-Kessler key profiles correlated with a mean chroma vector.

``estimate_key(chroma_mean)`` is pure numpy (unit-testable on synthetic chroma).
``harmonic_chroma(y, sr)`` computes its input from audio with librosa; the notes
stage feeds it the drum-less stems (bass + other + vocals).

How it works: the 12-bin chroma is Pearson-correlated with the major and minor
probe-tone profiles rotated to all 12 tonics (24 keys); the best key wins.
A relative major/minor pair (C major / A minor) uses the same seven notes, so
the two are often within ~0.1 of each other and the runner-up is reported too.

Output of ``estimate_key``::

    {"label": "A minor", "tonic": "A", "mode": "minor", "confidence": 0..1,
     "correlation": r, "runner_up": "C major", "runner_up_correlation": r,
     "relative": "C major", "method": "..."}

Pitch classes use ASCII sharps (C, C#, D, ... B), never flats.
"""

from __future__ import annotations

import numpy as np

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Krumhansl & Kessler (1982) probe-tone ratings; index 0 is the tonic.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# confidence = fit * separation, both 0..1:
# fit: a random 12-bin chroma vector already reaches r ~= 0.6 with its best of the 24 keys
# (median over uniform Dirichlet draws), so that maps to 0. Clearly tonal material scores higher:
# an evenly weighted Am-F-C-G triad loop 0.93, the drum-less stems of the synthetic test clip
# 0.81, a bare major scale 0.76. r >= 0.9 maps to 1.
R_CHANCE = 0.6
R_CLEAR = 0.9
# separation: how far the winner is ahead of the runner-up. A lead of 0.1 or more counts as
# decisive; a relative major/minor or fifth-related key within a few hundredths pulls it toward 0,
# because then the label itself is a coin toss even when the scale is clear.
MARGIN_CLEAR = 0.1

METHOD = ("estimate: Krumhansl-Kessler major/minor profiles correlated with the mean CQT chroma of the "
          "drum-less stems; one key for the whole clip, relative major/minor are easily confused (see runner_up)")


def key_name(tonic: int, mode: str) -> str:
    return f"{PITCH_CLASSES[tonic % 12]} {mode}"


def relative_key(tonic: int, mode: str) -> tuple[int, str]:
    """Relative minor of a major key (down 3 semitones) or relative major of a minor key (up 3)."""
    return ((tonic + 9) % 12, "minor") if mode == "major" else ((tonic + 3) % 12, "major")


def _profiles() -> tuple[np.ndarray, list[tuple[int, str]]]:
    """(24, 12) profile matrix and the (tonic, mode) of each row: 12 majors then 12 minors."""
    rows = [np.roll(KK_MAJOR, t) for t in range(12)] + [np.roll(KK_MINOR, t) for t in range(12)]
    keys = [(t, "major") for t in range(12)] + [(t, "minor") for t in range(12)]
    return np.array(rows), keys


def key_correlations(chroma_mean: np.ndarray) -> np.ndarray | None:
    """Pearson r of the chroma with each of the 24 profiles (order of ``_profiles``), or None if flat."""
    c = np.asarray(chroma_mean, dtype=np.float64).reshape(-1)
    if c.shape != (12,):
        raise ValueError(f"expected 12 chroma bins, got {c.shape}")
    if not np.all(np.isfinite(c)) or c.std() < 1e-9:
        return None  # silence or a perfectly flat spectrum: no tonal centre to find
    prof, _ = _profiles()
    pz = (prof - prof.mean(axis=1, keepdims=True)) / prof.std(axis=1, keepdims=True)
    cz = (c - c.mean()) / c.std()
    return pz @ cz / 12


def estimate_key(chroma_mean: np.ndarray) -> dict:
    """Best of the 24 major/minor keys for a 12-bin chroma vector (index 0 = C). Pure function."""
    r = key_correlations(chroma_mean)
    if r is None:
        return {"label": None, "tonic": None, "mode": None, "confidence": 0.0, "correlation": None,
                "runner_up": None, "runner_up_correlation": None, "relative": None,
                "method": METHOD + "; no tonal content found"}
    _, keys = _profiles()
    order = np.argsort(-r, kind="stable")
    best, second = int(order[0]), int(order[1])
    tonic, mode = keys[best]
    fit = np.clip((r[best] - R_CHANCE) / (R_CLEAR - R_CHANCE), 0.0, 1.0)
    separation = np.clip((r[best] - r[second]) / MARGIN_CLEAR, 0.0, 1.0)
    return {
        "label": key_name(tonic, mode),
        "tonic": PITCH_CLASSES[tonic],
        "mode": mode,
        "confidence": round(float(fit * separation), 2),
        "correlation": round(float(r[best]), 3),
        "runner_up": key_name(*keys[second]),
        "runner_up_correlation": round(float(r[second]), 3),
        "relative": key_name(*relative_key(tonic, mode)),
        "method": METHOD,
    }


def harmonic_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    """Mean 12-bin CQT chroma of mono audio, weighted by loudness (index 0 = C).

    ``norm=None`` keeps each frame's magnitude, so loud sustained harmony counts more than
    near-silent frames (whose per-frame-normalised chroma would be mostly noise).
    """
    import librosa

    chroma = librosa.feature.chroma_cqt(y=np.asarray(y, dtype=np.float32), sr=sr, hop_length=512, norm=None)
    total = chroma.sum(axis=1)
    return total / total.sum() if total.sum() > 0 else total
