"""Drum-stem onsets and kick / snare / hat estimates (called by the energy stage).

Contract:
    analyze(mono, sr, n_frames, frame_hz) -> (hits_json, bands)

    hits_json = {
        "method": "<one line describing the approach>",
        "onsets": [t, ...],                       # all drum-stem onsets, seconds
        "hits": [{"t": 1.234, "type": "kick"|"snare"|"hat", "strength": 0..1}, ...],  # sorted by t
        "counts": {"kick": n, "snare": n, "hat": n},
    }
    bands = {"low": arr, "mid": arr, "high": arr}  # 0..1 band envelopes, len n_frames, at frame_hz

TODO: placeholder until implemented.
"""

from __future__ import annotations

import numpy as np


def analyze(mono: np.ndarray, sr: int, n_frames: int, frame_hz: int) -> tuple[dict, dict[str, np.ndarray]]:
    z = np.zeros(n_frames, dtype=np.float32)
    return {"method": "not implemented", "onsets": [], "hits": [], "counts": {}}, {"low": z, "mid": z, "high": z}
