"""Stage 3 - energy, loudness, stem activity and drum hits.

Everything here is plain DSP on input.wav and the stems (no models).

Writes:
  cache/energy.npz   dense 100 Hz curves (see CURVE KEYS below)
  cache/energy.json  loudness stats, per-stem levels, active regions, has_vocals
  cache/hits.json    drum-stem onsets and kick/snare/hat estimates (from drums.py)

Curve keys in energy.npz (all arrays share ``times``):
  times                frame centre times, s (FRAME_HZ per second)
  db_<name>            RMS level in dBFS, name in mix|vocals|drums|bass|other
  energy_<name>        same, normalised to 0..1 (see normalize_db)
  lufs                 momentary loudness of the mix (400 ms window, BS.1770 K-weighting)
  brightness           mix spectral centroid mapped to 0..1 on a log scale (200 Hz..8 kHz)
  onset                mix onset strength, 0..1
  drums_low|mid|high   drum-stem band envelopes 0..1 (kick / snare / hat bands)
"""

from __future__ import annotations

import logging

import numpy as np

from .. import audio
from ..config import STEM_NAMES
from ..context import RunContext
from ..pipeline import Stage

log = logging.getLogger("audio4llm")

FRAME_HZ = 100
RANGE_DB = 30.0  # energy 0..1 spans the top 30 dB of each stem's level
SILENCE_DB = -90.0

# activity detection
ACT_SMOOTH_S = 0.5  # smoothing for the stem level
ACT_MIX_SMOOTH_S = 1.5  # smoothing for the local mix level it is compared to
ACT_REL_DB = 18.0  # active if within this many dB of the local mix level...
ACT_ABS_DB = -60.0  # ...and above this absolute level
ACT_HYST_DB = 3.0  # hysteresis: stay active until 3 dB below the threshold
ACT_MIN_S = 0.4  # drop regions shorter than this
ACT_GAP_S = 0.75  # merge regions separated by less than this
HAS_VOCALS_MIN_S = 1.0


# ---- helpers shared with other stages ---------------------------------------------

def frame_count(n_samples: int, sr: int) -> int:
    hop = sr // FRAME_HZ
    return 1 + n_samples // hop


def frame_power(y: np.ndarray, sr: int, win_s: float = 0.046) -> np.ndarray:
    """Mean-square power per frame (centred windows), averaged over channels. y: (ch, n) or (n,)."""
    y = np.atleast_2d(y).astype(np.float64)
    hop = sr // FRAME_HZ
    win = int(win_s * sr)
    sq = (y ** 2).mean(axis=0)
    c = np.concatenate([[0.0], np.cumsum(sq)])
    centres = np.arange(frame_count(y.shape[1], sr)) * hop
    lo = np.clip(centres - win // 2, 0, len(sq))
    hi = np.clip(centres + win // 2, 0, len(sq))
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def smooth(x: np.ndarray, seconds: float) -> np.ndarray:
    n = max(1, int(round(seconds * FRAME_HZ)))
    if n == 1:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def to_db(power: np.ndarray) -> np.ndarray:
    return np.maximum(10 * np.log10(np.maximum(power, 1e-12)), SILENCE_DB)


def normalize_db(db: np.ndarray | float, ref_db: float, range_db: float = RANGE_DB) -> np.ndarray:
    """0 at ref_db - range_db or below, 1 at ref_db; linear in dB in between."""
    return np.clip((np.asarray(db) - (ref_db - range_db)) / range_db, 0.0, 1.0)


def regions_from_mask(mask: np.ndarray, times: np.ndarray, min_s: float, gap_s: float) -> list[list[float]]:
    """Contiguous True runs as [start, end] seconds, gaps < gap_s merged, runs < min_s dropped."""
    out: list[list[float]] = []
    if not mask.any():
        return out
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    dt = 1.0 / FRAME_HZ
    for a, b in zip(edges[::2], edges[1::2]):
        s, e = float(times[a]), float(times[b - 1] + dt)
        if out and s - out[-1][1] < gap_s:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [[round(s, 3), round(e, 3)] for s, e in out if e - s >= min_s]


def active_regions(stem_db: np.ndarray, mix_db: np.ndarray, times: np.ndarray) -> list[list[float]]:
    level = to_db(smooth(10 ** (stem_db / 10), ACT_SMOOTH_S))
    local_mix = to_db(smooth(10 ** (mix_db / 10), ACT_MIX_SMOOTH_S))
    on = np.maximum(local_mix - ACT_REL_DB, ACT_ABS_DB)
    off = on - ACT_HYST_DB
    mask = np.zeros(len(level), dtype=bool)
    state = False
    for i, (v, a, b) in enumerate(zip(level, on, off)):
        state = v >= a if not state else v >= b
        mask[i] = state
    return regions_from_mask(mask, times, ACT_MIN_S, ACT_GAP_S)


# ---- loudness ------------------------------------------------------------------------

def k_weighted(y: np.ndarray, sr: int) -> np.ndarray:
    """BS.1770 K-weighting via pyloudnorm's filters. y: (ch, n) -> (ch, n)."""
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    out = np.array(y, dtype=np.float64)
    for filt in meter._filters.values():
        for ch in range(out.shape[0]):
            out[ch] = filt.apply_filter(out[ch])
    return out


def windowed_lufs(kw: np.ndarray, sr: int, window_s: float, hop_s: float | None = None) -> np.ndarray:
    """Loudness (LUFS) of K-weighted audio in sliding windows centred on each frame (or every hop_s)."""
    hop = int(sr / FRAME_HZ) if hop_s is None else int(hop_s * sr)
    win = int(window_s * sr)
    ms = (kw ** 2).sum(axis=0)  # channel gains are 1.0 for L/R
    c = np.concatenate([[0.0], np.cumsum(ms)])
    n = kw.shape[1]
    centres = np.arange(0, n, hop) if hop_s is not None else np.arange(frame_count(n, sr)) * hop
    lo = np.clip(centres - win // 2, 0, n)
    hi = np.clip(centres + win // 2, 0, n)
    z = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    return np.maximum(-0.691 + 10 * np.log10(np.maximum(z, 1e-12)), -70.0)


def loudness_stats(y: np.ndarray, sr: int, kw: np.ndarray) -> dict:
    import pyloudnorm as pyln

    integrated = float(pyln.Meter(sr).integrated_loudness(y.T.astype(np.float64)))
    st = windowed_lufs(kw, sr, 3.0, hop_s=0.1)  # short-term loudness for the loudness range (EBU 3342)
    st = st[st > -70]
    lra = None
    if len(st):
        st = st[st > 10 * np.log10(np.mean(10 ** (st / 10))) - 20]
        lra = float(np.percentile(st, 95) - np.percentile(st, 10)) if len(st) else None
    peak = float(np.abs(y).max())
    return {
        "integrated_lufs": round(integrated, 1) if np.isfinite(integrated) else None,
        "loudness_range_lu": round(lra, 1) if lra is not None else None,
        "peak_dbfs": round(20 * np.log10(max(peak, 1e-9)), 1),
        "rms_dbfs": round(10 * np.log10(max(float((y.astype(np.float64) ** 2).mean()), 1e-12)), 1),
    }


# ---- spectral curves -----------------------------------------------------------------

def brightness_and_onset(mono: np.ndarray, sr: int, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    hop = sr // FRAME_HZ
    cent = librosa.feature.spectral_centroid(y=mono, sr=sr, n_fft=2048, hop_length=hop)[0]
    bright = np.clip((np.log2(np.maximum(cent, 1.0)) - np.log2(200)) / (np.log2(8000) - np.log2(200)), 0, 1)
    onset = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    onset = np.clip(onset / (np.percentile(onset, 99) + 1e-9), 0, 1)
    fit = lambda x: np.pad(x, (0, max(0, n_frames - len(x))), mode="edge")[:n_frames]
    return fit(bright), fit(onset)


# ---- stage --------------------------------------------------------------------------

def run(ctx: RunContext) -> None:
    from .. import drums

    y, sr = audio.load(ctx.input_wav)
    n_frames = frame_count(y.shape[1], sr)
    times = np.arange(n_frames) / FRAME_HZ
    curves: dict[str, np.ndarray] = {"times": times}

    # levels
    sources = {"mix": y}
    for s in STEM_NAMES:
        sources[s] = audio.load(ctx.stem_path(s))[0]
    for name, sig in sources.items():
        curves[f"db_{name}"] = to_db(frame_power(sig, sr))[:n_frames]

    mix_ref = float(np.percentile(curves["db_mix"], 99))
    levels, active = {}, {}
    for name in sources:
        db = curves[f"db_{name}"]
        own_ref = float(np.percentile(db, 99))
        # a stem that is nearly silent everywhere must not be stretched up to 1.0
        ref = own_ref if name == "mix" else max(own_ref, mix_ref - RANGE_DB)
        curves[f"energy_{name}"] = normalize_db(db, ref)
        levels[name] = {"ref_db": round(ref, 1), "p99_db": round(own_ref, 1), "mean_db": round(float(to_db(np.mean(10 ** (db / 10)))), 1)}
        if name != "mix":
            active[name] = active_regions(db, curves["db_mix"], times)

    # loudness
    kw = k_weighted(y, sr)
    curves["lufs"] = windowed_lufs(kw, sr, 0.4)[:n_frames]
    loud = loudness_stats(y, sr, kw)
    loud["max_momentary_lufs"] = round(float(curves["lufs"].max()), 1)

    # spectral curves of the mix
    mono = y.mean(axis=0)
    curves["brightness"], curves["onset"] = brightness_and_onset(mono, sr, n_frames)

    # drum hits + band envelopes
    drum_y = sources["drums"]
    hits, bands = drums.analyze(drum_y.mean(axis=0), sr, n_frames, FRAME_HZ)
    for k, v in bands.items():
        curves[f"drums_{k}"] = v[:n_frames]

    vocal_s = sum(e - s for s, e in active.get("vocals", []))
    np.savez_compressed(ctx.cache_path("energy.npz"), **{k: v.astype(np.float32) for k, v in curves.items()})
    ctx.write_json("energy.json", {
        "frame_hz": FRAME_HZ,
        "range_db": RANGE_DB,
        "duration": round(y.shape[1] / sr, 3),
        "loudness": loud,
        "levels": levels,
        "active": active,
        "has_vocals": vocal_s >= HAS_VOCALS_MIN_S,
        "vocal_seconds": round(vocal_s, 2),
    })
    ctx.write_json("hits.json", hits)
    log.info("energy: %.1f LUFS, active stems: %s, %d drum hits", loud["integrated_lufs"] or float("nan"),
             ", ".join(f"{k} {sum(e - s for s, e in v):.0f}s" for k, v in active.items()), len(hits.get("hits", [])))


def load_curves(ctx: RunContext) -> dict[str, np.ndarray]:
    with np.load(ctx.cache_path("energy.npz")) as z:
        return {k: z[k] for k in z.files}


STAGE = Stage(
    name="energy",
    run=run,
    outputs=lambda ctx: [ctx.cache_path(f) for f in ("energy.npz", "energy.json", "hits.json")],
    deps=("ingest", "stems"),
)
