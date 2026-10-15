"""Drum-stem onsets and kick / snare / hat estimates (called by the energy stage).

Method: one STFT of the drum stem at the curve frame rate, summed into three bands
(low 30-150 Hz = kick, mid 1-5 kHz = snare crack / clap, high 6-16 kHz = hats). Each
band gets a level curve in dB and an onset function (the rise of that level over
20 ms). Peaks that rise sharply and come within a few tens of dB of the band's loudest
moments are candidates. Cross-band rules then remove the usual leakage:

* bass bleed: a low-band onset counts as a kick only if something also rises in the
  mid or high band at the same moment (the beater / click transient), or if it is
  nearly as loud as the loudest kicks. It must also not be far quieter than the
  loudest kick within a couple of seconds.
* kick click: a kick's click is a burst of a few ms in the mid and high bands. A
  mid onset on a kick is a snare only if the mid band is still loud 30-60 ms later
  (snare noise rings), and a high onset on a kick is a hat only if the high band
  has not collapsed 30-40 ms later.
* snare noise: a snare also lights up the high band, and snare + hat cannot be told
  apart from snare alone by spectrum. A high onset on a snare is reported as a hat
  only if it continues an evenly spaced run of hats at that run's own spacing (8th
  hats with a backbeat snare), or if the high band is clearly hotter than this
  song's snares usually make it (e.g. a crash).

So a kick and a hat on the same beat are both reported. Types are estimates from band
energy, not a trained drum transcriber.

Contract:
    analyze(mono, sr, n_frames, frame_hz) -> (hits_json, bands)

    hits_json = {
        "method": "<one line describing the approach>",
        "onsets": [t, ...],                       # all drum-stem onsets (full-band flux, plus every hit), seconds
        "hits": [{"t": 1.234, "type": "kick"|"snare"|"hat", "strength": 0..1}, ...],  # sorted by t
        "counts": {"kick": n, "snare": n, "hat": n},
    }
    bands = {"low": arr, "mid": arr, "high": arr}  # 0..1 band envelopes, len n_frames, at frame_hz

Envelopes are band level in dB mapped linearly to 0..1 over the 30 dB below the
band's loudest moments (99.5th percentile), like the energy_* curves. A hit's
``strength`` is its band envelope's peak within 50 ms of the hit, so it agrees with
``bands``.
"""

from __future__ import annotations

import logging

import numpy as np

from .jsonfmt import r3

log = logging.getLogger("audio4llm")

METHOD = ("estimate: drum-stem STFT split into low 30-150 Hz (kick) / mid 1-5 kHz (snare) / high 6-16 kHz (hat) "
          "bands; per-band onset peaks gated by level, with rules against bass bleed, kick-click and snare-noise leakage")

N_FFT = 2048  # 46 ms at 44.1 kHz: enough frequency resolution to keep the kick band below 150 Hz
LOW_HZ = (30.0, 150.0)  # kick fundamental (bass bleed also lives here)
MID_HZ = (1000.0, 5000.0)  # snare crack / clap; skips 150-1000 Hz where kick and bass harmonics sit
HIGH_HZ = (6000.0, 16000.0)  # hats, shakers, cymbals
ENV_RANGE_DB = 30.0  # envelopes span the top 30 dB of each band, same as the energy_* curves
REF_PCT = 99.5  # band reference level = its loudest moments, robust to a single spike
FLOOR_DB = 45.0  # level changes more than 45 dB below the reference are noise floor / bleed

ONSET_LAG = 2  # onset function = level rise over 2 frames (20 ms at 100 Hz)
RISE_WIN_S = 0.04  # a candidate's rise is measured from the minimum in the 40 ms before it
COINCIDE_S = 0.035  # onsets in different bands within 35 ms are the same moment
MIN_GAP_S = 0.05  # at most one hit per band per 50 ms (32nd notes at 150 BPM)

# Per type: minimum rise in dB, and how far below the band reference a hit may peak. Kicks and
# snares are the loudest things in their bands, so much quieter onsets there are mostly bleed;
# the kick gate is the tightest because Demucs bass bleed shares its band. Hats are played with
# more dynamics (ghost notes, open/closed), so their gate is wider.
RULES = {
    "kick": {"band": "low", "rise_db": 6.0, "gate_db": 18.0},
    "snare": {"band": "mid", "rise_db": 6.0, "gate_db": 24.0},
    "hat": {"band": "high", "rise_db": 4.0, "gate_db": 30.0},
}
TRANSIENT_DB = 6.0  # a kick's beater/click: mid or high band rises >= 6 dB at the same moment...
KICK_SOLO_DB = 10.0  # ...unless the kick is within 10 dB of the loudest kicks (sub kicks with no click)
KICK_LOCAL_S = 2.0  # and no kick may be more than KICK_LOCAL_DB below the loudest kick within +-2 s
KICK_LOCAL_DB = 15.0  # (ghost kicks are rarely that soft; bass bleed usually is)
SNARE_RING_S = (0.03, 0.06)  # snare-on-kick: after 30-60 ms the click is gone but snare noise rings...
SNARE_RING_DB = 18.0  # ...within 18 dB of the mid band's reference
HAT_RING_S = (0.03, 0.04)  # hat-on-kick: 30-40 ms after the peak, a click has collapsed (> 40 dB down)...
HAT_RING_DB = 24.0  # ...while even a short closed hat is still within ~20 dB of its peak
HAT_RUN_MAX_S = 0.5  # hat-on-snare: neighbouring hats at most 0.5 s away (8ths down to 60 BPM)...
HAT_RUN_TOL = 0.2  # ...and gaps equal to each other and to the run's own spacing within 20%
HAT_HOTTER_DB = 6.0  # or high-minus-mid is 6 dB above its median over this song's snares


def band_db(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sel = (freqs >= lo) & (freqs < hi)
    if not sel.any():
        return np.full(power.shape[1], -120.0)
    return 10 * np.log10(power[sel].sum(axis=0) + 1e-12)


def envelope(db: np.ndarray, ref_db: float, range_db: float = ENV_RANGE_DB) -> np.ndarray:
    """0 at ref_db - range_db or below, 1 at ref_db; linear in dB in between."""
    return np.clip((db - (ref_db - range_db)) / range_db, 0.0, 1.0)


def onset_function(db: np.ndarray, ref_db: float, lag: int = ONSET_LAG) -> np.ndarray:
    """Positive rise of a band level (dB) over ``lag`` frames, ignoring the noise floor."""
    x = np.maximum(db, ref_db - FLOOR_DB)
    prev = np.concatenate([np.full(lag, ref_db - FLOOR_DB), x[:-lag]])  # a hit at t=0 rises from the floor
    return np.maximum(x - prev, 0.0)


def pick_peaks(x: np.ndarray, frame_hz: float, delta: float = 1.0) -> np.ndarray:
    import librosa

    w = lambda s: max(1, int(round(s * frame_hz)))
    return librosa.util.peak_pick(x, pre_max=w(0.03), post_max=w(0.03), pre_avg=w(0.1), post_avg=w(0.1),
                                  delta=delta, wait=w(MIN_GAP_S))


def band_candidates(db: np.ndarray, ref_db: float, frame_hz: float, rise_db: float, gate_db: float) -> list[dict]:
    """Onset peaks in one band that rise by >= rise_db and peak within gate_db of the reference.

    Each candidate: frame (onset), peak (frame of the band's peak just after), level (dB at peak).
    """
    floor = ref_db - FLOOR_DB
    win = max(1, round(RISE_WIN_S * frame_hz))
    padded = np.concatenate([np.full(win, floor), np.maximum(db, floor)])  # padded[i + win] == frame i
    out = []
    for p in pick_peaks(onset_function(db, ref_db), frame_hz):
        p = int(p)
        before = padded[p:p + win].min()
        seg = padded[p + win:p + 2 * win]
        level = float(seg.max())
        if level - before >= rise_db and level >= ref_db - gate_db:
            out.append({"frame": p, "peak": p + int(seg.argmax()), "level": level})
    return out


def _near(frames: np.ndarray, f: int, tol: int) -> bool:
    if len(frames) == 0:
        return False
    i = int(np.searchsorted(frames, f))
    return any(0 <= j < len(frames) and abs(int(frames[j]) - f) <= tol for j in (i - 1, i))


def _mean_db(db: np.ndarray, start: int, stop: int) -> float:
    """Power-mean level of frames [start, stop] (inclusive), -120 if out of range."""
    seg = db[max(0, start):max(0, stop + 1)]
    return float(10 * np.log10(np.mean(10 ** (seg / 10)) + 1e-12)) if len(seg) else -120.0


def continues_run(f: int, accepted: np.ndarray, max_gap: float, tol: float) -> bool:
    """True if frame f sits in an evenly spaced run of accepted hats at the run's own spacing."""
    i = int(np.searchsorted(accepted, f))
    if i < 1 or i >= len(accepted):
        return False
    g1, g2 = f - accepted[i - 1], accepted[i] - f
    close = lambda a, b: abs(a - b) <= tol * max(a, b) + 1  # +1 frame of slack for quantisation
    if max(g1, g2) > max_gap or not close(g1, g2):
        return False
    before = accepted[i - 1] - accepted[i - 2] if i >= 2 else None
    after = accepted[i + 1] - accepted[i] if i + 1 < len(accepted) else None
    return (before is not None and close(before, g1)) or (after is not None and close(after, g2))


def classify(dbs: dict[str, np.ndarray], refs: dict[str, float], frame_hz: float) -> dict[str, list[int]]:
    """Kick / snare / hat onset frames from per-band candidates plus the cross-band leakage rules."""
    tol = max(1, round(COINCIDE_S * frame_hz))
    sec = lambda s: max(1, round(s * frame_hz))
    cands = {t: band_candidates(dbs[r["band"]], refs[r["band"]], frame_hz, r["rise_db"], r["gate_db"])
             for t, r in RULES.items()}

    # kicks: need a click/beater transient (any clear mid or high rise) unless they are loud;
    # then drop anything far below the loudest kick nearby. Both target bass bleed.
    transients = np.array(sorted(
        c["frame"] for b in ("mid", "high")
        for c in band_candidates(dbs[b], refs[b], frame_hz, TRANSIENT_DB, FLOOR_DB)), dtype=int)
    kc = [c for c in cands["kick"]
          if _near(transients, c["frame"], tol) or c["level"] >= refs["low"] - KICK_SOLO_DB]
    kf = np.array([c["frame"] for c in kc], dtype=int)
    kl = np.array([c["level"] for c in kc])
    span = sec(KICK_LOCAL_S)
    kicks = [int(f) for f, lv in zip(kf, kl) if lv >= kl[np.abs(kf - f) <= span].max() - KICK_LOCAL_DB]
    kick_arr = np.array(kicks, dtype=int)

    # snares: on a kick, the mid band must still be ringing 30-60 ms later
    a, b = sec(SNARE_RING_S[0]), sec(SNARE_RING_S[1])
    snares = [c["frame"] for c in cands["snare"]
              if not _near(kick_arr, c["frame"], tol)
              or _mean_db(dbs["mid"], c["frame"] + a, c["frame"] + b) >= refs["mid"] - SNARE_RING_DB]
    snare_arr = np.array(snares, dtype=int)

    # hats: clean ones first, then on-kick ones that ring, then on-snare ones that continue a run
    a, b = sec(HAT_RING_S[0]), sec(HAT_RING_S[1])
    accepted, on_snare = [], []
    for c in cands["hat"]:
        f = c["frame"]
        if _near(snare_arr, f, tol):
            on_snare.append(c)
        elif not _near(kick_arr, f, tol) or _mean_db(dbs["high"], c["peak"] + a, c["peak"] + b) >= c["level"] - HAT_RING_DB:
            accepted.append(f)
    acc = np.array(sorted(accepted), dtype=int)
    tilt = [_mean_db(dbs["high"], f, f + 2) - _mean_db(dbs["mid"], f, f + 2) for f in snares]
    snare_tilt = float(np.median(tilt)) if tilt else 0.0
    hats = list(accepted)
    for c in on_snare:
        f = c["frame"]
        hotter = _mean_db(dbs["high"], f, f + 2) - _mean_db(dbs["mid"], f, f + 2) >= snare_tilt + HAT_HOTTER_DB
        if hotter or continues_run(f, acc, HAT_RUN_MAX_S * frame_hz, HAT_RUN_TOL):
            hats.append(f)
    return {"kick": kicks, "snare": snares, "hat": sorted(hats)}


def full_band_onsets(power: np.ndarray, sr: int, hop: int, full_db: np.ndarray, frame_hz: float) -> np.ndarray:
    """Generic onset frames from log-mel spectral flux (librosa defaults), skipping the noise floor."""
    import librosa

    mel = librosa.feature.melspectrogram(S=power, sr=sr, n_mels=64, fmax=min(16000.0, sr / 2))
    env = librosa.onset.onset_strength(S=librosa.power_to_db(mel, ref=np.max, top_db=FLOOR_DB), sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, units="frames")
    ref = float(np.percentile(full_db, REF_PCT))
    look = max(1, round(0.04 * frame_hz))
    # same idea as the band gates: an onset that never gets within 40 dB of the stem's loudest is bleed
    return np.array([f for f in frames if full_db[f:f + look].max(initial=-120.0) >= ref - 40.0], dtype=int)


def analyze(mono: np.ndarray, sr: int, n_frames: int, frame_hz: int) -> tuple[dict, dict[str, np.ndarray]]:
    import librosa

    hop = int(round(sr / frame_hz))
    mono = np.asarray(mono, dtype=np.float32)
    fit = lambda x: np.pad(x, (0, max(0, n_frames - len(x))), mode="edge")[:n_frames]
    names = ("kick", "snare", "hat")

    if len(mono) < N_FFT or float(np.max(np.abs(mono))) < 1e-4:  # no drums (or no audio) at all
        z = np.zeros(n_frames, dtype=np.float32)
        return {"method": METHOD, "onsets": [], "hits": [], "counts": {t: 0 for t in names}}, \
            {"low": z, "mid": z.copy(), "high": z.copy()}

    power = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    dbs = {name: fit(band_db(power, freqs, lo, min(hi, sr / 2)))
           for name, (lo, hi) in (("low", LOW_HZ), ("mid", MID_HZ), ("high", HIGH_HZ))}
    refs = {k: float(np.percentile(v, REF_PCT)) for k, v in dbs.items()}
    bands = {k: envelope(v, refs[k]).astype(np.float32) for k, v in dbs.items()}

    frames = classify(dbs, refs, frame_hz)
    look = max(1, round(0.05 * frame_hz))
    hits = []
    for typ in names:
        env = bands[RULES[typ]["band"]]
        for f in frames[typ]:
            hits.append({"t": r3(f / frame_hz), "type": typ, "strength": round(float(env[f:f + look].max()), 2)})
    hits.sort(key=lambda h: (h["t"], names.index(h["type"])))

    full_db = fit(10 * np.log10(power.sum(axis=0) + 1e-12))
    cand = sorted(set(full_band_onsets(power, sr, hop, full_db, frame_hz).tolist())
                  | {f for fs in frames.values() for f in fs})
    tol = max(1, round(COINCIDE_S * frame_hz))
    onsets: list[int] = []
    for f in cand:  # one onset per moment: keep the earliest of a cluster
        if not onsets or f - onsets[-1] > tol:
            onsets.append(f)
    counts = {t: len(frames[t]) for t in names}
    return {"method": METHOD, "onsets": [r3(f / frame_hz) for f in onsets], "hits": hits, "counts": counts}, bands
