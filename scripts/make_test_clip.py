#!/usr/bin/env python
"""Synthesize an ~80 s, 120 BPM test clip with known structure, plus ground_truth.json.

Structure (4/4, 2.0 s per bar, bars numbered from 1):

  bars  1-8   0-16 s  intro      closed hats on 8ths + quiet pad
  bars  9-16 16-32 s  build      16th hats, snare roll speeding up, noise riser, pad swells;
                                 the last beat (31.5-32.0 s) is a full stop before the drop
  bars 17-28 32-56 s  full       kick on every beat, snare on 2 and 4, hats, bass line, chords
                                 + spoken phrase 1 (macOS `say`) at bar 19
  bars 29-36 56-72 s  breakdown  pad only + spoken phrase 2 at bar 31
  bars 37-40 72-80 s  outro      kick, hats, bass and pad, fading out

Chords cycle Am | F | C | G (one per bar), so the key is A minor (relative: C major).
The bass plays the chord root (MIDI 45/41/48/43) as staccato quarter notes.

Usage:  python scripts/make_test_clip.py [--out-dir test_data] [--no-vocals] [--voice Samantha]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100
BPM = 120.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
N_BARS = 40
DURATION = N_BARS * BAR

SECTIONS = [  # name, first bar, last bar, character we expect the analyzer to report
    ("intro", 1, 8, "sparse"),
    ("build", 9, 16, "build"),
    ("full", 17, 28, "full"),
    ("breakdown", 29, 36, "breakdown"),
    ("outro", 37, 40, "outro"),
]
CHORDS = [  # name, bass midi, pad midi notes
    ("Am", 45, [57, 60, 64]),
    ("F", 41, [53, 57, 60]),
    ("C", 48, [55, 60, 64]),
    ("G", 43, [55, 59, 62]),
]
PHRASES = [  # bar the phrase starts on, text
    (19, "Hold on to the night, we are dancing in the light."),
    (31, "Never let me go, the music sets us free."),
]
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

rng = np.random.default_rng(1234)


def bar_t(bar: int) -> float:
    return (bar - 1) * BAR


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def midi_name(m: int) -> str:
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def env_exp(n: int, tau: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (tau * SR))


def sos(kind: str, freq, order: int = 4):
    return signal.butter(order, freq, btype=kind, fs=SR, output="sos")


# ---- instruments (all mono, float64) ---------------------------------------

def kick() -> np.ndarray:
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 48 + 110 * np.exp(-t / 0.03)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(n, 0.16)
    click = signal.sosfilt(sos("highpass", 2000, 2), rng.standard_normal(n)) * env_exp(n, 0.002) * 0.3
    return body + click


def snare() -> np.ndarray:
    n = int(0.25 * SR)
    noise = signal.sosfilt(sos("bandpass", [1500, 9000], 2), rng.standard_normal(n)) * env_exp(n, 0.07)
    tone = np.sin(2 * np.pi * 190 * np.arange(n) / SR) * env_exp(n, 0.05)
    return 0.8 * noise + 0.5 * tone


def hat() -> np.ndarray:
    n = int(0.07 * SR)
    return signal.sosfilt(sos("highpass", 7500, 4), rng.standard_normal(n)) * env_exp(n, 0.018)


def saw(freq: float, n: int, detune_cents: float = 0.0) -> np.ndarray:
    f = freq * 2 ** (detune_cents / 1200)
    phase = (np.arange(n) * f / SR + rng.random()) % 1.0
    return 2 * phase - 1


def bass_note(midi: int, dur: float) -> np.ndarray:
    n = int(dur * SR)
    f = midi_hz(midi)
    t = np.arange(n) / SR
    tone = 0.6 * np.sin(2 * np.pi * f * t) + 0.4 * saw(f, n)
    tone = signal.sosfilt(sos("lowpass", 900, 2), tone)
    a = np.minimum(1, t / 0.005) * np.minimum(1, (dur - t) / 0.03).clip(0, 1)
    return tone * a


def pad_chord(notes: list[int], dur: float) -> np.ndarray:
    """Stereo (2, n) detuned-saw pad with slow attack/release."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for m in notes:
        f = midi_hz(m)
        out[0] += saw(f, n, -7) + 0.5 * np.sin(2 * np.pi * f * t)
        out[1] += saw(f, n, +7) + 0.5 * np.sin(2 * np.pi * f * t)
    out = signal.sosfilt(sos("lowpass", 1800, 2), out, axis=1)
    amp = np.minimum(1, t / 0.35) * np.minimum(1, (dur - t) / 0.25).clip(0, 1)
    return out * amp / len(notes)


# ---- arrangement -------------------------------------------------------------

class Track:
    def __init__(self) -> None:
        self.buf = np.zeros((2, int(DURATION * SR) + SR))

    def add(self, t: float, x: np.ndarray, gain: float = 1.0, pan: float = 0.0) -> None:
        i = int(round(t * SR))
        if x.ndim == 1:
            left, right = np.sqrt((1 - pan) / 2), np.sqrt((1 + pan) / 2)
            x = np.vstack([x * left * np.sqrt(2), x * right * np.sqrt(2)])
        n = min(x.shape[1], self.buf.shape[1] - i)
        self.buf[:, i:i + n] += gain * x[:, :n]


def say_phrase(text: str, voice: str) -> np.ndarray | None:
    if shutil.which("say") is None or shutil.which("ffmpeg") is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        aiff, wav = Path(td) / "p.aiff", Path(td) / "p.wav"
        r = subprocess.run(["say", "-v", voice, "-r", "165", "-o", str(aiff), text], capture_output=True)
        if r.returncode != 0:
            r = subprocess.run(["say", "-r", "165", "-o", str(aiff), text], capture_output=True)  # default voice
            if r.returncode != 0:
                return None
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(aiff), "-ac", "1", "-ar", str(SR), str(wav)], check=True)
        y, _ = sf.read(str(wav), dtype="float64")
    # trim silence so the ground-truth start/end are the audible phrase
    env = np.convolve(np.abs(y), np.ones(441) / 441, mode="same")
    on = np.where(env > 0.02 * env.max())[0]
    y = y[on[0]:on[-1] + 1]
    return y / (np.abs(y).max() + 1e-9)


def build(with_vocals: bool, voice: str) -> tuple[np.ndarray, dict]:
    drums, bass, pad, fx, vox = Track(), Track(), Track(), Track(), Track()
    K, S, H = kick(), snare(), hat()
    hits = {"kick": [], "snare": [], "hat": []}
    bass_notes, chords, vocals = [], [], []

    def section_of(bar: int) -> str:
        return next(name for name, a, b, _ in SECTIONS if a <= bar <= b)

    for bar in range(1, N_BARS + 1):
        t0 = bar_t(bar)
        sec = section_of(bar)
        name, root, notes = CHORDS[(bar - 1) % 4]
        fade = 1.0 - 0.85 * (bar - 37) / 4 if sec == "outro" else 1.0
        build_pos = (bar - 9) / 8 if sec == "build" else 0.0  # 0 -> 1 across the build

        # pad (every section)
        pad_gain = {"intro": 0.10, "build": 0.10 + 0.15 * build_pos, "full": 0.22, "breakdown": 0.20, "outro": 0.22 * fade}[sec]
        pad.add(t0, pad_chord(notes, BAR), pad_gain)
        chords.append({"start": t0, "end": t0 + BAR, "name": name, "midi": notes})

        # hats
        if sec in ("intro", "full", "outro"):
            g = {"intro": 0.07, "full": 0.16, "outro": 0.16 * fade}[sec]
            for k in range(8):
                t = t0 + k * BEAT / 2
                drums.add(t, H, g * (1.0 if k % 2 == 0 else 0.7), pan=0.3)
                hits["hat"].append(t)
        elif sec == "build":
            for k in range(16):
                t = t0 + k * BEAT / 4
                if bar == 16 and t >= bar_t(17) - BEAT:  # stop before the drop
                    continue
                drums.add(t, H, (0.06 + 0.10 * build_pos) * (1.0 if k % 2 == 0 else 0.6), pan=0.3)
                hits["hat"].append(t)

        # kick + snare groove
        if sec in ("full", "outro"):
            for k in range(4):
                drums.add(t0 + k * BEAT, K, 0.9 * fade)
                hits["kick"].append(t0 + k * BEAT)
            for k in (1, 3):
                drums.add(t0 + k * BEAT, S, 0.45 * fade)
                hits["snare"].append(t0 + k * BEAT)
        elif sec == "build":  # snare roll: quarters -> 8ths -> 16ths, getting louder
            div = 1 if bar <= 12 else (2 if bar <= 14 else 4)
            for k in range(4 * div):
                t = t0 + k * BEAT / div
                if bar == 16 and t >= bar_t(16) + 3 * BEAT:
                    continue
                drums.add(t, S, 0.12 + 0.35 * build_pos)
                hits["snare"].append(t)

        # bass: root as staccato quarter notes
        if sec in ("full", "outro"):
            for k in range(4):
                t = t0 + k * BEAT
                bass.add(t, bass_note(root, 0.4), 0.55 * fade)
                bass_notes.append({"start": round(t, 3), "end": round(t + 0.4, 3), "midi": root, "name": midi_name(root)})

    # riser through the build, cut dead at the stop
    t_start, t_stop = bar_t(9), bar_t(16) + 3 * BEAT
    n = int((t_stop - t_start) * SR)
    ramp = np.linspace(0, 1, n) ** 2
    chirp = signal.chirp(np.arange(n) / SR, 300, n / SR, 3000, method="logarithmic")
    noise = signal.sosfilt(sos("highpass", 3000, 2), rng.standard_normal(n))
    fx.add(t_start, (0.25 * chirp + 0.2 * noise) * ramp, 0.6)

    # silence the pad during the pre-drop stop (31.5-32.0 s) so it's a real gap
    i0, i1 = int(t_stop * SR), int(bar_t(17) * SR)
    pad.buf[:, i0:i1] *= np.linspace(1, 0, i1 - i0) ** 8

    if with_vocals:
        for bar, text in PHRASES:
            y = say_phrase(text, voice)
            if y is None:
                print("warning: macOS `say` unavailable, building the clip without vocals")
                vocals = []
                break
            t = bar_t(bar)
            vox.add(t, y, 0.42)
            vocals.append({
                "start": round(t, 3), "end": round(t + len(y) / SR, 3), "bar": bar, "text": text,
                "words": [w.strip(".,").lower() for w in text.split()],
            })

    stems = {"drums": drums.buf, "bass": bass.buf, "other": pad.buf + fx.buf, "vocals": vox.buf}
    mix = sum(stems.values())[:, : int(DURATION * SR)]
    peak = np.abs(mix).max()
    mix = mix / peak * 10 ** (-1 / 20)  # -1 dBFS peak

    beats = [round(i * BEAT, 3) for i in range(int(DURATION / BEAT))]
    truth = {
        "description": "Synthetic 120 BPM test clip made by scripts/make_test_clip.py",
        "duration": DURATION,
        "sample_rate": SR,
        "tempo_bpm": BPM,
        "time_signature": "4/4",
        "key": "A minor",
        "key_equivalents": ["A minor", "C major"],
        "beats": beats,
        "downbeats": [round(bar_t(b), 3) for b in range(1, N_BARS + 1)],
        "sections": [
            {"name": name, "start": bar_t(a), "end": bar_t(b + 1), "first_bar": a, "last_bar": b, "character": ch}
            for name, a, b, ch in SECTIONS
        ],
        "moments": [
            {"type": "stop", "t": t_stop, "note": "everything cuts out for the last beat of the build"},
            {"type": "drop", "t": bar_t(17), "note": "full groove starts"},
            {"type": "breakdown", "t": bar_t(29), "note": "drums and bass drop out"},
        ],
        "hits": {k: [round(t, 3) for t in sorted(v)] for k, v in hits.items()},
        "bass_notes": bass_notes,
        "chords": chords,
        "vocals": vocals,
        "stem_activity": {
            "drums": [[0.0, t_stop], [bar_t(17), bar_t(29)], [bar_t(37), DURATION]],
            "bass": [[bar_t(17), bar_t(29)], [bar_t(37), DURATION]],
            "vocals": [[v["start"], v["end"]] for v in vocals],
        },
    }
    return mix.astype(np.float32), truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="test_data", help="where to write test_clip.wav and ground_truth.json")
    ap.add_argument("--no-vocals", action="store_true", help="skip the spoken phrases")
    ap.add_argument("--voice", default="Samantha", help="macOS `say` voice (default: Samantha)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mix, truth = build(not args.no_vocals, args.voice)
    sf.write(str(out / "test_clip.wav"), mix.T, SR, subtype="PCM_16")
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=1) + "\n")
    print(f"wrote {out / 'test_clip.wav'} ({DURATION:.0f}s) and {out / 'ground_truth.json'}")


if __name__ == "__main__":
    main()
