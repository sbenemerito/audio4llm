"""Stage 2 - rhythm: beats, downbeats and tempo with beat_this (CPJKU).

beat_this runs on the full mix with its default "minimal" postprocessing (no
madmom/DBN). Bars, time signature and bar.beat numbering are derived in
grid.py: 4/4 unless the downbeat spacing clearly says otherwise.

Each beat is then nudged onto the strongest mix onset within +/-40 ms, when
there is a clear one (see refine_to_onsets).

Writes cache/rhythm.json:
  beats                        beat times after onset refinement, seconds
  beats_tracker, beats_refined beat_this's own times, and how many beats were moved
  downbeats_raw                beat_this downbeats, seconds
  downbeat_idx                 indices into beats that start a bar (after filling obvious gaps)
  beats_per_bar, time_signature, tempo_bpm, tempo_std_bpm, duration, model, device
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np

from .. import audio
from ..context import RunContext
from ..grid import BeatGrid
from ..pipeline import Stage

log = logging.getLogger("audio4llm")

CHECKPOINT = "final0"
REFINE_WINDOW_S = 0.04  # a 16th note is 125 ms at 120 BPM, 83 ms at 180: +/-40 ms cannot reach the next one
# beat_this downloads from this (slow, sometimes stalling) server via torch.hub, which can leave a
# truncated file behind. We fetch it ourselves with resume + retries and verify the hash.
CHECKPOINT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
CHECKPOINT_SHA256 = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
CHECKPOINT_BYTES = 81_058_141


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_checkpoint() -> Path:
    """Path to a verified beat_this final0 checkpoint in the torch hub cache, downloading it if needed."""
    import torch

    path = Path(torch.hub.get_dir()) / "checkpoints" / f"beat_this-{CHECKPOINT}.ckpt"
    if path.exists() and path.stat().st_size == CHECKPOINT_BYTES and _sha256(path) == CHECKPOINT_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    log.info("rhythm: downloading beat_this checkpoint (%d MB) to %s", CHECKPOINT_BYTES >> 20, path)
    for attempt in range(1, 9):
        have = part.stat().st_size if part.exists() else 0
        if have >= CHECKPOINT_BYTES:
            break
        req = urllib.request.Request(CHECKPOINT_URL, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(part, "ab" if resp.status == 206 else "wb") as f:
                shutil.copyfileobj(resp, f, 1 << 20)
        except OSError as exc:
            log.warning("rhythm: checkpoint download interrupted (attempt %d: %s); resuming", attempt, exc)
            time.sleep(min(2 * attempt, 10))
    if not part.exists() or _sha256(part) != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"could not download a valid beat_this checkpoint. Fetch it manually:\n"
            f"  curl -L -C - --retry 8 -o '{path}' '{CHECKPOINT_URL}'"
        )
    part.replace(path)
    return path


def track(y: np.ndarray, sr: int, device: str) -> tuple[np.ndarray, np.ndarray, str]:
    """y: (ch, n). Returns (beats, downbeats, device used)."""
    from beat_this.inference import Audio2Beats

    ckpt = str(ensure_checkpoint())
    sig = np.ascontiguousarray(y.T)  # beat_this wants (samples, channels) and averages to mono

    def _run(dev: str):
        a2b = Audio2Beats(checkpoint_path=ckpt, device=dev, float16=False, dbn=False)
        return a2b(sig, sr)

    try:
        beats, downbeats = _run(device)
    except Exception as exc:
        if device == "cpu":
            raise
        log.warning("rhythm: beat_this failed on %s (%s); falling back to CPU", device, exc)
        device = "cpu"
        beats, downbeats = _run(device)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float), device


def refine_to_onsets(beats: np.ndarray, mono: np.ndarray, sr: int, window_s: float = REFINE_WINDOW_S) -> np.ndarray:
    """Move each beat to the strongest mix onset within +/- window_s, if there is a clear one.

    beat_this works on 20 ms frames and, on sharp material, reports beats about one frame late.
    A 5 ms onset envelope pulls them onto the transient, which is what a flash or pulse should hit.
    Beats with no clear onset nearby (pads, rests) keep the tracker's time.
    """
    import librosa

    hop = sr // 200
    env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop, n_fft=1024)
    t = np.arange(len(env)) * hop / sr
    clear = 2.0 * np.percentile(env, 75)  # well above the typical frame-to-frame flux
    out = beats.copy()
    for i, b in enumerate(beats):
        idx = np.flatnonzero(np.abs(t - b) <= window_s)
        if len(idx) < 3:
            continue
        k = idx[np.argmax(env[idx])]
        is_peak = 0 < k < len(env) - 1 and env[k] >= env[k - 1] and env[k] >= env[k + 1]
        if is_peak and env[k] > clear:
            out[i] = t[k]
    if np.any(np.diff(out) <= 0):  # never reorder beats; fall back to the tracker where it would
        bad = np.flatnonzero(np.diff(out) <= 0)
        out[bad] = beats[bad]
        out[bad + 1] = beats[bad + 1]
    return out


def run(ctx: RunContext) -> None:
    y, sr = audio.load(ctx.input_wav)
    duration = y.shape[1] / sr
    beats, downbeats, device = track(y, sr, ctx.torch_device())
    grid = BeatGrid.from_tracker(beats, downbeats, duration)
    tracker_beats = grid.beats.copy()
    grid.beats = refine_to_onsets(tracker_beats, y.mean(axis=0), sr)
    local_bpm = grid.local_tempi()
    out = {
        "model": f"beat_this {CHECKPOINT}",
        "device": device,
        "duration": round(duration, 3),
        "beats": [round(float(t), 3) for t in grid.beats],
        "beats_tracker": [round(float(t), 3) for t in tracker_beats],
        "beats_refined": int(np.sum(np.abs(grid.beats - tracker_beats) > 1e-6)),
        "downbeats_raw": [round(float(t), 3) for t in downbeats],
        "downbeat_idx": [int(i) for i in grid.downbeat_idx],
        "beats_per_bar": grid.bpb,
        "time_signature": f"{grid.bpb}/4",
        "tempo_bpm": round(grid.tempo_bpm, 2) if grid.tempo_bpm else None,
        "tempo_std_bpm": round(float(np.std(local_bpm)), 2) if len(local_bpm) else None,
    }
    ctx.write_json("rhythm.json", out)
    log.info("rhythm: %d beats, %d bars, %s, %.1f BPM (%s)", len(grid), len(grid.downbeat_idx),
             out["time_signature"], out["tempo_bpm"] or 0, device)


STAGE = Stage(
    name="rhythm",
    run=run,
    outputs=lambda ctx: [ctx.cache_path("rhythm.json")],
    deps=("ingest",),
    version=3,
)
