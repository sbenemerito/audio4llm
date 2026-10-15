"""Stage 0 - ingest: decode any input with ffmpeg to runs/<clip>/input.wav.

input.wav (44.1 kHz, stereo, 16-bit PCM) is the single timeline everything else
is measured against, and the file a visualizer should mux into its video.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path

from ..config import CHANNELS, SAMPLE_RATE
from ..context import RunContext
from ..pipeline import Stage

log = logging.getLogger("audio4llm")

LONG_INPUT_WARN_S = 90.0
MIN_DURATION_S = 2.0  # Demucs, loudness (400 ms blocks) and bar-level sections need at least this much


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(tool: str) -> str:
    exe = shutil.which(tool)
    if exe is None:
        raise RuntimeError(f"{tool} not found on PATH. Install it with `brew install ffmpeg`.")
    return exe


def probe(path: Path) -> dict:
    out = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", "-select_streams", "a:0", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe could not read {path}: {out.stderr.strip()}")
    info = json.loads(out.stdout)
    if not info.get("streams"):
        raise RuntimeError(f"{path} has no audio stream")
    st, fmt = info["streams"][0], info.get("format", {})
    dur = st.get("duration") or fmt.get("duration")
    return {
        "codec": st.get("codec_name"),
        "sample_rate": int(st["sample_rate"]) if st.get("sample_rate") else None,
        "channels": st.get("channels"),
        "duration": round(float(dur), 3) if dur else None,
    }


def fingerprint(ctx: RunContext) -> dict:
    return {"source_sha1": _sha1(ctx.source)} if ctx.source else {}


def run(ctx: RunContext) -> None:
    import soundfile as sf

    src = ctx.source
    cfg = ctx.config
    meta = probe(src)
    if cfg.start is None and cfg.duration is None and (meta["duration"] or 0) > LONG_INPUT_WARN_S:
        log.warning(
            "input is %.0fs long; audio4llm is tuned for clips up to ~80s. Continuing anyway "
            "(use --start/--duration to analyse an excerpt).", meta["duration"],
        )
    if cfg.start is not None and meta["duration"] and cfg.start >= meta["duration"]:
        raise ValueError(f"--start {cfg.start}s is past the end of the input ({meta['duration']}s)")

    cmd = [_require("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if cfg.start:
        cmd += ["-ss", f"{cfg.start:.3f}"]
    cmd += ["-i", str(src)]
    if cfg.duration is not None:
        cmd += ["-t", f"{cfg.duration:.3f}"]
    tmp = ctx.input_wav.with_name("input.tmp.wav")
    cmd += ["-map", "0:a:0", "-vn", "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")
    tmp.replace(ctx.input_wav)

    info = sf.info(str(ctx.input_wav))
    got = info.frames / info.samplerate
    if got < MIN_DURATION_S:
        ctx.input_wav.unlink()
        raise RuntimeError(f"decoded audio is only {got:.2f}s; audio4llm needs at least {MIN_DURATION_S:g}s "
                           "(check --start/--duration)")
    if cfg.duration is not None and got < cfg.duration - 0.5:
        log.warning("ingest: asked for %.1fs from %.1fs but the input ends there, so the clip is %.1fs",
                    cfg.duration, cfg.start or 0.0, got)
    ctx.write_json("ingest.json", {
        "source": str(src),
        "source_sha1": _sha1(src),
        "source_meta": meta,
        "start": cfg.start or 0.0,
        "requested_duration": cfg.duration,
        "duration": round(info.frames / info.samplerate, 3),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
    })
    log.info("ingest: %s -> input.wav (%.1fs, %d Hz, %d ch)", src.name, info.frames / info.samplerate, info.samplerate, info.channels)


STAGE = Stage(
    name="ingest",
    run=run,
    outputs=lambda ctx: [ctx.input_wav, ctx.cache_path("ingest.json")],
    config_keys=("start", "duration"),
    needs_source=True,
    fingerprint=fingerprint,
)
