"""Stage 1 - stems: Demucs source separation into vocals / drums / bass / other.

Calls Demucs' model and apply functions directly (no CLI, no torchaudio I/O):
audio is read and written with soundfile. Tries MPS first and falls back to CPU
if MPS raises or produces non-finite output.

Writes stems/{vocals,drums,bass,other}.wav (44.1 kHz stereo float32) and
cache/stems.json (model, device actually used).
"""

from __future__ import annotations

import logging
import random

import numpy as np

from .. import audio
from ..config import STEM_NAMES
from ..context import RunContext
from ..pipeline import Stage

log = logging.getLogger("audio4llm")


def separate(y: np.ndarray, model_name: str, device: str) -> tuple[dict[str, np.ndarray], str]:
    """y: (2, n) float32 at 44.1 kHz -> ({source: (2, n)}, device used)."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model(model_name)
    model.eval()
    if model.samplerate != 44100 or model.audio_channels != 2:
        raise RuntimeError(f"unexpected Demucs model format: {model.samplerate} Hz, {model.audio_channels} ch")

    wav = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    x = ((wav - mean) / std)[None]

    def _apply(dev: str):
        random.seed(0)  # apply_model's shift trick uses `random`; keep runs reproducible
        with torch.no_grad():
            out = apply_model(model, x, shifts=1, split=True, overlap=0.25, progress=False, device=dev)
        out = out[0].cpu() * std + mean
        if not torch.isfinite(out).all():
            raise RuntimeError("non-finite values in Demucs output")
        return out

    try:
        out = _apply(device)
    except Exception as exc:
        if device == "cpu":
            raise
        log.warning("stems: Demucs failed on %s (%s); falling back to CPU", device, exc)
        device = "cpu"
        out = _apply(device)
    return {s: out[i].numpy() for i, s in enumerate(model.sources)}, device


def run(ctx: RunContext) -> None:
    y, sr = audio.load(ctx.input_wav)
    stems, device = separate(y, ctx.config.demucs_model, ctx.torch_device())
    missing = set(STEM_NAMES) - set(stems)
    if missing:
        raise RuntimeError(f"Demucs model did not return stems: {sorted(missing)}")
    for name in STEM_NAMES:
        audio.save(ctx.stem_path(name), stems[name], sr, subtype="FLOAT")
    ctx.write_json("stems.json", {
        "model": ctx.config.demucs_model,
        "device": device,
        "sources": list(STEM_NAMES),
        "peak_dbfs": {s: round(20 * float(np.log10(np.abs(stems[s]).max() + 1e-9)), 1) for s in STEM_NAMES},
    })
    log.info("stems: %s on %s -> %s", ctx.config.demucs_model, device, ", ".join(STEM_NAMES))


STAGE = Stage(
    name="stems",
    run=run,
    outputs=lambda ctx: [*ctx.stem_paths().values(), ctx.cache_path("stems.json")],
    deps=("ingest",),
    config_keys=("demucs_model",),
)
