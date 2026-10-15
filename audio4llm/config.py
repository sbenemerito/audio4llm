"""User-facing settings for one pipeline run.

Every CLI flag that changes analysis output lives here. Stages declare which of
these fields they depend on (``Stage.config_keys``) so that changing, say,
``whisper_model`` re-runs only the lyrics stage on the next invocation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SAMPLE_RATE = 44100  # input.wav and all stems use this rate
CHANNELS = 2

DEMUCS_MODELS = ("htdemucs", "htdemucs_ft")
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
STEM_NAMES = ("vocals", "drums", "bass", "other")


@dataclass
class Config:
    # ingest
    start: float | None = None  # seconds into the source to start from
    duration: float | None = None  # seconds to keep
    # vocals: False (--no-vocals) treats the clip as instrumental: no vocal activity, no lyrics, no vocal notes
    vocals: bool = True
    # stems
    demucs_model: str = "htdemucs"
    # lyrics
    whisper_model: str = DEFAULT_WHISPER_MODEL
    language: str | None = None  # None = auto-detect
    # assemble
    curves_hz: float = 20.0
    plot: bool = True
    # runtime (does not affect results, so no stage lists it in config_keys)
    device: str = "auto"  # auto | mps | cpu

    def subset(self, keys: tuple[str, ...]) -> dict:
        d = asdict(self)
        return {k: d[k] for k in keys}
