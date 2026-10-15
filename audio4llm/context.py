"""RunContext: the one object every stage receives.

It knows where the run folder is, which settings are active, which torch
device to try, and how to read/write the per-stage cache files. Stages never
build paths by hand; they ask the context.

Run folder layout::

    runs/<clip>/
      input.wav            canonical 44.1 kHz stereo audio (ingest)
      stems/<name>.wav     Demucs stems (stems)
      cache/<stage>.json   per-stage intermediate results
      cache/manifest.json  what ran, when, with which settings (pipeline)
      analysis.json, curves.json, summary.md, overview.png   (assemble)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import STEM_NAMES, Config
from .jsonfmt import dumps

log = logging.getLogger("audio4llm")


@dataclass
class RunContext:
    run_dir: Path
    config: Config = field(default_factory=Config)
    source: Path | None = None  # original input file; None when re-running an existing run folder

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stems_dir.mkdir(parents=True, exist_ok=True)

    # ---- paths -----------------------------------------------------------
    @property
    def name(self) -> str:
        return self.run_dir.name

    @property
    def input_wav(self) -> Path:
        return self.run_dir / "input.wav"

    @property
    def stems_dir(self) -> Path:
        return self.run_dir / "stems"

    @property
    def cache_dir(self) -> Path:
        return self.run_dir / "cache"

    def stem_path(self, stem: str) -> Path:
        return self.stems_dir / f"{stem}.wav"

    def stem_paths(self) -> dict[str, Path]:
        return {s: self.stem_path(s) for s in STEM_NAMES}

    def cache_path(self, filename: str) -> Path:
        return self.cache_dir / filename

    def out_path(self, filename: str) -> Path:
        return self.run_dir / filename

    # ---- cache I/O -------------------------------------------------------
    def read_json(self, filename: str) -> Any:
        return json.loads(self.cache_path(filename).read_text())

    def maybe_read_json(self, filename: str) -> Any | None:
        p = self.cache_path(filename)
        return json.loads(p.read_text()) if p.exists() else None

    def write_json(self, filename: str, data: Any) -> Path:
        return write_text_atomic(self.cache_path(filename), dumps(data))

    # ---- device ----------------------------------------------------------
    def torch_device(self) -> str:
        """Preferred torch device. Stages fall back to CPU themselves if MPS fails."""
        want = self.config.device
        if want != "auto":
            return want
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:  # torch missing or broken: CPU is still a valid answer
            pass
        return "cpu"


def write_text_atomic(path: Path, text: str) -> Path:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return path
