"""Command line entry point: ``audio4llm path/to/clip.mp3`` or ``python -m audio4llm ...``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import fields
from pathlib import Path

from . import __version__
from .config import DEFAULT_WHISPER_MODEL, DEMUCS_MODELS, Config
from .stages import STAGES

STAGE_NAMES = tuple(s.name for s in STAGES)
MAX_CURVES_HZ = 100.0  # the energy stage analyses at 100 frames/s; more would only repeat samples
# CLI flags that map 1:1 onto Config fields (None = not given on the command line)
CONFIG_FLAGS = ("start", "duration", "vocals", "demucs_model", "whisper_model", "language", "curves_hz", "plot", "device")

EPILOG = """\
examples:
  audio4llm song.mp3                            analyse the whole clip into runs/song/
  audio4llm song.mp3 --start 45 --duration 30   analyse 0:45-1:15 only
  audio4llm song.mp3 --force-stage lyrics       redo one stage (and whatever depends on it)
  audio4llm runs/song                           re-run on an existing run folder with the settings it was
                                                made with (only re-assembles the outputs)
  audio4llm song.mp3 --force                    ignore every cache

With an audio file, settings come from the flags (defaults for flags you leave out). With a run folder,
the folder's last settings are re-used and only the flags you pass change them.

stages, in order: """ + ", ".join(STAGE_NAMES)


def _number(kind: str):
    def parse(text: str) -> float:
        try:
            v = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {text!r}")
        if kind == "start" and v < 0:
            raise argparse.ArgumentTypeError("must be 0 or more")
        if kind == "duration" and v <= 0:
            raise argparse.ArgumentTypeError("must be more than 0")
        if kind == "hz" and not 0 < v <= MAX_CURVES_HZ:
            raise argparse.ArgumentTypeError(f"must be more than 0 and at most {MAX_CURVES_HZ:g}")
        return v

    return parse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio4llm",
        description="Turn a short music clip into beat-synced, LLM-readable analysis (analysis.json, curves.json, summary.md, overview.png).",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="audio file (mp3, m4a, wav, flac, ...) or an existing runs/<clip> folder")
    p.add_argument("--start", type=_number("start"), help="start offset into the input, in seconds")
    p.add_argument("--duration", type=_number("duration"), help="length to analyse, in seconds")
    p.add_argument("--name", help="run folder name (default: the input file name, so song.mp3 and song.m4a share runs/song)")
    p.add_argument("--runs-dir", default="runs", help="parent folder for run folders (default: ./runs)")
    p.add_argument("--force", action="store_true", help="re-run every stage, ignoring caches")
    p.add_argument("--force-stage", action="append", default=[], choices=STAGE_NAMES, metavar="STAGE",
                   help="re-run one stage (repeatable); downstream stages re-run automatically")
    p.add_argument("--demucs-model", choices=DEMUCS_MODELS,
                   help="stem separation model; htdemucs_ft is slightly better and ~4x slower (default: htdemucs)")
    p.add_argument("--whisper-model", help=f"mlx-whisper model repo (default: {DEFAULT_WHISPER_MODEL})")
    p.add_argument("--language", help="lyrics language code, e.g. en, es, ja (default: auto-detect)")
    p.add_argument("--vocals", action=argparse.BooleanOptionalAction, default=None,
                   help="--no-vocals treats the clip as instrumental: the vocal stem is never cut into regions, "
                        "Whisper (language detection + lyrics) and vocal note transcription are skipped. "
                        "--vocals goes back to auto-detection (default)")
    p.add_argument("--curves-hz", type=_number("hz"), help="sample rate of curves.json envelopes, up to 100 (default: 20)")
    p.add_argument("--device", choices=("auto", "mps", "cpu"),
                   help="torch device for Demucs and beat_this when they run (default: auto = MPS if available); "
                        "it does not invalidate the cache, so add --force-stage stems to redo cached stems")
    p.add_argument("--plot", action=argparse.BooleanOptionalAction, default=None,
                   help="write overview.png (default: yes; --no-plot skips it)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--version", action="version", version=f"audio4llm {__version__}")
    return p


def resolve_config(args: argparse.Namespace, saved: dict | None) -> Config:
    """Defaults, then the run folder's saved settings (if any), then the flags actually given."""
    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in (saved or {}).items() if k in known})
    for name in CONFIG_FLAGS:
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    return cfg


def settings_for(args: argparse.Namespace) -> Config:
    """The Config for this invocation. A run folder brings back the settings it was last run with."""
    from .pipeline import CONFIG_FILE

    saved = None
    run_folder = Path(args.input).expanduser()
    if run_folder.is_dir():
        if args.start is not None or args.duration is not None:
            logging.getLogger("audio4llm").warning(
                "--start/--duration only apply when an audio file is given; ignoring them for a run folder")
            args.start = args.duration = None
        cfg_file = run_folder / "cache" / CONFIG_FILE
        if cfg_file.exists():
            saved = json.loads(cfg_file.read_text())
    return resolve_config(args, saved)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Our own log at INFO; third-party libraries (httpx, huggingface_hub, numba, ...) only at WARNING.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("audio4llm")
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if not args.verbose:  # "unauthenticated requests to the HF Hub" is noise for public models
        for name in ("huggingface_hub", "huggingface_hub.utils._http"):
            logging.getLogger(name).setLevel(logging.ERROR)

    from .pipeline import PipelineError, format_results, run

    cfg = settings_for(args)
    try:
        ctx, results = run(args.input, cfg, runs_root=args.runs_dir, name=args.name,
                           force=args.force, force_stages=tuple(args.force_stage))
    except PipelineError as exc:
        if not getattr(exc, "logged", False):
            log.error("%s", exc)
        return 1
    print()
    print(format_results(results))
    print()
    for f in ("analysis.json", "curves.json", "summary.md", "overview.png"):
        p = ctx.out_path(f)
        if p.exists():
            print(f"  {p}")
    return 0 if all(r.status != "failed" for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
