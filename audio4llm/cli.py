"""Command line entry point: ``audio4llm path/to/clip.mp3`` or ``python -m audio4llm ...``."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .config import DEFAULT_WHISPER_MODEL, DEMUCS_MODELS, Config

STAGE_NAMES = ("ingest", "stems", "rhythm", "energy", "sections", "lyrics", "notes", "assemble")

EPILOG = """\
examples:
  audio4llm song.mp3                            analyse the whole clip into runs/song/
  audio4llm song.mp3 --start 45 --duration 30   analyse 0:45-1:15 only
  audio4llm song.mp3 --force-stage lyrics       redo one stage (and whatever depends on it)
  audio4llm runs/song                           re-run on an existing run folder (re-assembles outputs)
  audio4llm song.mp3 --force                    ignore every cache

stages, in order: """ + ", ".join(STAGE_NAMES)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio4llm",
        description="Turn a short music clip into beat-synced, LLM-readable analysis (analysis.json, curves.json, summary.md, overview.png).",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="audio file (mp3, m4a, wav, flac, ...) or an existing runs/<clip> folder")
    p.add_argument("--start", type=float, help="start offset into the input, in seconds")
    p.add_argument("--duration", type=float, help="length to analyse, in seconds")
    p.add_argument("--name", help="run folder name (default: input file name)")
    p.add_argument("--runs-dir", default="runs", help="parent folder for run folders (default: ./runs)")
    p.add_argument("--force", action="store_true", help="re-run every stage, ignoring caches")
    p.add_argument("--force-stage", action="append", default=[], choices=STAGE_NAMES, metavar="STAGE",
                   help="re-run one stage (repeatable); downstream stages re-run automatically")
    p.add_argument("--demucs-model", default="htdemucs", choices=DEMUCS_MODELS,
                   help="stem separation model; htdemucs_ft is slightly better and ~4x slower (default: htdemucs)")
    p.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL,
                   help=f"mlx-whisper model repo (default: {DEFAULT_WHISPER_MODEL})")
    p.add_argument("--language", help="lyrics language code, e.g. en, es, ja (default: auto-detect)")
    p.add_argument("--curves-hz", type=float, default=20.0, help="sample rate of curves.json envelopes (default: 20)")
    p.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"), help="torch device for Demucs and beat_this (default: auto)")
    p.add_argument("--no-plot", action="store_true", help="skip overview.png")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--version", action="version", version=f"audio4llm {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Our own log at INFO; third-party libraries (httpx, huggingface_hub, numba, ...) only at WARNING.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("audio4llm").setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if not args.verbose:  # "unauthenticated requests to the HF Hub" is noise for public models
        for name in ("huggingface_hub", "huggingface_hub.utils._http"):
            logging.getLogger(name).setLevel(logging.ERROR)

    from .pipeline import PipelineError, format_results, run

    cfg = Config(
        start=args.start,
        duration=args.duration,
        demucs_model=args.demucs_model,
        whisper_model=args.whisper_model,
        language=args.language,
        curves_hz=args.curves_hz,
        plot=not args.no_plot,
        device=args.device,
    )
    try:
        ctx, results = run(args.input, cfg, runs_root=args.runs_dir, name=args.name,
                           force=args.force, force_stages=tuple(args.force_stage))
    except PipelineError as exc:
        logging.getLogger("audio4llm").error("%s", exc)
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
