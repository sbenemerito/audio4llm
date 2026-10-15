"""Stage interface, caching, and the run loop.

A stage is a ``Stage`` record: a name, a ``run(ctx)`` function, the files it
produces, the stages it reads from, and the config fields that affect it.

Caching rule: a stage is skipped when all of its outputs exist AND
``cache/manifest.json`` says it last ran with the same settings (config subset +
stage version + input fingerprint) AND every upstream stage it depends on still
has the same run token as when it ran. So re-running an upstream stage (or
changing a setting) automatically re-runs everything downstream of it, and
nothing else.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import Config
from .context import RunContext, write_text_atomic
from .jsonfmt import dumps

log = logging.getLogger("audio4llm")

MANIFEST = "manifest.json"
CONFIG_FILE = "config.json"  # the settings of the last successful run; re-used when re-running a run folder


@dataclass(frozen=True)
class Stage:
    name: str
    run: Callable[[RunContext], None]
    outputs: Callable[[RunContext], list[Path]]
    deps: tuple[str, ...] = ()
    config_keys: tuple[str, ...] = ()
    version: int = 1  # bump when the stage's algorithm or cache format changes
    required: bool = True  # False: on failure, log it and carry on without this stage
    always_run: bool = False  # assemble is cheap and is what you iterate on
    needs_source: bool = False  # ingest: cannot re-run from a run folder alone
    fingerprint: Callable[[RunContext], dict] | None = None  # extra cache key (e.g. source hash)


@dataclass
class StageResult:
    name: str
    status: str  # ran | cached | failed | skipped
    seconds: float = 0.0
    reason: str = ""


class PipelineError(RuntimeError):
    pass


@dataclass
class Manifest:
    path: Path
    stages: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        if path.exists():
            try:
                return cls(path, json.loads(path.read_text()).get("stages", {}))
            except json.JSONDecodeError:
                log.warning("manifest.json is corrupt; treating every stage as stale")
        return cls(path)

    def save(self) -> None:
        write_text_atomic(self.path, dumps({"stages": self.stages}))


def _cache_key(stage: Stage, ctx: RunContext) -> dict:
    key = {"version": stage.version, "config": ctx.config.subset(stage.config_keys)}
    if stage.fingerprint is not None:
        key["inputs"] = stage.fingerprint(ctx)
    return key


def _short(v) -> str:
    is_hash = isinstance(v, str) and len(v) >= 32 and all(c in "0123456789abcdef" for c in v)
    return repr(v[:10] + "…") if is_hash else repr(v)


def _diff(old: dict, new: dict) -> str:
    changed = []
    for section in ("version", "config", "inputs"):
        a, b = old.get(section), new.get(section)
        if isinstance(a, dict) and isinstance(b, dict):
            changed += [f"{k}: {_short(a.get(k))} -> {_short(b.get(k))}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        elif a != b:
            changed.append(f"{section}: {_short(a)} -> {_short(b)}")
    return "; ".join(changed) or "settings changed"


def stale_reason(stage: Stage, ctx: RunContext, manifest: Manifest, force: bool, force_stages: set[str]) -> str | None:
    """Why this stage must run, or None if its cached outputs are still valid."""
    if stage.needs_source and ctx.source is None:
        missing = [p for p in stage.outputs(ctx) if not p.exists()]
        if missing:
            raise PipelineError(f"{stage.name}: {missing[0]} is missing and no source file was given")
        if force or stage.name in force_stages:
            log.warning("%s: cannot re-run without the original file; keeping the existing input.wav", stage.name)
        return None
    if force:
        return "--force"
    if stage.name in force_stages:
        return "--force-stage"
    if stage.always_run:
        return "always runs"
    missing = [p for p in stage.outputs(ctx) if not p.exists()]
    if missing:
        return f"missing {missing[0].relative_to(ctx.run_dir)}"
    entry = manifest.stages.get(stage.name)
    if entry is None:
        return "not in manifest"
    key = _cache_key(stage, ctx)
    if entry.get("key") != key:
        return _diff(entry.get("key", {}), key)
    for dep in stage.deps:
        if entry.get("deps", {}).get(dep) != manifest.stages.get(dep, {}).get("token"):
            return f"upstream '{dep}' changed"
    return None


def execute(ctx: RunContext, stages: list[Stage], force: bool = False, force_stages: tuple[str, ...] = ()) -> list[StageResult]:
    names = {s.name for s in stages}
    unknown = set(force_stages) - names
    if unknown:
        raise PipelineError(f"unknown stage(s) for --force-stage: {', '.join(sorted(unknown))}. Stages: {', '.join(s.name for s in stages)}")

    manifest = Manifest.load(ctx.cache_path(MANIFEST))
    results: list[StageResult] = []
    failed: set[str] = set()

    for stage in stages:
        bad_deps = [d for d in stage.deps if d in failed]
        if bad_deps and not stage.always_run:
            log.warning("%-8s skipped (upstream %s failed)", stage.name, ", ".join(bad_deps))
            results.append(StageResult(stage.name, "skipped", reason=f"upstream {bad_deps[0]} failed"))
            failed.add(stage.name)
            continue

        reason = stale_reason(stage, ctx, manifest, force, set(force_stages))
        if reason is None:
            log.info("%-8s cached", stage.name)
            results.append(StageResult(stage.name, "cached"))
            continue

        log.info("%-8s running (%s)", stage.name, reason)
        key = _cache_key(stage, ctx)
        manifest.stages.pop(stage.name, None)
        t0 = time.perf_counter()
        try:
            stage.run(ctx)
        except Exception as exc:
            dt = time.perf_counter() - t0
            manifest.save()
            if stage.required:
                log.error("%-8s FAILED after %.1fs: %s", stage.name, dt, exc)
                err = PipelineError(f"stage '{stage.name}' failed: {exc}")
                err.logged = True
                raise err from exc
            log.exception("%-8s FAILED after %.1fs (optional stage, continuing without it)", stage.name, dt)
            for p in stage.outputs(ctx):  # never leave a half-written cache behind
                if p.is_file():
                    p.unlink()
            results.append(StageResult(stage.name, "failed", dt, str(exc)))
            failed.add(stage.name)
            continue
        dt = time.perf_counter() - t0
        manifest.stages[stage.name] = {
            "token": uuid.uuid4().hex[:12],
            "key": key,
            "deps": {d: manifest.stages.get(d, {}).get("token") for d in stage.deps},
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(dt, 2),
        }
        manifest.save()
        log.info("%-8s done in %.1fs", stage.name, dt)
        results.append(StageResult(stage.name, "ran", dt, reason))
    return results


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._").lower()
    return s or "clip"


def resolve_run_dir(source: Path, runs_root: Path, name: str | None = None) -> tuple[Path | None, Path]:
    """Return (source_file_or_None, run_dir). A run folder can be passed instead of an audio file."""
    source = Path(source).expanduser()
    if source.is_dir():
        if not (source / "input.wav").exists():
            raise PipelineError(f"{source} is a directory but has no input.wav; pass an audio file instead")
        return None, source
    if not source.exists():
        raise PipelineError(f"input not found: {source}")
    return source.resolve(), Path(runs_root) / (name or slugify(source.stem))


def run(
    source: str | Path,
    config: Config | None = None,
    *,
    runs_root: str | Path = "runs",
    name: str | None = None,
    force: bool = False,
    force_stages: tuple[str, ...] = (),
) -> tuple[RunContext, list[StageResult]]:
    """Programmatic entry point (the CLI and the tests both call this)."""
    from .stages import STAGES

    src, run_dir = resolve_run_dir(Path(source), Path(runs_root), name)
    is_new = not run_dir.exists()
    ctx = RunContext(run_dir=run_dir, config=config or Config(), source=src)
    log.info("run folder: %s", ctx.run_dir)
    try:
        results = execute(ctx, STAGES, force=force, force_stages=tuple(force_stages))
    except PipelineError:
        if is_new and not ctx.input_wav.exists():  # nothing usable was produced: don't leave an empty folder
            shutil.rmtree(ctx.run_dir, ignore_errors=True)
        raise
    write_text_atomic(ctx.cache_path(CONFIG_FILE), dumps(asdict(ctx.config)))
    return ctx, results


def format_results(results: list[StageResult]) -> str:
    lines = [f"{'stage':<9} {'status':<8} {'seconds':>8}  note"]
    for r in results:
        secs = f"{r.seconds:8.1f}" if r.status in ("ran", "failed") else f"{'-':>8}"
        lines.append(f"{r.name:<9} {r.status:<8} {secs}  {r.reason}")
    total = sum(r.seconds for r in results)
    lines.append(f"{'total':<9} {'':<8} {total:8.1f}")
    return "\n".join(lines)
