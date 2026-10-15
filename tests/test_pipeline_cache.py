"""Caching rules of pipeline.execute, using tiny fake stages (no audio, no models)."""

import json

import pytest

from audio4llm.config import Config
from audio4llm.context import RunContext
from audio4llm.pipeline import PipelineError, Stage, execute


def make_stages(calls, fail=()):
    def stage(name, deps=(), keys=(), required=True, always=False):
        def run(ctx):
            calls.append(name)
            if name in fail:
                raise RuntimeError("boom")
            ctx.write_json(f"{name}.json", {"ok": True})

        return Stage(name=name, run=run, outputs=lambda ctx: [ctx.cache_path(f"{name}.json")],
                     deps=deps, config_keys=keys, required=required, always_run=always)

    return [
        stage("a"),
        stage("b", deps=("a",), keys=("demucs_model",)),
        stage("c", deps=("a",), required=False),
        stage("d", deps=("b", "c"), always=True),
    ]


def statuses(results):
    return {r.name: r.status for r in results}


@pytest.fixture
def ctx(tmp_path):
    return RunContext(run_dir=tmp_path / "run")


def test_second_run_uses_cache(ctx):
    calls = []
    execute(ctx, make_stages(calls))
    calls.clear()
    res = execute(ctx, make_stages(calls))
    assert statuses(res) == {"a": "cached", "b": "cached", "c": "cached", "d": "ran"}
    assert calls == ["d"]


def test_force_stage_reruns_dependents_only(ctx):
    calls = []
    execute(ctx, make_stages(calls))
    calls.clear()
    execute(ctx, make_stages(calls), force_stages=("b",))
    assert calls == ["b", "d"]


def test_upstream_rerun_invalidates_downstream(ctx):
    calls = []
    execute(ctx, make_stages(calls))
    calls.clear()
    execute(ctx, make_stages(calls), force_stages=("a",))
    assert calls == ["a", "b", "c", "d"]


def test_config_change_invalidates_stage(ctx):
    calls = []
    execute(ctx, make_stages(calls))
    calls.clear()
    ctx.config = Config(demucs_model="htdemucs_ft")
    execute(ctx, make_stages(calls))
    assert calls == ["b", "d"]


def test_deleted_output_reruns(ctx):
    calls = []
    execute(ctx, make_stages(calls))
    calls.clear()
    ctx.cache_path("c.json").unlink()
    execute(ctx, make_stages(calls))
    assert calls == ["c", "d"]


def test_force_reruns_everything(ctx):
    calls = []
    execute(ctx, make_stages(calls))
    calls.clear()
    execute(ctx, make_stages(calls), force=True)
    assert calls == ["a", "b", "c", "d"]


def test_optional_failure_is_contained(ctx):
    calls = []
    res = execute(ctx, make_stages(calls, fail=("c",)))
    assert statuses(res) == {"a": "ran", "b": "ran", "c": "failed", "d": "ran"}
    manifest = json.loads(ctx.cache_path("manifest.json").read_text())
    assert "c" not in manifest["stages"]  # retried next time


def test_required_failure_aborts(ctx):
    with pytest.raises(PipelineError):
        execute(ctx, make_stages([], fail=("b",)))


def test_unknown_force_stage(ctx):
    with pytest.raises(PipelineError):
        execute(ctx, make_stages([]), force_stages=("nope",))
