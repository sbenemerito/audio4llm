"""Shared fixtures: build the synthetic test clip once and run the full pipeline on it once.

The run goes to runs/test_clip/ with default settings, so it shares the cache
with `audio4llm test_data/test_clip.wav`: the first session takes a few minutes
(model downloads + inference), later sessions only re-run `assemble`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "test_data"
CLIP = TEST_DATA / "test_clip.wav"


@pytest.fixture(scope="session")
def ground_truth() -> dict:
    gt = TEST_DATA / "ground_truth.json"
    if not (CLIP.exists() and gt.exists()):
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_clip.py"), "--out-dir", str(TEST_DATA)], check=True)
    return json.loads(gt.read_text())


@pytest.fixture(scope="session")
def pipeline_run(ground_truth):
    from audio4llm.pipeline import run

    return run(CLIP, runs_root=ROOT / "runs")


@pytest.fixture(scope="session")
def run_dir(pipeline_run) -> Path:
    return pipeline_run[0].run_dir


@pytest.fixture(scope="session")
def analysis(run_dir) -> dict:
    return json.loads((run_dir / "analysis.json").read_text())


@pytest.fixture(scope="session")
def curves(run_dir) -> dict:
    return json.loads((run_dir / "curves.json").read_text())
