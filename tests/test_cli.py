import json

import pytest

from audio4llm.cli import build_parser, settings_for
from audio4llm.pipeline import CONFIG_FILE


@pytest.mark.parametrize("argv", [
    ["x.mp3", "--curves-hz", "0"], ["x.mp3", "--curves-hz", "-5"], ["x.mp3", "--curves-hz", "500"],
    ["x.mp3", "--start", "-1"], ["x.mp3", "--duration", "0"], ["x.mp3", "--force-stage", "nope"],
])
def test_bad_values_are_rejected(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_file_input_uses_flags_and_defaults():
    cfg = settings_for(build_parser().parse_args(["x.mp3", "--start", "5", "--no-plot"]))
    assert cfg.start == 5 and cfg.plot is False and cfg.demucs_model == "htdemucs" and cfg.curves_hz == 20


def test_run_folder_reuses_saved_settings(tmp_path):
    run = tmp_path / "song"
    (run / "cache").mkdir(parents=True)
    saved = {"start": 10.0, "duration": 30.0, "demucs_model": "htdemucs_ft", "whisper_model": "small",
             "language": "es", "curves_hz": 30.0, "plot": False, "device": "cpu"}
    (run / "cache" / CONFIG_FILE).write_text(json.dumps(saved))
    cfg = settings_for(build_parser().parse_args([str(run), "--language", "en", "--start", "99"]))
    assert cfg.demucs_model == "htdemucs_ft" and cfg.whisper_model == "small" and cfg.plot is False
    assert cfg.language == "en"  # an explicit flag wins
    assert cfg.start == 10.0  # trimming can't change for an existing run folder
