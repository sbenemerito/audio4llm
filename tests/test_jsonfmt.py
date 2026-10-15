import json
import math

import numpy as np

from audio4llm.jsonfmt import dumps


def test_round_trip_and_strictness():
    obj = {"a": [1, 2.5, None], "n": np.float32(0.25), "i": np.int64(3), "bad": math.nan, "arr": np.arange(3)}
    text = dumps(obj)
    assert json.loads(text) == {"a": [1, 2.5, None], "n": 0.25, "i": 3, "bad": None, "arr": [0, 1, 2]}


def test_small_objects_one_per_line_and_scalar_arrays_inline():
    beats = [{"t": i * 0.5, "bar": 1 + i // 4, "beat": 1 + i % 4, "is_downbeat": i % 4 == 0} for i in range(40)]
    text = dumps({"beats": beats, "energy": list(range(200))})
    lines = text.splitlines()
    assert sum(1 for ln in lines if ln.strip().startswith('{"t"')) == 40
    assert any(ln.strip().startswith('"energy": [0, 1, 2') for ln in lines)
