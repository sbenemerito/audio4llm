import numpy as np

from audio4llm.stages.energy import FRAME_HZ, gate_to_regions, regions_from_mask


def test_hits_outside_drum_activity_are_dropped():
    hits = {"method": "m", "onsets": [1.0, 5.0], "counts": {"kick": 2, "snare": 0, "hat": 0},
            "hits": [{"t": 1.0, "type": "kick", "strength": 1.0}, {"t": 5.0, "type": "kick", "strength": 1.0}]}
    times = np.arange(10 * FRAME_HZ) / FRAME_HZ
    out, mask = gate_to_regions(hits, [[0.5, 2.0]], times)
    assert [h["t"] for h in out["hits"]] == [1.0] and out["onsets"] == [1.0] and out["counts"]["kick"] == 1
    assert mask[int(1.0 * FRAME_HZ)] == 1.0 and mask[int(5.0 * FRAME_HZ)] == 0.0
    out, mask = gate_to_regions(hits, [], times)  # no drums at all
    assert out["hits"] == [] and mask.max() == 0.0


def test_regions_merge_gaps_and_drop_blips():
    times = np.arange(500) / FRAME_HZ
    mask = np.zeros(500, dtype=bool)
    mask[100:200] = mask[250:300] = True  # 0.5 s gap -> merged
    mask[400:420] = True  # 0.2 s blip -> dropped
    assert regions_from_mask(mask, times, min_s=0.4, gap_s=0.75) == [[1.0, 3.0]]
