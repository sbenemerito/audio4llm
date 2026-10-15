import numpy as np
import pytest

from audio4llm.grid import BeatGrid, infer_beats_per_bar, regularize_downbeats, snap_downbeats


def steady(n=33, period=0.5, t0=0.1):
    return t0 + period * np.arange(n)


def test_numbering_with_pickup():
    beats = steady()
    grid = BeatGrid.from_tracker(beats, beats[1::4])  # first downbeat is beat index 1
    assert grid.bpb == 4
    assert (grid.bar_of_beat[0], grid.pos_of_beat[0]) == (0, 4)  # pickup beat
    assert (grid.bar_of_beat[1], grid.pos_of_beat[1]) == (1, 1)
    assert (grid.bar_of_beat[6], grid.pos_of_beat[6]) == (2, 2)


def test_locate_fractional_and_extrapolated():
    beats = steady(t0=1.0)
    grid = BeatGrid.from_tracker(beats, beats[::4])
    assert grid.locate(1.0) == (1, 1.0)
    assert grid.locate(1.75) == (1, 2.5)  # halfway between beats 2 and 3
    assert grid.locate(2.92) == (1, 4.84)  # off the 16th grid: kept as is
    assert grid.locate(0.5) == (0, 4.0)  # one beat before the first downbeat
    assert grid.locate(beats[-1]) == (9, 1.0)  # 33 beats: the last one starts bar 9
    assert grid.locate(beats[-1] + 0.5) == (9, 2.0)  # one beat past the end continues the count


def test_locate_snaps_near_grid_points():
    beats = steady(t0=1.0)  # 120 BPM: 0.06 beats = 30 ms
    grid = BeatGrid.from_tracker(beats, beats[::4])
    assert grid.locate(2.98) == (2, 1.0)  # 20 ms before bar 2 belongs to bar 2, not "bar 1 beat 4.96"
    assert grid.locate(1.765) == (1, 2.5)  # 15 ms late off-beat
    assert grid.locate(1.0 + 0.5 / 3) == (1, 1.33)  # a triplet is not pulled onto the 16th grid


def test_three_four_detected():
    beats = steady(n=31)
    grid = BeatGrid.from_tracker(beats, beats[::3])
    assert grid.bpb == 3
    assert grid.pos_of_beat[:6].tolist() == [1, 2, 3, 1, 2, 3]


def test_default_four_four_when_unclear():
    assert infer_beats_per_bar(np.array([0, 4])) == 4  # too few bars to tell
    assert infer_beats_per_bar(np.array([0, 3, 7, 12, 14])) == 4  # no clear agreement


def test_regularize_fills_missed_downbeats():
    idx = regularize_downbeats(np.array([4, 8, 16]), n_beats=26, bpb=4)
    assert idx.tolist() == [0, 4, 8, 12, 16, 20, 24]


def test_snap_downbeats_tolerance():
    beats = steady()
    snapped = snap_downbeats(beats, np.array([beats[4] + 0.03, beats[8] + 0.2]))
    assert snapped.tolist() == [4]


def test_end_of_clip_beat_dropped():
    beats = steady(n=21, t0=0.0)  # last beat at exactly 10.0 s
    grid = BeatGrid.from_tracker(beats, beats[::4], duration=10.0)
    assert grid.beats[-1] == pytest.approx(9.5)


def test_bars_and_tempo():
    beats = steady()
    grid = BeatGrid.from_tracker(beats, beats[::4], duration=17.0)
    bars = grid.bars()
    assert bars[0] == {"bar": 1, "start": pytest.approx(0.1), "end": pytest.approx(2.1), "beats": 4}
    assert grid.tempo_bpm == pytest.approx(120.0)
    assert grid.nearest_bar_start(4.0) == pytest.approx(4.1)


def test_bar_range_of_bar_aligned_span():
    beats = steady(t0=0.04)  # bars start at 0.04, 2.04, 4.04, ...
    grid = BeatGrid.from_tracker(beats, beats[::4])
    assert grid.bar_range(0.0, 4.044) == (1, 2)  # edges a few ms off the bar lines
    assert grid.bar_range(4.04, 8.03) == (3, 4)
