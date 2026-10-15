"""Beat grid math: bar/beat numbering and time -> (bar, beat) lookup.

Conventions used everywhere in the outputs:

* Bars are numbered from 1 at the first downbeat. Beats before it (a pickup)
  belong to bar 0. Events before the first detected beat are located by
  extrapolating the grid backwards, so they can land in bar 0 or below.
* ``beat`` is 1-based and fractional: 1.0 is the downbeat, 2.5 is the "and"
  after beat 2. The integer part is the beat in the bar, the fraction is the
  position between that beat and the next one. Positions within SNAP_BEATS of
  a 16th-note point are snapped to it (the exact time is always in ``t``).
"""

from __future__ import annotations

import numpy as np

# Event positions within this many beats of a 16th-note grid point are labelled at that point, so an
# onset detected 10-30 ms off the beat reads "bar 17, beat 1.0" instead of "bar 16, beat 4.97".
# 0.06 beats is 30 ms at 120 BPM and stays clear of triplet positions (0.083 beats from a 16th).
SNAP_BEATS = 0.06
SNAP_GRID = 0.25


def snap_downbeats(beats: np.ndarray, downbeats: np.ndarray, tol: float = 0.07) -> np.ndarray:
    """Indices into ``beats`` of the beats closest to each downbeat (within ``tol`` seconds)."""
    if len(beats) == 0 or len(downbeats) == 0:
        return np.zeros(0, dtype=int)
    idx = np.searchsorted(beats, downbeats)
    out = []
    for d, i in zip(downbeats, idx):
        cands = [j for j in (i - 1, i) if 0 <= j < len(beats)]
        j = min(cands, key=lambda j: abs(beats[j] - d))
        if abs(beats[j] - d) <= tol:
            out.append(j)
    return np.unique(np.asarray(out, dtype=int))


def infer_beats_per_bar(downbeat_idx: np.ndarray, default: int = 4, min_agreement: float = 0.6) -> int:
    """Most common bar length in beats, if the downbeats agree on it clearly; else ``default`` (4/4)."""
    counts = np.diff(downbeat_idx)
    counts = counts[(counts >= 2) & (counts <= 7)]
    if len(counts) < 2:
        return default
    vals, freq = np.unique(counts, return_counts=True)
    mode = int(vals[np.argmax(freq)])
    return mode if freq.max() / len(counts) >= min_agreement else default


def regularize_downbeats(downbeat_idx: np.ndarray, n_beats: int, bpb: int) -> np.ndarray:
    """Fill in downbeats the tracker skipped, where the spacing makes it unambiguous.

    - a gap that is an exact multiple of the bar length gets the missing downbeats,
    - beats after the last downbeat get a downbeat every ``bpb`` beats,
    - a lead-in longer than one bar is counted backwards from the first downbeat,
    - with no downbeats at all, the first beat is taken as the first downbeat.
    """
    idx = [int(i) for i in downbeat_idx]
    if not idx:
        idx = [0] if n_beats else []
    out: list[int] = []
    if idx and idx[0] >= bpb:  # count whole bars back from the first detected downbeat
        out += list(range(idx[0] % bpb, idx[0], bpb))
    for a, b in zip(idx, idx[1:] + [None]):
        out.append(a)
        if b is None:
            out += list(range(a + bpb, n_beats, bpb))
        elif (b - a) > bpb and (b - a) % bpb == 0:
            out += list(range(a + bpb, b, bpb))
    return np.unique(np.asarray([i for i in out if 0 <= i < n_beats], dtype=int))


class BeatGrid:
    def __init__(self, beats, downbeat_idx, beats_per_bar: int = 4, duration: float | None = None):
        self.beats = np.asarray(beats, dtype=float)
        self.downbeat_idx = np.asarray(downbeat_idx, dtype=int)
        self.bpb = int(beats_per_bar)
        self.duration = duration
        n = len(self.beats)
        self.bar_of_beat = np.zeros(n, dtype=int)
        self.pos_of_beat = np.zeros(n, dtype=int)
        self._number()

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_tracker(cls, beats, downbeats, duration: float | None = None) -> "BeatGrid":
        beats = np.sort(np.asarray(beats, dtype=float))
        beats = beats[beats >= 0]
        if duration is not None:  # trackers can emit a beat at the very end of the audio; it has no musical span
            beats = beats[beats < duration - 0.05]
        raw = snap_downbeats(beats, np.asarray(downbeats, dtype=float))
        bpb = infer_beats_per_bar(raw)
        return cls(beats, regularize_downbeats(raw, len(beats), bpb), bpb, duration)

    @classmethod
    def from_rhythm(cls, rhythm: dict) -> "BeatGrid":
        """Rebuild from cache/rhythm.json."""
        return cls(rhythm["beats"], rhythm["downbeat_idx"], rhythm["beats_per_bar"], rhythm.get("duration"))

    def _number(self) -> None:
        n, d = len(self.beats), self.downbeat_idx
        if n == 0:
            return
        if len(d) == 0:
            self.downbeat_idx = d = np.array([0])
        for k, start in enumerate(d):
            stop = d[k + 1] if k + 1 < len(d) else n
            self.bar_of_beat[start:stop] = k + 1
            self.pos_of_beat[start:stop] = np.arange(1, stop - start + 1)
        for i in range(d[0]):  # pickup beats
            back = d[0] - i
            self.bar_of_beat[i] = -((back - 1) // self.bpb)
            self.pos_of_beat[i] = self.bpb - ((back - 1) % self.bpb)

    # ---- basic properties --------------------------------------------------
    def __len__(self) -> int:
        return len(self.beats)

    @property
    def ibis(self) -> np.ndarray:
        return np.diff(self.beats)

    def local_tempi(self, span: int = 4) -> np.ndarray:
        """BPM measured over `span`-beat windows (less sensitive to the tracker's frame quantisation than single IBIs)."""
        b = self.beats
        span = min(span, len(b) - 1)
        if span < 1:
            return np.zeros(0)
        return 60.0 * span / (b[span:] - b[:-span])

    @property
    def tempo_bpm(self) -> float | None:
        tempi = self.local_tempi()
        return float(np.median(tempi)) if len(tempi) else None

    def _edge_ibi(self, at_start: bool) -> float:
        ib = self.ibis
        if len(ib) == 0:
            return 0.5
        part = ib[:4] if at_start else ib[-4:]
        return float(np.median(part))

    def is_downbeat(self) -> np.ndarray:
        mask = np.zeros(len(self.beats), dtype=bool)
        mask[self.downbeat_idx] = True
        return mask

    def beat_edges(self) -> np.ndarray:
        """n+1 edges: beat i spans [edges[i], edges[i+1]). The last beat lasts one local IBI."""
        if len(self.beats) == 0:
            return np.zeros(1)
        end = self.beats[-1] + self._edge_ibi(at_start=False)
        if self.duration is not None:
            end = min(end, max(self.duration, self.beats[-1] + 1e-3))
        return np.append(self.beats, end)

    def bars(self) -> list[dict]:
        """One entry per bar: number, start, end, beat count. Bar 0 is a pickup, if there is one."""
        out = []
        d, n = self.downbeat_idx, len(self.beats)
        if n == 0:
            return out
        edges = self.beat_edges()
        if d[0] > 0:
            out.append({"bar": 0, "start": float(self.beats[0]), "end": float(self.beats[d[0]]), "beats": int(d[0])})
        for k, start in enumerate(d):
            stop = d[k + 1] if k + 1 < len(d) else n
            out.append({"bar": k + 1, "start": float(self.beats[start]), "end": float(edges[stop]), "beats": int(stop - start)})
        return out

    def bar_starts(self) -> np.ndarray:
        return self.beats[self.downbeat_idx] if len(self.beats) else np.zeros(0)

    # ---- lookup ------------------------------------------------------------
    def beat_position(self, t: float) -> float:
        """Continuous beat index at time t (0.0 = first beat), extrapolated outside the grid."""
        b = self.beats
        if len(b) == 0:
            return 0.0
        if t < b[0]:
            return (t - b[0]) / self._edge_ibi(at_start=True)
        if t >= b[-1]:
            return (len(b) - 1) + (t - b[-1]) / self._edge_ibi(at_start=False)
        i = int(np.searchsorted(b, t, side="right") - 1)
        return i + (t - b[i]) / (b[i + 1] - b[i])

    def _label(self, j: int) -> tuple[int, int]:
        """(bar, beat) of integer beat index j, continuing the numbering past either end."""
        n = len(self.beats)
        if n == 0:  # no grid at all (e.g. a drone or silence): everything is bar 1
            return 1, 1
        if 0 <= j < n:
            return int(self.bar_of_beat[j]), int(self.pos_of_beat[j])
        if j < 0:
            bar, pos = int(self.bar_of_beat[0]), int(self.pos_of_beat[0])
            for _ in range(-j):
                pos -= 1
                if pos < 1:
                    bar, pos = bar - 1, self.bpb
            return bar, pos
        bar, pos = int(self.bar_of_beat[-1]), int(self.pos_of_beat[-1])
        for _ in range(j - (n - 1)):
            pos += 1
            if pos > self.bpb:
                bar, pos = bar + 1, 1
        return bar, pos

    def locate(self, t: float) -> tuple[int, float]:
        """(bar, fractional beat) for a time in seconds. See module docstring for conventions."""
        if len(self.beats) == 0:
            return 1, 1.0
        x = self.beat_position(float(t))
        q = round(x / SNAP_GRID) * SNAP_GRID
        if abs(x - q) <= SNAP_BEATS:
            x = q
        j = int(np.floor(x + 1e-9))
        frac = min(max(x - j, 0.0), 0.99)
        bar, pos = self._label(j)
        return bar, round(pos + frac, 2)

    def bar_range(self, start: float, end: float) -> tuple[int, int]:
        """First and last bar of a span whose edges sit (roughly) on bar lines; no snapping involved."""
        if len(self.beats) == 0:
            return 1, 1
        first = int(np.floor(self.beat_position(float(start)) + 0.25))
        last = int(np.floor(self.beat_position(float(end)) - 0.25))
        return self._label(first)[0], self._label(max(last, first))[0]

    def nearest_bar_start(self, t: float) -> float:
        starts = self.bar_starts()
        if len(starts) == 0:
            return float(t)
        return float(starts[np.argmin(np.abs(starts - t))])

    def bar_of_time(self, t: float) -> int:
        return self.locate(t)[0]
