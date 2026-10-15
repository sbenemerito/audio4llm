"""Stage 4 - sections and moments (plain DSP on the energy-stage curves, no models).

Sections describe what the music is doing (its character), not what the song part is
called: no verse/chorus labels.

Segmentation. Every bar (from the beat grid) gets a feature vector: loudness, the
level of each stem, kick / snare / hat hits per beat and brightness, each scaled so
that one unit is a clearly audible change (SCALES). The clip is then cut at bar
starts into the fewest sections that each follow a straight line in those features:
a dynamic programme minimises the squared residual of each section's fit +
SECTION_PENALTY per section + SHORT_PENALTY per section shorter than MIN_BARS. Each
section is fitted with a constant, or with a straight line at an extra SLOPE_PENALTY:
builds and fades stay in one piece, but a step change is not smeared into a fake
trend. Features first go through a 3-bar running median, so a one-bar fill or dropout
cannot start a section, and the short-section penalty means fewer than 4 bars only
appear when the change is dramatic. At most MAX_SECTIONS sections. Without a usable
beat grid, 2-second blocks stand in for bars.

Characters (first matching rule wins; "loud" is relative to the clip's loudest bar):
  silence    mean loudness more than 40 LU below the loudest bar
  build      loudness rises >= 5 LU across the section (linear fit), or it rises
             >= 3 LU while drum hits per beat at least double
  outro      last section, falling >= 4 LU across it (a fade or wind-down)
  full       within 3 LU of the loudest section, with a beat (kick/snare >= 0.5 per
             beat) or at least three stems active
  breakdown  >= 4 LU quieter than the section before it, which was full / groove /
             build, and the beat has mostly gone (kick/snare under half its level there)
  groove     a steady beat (kick/snare >= 0.5 per beat) but not full
  sparse     anything else: quiet and/or few elements, no steady beat

Moments (types):
  silence      mix more than 50 dB below its loudest for >= 1 s (incl. leading/trailing)
  stop         sudden gap: the mix falls >= 12 dB below its previous 2 s for 0.25-4 s and
               then comes back; those 2 s must be within 12 dB of the clip's loudest 2 s.
               A near-silent gap of 1 s or more is reported as silence instead
  build_peak   loudest point (momentary loudness) in the last bar of a build, before any stop
  drop         start of a full section that follows a build, a stop or a silence, or whose
               first bar is >= 6 LU louder than the bar before; snapped to a kick (or any
               drum onset) within 80 ms
  breakdown    start of a breakdown section
  loudest      peak of the 1 s loudness curve (one per clip)
  vocal_entry  vocals become active after >= 3 s without them
  drums_in / drums_out, bass_in / bass_out
               stem becomes active / inactive with >= 2 s of inactivity on the other side
  kick_in      first kick after >= 4 s without kicks
Entries/exits within 0.3 s of a drop or breakdown are folded into its description.

Contract: writes cache/sections.json
{
  "method": "<one line>",
  "sections": [   # contiguous, cover the clip, internal boundaries at bar starts; first starts at 0.0, last ends at duration
    {"start": 0.0, "end": 16.0, "bars": [1, 8],   # bars: first and last bar number (omitted without a beat grid)
     "character": "silence"|"sparse"|"build"|"groove"|"full"|"breakdown"|"outro",
     "energy": 0..1,              # mean loudness, 0 = 30 LU below the loudest bar, 1 = the loudest bar
     "loudness_lufs": -21.3,      # mean momentary loudness over the section
     "trend": "rising"|"falling"|"flat",   # linear-fit loudness change across the section, +-3 LU
     "change_lu": 12.5,           # that change, in LU
     "active_stems": ["drums", ...],       # stems active (energy.json) for >= 2 s or 25% of the section, whichever is shorter
     "hits_per_beat": {"kick": 1.0, "snare": 0.5, "hat": 2.0, "all": 3.5},
     "description": "<short phrase>"}
  ],
  "moments": [    # sorted by t
    {"t": 32.0, "type": "<one of the moment types above>", "description": "<short phrase>", "strength": 0..1,
     "end": 32.5}             # silence and stop only: when the gap ends
  ]
}
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import STEM_NAMES
from ..context import RunContext
from ..grid import BeatGrid
from ..jsonfmt import r2, r3
from ..pipeline import Stage
from .energy import FRAME_HZ, load_curves, smooth

log = logging.getLogger("audio4llm")

METHOD = ("bar-level features (loudness, stem levels, kick/snare/hat density, brightness) cut into sections by a "
          "piecewise-linear dynamic programme at bar starts; characters and moments from documented rules")

DRUM_TYPES = ("kick", "snare", "hat")

# ---- segmentation ----------------------------------------------------------------------
FALLBACK_BAR_S = 2.0  # pseudo-bar length when there is no usable beat grid (one 4/4 bar at 120 BPM)
FLOOR_LUFS = -70.0  # the momentary-loudness floor used by the energy stage
STEM_FLOOR_DB = 40.0  # stem levels count from 40 dB below the stem's reference (below that = absent)
# One unit of each feature is a clearly audible change; residuals are measured in these units.
SCALES = {
    "loudness": 3.0,  # LU
    "drums": 8.0, "bass": 8.0, "other": 8.0,  # dB
    "vocals": 16.0,  # dB: vocals come and go within sections, so they weigh half as much
    "kick": 0.5, "snare": 0.5,  # hits per beat
    "hat": 1.0,  # hits per beat (hats are busier and noisier)
    "brightness": 0.1,  # 0..1 log-centroid scale
}
SECTION_PENALTY = 12.0  # cost of one more section in squared feature units: about a 1-unit change held
#                         over 8+8 bars. Checked on simulated songs with realistic bar-to-bar jitter,
#                         fills and vocal gaps: stationary stretches stay one section, 3 LU steps split
SLOPE_PENALTY = 12.0  # cost of letting a section follow a trend instead of a constant (same scale)
SHORT_PENALTY = 25.0  # extra cost for a section shorter than MIN_BARS: only dramatic changes pay it
MIN_BARS = 4  # phrases are almost always 4+ bars
HARD_MIN_BARS = 2  # anything shorter is a moment (stop, fill), not a section
MAX_SECTIONS = 10  # more than ~10 sections in 80 s would read as noise

# ---- characters ------------------------------------------------------------------------
ENERGY_RANGE_LU = 30.0  # section energy 0..1 spans 30 LU below the loudest bar (same range as energy_*)
TREND_LU = 3.0  # rising / falling if the linear fit changes by >= 3 LU (a clearly audible change)
BUILD_LU = 5.0  # a build is an obvious ramp, well above bar-to-bar loudness jitter (~1 LU)
BUILD_WITH_DENSITY_LU = 3.0  # ...or a gentler ramp that is clearly getting busier (rolls, 8ths -> 16ths)
OUTRO_FALL_LU = 4.0  # a fade or wind-down at the end; a clip that is just cut off stays flat
FULL_WITHIN_LU = 3.0  # "full" is reserved for the top of the clip's dynamics (verse vs chorus is ~3 LU)
BREAKDOWN_DROP_LU = 4.0  # clearly quieter than what came before, not just a lighter bar
BEAT_PER_BEAT = 0.5  # kick+snare hits per beat that make a steady beat (e.g. kick on 1 and 3)
SILENCE_LU = 40.0  # 40 LU below the loudest bar is inaudible next to the rest of the clip
STEM_ACTIVE_FRAC = 0.25  # a stem counts for a section if it plays a quarter of it...
STEM_ACTIVE_S = 2.0  # ...or 2 s, whichever is shorter (a vocal phrase in a long section still counts)

# ---- moments ---------------------------------------------------------------------------
STOP_DROP_DB = 12.0  # a stop is a real cut: >= 12 dB below the last 2 s, not a softer bar
STOP_MIN_S = 0.25  # at least half a beat at 120 BPM; shorter dips are gaps between notes
STOP_CONTEXT_S = 2.0  # compare with the last ~bar of music
STOP_LOUD_WITHIN_DB = 12.0  # only cuts out of loud music count (quiet passages have natural gaps)
SILENCE_DB = 50.0  # 50 dB below the mix's loudest frames is effectively silent
SILENCE_MIN_S = 1.0  # shorter near-silent gaps are stops
DROP_JUMP_LU = 6.0  # a first bar 6 LU above the bar before is a drop even without a build or stop
STEM_GAP_S = 2.0  # drums / bass must be absent for ~a bar to "come in" again
VOCAL_GAP_S = 3.0  # vocals pause between lines; a new entry needs a longer gap
KICK_GAP_S = 4.0  # ~2 bars without kicks before a "kick comes in"
FOLD_S = 0.3  # entries this close to a drop / breakdown are part of it


# ---- small helpers -----------------------------------------------------------------------

def power_mean_db(db: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(10 ** (np.asarray(db, dtype=np.float64) / 10)) + 1e-12)) if len(db) else FLOOR_LUFS


def frames_of(t0: float, t1: float, n: int) -> slice:
    a = int(np.clip(round(t0 * FRAME_HZ), 0, n))
    b = int(np.clip(round(t1 * FRAME_HZ), a, n))
    return slice(a, max(b, min(a + 1, n)))


def linear_change(y: np.ndarray) -> float:
    """Change of the least-squares line through y across its length (last fitted - first fitted)."""
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope * (len(y) - 1))


def overlap(regions: list[list[float]], t0: float, t1: float) -> float:
    return sum(max(0.0, min(e, t1) - max(s, t0)) for s, e in regions)


# ---- bars ------------------------------------------------------------------------------

def bar_edges(starts: np.ndarray, duration: float) -> np.ndarray:
    """Contiguous bar edges from 0 to duration, built on the grid's bar starts.

    A lead-in shorter than half a bar joins bar 1; a longer one, a tail after the last
    detected bar, or a clip with no grid at all, is filled with bar-length blocks.
    """
    starts = np.asarray([s for s in np.sort(starts) if 0.0 <= s < duration], dtype=float)
    if len(starts) < 2:
        return np.append(np.arange(0.0, duration - 0.5 * FALLBACK_BAR_S, FALLBACK_BAR_S), duration)
    bar = float(np.median(np.diff(starts)))
    edges = list(starts)
    while edges[0] > 0.5 * bar:  # count whole bars back to the start
        edges.insert(0, edges[0] - bar)
    edges[0] = 0.0
    while edges[-1] + 1.5 * bar < duration:  # the grid stopped early
        edges.append(edges[-1] + bar)
    if duration - edges[-1] < 0.5 * bar and len(edges) > 1:
        edges.pop()  # a short tail joins the last bar
    return np.asarray(edges + [duration])




LEAD_S = 0.05  # bar windows end 50 ms early: a hit on the next downbeat (centred analysis windows,
#               tracker a few ms late) belongs to the next bar


def bar_features(edges: np.ndarray, curves: dict, energy: dict, hits: list[dict], beats: np.ndarray) -> dict[str, np.ndarray]:
    """Per-bar features in natural units (LUFS, dB re stem reference, hits per beat, 0..1) plus beat counts."""
    n = len(curves["times"])
    nb = len(edges) - 1
    beats = np.asarray(beats, dtype=float)
    beat_s = float(np.median(np.diff(beats))) if len(beats) > 1 else FALLBACK_BAR_S / 4
    out = {k: np.zeros(nb) for k in ("loudness", *STEM_NAMES, *DRUM_TYPES, "brightness", "beats")}
    hit_t = {k: np.array([h["t"] for h in hits if h["type"] == k]) for k in DRUM_TYPES}
    for i in range(nb):
        a = edges[i] - LEAD_S if i else 0.0
        b = edges[i + 1] - LEAD_S if i < nb - 1 else edges[i + 1]
        sl = frames_of(a, b, n)
        out["loudness"][i] = max(power_mean_db(curves["lufs"][sl]), FLOOR_LUFS)
        for s in STEM_NAMES:
            ref = energy["levels"][s]["ref_db"]
            out[s][i] = max(power_mean_db(curves[f"db_{s}"][sl]) - ref, -STEM_FLOOR_DB)
        n_beats = int(np.sum((beats >= a) & (beats < b))) or max(1, round((b - a) / beat_s))
        out["beats"][i] = n_beats
        for k in DRUM_TYPES:
            out[k][i] = np.sum((hit_t[k] >= a) & (hit_t[k] < b)) / n_beats if len(hit_t[k]) else 0.0
        out["brightness"][i] = float(np.mean(curves["brightness"][sl]))
    return out


# ---- segmentation ------------------------------------------------------------------------

def median3(x: np.ndarray) -> np.ndarray:
    """3-bar running median: removes one-bar outliers (fills, a missed bar) but keeps steps and ramps."""
    if len(x) < 3:
        return np.asarray(x, dtype=float)
    p = np.concatenate([[x[0]], x, [x[-1]]])
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def feature_matrix(feats: dict[str, np.ndarray]) -> np.ndarray:
    """Scaled, one-bar-outlier-free features, one row per bar (what the segmentation sees)."""
    return np.stack([median3(feats[k]) / s for k, s in SCALES.items()], axis=1)


def segment_costs(X: np.ndarray, slope_penalty: float = SLOPE_PENALTY) -> np.ndarray:
    """cost[i, j] for bars i..j-1: squared residual of the cheaper model, a constant per feature or
    a straight line per feature + slope_penalty (so a step is not absorbed into a fake trend)."""
    nb = len(X)
    cost = np.full((nb + 1, nb + 1), np.inf)
    for i in range(nb):
        for j in range(i + 1, nb + 1):
            m = j - i
            centred = X[i:j] - X[i:j].mean(axis=0)
            flat = float(np.sum(centred ** 2))
            if m <= 2:
                cost[i, j] = flat
                continue
            x = np.arange(m, dtype=float) - (m - 1) / 2
            slope = (x @ centred) / (x @ x)
            cost[i, j] = min(flat, float(np.sum((centred - np.outer(x, slope)) ** 2)) + slope_penalty)
    return cost


def segment(X: np.ndarray, penalty: float = SECTION_PENALTY, short_penalty: float = SHORT_PENALTY,
            min_bars: int = MIN_BARS, hard_min: int = HARD_MIN_BARS, max_sections: int = MAX_SECTIONS) -> list[tuple[int, int]]:
    """Optimal piecewise-linear segmentation of per-bar features: [(first_bar, end_bar_exclusive), ...]."""
    nb = len(X)
    if nb == 0:
        return []
    hard_min = max(1, min(hard_min, nb))
    cost = segment_costs(X)
    K = max(1, min(max_sections, nb // hard_min))
    best = np.full((K + 1, nb + 1), np.inf)  # best[k, j]: bars 0..j-1 in exactly k sections
    back = np.zeros((K + 1, nb + 1), dtype=int)
    best[0, 0] = 0.0
    for k in range(1, K + 1):
        for j in range(hard_min, nb + 1):
            for i in range(0, j - hard_min + 1):
                if not np.isfinite(best[k - 1, i]):
                    continue
                c = best[k - 1, i] + cost[i, j] + penalty + (short_penalty if j - i < min_bars else 0.0)
                if c < best[k, j]:
                    best[k, j], back[k, j] = c, i
    k = int(np.argmin(best[:, nb]))
    segs, j = [], nb
    while k > 0:
        i = int(back[k, j])
        segs.append((i, j))
        j, k = i, k - 1
    return segs[::-1]


# ---- characters --------------------------------------------------------------------------

def section_stats(i: int, j: int, edges: np.ndarray, feats: dict, curves: dict, energy: dict) -> dict:
    """Numbers the character rules and the output need, for bars i..j-1."""
    start, end = float(edges[i]), float(edges[j])
    lufs = power_mean_db(curves["lufs"][frames_of(start, end, len(curves["times"]))])
    beats = feats["beats"][i:j]
    per_beat = {k: float(np.sum(feats[k][i:j] * beats) / max(beats.sum(), 1)) for k in DRUM_TYPES}
    dens = sum(feats[k][i:j] for k in DRUM_TYPES)
    x = np.arange(j - i)
    fit = np.polyval(np.polyfit(x, dens, 1), [0, j - i - 1]) if j - i >= 2 else [dens[0], dens[0]]
    dur = end - start
    need = min(STEM_ACTIVE_S, STEM_ACTIVE_FRAC * dur)
    active = [s for s in STEM_NAMES if overlap(energy["active"].get(s, []), start, end) >= need]
    return {
        "start": start, "end": end, "lufs": lufs,
        "change": linear_change(feats["loudness"][i:j]),
        "beat": per_beat["kick"] + per_beat["snare"],
        "per_beat": per_beat,
        "dens_start": float(max(fit[0], 0.0)), "dens_end": float(max(fit[1], 0.0)),
        "active": active,
    }


def label_sections(stats: list[dict], loudest_bar: float) -> list[str]:
    """Character of each section by the rules in the module docstring (first match wins)."""
    loudest_section = max((s["lufs"] for s in stats), default=loudest_bar)
    labels: list[str] = []
    for k, s in enumerate(stats):
        prev = stats[k - 1] if k else None
        last = all(t["lufs"] < loudest_bar - SILENCE_LU for t in stats[k + 1:])  # only silence after it
        if s["lufs"] < loudest_bar - SILENCE_LU:
            lab = "silence"
        elif s["change"] >= BUILD_LU or (s["change"] >= BUILD_WITH_DENSITY_LU and s["dens_end"] >= max(2 * s["dens_start"], s["dens_start"] + 1)):
            lab = "build"
        elif last and k > 0 and s["change"] <= -OUTRO_FALL_LU:
            lab = "outro"
        elif s["lufs"] >= loudest_section - FULL_WITHIN_LU and (s["beat"] >= BEAT_PER_BEAT or len(s["active"]) >= 3):
            lab = "full"
        elif (prev is not None and labels[-1] in ("full", "groove", "build") and s["lufs"] <= prev["lufs"] - BREAKDOWN_DROP_LU
              and prev["beat"] >= BEAT_PER_BEAT and s["beat"] < 0.5 * prev["beat"]):
            lab = "breakdown"
        elif s["beat"] >= BEAT_PER_BEAT:
            lab = "groove"
        else:
            lab = "sparse"
        labels.append(lab)
    return labels


STEM_WORDS = {"drums": "drums", "bass": "bass", "other": "other instruments", "vocals": "vocals"}


def describe_section(s: dict, label: str, prev: dict | None) -> str:
    lead = {
        "silence": "silence",
        "sparse": "sparse",
        "build": f"builds {s['change']:+.0f} LU",
        "full": "full energy",
        "groove": "steady groove",
        "breakdown": f"breakdown, {prev['lufs'] - s['lufs']:.0f} LU down" if prev else "breakdown",
        "outro": f"winds down {s['change']:+.0f} LU",
    }[label]
    parts = [lead]
    if s["active"]:
        parts.append(", ".join(STEM_WORDS[x] for x in s["active"]))
    pb = s["per_beat"]
    drums = " / ".join(f"{k} {pb[k]:.1f}" for k in DRUM_TYPES if pb[k] >= 0.1)
    if label == "build" and s["dens_end"] >= s["dens_start"] + 1:
        parts.append(f"drum hits per beat {s['dens_start']:.1f} -> {s['dens_end']:.1f}")
    elif drums:
        parts.append(f"{drums} per beat")
    return "; ".join(parts)


# ---- moments -----------------------------------------------------------------------------

def smoothed_db(db: np.ndarray, seconds: float) -> np.ndarray:
    return 10 * np.log10(smooth(10 ** (np.asarray(db, dtype=np.float64) / 10), seconds) + 1e-12)


def trailing_context(level: np.ndarray, window_s: float = STOP_CONTEXT_S, gap_s: float = 0.1) -> np.ndarray:
    """Power-mean level of the window_s seconds ending gap_s before each frame (-inf with no history)."""
    n = len(level)
    c = np.concatenate([[0.0], np.cumsum(10 ** (np.asarray(level, dtype=np.float64) / 10))])
    idx = np.arange(n)
    hi = np.clip(idx - round(gap_s * FRAME_HZ), 0, n)
    lo = np.clip(hi - round(window_s * FRAME_HZ), 0, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        ctx = 10 * np.log10((c[hi] - c[lo]) / np.maximum(hi - lo, 1))
    ctx[hi - lo < round(0.5 * window_s * FRAME_HZ)] = -np.inf  # need at least half a window of history
    return ctx


def find_silences(level: np.ndarray, ref_db: float, min_s: float = SILENCE_MIN_S) -> list[tuple[int, int]]:
    """Runs of frames more than SILENCE_DB below ref_db lasting >= min_s, as (first, end_exclusive)."""
    quiet = np.concatenate([[False], level < ref_db - SILENCE_DB, [False]])
    edges = np.flatnonzero(np.diff(quiet.astype(int)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if (b - a) >= min_s * FRAME_HZ]


def find_stops(level: np.ndarray, loud_ref: float) -> list[dict]:
    """Sudden gaps: >= STOP_DROP_DB below the previous 2 s for STOP_MIN_S..4 s, music loud before and back after."""
    n = len(level)
    ctx = trailing_context(level)
    out, i = [], 0
    min_len, max_len, after_len = round(STOP_MIN_S * FRAME_HZ), round(4.0 * FRAME_HZ), round(0.5 * FRAME_HZ)
    while i < n:
        if not (np.isfinite(ctx[i]) and ctx[i] >= loud_ref - STOP_LOUD_WITHIN_DB and level[i] < ctx[i] - STOP_DROP_DB):
            i += 1
            continue
        c0, j = ctx[i], i
        while j < n and level[j] < c0 - STOP_DROP_DB + 3.0:  # 3 dB hysteresis
            j += 1
        dip = power_mean_db(level[i:j])
        back = power_mean_db(level[j:j + after_len]) if j < n else -np.inf
        if min_len <= j - i <= max_len and back >= dip + STOP_DROP_DB:
            a = i  # report the cut where the level is halfway down (-6 dB), not where it crossed -12 dB
            while a > 0 and level[a - 1] < c0 - 6.0:
                a -= 1
            out.append({"start": a, "end": j, "depth_db": float(c0 - dip), "near_silent": dip < loud_ref - SILENCE_DB})
        i = j + 1
    return out


def stem_events(regions: list[list[float]], duration: float, gap_s: float) -> list[tuple[str, float]]:
    """("in", t) / ("out", t) where a stem starts / stops with >= gap_s of inactivity on the other side."""
    ev = []
    for k, (s, e) in enumerate(regions):
        if s - (regions[k - 1][1] if k else 0.0) >= gap_s:
            ev.append(("in", float(s)))
        if (regions[k + 1][0] if k + 1 < len(regions) else duration) - e >= gap_s:
            ev.append(("out", float(e)))
    return ev


def refine_edge(db: np.ndarray, t: float, entering: bool) -> float:
    """Sharpen an activity-region edge (smoothed, so ~0.25 s early/late) on the raw stem level."""
    n = len(db)
    if entering:
        target = power_mean_db(db[frames_of(t, t + 1.0, n)]) - 10.0
        sl = frames_of(t - 0.5, t + 1.0, n)
        above = np.flatnonzero(db[sl] >= target)
        return (sl.start + above[0]) / FRAME_HZ if len(above) else t
    target = power_mean_db(db[frames_of(t - 1.0, t, n)]) - 10.0
    sl = frames_of(t - 1.0, t + 0.5, n)
    above = np.flatnonzero(db[sl] >= target)
    return (sl.start + above[-1] + 1) / FRAME_HZ if len(above) else t


def snap_to(t: float, candidates: np.ndarray, tol: float = 0.08) -> float:
    if len(candidates) == 0:
        return t
    k = int(np.argmin(np.abs(candidates - t)))
    return float(candidates[k]) if abs(candidates[k] - t) <= tol else t


def mean_of(curve: np.ndarray, t0: float, t1: float) -> float:
    seg = curve[frames_of(t0, t1, len(curve))]
    return float(np.mean(seg)) if len(seg) else 0.0


def find_moments(sections: list[dict], stats: list[dict], feats: dict, edges: np.ndarray, curves: dict,
                 energy: dict, hits: dict, duration: float) -> list[dict]:
    moments: list[dict] = []

    def add(t, typ, desc, strength, **extra):
        moments.append({"t": float(t), "type": typ, "description": desc, "strength": float(np.clip(strength, 0, 1)), **extra})

    level = smoothed_db(curves["db_mix"], 0.05)
    lufs_1s = smoothed_db(curves["lufs"], 1.0)
    mix_ref = float(np.percentile(curves["db_mix"], 99))
    ctx = trailing_context(level)
    loud_ref = float(np.max(ctx[np.isfinite(ctx)], initial=mix_ref))  # loudest 2 s stretch
    onset_t = np.asarray(hits.get("onsets", []), dtype=float)
    kick_t = np.array(sorted(h["t"] for h in hits.get("hits", []) if h["type"] == "kick"))

    # silence and stops
    for a, b in find_silences(level, mix_ref):
        where = "until the music starts" if a == 0 else ("to the end" if b >= len(level) else "")
        add(a / FRAME_HZ, "silence", f"silence for {(b - a) / FRAME_HZ:.1f} s {where}".strip(), 1.0, end=b / FRAME_HZ)
    stops = [s for s in find_stops(level, loud_ref) if not (s["near_silent"] and (s["end"] - s["start"]) >= SILENCE_MIN_S * FRAME_HZ)]
    for s in stops:
        dur = (s["end"] - s["start"]) / FRAME_HZ
        add(s["start"] / FRAME_HZ, "stop", f"music cuts out ({s['depth_db']:.0f} dB down) for {dur:.2f} s", s["depth_db"] / 30, end=s["end"] / FRAME_HZ)
    gap_ends = [m["end"] for m in moments if m["type"] in ("stop", "silence")]

    # section-driven moments
    bar_len = float(np.median(np.diff(edges))) if len(edges) > 1 else FALLBACK_BAR_S
    for k, sec in enumerate(sections):
        st, prev = stats[k], stats[k - 1] if k else None
        start = sec["start"]
        if sec["character"] == "build":
            sl = frames_of(max(st["start"], st["end"] - bar_len), st["end"], len(level))
            stop_in = [s["start"] for s in stops if sl.start <= s["start"] < sl.stop]
            if stop_in:
                sl = slice(sl.start, max(stop_in[0], sl.start + 1))
            pk = sl.start + int(np.argmax(curves["lufs"][sl]))
            add(pk / FRAME_HZ, "build_peak", f"top of the build ({st['change']:+.0f} LU over {st['end'] - st['start']:.0f} s)",
                st["change"] / 15)
        if k and sec["character"] == "full":
            i0 = int(np.searchsorted(edges, start - 1e-6))
            jump = feats["loudness"][i0] - feats["loudness"][i0 - 1] if i0 > 0 else 0.0
            after_gap = any(start - bar_len <= g <= start + 0.1 for g in gap_ends)
            if sections[k - 1]["character"] == "build" or after_gap or jump >= DROP_JUMP_LU:
                t = snap_to(start, kick_t) if len(kick_t) else snap_to(start, onset_t)
                # strength: half how loud the section is, half the contrast with the quietest moment just before
                before = float(np.min(curves["lufs"][frames_of(t - 0.5, t, len(level))], initial=st["lufs"]))
                why = "after the build" if sections[k - 1]["character"] == "build" else ("after a stop" if after_gap else f"+{jump:.0f} LU jump")
                add(t, "drop", f"full section hits {why}", 0.5 * sec["energy"] + 0.5 * min(1.0, (st["lufs"] - before) / 12))
        if prev is not None and sec["character"] == "breakdown":
            add(start, "breakdown", f"breakdown: {prev['lufs'] - st['lufs']:.0f} LU quieter", (prev["lufs"] - st["lufs"]) / 18)

    # loudest moment
    pk = int(np.argmax(lufs_1s))
    add(pk / FRAME_HZ, "loudest", f"loudest point ({float(curves['lufs'][max(0, pk - 50):pk + 50].max()):.1f} LUFS momentary)", 1.0)

    # stem entries / exits
    events = []
    for stem, gap in (("drums", STEM_GAP_S), ("bass", STEM_GAP_S), ("vocals", VOCAL_GAP_S)):
        regions = energy["active"].get(stem, [])
        db = curves[f"db_{stem}"]
        for kind, t in stem_events(regions, duration, gap):
            t = refine_edge(db, t, kind == "in")
            e = mean_of(curves[f"energy_{stem}"], t, t + 1.0) if kind == "in" else mean_of(curves[f"energy_{stem}"], t - 1.0, t)
            if stem == "vocals":
                if kind == "in":
                    returning = any(r[1] <= t for r in regions)
                    events.append({"t": t, "type": "vocal_entry", "description": "vocals return" if returning else "vocals come in", "strength": e})
            else:
                text = {("drums", "in"): "drums come in", ("drums", "out"): "drums drop out",
                        ("bass", "in"): "bass comes in", ("bass", "out"): "bass drops out"}[(stem, kind)]
                events.append({"t": t, "type": f"{stem}_{kind}", "description": text, "strength": e})
    for k, t in enumerate(kick_t):
        if t - (kick_t[k - 1] if k else 0.0) >= KICK_GAP_S:
            s = next((h["strength"] for h in hits["hits"] if h["type"] == "kick" and h["t"] == t), 1.0)
            events.append({"t": float(t), "type": "kick_in", "description": "kick comes in", "strength": s})

    # fold entries/exits that coincide with a drop / breakdown (or kick_in with drums_in) into that moment
    def fold_into(target_types, ev_types, joiner):
        for m in [m for m in moments + events if m["type"] in target_types]:
            folded = [e for e in events if e["type"] in ev_types and abs(e["t"] - m["t"]) <= FOLD_S and e is not m]
            if folded:
                names = sorted({e["type"].split("_")[0] for e in folded}, key=["kick", "drums", "bass"].index)
                m["description"] += joiner(names)
                for e in folded:
                    events.remove(e)

    def and_list(names):
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]

    def verb(names, base):  # "bass comes in", "drums come in", "kick and bass come in"
        return base if len(names) > 1 or names[0] == "drums" else base.replace("come", "comes").replace("drop", "drops")
    fold_into(("drop",), ("drums_in", "bass_in", "kick_in"), lambda ns: f"; {and_list(ns)} {verb(ns, 'come in')}")
    fold_into(("breakdown",), ("drums_out", "bass_out"), lambda ns: f"; {and_list(ns)} {verb(ns, 'drop out')}")
    fold_into(("drums_in",), ("kick_in",), lambda ns: ", with kick")
    moments += events

    for m in moments:
        m["t"] = r3(m["t"])
        m["strength"] = r2(m["strength"])
        if "end" in m:
            m["end"] = r3(m["end"])
    moments.sort(key=lambda m: (m["t"], m["type"]))
    return moments


# ---- stage -------------------------------------------------------------------------------

def run(ctx: RunContext) -> None:
    rhythm = ctx.read_json("rhythm.json")
    energy = ctx.read_json("energy.json")
    hits = ctx.maybe_read_json("hits.json") or {"onsets": [], "hits": []}
    curves = load_curves(ctx)
    duration = float(energy["duration"])
    grid = BeatGrid.from_rhythm(rhythm)

    edges = bar_edges(grid.bar_starts(), duration)
    feats = bar_features(edges, curves, energy, hits.get("hits", []), grid.beats)
    segs = segment(feature_matrix(feats))
    stats = [section_stats(i, j, edges, feats, curves, energy) for i, j in segs]
    loudest_bar = float(np.max(feats["loudness"]))
    labels = label_sections(stats, loudest_bar)

    sections = []
    for k, ((i, j), st, lab) in enumerate(zip(segs, stats, labels)):
        sec = {"start": r3(st["start"]), "end": r3(st["end"])}
        if len(grid):
            sec["bars"] = [grid.bar_of_time(st["start"] + 0.06), grid.bar_of_time(st["end"] - 0.06)]
        sec.update({
            "character": lab,
            "energy": r2(np.clip((st["lufs"] - (loudest_bar - ENERGY_RANGE_LU)) / ENERGY_RANGE_LU, 0, 1)),
            "loudness_lufs": round(st["lufs"], 1),
            "trend": "rising" if st["change"] >= TREND_LU else ("falling" if st["change"] <= -TREND_LU else "flat"),
            "change_lu": round(st["change"], 1),
            "active_stems": st["active"],
            "hits_per_beat": {**{k: r2(v) for k, v in st["per_beat"].items()}, "all": r2(sum(st["per_beat"].values()))},
            "description": describe_section(st, lab, stats[k - 1] if k else None),
        })
        sections.append(sec)
    moments = find_moments(sections, stats, feats, edges, curves, energy, hits, duration)

    ctx.write_json("sections.json", {"method": METHOD, "sections": sections, "moments": moments})
    log.info("sections: %s; %d moments", " | ".join(f"{s['character']} {s['start']:.0f}s" for s in sections), len(moments))


STAGE = Stage(
    name="sections",
    run=run,
    outputs=lambda ctx: [ctx.cache_path("sections.json")],
    deps=("rhythm", "energy"),
    required=False,
)
