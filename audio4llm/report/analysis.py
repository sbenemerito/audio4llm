"""Build analysis.json and curves.json from the cached stage outputs.

This is the part of the project most likely to change, so it only reads cache
files (never audio or models) and is cheap to re-run: `audio4llm runs/<clip>`.
Every event is located on the beat grid here, in one place.
"""

from __future__ import annotations

import numpy as np

from .. import SCHEMA_VERSION, __version__
from ..config import STEM_NAMES
from ..context import RunContext
from ..grid import BeatGrid
from ..jsonfmt import r2, r3
from ..stages.energy import FRAME_HZ, load_curves, normalize_db, smooth, to_db

HIT_TYPES = ("kick", "snare", "hat")
STEPS_PER_BEAT = 4  # drum pattern resolution: 16th notes in 4/4
SMOOTH_S = 1.0  # window for curves.json mix_smooth

CONVENTIONS = {
    "time": "seconds from the start of input.wav (the audio to play/mux), 3 decimals",
    "bar": "1-based, counted from the first downbeat; 0 (or below) = before it",
    "beat": "1-based position within the bar, fractional: 1.0 = downbeat, 2.5 = halfway between beats 2 and 3",
    "energy": "0..1 within this clip, linear in dB over the top 30 dB of each stem (1 = that stem's loudest)",
    "estimates": "key, drum hit types, notes, section characters and moments are heuristic estimates",
    "dense_curves": "curves.json has fixed-rate envelopes for code; this file is the readable summary",
}


class Locator:
    """Puts bar/beat on events. One instance per assemble run."""

    def __init__(self, grid: BeatGrid):
        self.grid = grid

    def __call__(self, t: float) -> dict:
        bar, beat = self.grid.locate(t)
        return {"t": r3(t), "bar": bar, "beat": beat}

    def bar(self, t: float) -> int:
        return self.grid.locate(t)[0]



# ---- per-beat and dense curves ------------------------------------------------------

def _power_mean_db(db: np.ndarray) -> float:
    return float(to_db(np.mean(10 ** (db / 10)))) if len(db) else float("-inf")


def per_beat_energy(curves: dict, levels: dict, grid: BeatGrid) -> dict[str, list[float]]:
    edges = grid.beat_edges()
    idx = np.clip(np.round(edges * FRAME_HZ).astype(int), 0, len(curves["times"]))
    out = {}
    for name in ("mix", *STEM_NAMES):
        db, ref = curves[f"db_{name}"], levels[name]["ref_db"]
        vals = []
        for a, b in zip(idx[:-1], idx[1:]):
            seg = db[a:max(b, a + 1)]
            vals.append(r2(normalize_db(_power_mean_db(seg), ref)))
        out[name] = vals
    return out


def _resample(x: np.ndarray, hz: float, n_out: int, how: str) -> np.ndarray:
    """Aggregate a FRAME_HZ curve into n_out bins centred on i / hz."""
    centres = np.arange(n_out) / hz
    lo = np.clip(np.round((centres - 0.5 / hz) * FRAME_HZ).astype(int), 0, len(x) - 1)
    hi = np.clip(np.round((centres + 0.5 / hz) * FRAME_HZ).astype(int), lo + 1, len(x))
    out = np.empty(n_out)
    for i, (a, b) in enumerate(zip(lo, hi)):
        seg = x[a:b]
        if how == "max":
            out[i] = seg.max()
        elif how == "power_db":
            out[i] = 10 * np.log10(max(np.mean(10 ** (seg / 10)), 1e-12))
        else:
            out[i] = seg.mean()
    return out


def build_curves(ctx: RunContext, curves: dict, energy: dict, duration: float) -> dict:
    hz = float(ctx.config.curves_hz)
    n_out = int(np.floor(duration * hz)) + 1
    levels = energy["levels"]
    out: dict[str, list] = {}
    for name in ("mix", *STEM_NAMES):
        db = _resample(curves[f"db_{name}"], hz, n_out, "power_db")
        out[name] = np.round(normalize_db(db, levels[name]["ref_db"]), 3).tolist()
    slow = to_db(smooth(10 ** (curves["db_mix"] / 10), SMOOTH_S))
    out["mix_smooth"] = np.round(normalize_db(_resample(slow, hz, n_out, "power_db"), levels["mix"]["ref_db"]), 3).tolist()
    out["loudness_lufs"] = np.round(np.maximum(_resample(curves["lufs"], hz, n_out, "power_db"), -70), 1).tolist()
    out["brightness"] = np.round(_resample(curves["brightness"], hz, n_out, "mean"), 3).tolist()
    out["onset"] = np.round(_resample(curves["onset"], hz, n_out, "max"), 3).tolist()
    for band in ("low", "mid", "high"):
        key = f"drums_{band}"
        if key in curves:
            out[key] = np.round(_resample(curves[key], hz, n_out, "max"), 3).tolist()
    return {
        "schema_version": SCHEMA_VERSION,
        "hz": hz,
        "n": n_out,
        "duration": r3(duration),
        "about": "Sample i is at t = i / hz seconds from the start of input.wav. Each value summarises the "
                 "window t +/- 0.5/hz. Energies are 0..1 (see analysis.json conventions).",
        "curves_info": {
            "mix|vocals|drums|bass|other": "energy 0..1 (power-averaged per window, linear in dB over the top 30 dB)",
            "mix_smooth": f"mix energy smoothed over {SMOOTH_S:g} s: overall intensity without the per-beat pumping",
            "loudness_lufs": "momentary loudness of the mix (400 ms, BS.1770), LUFS, floor -70",
            "brightness": "spectral centroid of the mix, 0..1 on a log scale from 200 Hz to 8 kHz",
            "onset": "transient strength of the mix, 0..1 (window max)",
            "drums_low|drums_mid|drums_high": "drum-stem band envelopes 0..1 (window max): ~kick / ~snare / ~hats",
        },
        "curves": out,
    }


# ---- sections / moments ---------------------------------------------------------------

def build_sections(sections: dict | None, loc: Locator) -> list[dict] | None:
    if not sections:
        return None
    out = []
    for i, s in enumerate(sections.get("sections", [])):
        bars = s.get("bars") or (list(loc.grid.bar_range(s["start"], s["end"])) if len(loc.grid) else None)
        item = {"index": i, "start": r3(s["start"]), "end": r3(s["end"]), "bars": bars}
        item.update({k: v for k, v in s.items() if k not in ("start", "end", "bars")})
        out.append(item)
    return out


def build_moments(sections: dict | None, loc: Locator) -> list[dict] | None:
    if not sections:
        return None
    out = []
    for m in sorted(sections.get("moments", []), key=lambda m: m["t"]):
        item = loc(m["t"])
        item.update({k: v for k, v in m.items() if k != "t"})
        out.append(item)
    return out


# ---- hits -----------------------------------------------------------------------------

def drum_patterns(rows: dict[str, list[list]], grid: BeatGrid) -> list[dict]:
    """Per-bar 16th-note step strings ("x...x..."), consecutive identical bars merged."""
    bars = [b for b in grid.bars() if b["bar"] >= 1]
    if not bars:
        return []
    steps = grid.bpb * STEPS_PER_BEAT
    pats: dict[int, dict[str, list[str]]] = {b["bar"]: {h: ["."] * steps for h in HIT_TYPES} for b in bars}
    for h in HIT_TYPES:
        for _, bar, beat, *_ in rows.get(h, []):
            step = int(round((beat - 1) * STEPS_PER_BEAT))
            if step == steps:  # e.g. beat 4.9 rounds up to the next bar's first 16th
                bar, step = bar + 1, 0
            if bar in pats and 0 <= step < steps:
                pats[bar][h][step] = "x"
    out: list[dict] = []
    for bar in sorted(pats):
        p = {h: "".join(v) for h, v in pats[bar].items()}
        if all(set(s) == {"."} for s in p.values()):
            continue
        if out and out[-1]["bars"][1] == bar - 1 and all(out[-1][h] == p[h] for h in HIT_TYPES):
            out[-1]["bars"][1] = bar
        else:
            out.append({"bars": [bar, bar], **p})
    return out


def build_hits(hits: dict | None, grid: BeatGrid, loc: Locator) -> dict | None:
    if hits is None:
        return None
    rows: dict[str, list[list]] = {h: [] for h in HIT_TYPES}
    for h in hits.get("hits", []):
        if h["type"] in rows:
            p = loc(h["t"])
            rows[h["type"]].append([p["t"], p["bar"], p["beat"], r2(h.get("strength", 1.0))])
    onsets = [[p["t"], p["bar"], p["beat"]] for p in map(loc, hits.get("onsets", []))]
    return {
        "about": hits.get("method", "estimate from the drum stem") + ". Rows are [t, bar, beat, strength 0..1]. "
                 "Only counted where the drum stem is active.",
        "columns": ["t", "bar", "beat", "strength"],
        "counts": {**{h: len(v) for h, v in rows.items()}, "onsets": len(onsets)},
        **rows,
        "onsets_about": "every drum-stem onset as [t, bar, beat], typed or not (toms, claps, fills included): "
                        "use it for 'any drum hit' effects",
        "onsets": onsets,
        "patterns_about": f"One character per 16th note (x = hit), {grid.bpb * STEPS_PER_BEAT} per bar; "
                          "consecutive bars with the same pattern are merged into one entry.",
        "patterns": drum_patterns(rows, grid),
    }


# ---- stems -----------------------------------------------------------------------------

def build_stems(energy: dict, per_beat: dict, loc: Locator, duration: float) -> dict:
    out: dict = {
        "about": "per_beat[i] is the stem's energy (0..1) during grid.beats[i] (until the next beat). "
                 "active lists the regions where the stem is clearly audible.",
        "mix": {"per_beat": per_beat["mix"]},
    }
    for name in STEM_NAMES:
        regions = energy["active"].get(name, [])
        act = []
        for s, e in regions:
            a, b = loc(s), loc(e)
            act.append({"start": a["t"], "end": b["t"], "bar": a["bar"], "beat": a["beat"],
                        "end_bar": b["bar"], "end_beat": b["beat"]})
        out[name] = {
            "per_beat": per_beat[name],
            "active": act,
            "active_fraction": r2(sum(e - s for s, e in regions) / duration) if duration else 0.0,
        }
    return out


# ---- notes -----------------------------------------------------------------------------

def _collapse(names: list[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        if not out or out[-1] != n:
            out.append(n)
    return out


def build_notes(notes: dict | None, loc: Locator) -> dict | None:
    if not notes:
        return None
    out: dict = {
        "about": "Basic Pitch transcription of the bass and vocal stems (estimates). "
                 "Rows are [t, bar, beat, duration, midi, name, velocity 1..127].",
        "backend": notes.get("backend"),
        "columns": ["t", "bar", "beat", "dur", "midi", "name", "velocity"],
    }
    per_bar: dict[int, dict] = {}
    for stem, info in notes.get("stems", {}).items():
        rows = []
        for n in info.get("notes", []):
            p = loc(n["start"])
            dur = r3(n["end"] - n["start"]) if "end" in n else r3(n.get("duration", 0.0))
            rows.append([p["t"], p["bar"], p["beat"], dur, n["midi"], n["name"], n.get("velocity")])
            per_bar.setdefault(p["bar"], {}).setdefault(stem, []).append((n["name"], n["midi"], dur))
        out[stem] = rows
        if info.get("skipped") or info.get("reason"):
            out[f"{stem}_note"] = info.get("reason") or "skipped"
    summary = []
    for bar in sorted(per_bar):
        entry: dict = {"bar": bar}
        for stem, items in per_bar[bar].items():
            entry[stem] = " ".join(_collapse([n for n, _, _ in items]))
            if stem == "bass":
                weight: dict[str, float] = {}
                for name, midi, dur in items:
                    pc = name.rstrip("-0123456789")
                    weight[pc] = weight.get(pc, 0.0) + (dur or 0.0)
                entry["bass_root"] = max(weight, key=weight.get)
        summary.append(entry)
    out["per_bar_about"] = "Notes starting in each bar, in order (repeats collapsed); bass_root = bass pitch class held longest."
    out["per_bar"] = summary
    return out


# ---- lyrics ----------------------------------------------------------------------------

def build_lyrics(lyrics: dict | None, loc: Locator) -> dict | None:
    if not lyrics:
        return None
    out = {k: lyrics.get(k) for k in ("status", "reason", "language", "model")}  # always present, null if n/a
    lines = []
    for ln in lyrics.get("lines", []):
        p = loc(ln["start"])
        words = []
        for w in ln.get("words", []):
            q = loc(w["start"])
            words.append({"w": w["w"], "start": q["t"], "end": r3(w["end"]), "bar": q["bar"], "beat": q["beat"]})
        lines.append({"start": p["t"], "end": r3(ln["end"]), "bar": p["bar"], "beat": p["beat"],
                      "text": ln["text"], "words": words})
    out["lines"] = lines
    return out


# ---- top level -------------------------------------------------------------------------

def build(ctx: RunContext) -> tuple[dict, dict]:
    ingest = ctx.read_json("ingest.json")
    rhythm = ctx.read_json("rhythm.json")
    energy = ctx.read_json("energy.json")
    stems_meta = ctx.read_json("stems.json")
    hits = ctx.maybe_read_json("hits.json")
    sections = ctx.maybe_read_json("sections.json")
    lyrics = ctx.maybe_read_json("lyrics.json")
    notes = ctx.maybe_read_json("notes.json")
    curves = load_curves(ctx)

    duration = float(ingest["duration"])
    grid = BeatGrid.from_rhythm(rhythm)
    loc = Locator(grid)
    is_db = grid.is_downbeat()
    warnings = []
    for name, data in (("sections", sections), ("lyrics", lyrics), ("notes", notes)):
        if data is None:
            warnings.append(f"{name} stage output missing (stage failed or was skipped); '{name}' is null")

    key = (notes or {}).get("key")
    lang = (lyrics or {}).get("language") if (lyrics or {}).get("status") == "ok" else None
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "conventions": CONVENTIONS,
        "source": {
            "file": ingest["source"],
            "input_wav": str(ctx.input_wav.resolve()),
            "start_offset": ingest.get("start", 0.0),
            "duration": r3(duration),
            "sample_rate": ingest["sample_rate"],
            "channels": ingest["channels"],
        },
        "global": {
            "tempo_bpm": rhythm.get("tempo_bpm"),
            "tempo_std_bpm": rhythm.get("tempo_std_bpm"),
            "time_signature": rhythm.get("time_signature"),
            "beats_per_bar": rhythm.get("beats_per_bar"),
            "beat_count": len(grid),
            "bar_count": sum(1 for b in grid.bars() if b["bar"] >= 1),
            "key": ({k: key.get(k) for k in ("label", "confidence", "runner_up")} | {"estimate": True}) if key else None,
            "loudness": energy["loudness"],
            "has_vocals": bool(energy.get("has_vocals")),
            "vocals_disabled": bool(energy.get("vocals_disabled")),
            "language": lang,
        },
        "grid": {
            "beats": [{"t": r3(t), "bar": int(b), "beat": int(p), "is_downbeat": bool(d)}
                      for t, b, p, d in zip(grid.beats, grid.bar_of_beat, grid.pos_of_beat, is_db)],
            "bars": [{"bar": b["bar"], "start": r3(b["start"]), "end": r3(b["end"])} for b in grid.bars()],
        },
        "sections": build_sections(sections, loc),
        "moments": build_moments(sections, loc),
        "hits": build_hits(hits, grid, loc),
        "stems": build_stems(energy, per_beat_energy(curves, energy["levels"], grid), loc, duration),
        "notes": build_notes(notes, loc),
        "lyrics": build_lyrics(lyrics, loc),
        "curves": {
            "file": "curves.json",
            "hz": float(ctx.config.curves_hz),
            "keys": ["mix", *STEM_NAMES, "mix_smooth", "loudness_lufs", "brightness", "onset", "drums_low", "drums_mid", "drums_high"],
            "about": "dense fixed-rate envelopes for code to sample at render time (not meant to be read)",
        },
        "meta": {
            "generator": f"audio4llm {__version__}",
            "models": {
                "stems": stems_meta.get("model"),
                "rhythm": rhythm.get("model"),
                "lyrics": (lyrics or {}).get("model"),
                "notes": f"basic-pitch ({(notes or {}).get('backend')})" if notes else None,
            },
            "devices": {"stems": stems_meta.get("device"), "rhythm": rhythm.get("device")},
            "warnings": warnings,
        },
    }
    return analysis, build_curves(ctx, curves, energy, duration)
