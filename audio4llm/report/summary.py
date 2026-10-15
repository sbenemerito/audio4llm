"""summary.md: a short description of the clip, generated from analysis.json with templates (no LLM)."""

from __future__ import annotations

import numpy as np

from ..config import STEM_NAMES

BLOCKS = "▁▂▃▄▅▆▇█"
STEM_LABEL = {"vocals": "Vocals", "drums": "Drums", "bass": "Bass", "other": "Other (pads, keys, guitars, fx)"}


def ts(t: float) -> str:
    m, s = divmod(max(float(t), 0.0), 60)
    return f"{int(m)}:{s:04.1f}"


def meter(x: float | None) -> str:
    if x is None:
        return ""
    return BLOCKS[min(int(x * len(BLOCKS)), len(BLOCKS) - 1)] + f" {x:.2f}"


def where(item: dict) -> str:
    return f"{ts(item['t'])} (bar {item['bar']}, beat {item['beat']:g})"


def _bar_loudness(analysis: dict, curves: dict) -> list[tuple[int, float, float]]:
    """(bar, start, LUFS) per full bar, power-averaged from the dense loudness curve."""
    hz = curves["hz"]
    lufs = np.asarray(curves["curves"]["loudness_lufs"])
    out = []
    for b in analysis["grid"]["bars"]:
        if b["bar"] < 1:
            continue
        seg = lufs[int(b["start"] * hz):max(int(b["end"] * hz), int(b["start"] * hz) + 1)]
        if len(seg):
            out.append((b["bar"], b["start"], float(10 * np.log10(np.mean(10 ** (seg / 10))))))
    return out


def render(analysis: dict, curves: dict, name: str) -> str:
    g, src = analysis["global"], analysis["source"]
    L: list[str] = [f"# {name}", ""]

    key = g.get("key")
    head = [f"**{src['duration']:.1f} s**", f"**{g['tempo_bpm']:.1f} BPM**" if g.get("tempo_bpm") else "tempo unknown",
            g.get("time_signature") or "?"]
    if key:
        head.append(f"key ≈ {key['label']} (estimate)")
    loud = g.get("loudness", {})
    if loud.get("integrated_lufs") is not None:
        head.append(f"{loud['integrated_lufs']:.1f} LUFS")
    head.append(f"vocals: {'yes' + (' (' + g['language'] + ')' if g.get('language') else '') if g.get('has_vocals') else 'no'}")
    L += [" · ".join(head), ""]
    L += [f"{g['bar_count']} bars, {g['beat_count']} beats. All times are seconds into `input.wav`. "
          "Bars count from the first downbeat; `bar.beat` positions are 1-based.", ""]

    # timeline
    secs = analysis.get("sections")
    if secs:
        L += ["## Timeline", "", "| Time | Bars | Character | Energy | Active stems | Notes |", "|---|---|---|---|---|---|"]
        for s in secs:
            stems = ", ".join(s.get("active_stems", [])) or "-"
            L.append(f"| {ts(s['start'])}–{ts(s['end'])} | {s['bars'][0]}–{s['bars'][1]} | **{s.get('character', '?')}** "
                     f"| {meter(s.get('energy'))} | {stems} | {s.get('description', '')} |")
        L.append("")

    # moments
    moms = analysis.get("moments")
    if moms:
        L += ["## Moments", ""]
        for m in moms:
            desc = f": {m['description']}" if m.get("description") else ""
            L.append(f"- {where(m)} **{m['type']}**{desc}")
        L.append("")

    # stems
    L += ["## When things come in", ""]
    stems = analysis["stems"]
    for s in STEM_NAMES:
        act = stems[s]["active"]
        if not act:
            L.append(f"- **{STEM_LABEL[s]}**: not detected")
            continue
        spans = ", ".join(f"{ts(a['start'])}–{ts(a['end'])}" for a in act[:6]) + (" …" if len(act) > 6 else "")
        L.append(f"- **{STEM_LABEL[s]}**: first at {ts(act[0]['start'])} (bar {act[0]['bar']}, beat {act[0]['beat']:g}); "
                 f"active {stems[s]['active_fraction'] * 100:.0f}% of the clip: {spans}")
    L.append("")

    # loud / quiet
    bars = _bar_loudness(analysis, curves)
    L += ["## Loud and quiet", ""]
    if bars:
        loudest = max(bars, key=lambda b: b[2])
        quietest = min(bars, key=lambda b: b[2])
        L.append(f"- Loudest bar: bar {loudest[0]} at {ts(loudest[1])} ({loudest[2]:.1f} LUFS momentary average)")
        L.append(f"- Quietest bar: bar {quietest[0]} at {ts(quietest[1])} ({quietest[2]:.1f} LUFS)")
    if loud:
        L.append(f"- Whole clip: {loud.get('integrated_lufs')} LUFS integrated, loudness range {loud.get('loudness_range_lu')} LU, "
                 f"peak {loud.get('peak_dbfs')} dBFS")
    L.append("")

    # drums
    hits = analysis.get("hits")
    if hits and hits.get("patterns"):
        c = hits["counts"]
        L += ["## Drum patterns (estimated)", "",
              f"{c.get('kick', 0)} kicks, {c.get('snare', 0)} snares, {c.get('hat', 0)} hats detected. "
              "One character per 16th note, `x` = hit:", "", "```"]
        for p in hits["patterns"][:8]:
            label = f"bars {p['bars'][0]}–{p['bars'][1]}" if p["bars"][0] != p["bars"][1] else f"bar {p['bars'][0]}"
            for i, h in enumerate(("kick", "snare", "hat")):
                L.append(f"{label if i == 0 else '':<12} {h:<5} {p[h]}")
        if len(hits["patterns"]) > 8:
            L.append(f"... {len(hits['patterns']) - 8} more pattern changes in analysis.json")
        L += ["```", ""]

    # harmony
    notes = analysis.get("notes")
    if key or notes:
        L += ["## Key and bass", ""]
        if key:
            ru = f"; runner-up {key['runner_up']}" if key.get("runner_up") else ""
            L.append(f"- Key estimate: **{key['label']}** (confidence {key.get('confidence')}{ru})")
        if notes and notes.get("per_bar"):
            roots = [(b["bar"], b["bass_root"]) for b in notes["per_bar"] if b.get("bass_root")]
            if roots:
                seq = " ".join(f"{r}" for _, r in roots[:16])
                L.append(f"- Bass roots by bar from bar {roots[0][0]}: {seq}" + (" …" if len(roots) > 16 else ""))
            nv = len(notes.get("vocals", []) or [])
            nb = len(notes.get("bass", []) or [])
            L.append(f"- {nb} bass notes and {nv} vocal notes transcribed")
        L.append("")

    # lyrics
    lyr = analysis.get("lyrics")
    L += ["## Lyrics", ""]
    if lyr and lyr.get("lines"):
        for ln in lyr["lines"][:6]:
            L.append(f"- {ts(ln['start'])} (bar {ln['bar']}): “{ln['text'].strip()}”")
        if len(lyr["lines"]) > 6:
            L.append(f"- … {len(lyr['lines']) - 6} more lines in analysis.json")
    elif lyr:
        L.append(f"- none ({lyr.get('reason') or lyr.get('status')})")
    else:
        L.append("- not available (lyrics stage did not run)")
    L.append("")

    L += ["## Files", "",
          "- `analysis.json`: the full structured analysis (grid, sections, moments, hits, stems, notes, lyrics)",
          f"- `curves.json`: {curves['hz']:g} Hz envelopes for code",
          "- `overview.png`: plot for a visual sanity check", ""]
    return "\n".join(L)
