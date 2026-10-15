"""overview.png: one figure to sanity-check the whole analysis at a glance.

Panels share the time axis: waveform with sections and moments, mix energy,
per-stem energy lanes with active regions, drum hits, transcribed notes, and
lyric lines. Downbeats are drawn as hairlines through every panel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import audio
from ..config import STEM_NAMES

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SHADE = "#f0efec"
WAVE = "#a3a29b"
# reference categorical order (blue, orange, aqua, yellow); lanes + direct labels carry identity too
STEM_COLOR = {"vocals": "#2a78d6", "drums": "#eb6834", "bass": "#1baf7a", "other": "#eda100"}


def _mmss(t: float, _pos=None) -> str:
    m, s = divmod(int(round(t)), 60)
    return f"{m}:{s:02d}"


def _waveform_envelope(path: Path, bins: int = 3000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, sr = audio.load(path)
    mono = y.mean(axis=0)
    n = len(mono) // bins * bins
    if n == 0:
        return np.zeros(1), np.zeros(1), np.zeros(1)
    blocks = mono[:n].reshape(bins, -1)
    t = (np.arange(bins) + 0.5) * blocks.shape[1] / sr
    return t, blocks.min(axis=1), blocks.max(axis=1)


def render(analysis: dict, curves: dict, input_wav: Path, out_path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.6,
    })
    dur = analysis["source"]["duration"]
    hz = curves["hz"]
    ct = np.arange(curves["n"]) / hz
    C = curves["curves"]
    downbeats = [b["start"] for b in analysis["grid"]["bars"] if b["bar"] >= 1]
    beats = [b["t"] for b in analysis["grid"]["beats"]]

    ratios = [2.2, 1.0, 3.2, 1.5, 1.8, 0.9]
    fig, axes = plt.subplots(len(ratios), 1, figsize=(18, 12.5), sharex=True,
                             gridspec_kw={"height_ratios": ratios, "hspace": 0.12})
    fig.patch.set_facecolor(SURFACE)
    ax_wave, ax_mix, ax_stems, ax_hits, ax_notes, ax_lyr = axes

    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for t in downbeats:
            ax.axvline(t, color=GRID, lw=0.6, zorder=0)
    # sections: alternating shade + boundary lines through every panel
    secs = analysis.get("sections") or []
    for i, s in enumerate(secs):
        for ax in axes:
            if i % 2 == 1:
                ax.axvspan(s["start"], s["end"], color=SHADE, lw=0, zorder=-1)
            ax.axvline(s["start"], color=BASELINE, lw=1.0, zorder=1)

    # 1. waveform + sections + moments + bar numbers
    t, lo, hi = _waveform_envelope(input_wav)
    ax_wave.fill_between(t, lo, hi, color=WAVE, lw=0, zorder=2)
    ax_wave.set_ylim(-1.05, 1.6)
    ax_wave.set_yticks([])
    for s in secs:
        ax_wave.text((s["start"] + s["end"]) / 2, 1.42, s.get("character", ""), ha="center", va="center",
                     fontsize=10, color=INK, fontweight="bold")
        ax_wave.text((s["start"] + s["end"]) / 2, 1.2, f"bars {s['bars'][0]}–{s['bars'][1]}", ha="center",
                     va="center", fontsize=8, color=INK_2)
    step = 4 if len(downbeats) > 24 else 1
    for b in analysis["grid"]["bars"]:
        if b["bar"] >= 1 and (b["bar"] - 1) % step == 0:
            ax_wave.text(b["start"] + 0.08, -1.0, str(b["bar"]), fontsize=7, color=MUTED, va="bottom")
    ax_wave.vlines(beats, 0.95, 1.02, color=MUTED, lw=0.6)
    for k, m in enumerate(analysis.get("moments") or []):
        y = 0.82 - 0.17 * (k % 4)
        ax_wave.plot([m["t"]], [y], marker="v", ms=6, color=INK, zorder=4)
        ax_wave.text(m["t"] + 0.25, y, m["type"], fontsize=8, color=INK, va="center", zorder=4,
                     bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.5, "alpha": 0.85})
    ax_wave.set_title(title, loc="left", fontsize=12, color=INK, pad=6)
    g = analysis["global"]
    key = (g.get("key") or {}).get("label")
    sub = f"{g.get('tempo_bpm') or 0:.1f} BPM · {g.get('time_signature')} · " + (f"key ≈ {key} · " if key else "") + \
          f"{g['loudness'].get('integrated_lufs')} LUFS · thin vertical lines = downbeats, ticks at top = beats"
    ax_wave.text(1.0, 1.02, sub, transform=ax_wave.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK_2)
    ax_wave.set_ylabel("waveform", rotation=0, ha="right", va="center")

    # 2. mix energy
    ax_mix.fill_between(ct, 0, C["mix"], color=INK_2, alpha=0.16, lw=0)
    if "mix_smooth" in C:
        ax_mix.plot(ct, C["mix_smooth"], color=INK, lw=1.4)
    ax_mix.set_ylim(0, 1.05)
    ax_mix.set_yticks([0, 1])
    ax_mix.set_ylabel("mix energy", rotation=0, ha="right", va="center")
    ax_mix.text(0.0, 1.01, "mix energy 0–1 (shaded = 20 Hz curve, line = 1 s smoothed)",
                transform=ax_mix.transAxes, fontsize=8, color=INK_2, va="bottom")

    # 3. stem energy lanes + active regions
    lane_h = 0.78
    stems = analysis["stems"]
    for k, name in enumerate(STEM_NAMES):
        base = len(STEM_NAMES) - 1 - k
        col = STEM_COLOR[name]
        e = np.asarray(C[name])
        ax_stems.fill_between(ct, base + 0.12, base + 0.12 + lane_h * e, color=col, alpha=0.28, lw=0)
        ax_stems.plot(ct, base + 0.12 + lane_h * e, color=col, lw=1.0)
        for a in stems[name]["active"]:
            ax_stems.plot([a["start"], a["end"]], [base + 0.05, base + 0.05], color=col, lw=3, solid_capstyle="round")
        ax_stems.axhline(base + 0.12, color=BASELINE, lw=0.5)
        ax_stems.text(-0.004 * dur, base + 0.5, name, ha="right", va="center", fontsize=9, color=INK)
    ax_stems.set_ylim(0, len(STEM_NAMES))
    ax_stems.set_yticks([])
    ax_stems.text(0.0, 1.01, "stem energy (0–1 per stem) · thick bar under each lane = active region",
                  transform=ax_stems.transAxes, fontsize=8, color=INK_2, va="bottom")

    # 4. drum hits, one lane per type, tick height = strength
    hits = analysis.get("hits") or {}
    for k, h in enumerate(("kick", "snare", "hat")):
        base = 2 - k
        rows = hits.get(h) or []
        if rows:
            ts = np.array([r[0] for r in rows])
            st = np.array([r[3] if r[3] is not None else 1.0 for r in rows])
            ax_hits.vlines(ts, base + 0.1, base + 0.1 + 0.8 * np.clip(st, 0.1, 1.0), color=STEM_COLOR["drums"], lw=1.1)
        ax_hits.axhline(base + 0.1, color=BASELINE, lw=0.5)
        ax_hits.text(-0.004 * dur, base + 0.5, f"{h} ({len(rows)})", ha="right", va="center", fontsize=9, color=INK)
    ax_hits.set_ylim(0, 3)
    ax_hits.set_yticks([])
    ax_hits.text(0.0, 1.01, "drum hits (estimated from the drum stem; height = strength)",
                 transform=ax_hits.transAxes, fontsize=8, color=INK_2, va="bottom")

    # 5. notes piano roll
    notes = analysis.get("notes") or {}
    pitches = []
    for stem in ("bass", "vocals"):
        rows = notes.get(stem) or []
        for r in rows:
            ax_notes.plot([r[0], r[0] + r[3]], [r[4], r[4]], color=STEM_COLOR[stem], lw=2.4, solid_capstyle="butt")
            pitches.append(r[4])
        if rows:
            ax_notes.text(-0.004 * dur, float(np.median([r[4] for r in rows])), f"{stem} ({len(rows)})",
                          ha="right", va="center", fontsize=9, color=INK)
    if pitches:
        lo_p, hi_p = min(pitches) - 2, max(pitches) + 2
        ax_notes.set_ylim(lo_p, hi_p)
        cs = [p for p in range(int(lo_p), int(hi_p) + 1) if p % 12 == 0]
        ax_notes.set_yticks(cs)
        ax_notes.set_yticklabels([f"C{p // 12 - 1}" for p in cs])
        ax_notes.yaxis.tick_right()
    else:
        ax_notes.set_yticks([])
        ax_notes.text(0.5, 0.5, "no notes", transform=ax_notes.transAxes, ha="center", color=MUTED)
    ax_notes.text(0.0, 1.01, "notes (Basic Pitch on bass and vocal stems; y = pitch)",
                  transform=ax_notes.transAxes, fontsize=8, color=INK_2, va="bottom")

    # 6. lyric lines
    lyr = analysis.get("lyrics") or {}
    lines = lyr.get("lines") or []
    for k, ln in enumerate(lines):
        y = 0.68 if k % 2 == 0 else 0.25
        ax_lyr.plot([ln["start"], ln["end"]], [y - 0.16, y - 0.16], color=STEM_COLOR["vocals"], lw=3, solid_capstyle="round")
        for w in ln.get("words", []):
            ax_lyr.plot([w["start"]], [y - 0.16], marker="|", ms=7, color=INK_2)
        ax_lyr.text(ln["start"], y, ln["text"].strip(), fontsize=8.5, color=INK, va="bottom", clip_on=True)
    if not lines:
        reason = lyr.get("reason") or lyr.get("status") or "lyrics stage did not run"
        ax_lyr.text(0.5, 0.5, f"no lyrics ({reason})", transform=ax_lyr.transAxes, ha="center", va="center", color=MUTED)
    ax_lyr.set_ylim(0, 1.1)
    ax_lyr.set_yticks([])
    ax_lyr.set_ylabel("lyrics", rotation=0, ha="right", va="center")

    ax_lyr.set_xlim(0, dur)
    ax_lyr.xaxis.set_major_locator(MultipleLocator(5 if dur <= 120 else 15))
    ax_lyr.xaxis.set_major_formatter(FuncFormatter(_mmss))
    ax_lyr.set_xlabel("time (m:ss into input.wav)")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.05)
    out_path = Path(out_path)
    tmp = out_path.with_name(out_path.stem + ".tmp.png")
    fig.savefig(tmp, dpi=100, facecolor=SURFACE)
    plt.close(fig)
    tmp.replace(out_path)
    return out_path
