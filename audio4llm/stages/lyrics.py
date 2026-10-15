"""Stage 5 - lyrics with mlx-whisper on the vocal stem.

Only the vocal-active regions from energy.json are transcribed. Each region is
padded, regions that end up close together are merged, and every merged region
is sliced out of the 16 kHz vocal stem and transcribed on its own (word
timestamps on, condition_on_previous_text off); times are shifted back to clip
time. The language is detected once on all the vocal audio together (more
reliable than per short region) unless ``config.language`` is set, and is then
forced for every region. Word boundaries that run into silence are pulled in to
the voiced audio of the vocal stem: whisper's alignment lets the first word of
a phrase swallow the pause before it (half a second early on the test clip).
If a large-v3-turbo checkpoint does not ship its own alignment heads, OpenAI's
tuned ones are installed (the mlx-community conversion already includes them).

Whisper hallucinates on music, so the output is filtered (thresholds below):
whole segments are dropped for known filler phrases ("Thank you.", "Thanks for
watching", subtitle credits, "[Music]"), a too-high compression ratio, likely
no speech, or low confidence; single words are dropped when they fall outside
the active vocal regions or belong to a decoder repetition loop. Everything
removed is listed under "dropped" with its reason. Kept segments are split at
pauses and punctuation into lyric-sized lines.

Writes cache/lyrics.json (every key is always present):
{
  "status": "ok"|"skipped"|"empty", "reason": "<why, when not ok>"|null,
  "model": "<hf repo>", "language": "en"|null,
  "language_source": "detected"|"user"|"model"|null,  # model: English-only model, always "en"
  "language_probability": 0.98|null,                    # detection confidence, null unless detected
  "regions": [[start, end], ...],   # padded + merged vocal regions that were transcribed, clip time
  "lines": [{"start": s, "end": e, "text": "...", "avg_logprob": x, "no_speech_prob": x,
             "words": [{"w": "night,", "start": s, "end": e, "p": 0.97}]}],
  "dropped": [{"start": s, "end": e, "text": "...", "reason": "..."}]
}
Times are clip seconds (3 decimals). "w" is the word as transcribed, with any
attached punctuation; "p" is whisper's word probability. avg_logprob and
no_speech_prob come from the whisper decode window the line was in.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from .. import audio
from ..context import RunContext
from ..jsonfmt import r2, r3
from ..pipeline import Stage

log = logging.getLogger("audio4llm")

SR = 16000  # whisper's input rate
NO_VOCALS_REASON = "vocals disabled with --no-vocals"

# regions
PAD_S = 0.4  # activity edges are smoothed; onsets/consonants sit just outside them
MERGE_GAP_S = 2.0  # padded regions closer than this share one whisper call (context, fewer calls)
LANG_MAX_S = 30.0  # language detection looks at up to one whisper window of vocal audio
LANG_JOIN_S = 0.2  # silence between regions when concatenating them for detection

# segment filters (whisper's decode-window stats)
COMPRESSION_MAX = 2.4  # gzip ratio above this = repetitive text (whisper's own fallback threshold)
LOGPROB_MIN = -1.0  # mean token log-prob below this = the decode failed even after temperature fallback
NO_SPEECH_MAX = 0.6  # no-speech probability above this...
NO_SPEECH_LOGPROB = -0.5  # ...with a log-prob below this = likely no speech (stricter than whisper's -1.0)

# word timing
VOICE_HZ = 100  # frame rate of the vocal-stem level used to tighten word boundaries
VOICE_REL_DB = 35.0  # voiced = within this many dB of the region's loud (95th pct) level...
VOICE_ABS_DB = -70.0  # ...and above this absolute level (stem bleed / noise floor)
TRIM_MIN_S = 0.15  # only move a word edge across silence at least this long (stop consonants are shorter)

# word filters
WORD_SLACK_S = 0.25  # a word must overlap an active vocal region widened by this much
LOOP_MIN_REPEATS = 4  # an n-gram (1-4 words) repeated this many times in a row is a loop candidate...
LOOP_MAX_WORD_S = 0.12  # ...and is a decoder loop if its words are crammed (median duration below this)
LOOP_MIN_P = 0.4  # ...or improbable (mean word probability below this)

# line splitting
LINE_GAP_S = 0.5  # a pause this long between words always starts a new line
LINE_MAX_WORDS = 8  # lines longer than this...
LINE_MAX_S = 5.0  # ...or than this many seconds are split at the best pause/punctuation
LINE_MIN_WORDS = 2  # a split never leaves a line shorter than this
SENTENCE_END = (".", "!", "?", "…", "。", "！", "？")
CLAUSE_END = (",", ";", ":", "—", "–", "-", "，", "、", "；", "：")

# Phrases whisper produces from music/silence (YouTube-caption training data). A
# segment whose whole text is one of these, possibly repeated, is dropped.
HALLUCINATION_PHRASES = (
    "thank you", "thank you so much", "thank you very much", "thanks", "thanks for watching",
    "thank you for watching", "thanks for watching and see you next time", "thank you for listening",
    "please subscribe", "please like and subscribe", "like and subscribe", "subscribe to my channel",
    "dont forget to like and subscribe", "see you next time", "see you in the next video",
    "bye", "bye bye", "you", "so", "music", "applause", "silence",
)
# Substrings that never occur in real lyrics (subtitle credits, mostly non-English).
HALLUCINATION_MARKERS = (
    "amara.org", "castingwords", "subtitles by", "captions by", "transcribed by", "transcription by",
    "untertitel", "sous-titres", "sous-titrage", "subtítulos", "sottotitoli", "legendas pela",
    "продолжение следует", "субтитры", "редактор субтитров", "ご視聴ありがとうございました", "字幕",
)
_PHRASE_RE = re.compile(r"^(?:(?:" + "|".join(re.escape(p) for p in HALLUCINATION_PHRASES) + r")\s*)+$")
_TAG_RE = re.compile(r"^[\[(*♪♫#].*[\])*♪♫#]$")  # [Music], (upbeat music), *laughs*, ♪ ... ♪

# OpenAI's tuned alignment heads for large-v3-turbo (openai/whisper, _ALIGNMENT_HEADS["large-v3-turbo"]):
# base85(gzip(bool mask of shape (4 layers, 20 heads))). mlx_whisper's own code never sets them and
# falls back to all 40 heads of the last two decoder layers, but the mlx-community checkpoint stores
# these same 6 heads in weights.safetensors, so this only matters for other turbo conversions.
TURBO_ALIGNMENT_HEADS = b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`"


# ---- regions -----------------------------------------------------------------------

def plan_regions(active: list[list[float]], duration: float, pad: float = PAD_S,
                 merge_gap: float = MERGE_GAP_S) -> list[list[float]]:
    """Pad each active region, clip to [0, duration], merge ones closer than merge_gap."""
    out: list[list[float]] = []
    for s, e in sorted(active):
        s, e = max(0.0, s - pad), min(duration, e + pad)
        if e <= s:
            continue
        if out and s - out[-1][1] < merge_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [[r3(s), r3(e)] for s, e in out]


def voiced_mask(y: np.ndarray, regions: list[list[float]]) -> np.ndarray:
    """VOICE_HZ frames of the 16 kHz vocal stem that carry voice, judged per region (False elsewhere)."""
    hop = SR // VOICE_HZ
    n = len(y) // hop
    db = 10 * np.log10((y[:n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
    mask = np.zeros(n, dtype=bool)
    for s, e in regions:
        a, b = int(s * VOICE_HZ), min(n, int(np.ceil(e * VOICE_HZ)))
        if b > a:
            thr = max(float(np.percentile(db[a:b], 95)) - VOICE_REL_DB, VOICE_ABS_DB)
            mask[a:b] = db[a:b] > thr
    return mask


def trim_to_voice(words: list[dict], voiced: np.ndarray, min_gap: float = TRIM_MIN_S) -> None:
    """Move word edges that sit in >= min_gap of silence onto the first/last voiced frame, in place.
    Words with no voiced frame at all are left alone (the outside-activity filter judges those)."""
    gap = int(round(min_gap * VOICE_HZ))
    for w in words:
        a, b = int(round(w["start"] * VOICE_HZ)), int(round(w["end"] * VOICE_HZ))
        idx = np.flatnonzero(voiced[a:b])
        if not len(idx):
            continue
        if idx[0] >= gap:
            w["start"] = (a + idx[0]) / VOICE_HZ
        if (b - a - 1) - idx[-1] >= gap:
            w["end"] = (a + idx[-1] + 1) / VOICE_HZ


def overlaps_active(start: float, end: float, active: list[list[float]], slack: float = WORD_SLACK_S) -> bool:
    return any(start <= e + slack and end >= s - slack for s, e in active)


# ---- filters -----------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase, apostrophes removed, other punctuation -> space, whitespace collapsed."""
    text = text.lower().replace("'", "").replace("’", "")
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def hallucination_reason(text: str) -> str | None:
    """Why a whole segment's text looks like a known whisper-on-music hallucination."""
    t = text.strip()
    low = t.lower()
    if any(m in low for m in HALLUCINATION_MARKERS):
        return "subtitle credit (known hallucination)"
    if _TAG_RE.match(t) or not normalize(t):
        return "sound tag / no words"
    if _PHRASE_RE.match(normalize(t)):
        return "filler phrase (known hallucination)"
    return None


def segment_drop_reason(seg: dict) -> str | None:
    """Why a whisper segment should be dropped as a whole, or None to keep it."""
    reason = hallucination_reason(seg["text"])
    if reason:
        return reason
    cr, lp, nsp = seg.get("compression_ratio", 0.0), seg.get("avg_logprob", 0.0), seg.get("no_speech_prob", 0.0)
    if cr > COMPRESSION_MAX:
        return f"repetitive (compression ratio {cr:.2f} > {COMPRESSION_MAX})"
    if nsp > NO_SPEECH_MAX and lp < NO_SPEECH_LOGPROB:
        return f"likely no speech (no_speech_prob {nsp:.2f}, avg_logprob {lp:.2f})"
    if lp < LOGPROB_MIN:
        return f"low confidence (avg_logprob {lp:.2f} < {LOGPROB_MIN})"
    return None


def find_repeats(tokens: list[str], min_repeats: int = LOOP_MIN_REPEATS, max_n: int = 4) -> list[tuple[int, int, int]]:
    """Runs where an n-gram (n <= max_n) repeats >= min_repeats times back to back: (start, end, n)."""
    out, i = [], 0
    while i < len(tokens):
        best = None
        for n in range(1, max_n + 1):
            gram = tokens[i:i + n]
            if len(gram) < n:
                break
            k = 1
            while tokens[i + k * n:i + (k + 1) * n] == gram:
                k += 1
            if k >= min_repeats and (best is None or k * n > best[1] - best[0]):
                best = (i, i + k * n, n)
        if best:
            out.append(best)
            i = best[1]
        else:
            i += 1
    return out


def drop_loops(words: list[dict]) -> tuple[list[dict], list[dict]]:
    """Remove decoder repetition loops: repeats after the first occurrence, when their timing or
    probabilities show they are not backed by audio (real lyrics repeat too: "no no no no")."""
    tokens = [normalize(w["raw"]) for w in words]
    drop = set()
    dropped = []
    for s, e, n in find_repeats(tokens):
        run = words[s:e]
        med_dur = float(np.median([w["end"] - w["start"] for w in run]))
        mean_p = float(np.mean([w["p"] for w in run]))
        if med_dur >= LOOP_MAX_WORD_S and mean_p >= LOOP_MIN_P:
            continue
        drop.update(range(s + n, e))
        gram = " ".join(tokens[s:s + n])
        dropped.append(_dropped(words[s + n:e], f"repetition loop ('{gram}' x{(e - s) // n}, "
                                                f"median word {med_dur:.2f}s, mean p {mean_p:.2f})"))
    return [w for i, w in enumerate(words) if i not in drop], dropped


# ---- lines -------------------------------------------------------------------------

def _break_score(words: list[dict], i: int) -> float:
    """How good a line break is between words[i-1] and words[i] (higher is better)."""
    gap = max(0.0, words[i]["start"] - words[i - 1]["end"])
    tail = words[i - 1]["raw"].rstrip()
    punct = 1.0 if tail.endswith(SENTENCE_END) else 0.6 if tail.endswith(CLAUSE_END) else 0.0
    balance = abs(i - len(words) / 2) / len(words)  # prefer halves of similar length
    return punct + 2.0 * min(gap, 1.0) - 0.8 * balance


def _split_long(words: list[dict]) -> list[list[dict]]:
    n = len(words)
    too_long = n > LINE_MAX_WORDS or words[-1]["end"] - words[0]["start"] > LINE_MAX_S
    if not too_long or n < 2 * LINE_MIN_WORDS:
        return [words]
    i = max(range(LINE_MIN_WORDS, n - LINE_MIN_WORDS + 1), key=lambda k: _break_score(words, k))
    return _split_long(words[:i]) + _split_long(words[i:])


def split_lines(words: list[dict]) -> list[list[dict]]:
    """Split one segment's words into lyric lines: always at pauses >= LINE_GAP_S, then long
    pieces at the best pause/punctuation."""
    if not words:
        return []
    pieces, cur = [], [words[0]]
    for prev, w in zip(words, words[1:]):
        if w["start"] - prev["end"] >= LINE_GAP_S:
            pieces.append(cur)
            cur = []
        cur.append(w)
    pieces.append(cur)
    return [line for piece in pieces for line in _split_long(piece)]


def _text(words: list[dict]) -> str:
    return "".join(w["raw"] for w in words).strip()


def _dropped(words: list[dict], reason: str) -> dict:
    return {"start": r3(words[0]["start"]), "end": r3(words[-1]["end"]), "text": _text(words), "reason": reason}


def process_result(result: dict, offset: float, active: list[list[float]],
                   voiced: np.ndarray | None = None) -> tuple[list[dict], list[dict]]:
    """Turn one region's whisper result into (lines, dropped), in clip time.

    ``offset`` is the region start (whisper times are relative to the slice); ``active`` are the
    unpadded vocal-active regions, used to drop words that land where there is no voice;
    ``voiced`` (from voiced_mask, optional) tightens word edges that run into silence.
    """
    dropped: list[dict] = []
    kept: list[dict] = []  # words of surviving segments, tagged with their segment
    for si, seg in enumerate(result.get("segments", [])):
        text = seg.get("text", "").strip()
        if not text:
            continue
        s0, s1 = offset + seg["start"], offset + seg["end"]
        reason = segment_drop_reason(seg)
        if reason is None and not seg.get("words"):
            reason = "no word timings"
        if reason:
            dropped.append({"start": r3(s0), "end": r3(s1), "text": text, "reason": reason})
            continue
        words = []
        for w in seg["words"]:
            ws = offset + float(w["start"])
            words.append({"raw": w["word"], "start": ws, "end": max(ws, offset + float(w["end"])),
                          "p": float(w.get("probability", 0.0)), "seg": si})
        if voiced is not None:
            trim_to_voice(words, voiced)
        outside: list[dict] = []
        for word in words:
            if overlaps_active(word["start"], word["end"], active):
                if outside:
                    dropped.append(_dropped(outside, "outside vocal activity"))
                    outside = []
                kept.append(word)
            else:
                outside.append(word)
        if outside:
            dropped.append(_dropped(outside, "outside vocal activity"))

    kept, loops = drop_loops(kept)
    dropped += loops

    segs = result.get("segments", [])
    lines = []
    for si in sorted({w["seg"] for w in kept}):
        seg_words = [w for w in kept if w["seg"] == si]
        for ln in split_lines(seg_words):
            lines.append({
                "start": r3(ln[0]["start"]),
                "end": r3(ln[-1]["end"]),
                "text": _text(ln),
                "avg_logprob": r2(segs[si].get("avg_logprob")),
                "no_speech_prob": r2(segs[si].get("no_speech_prob")),
                "words": [{"w": w["raw"].strip(), "start": r3(w["start"]), "end": r3(w["end"]), "p": r2(w["p"])}
                          for w in ln],
            })
    lines.sort(key=lambda ln: ln["start"])
    dropped.sort(key=lambda d: d["start"])
    return lines, dropped


# ---- whisper -----------------------------------------------------------------------

def normalize_language(lang: str) -> str:
    from mlx_whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

    code = lang.strip().lower()
    code = TO_LANGUAGE_CODE.get(code, code)
    if code not in LANGUAGES:
        raise ValueError(f"unsupported lyrics language {lang!r}; use a code like en, es, ja")
    return code


def load_model(repo: str):
    """Load (or reuse) the model through mlx_whisper's own cache, so transcribe() gets the same object."""
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
    from mlx_whisper.transcribe import ModelHolder

    try:  # already downloaded: skip the "Fetching 4 files" progress bar (keep it for real downloads)
        snapshot_download(repo, local_files_only=True)
        disable_progress_bars()
    except Exception:
        pass
    try:
        model = ModelHolder.get_model(repo, mx.float16)
    finally:
        enable_progress_bars()
    dims = model.dims
    untuned = model.alignment_heads.shape[0] == (dims.n_text_layer - dims.n_text_layer // 2) * dims.n_text_head
    if dims.n_text_layer == 4 and dims.n_text_head == 20 and untuned:  # turbo without shipped heads
        model.set_alignment_heads(TURBO_ALIGNMENT_HEADS)
        log.info("lyrics: installed the tuned large-v3-turbo alignment heads")
    return model


def release_model() -> None:
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder

    ModelHolder.model = None
    ModelHolder.model_path = None
    mx.clear_cache()


def detect_language(model, y: np.ndarray) -> tuple[str, float]:
    import mlx.core as mx
    from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

    mel = log_mel_spectrogram(y, n_mels=model.dims.n_mels, padding=N_SAMPLES)
    _, probs = model.detect_language(pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16))
    lang = max(probs, key=probs.get)
    return lang, float(probs[lang])


def transcribe(y: np.ndarray, repo: str, language: str) -> dict:
    import mlx.core as mx
    import mlx_whisper

    mx.random.seed(0)  # temperature fallback samples; keep reruns identical
    try:
        return mlx_whisper.transcribe(
            y,
            path_or_hf_repo=repo,
            language=language,
            task="transcribe",
            word_timestamps=True,
            condition_on_previous_text=False,  # stops one bad window from seeding the next
            compression_ratio_threshold=COMPRESSION_MAX,
            logprob_threshold=LOGPROB_MIN,
            no_speech_threshold=NO_SPEECH_MAX,
            verbose=None,
        )
    finally:
        mx.clear_cache()  # word-timestamp calls leak Metal buffers otherwise


# ---- stage -------------------------------------------------------------------------

def _write(ctx: RunContext, status: str, reason: str | None, **fields) -> None:
    out = {
        "status": status,
        "reason": reason,
        "model": ctx.config.whisper_model,
        "language": None,
        "language_source": None,
        "language_probability": None,
        "regions": [],
        "lines": [],
        "dropped": [],
    }
    out.update(fields)
    ctx.write_json("lyrics.json", out)


def run(ctx: RunContext) -> None:
    if not ctx.config.vocals:  # --no-vocals: nothing to cut, detect or transcribe
        _write(ctx, "skipped", NO_VOCALS_REASON)
        log.info("lyrics: skipped (%s)", NO_VOCALS_REASON)
        return
    energy = ctx.read_json("energy.json")
    active = sorted(energy.get("active", {}).get("vocals", []))
    if not energy.get("has_vocals") or not active:
        reason = "no vocal-active regions in the vocal stem" if not active else \
            f"energy.json has_vocals is false ({energy.get('vocal_seconds', 0)} s of vocal activity)"
        _write(ctx, "skipped", reason)
        log.info("lyrics: skipped (%s)", reason)
        return

    y, _ = audio.load_mono(ctx.stem_path("vocals"), sr=SR)
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    regions = plan_regions(active, len(y) / SR)
    voiced = voiced_mask(y, regions)
    repo = ctx.config.whisper_model

    try:
        model = load_model(repo)
        if ctx.config.language:
            language, source, prob = normalize_language(ctx.config.language), "user", None
        elif not model.is_multilingual:
            language, source, prob = "en", "model", None
        else:
            gap = np.zeros(int(LANG_JOIN_S * SR), dtype=np.float32)
            parts = [p for s, e in active for p in (y[int(s * SR):int(e * SR)], gap)]
            language, prob = detect_language(model, np.concatenate(parts)[:int(LANG_MAX_S * SR)])
            source = "detected"
            log.info("lyrics: detected language %s (p=%.2f)", language, prob)

        lines, dropped = [], []
        for s, e in regions:
            result = transcribe(y[int(s * SR):int(e * SR)], repo, language)
            ln, dr = process_result(result, s, active, voiced)
            lines += ln
            dropped += dr
    finally:
        release_model()

    status = "ok" if lines else "empty"
    reason = None if lines else ("everything whisper returned was filtered out" if dropped else "whisper returned no text")
    _write(ctx, status, reason, language=language, language_source=source, language_probability=r2(prob),
           regions=regions, lines=lines, dropped=dropped)
    n_words = sum(len(ln["words"]) for ln in lines)
    log.info("lyrics: %s, %d lines / %d words in %d region(s), %d dropped", language, len(lines), n_words,
             len(regions), len(dropped))


STAGE = Stage(
    name="lyrics",
    run=run,
    outputs=lambda ctx: [ctx.cache_path("lyrics.json")],
    deps=("stems", "energy"),
    config_keys=("whisper_model", "language", "vocals"),
    required=False,
)
