# Output schema (v1.0)

This is the field-by-field reference for the two machine-readable outputs in `runs/<clip>/`:

- [`analysis.json`](#analysisjson) is the compact, LLM-facing analysis.
- [`curves.json`](#curvesjson) holds dense fixed-rate envelopes for code to sample.

`summary.md` and `overview.png` are derived views of the same data; they are described in the README.

**Versioning.** `schema_version` is `"1.0"`. Adding a new optional field is not a breaking change. Renaming,
removing or retyping a field, or changing what a value means, bumps the version. All changes are listed in
[Changelog](#changelog).

## Conventions

These apply everywhere (they are also embedded in the file under `conventions`):

| Thing | Meaning |
|-------|---------|
| time (`t`, `start`, `end`) | seconds from the start of `input.wav`, 3 decimals. `input.wav` is the trimmed clip, so with `--start 60`, `t = 0` is 60 s into the original file. |
| `bar` | integer, 1-based, counted from the first downbeat. Beats before the first downbeat (a pickup) are in bar 0. Events before the first detected beat are placed by extrapolating the grid, so they can be in bar 0 or lower. |
| `beat` | 1-based position in the bar. In `grid.beats` it is an integer. On events it is fractional, with 2 decimals: `1.0` is the downbeat, `2.5` is halfway between beats 2 and 3, and `4.99` is just before the next bar. |
| energy (0..1) | Relative to this clip. It is linear in dB over the top 30 dB of each stem: 1.0 is that stem's loudest level (its 99th percentile), 0.0 is 30 dB or more below it. A stem that is nearly silent throughout is referenced to the mix instead, so its values stay near 0 rather than being stretched up to 1. |
| estimates | Key, drum-hit types, note transcriptions, section characters and moments come from heuristics or models and can be wrong. Each one says so in its `about`/`method` text. |
| `null` | A part whose stage failed or was skipped is `null`; the reason is in `meta.warnings`. |

## analysis.json

Top-level keys, in file order:

| Key | Type | Description |
|-----|------|-------------|
| `schema_version` | string | `"1.0"` |
| `conventions` | object | the conventions above, as short strings |
| `source` | object | where the audio came from |
| `global` | object | clip-wide facts: tempo, meter, key, loudness, vocals |
| `grid` | object | every beat and every bar |
| `sections` | array or null | contiguous segments covering the clip, with boundaries at bar starts |
| `moments` | array or null | notable instants (drop, stop, vocal entry, ...) |
| `hits` | object or null | drum hits by type, plus per-bar step patterns |
| `stems` | object | per-beat energy and active regions for the mix and each stem |
| `notes` | object or null | bass and vocal notes, plus a per-bar summary |
| `lyrics` | object or null | transcribed lines and words |
| `curves` | object | pointer to `curves.json` |
| `meta` | object | generator version, models, devices, warnings |

### `source`

| Field | Type | Description |
|-------|------|-------------|
| `file` | string | absolute path of the original input file |
| `input_wav` | string | path of the analysed audio. Play this, or mux it into the video. |
| `start_offset` | number | `--start` used for trimming, in seconds into `file` (0 if none) |
| `duration` | number | length of `input_wav` in seconds |
| `sample_rate` | int | 44100 |
| `channels` | int | 2 |

### `global`

| Field | Type | Description |
|-------|------|-------------|
| `tempo_bpm` | number | median tempo over 4-beat windows |
| `tempo_std_bpm` | number | spread of those local tempi: about 0–1 is steady; larger values mean drift, rubato or tracking trouble |
| `time_signature` | string | `"4/4"` unless the downbeats clearly say otherwise (`"3/4"`, ...); the denominator is always 4 |
| `beats_per_bar` | int | numerator of `time_signature` |
| `beat_count`, `bar_count` | int | `len(grid.beats)` and the number of bars numbered 1 and up |
| `key` | object or null | `{label: "A minor", confidence: 0..1, runner_up: "C major", estimate: true}` |
| `loudness` | object | `integrated_lufs` (BS.1770), `loudness_range_lu` (EBU R128 LRA, approximated), `peak_dbfs` (sample peak), `rms_dbfs`, `max_momentary_lufs` |
| `has_vocals` | bool | the vocal stem is clearly audible for at least 1 s |
| `language` | string or null | ISO code from Whisper (detected or `--language`); null when there are no lyrics |

### `grid`

| Field | Type | Description |
|-------|------|-------------|
| `beats` | array of `{t, bar, beat, is_downbeat}` | every beat beat_this found, in order; `beat` is an integer here |
| `bars` | array of `{bar, start, end}` | bar 0 (a pickup) only appears if there is one; the last bar ends one beat after its last beat, clipped to the clip end |

The downbeats come from beat_this. When the spacing makes it unambiguous, downbeats it skipped are filled in:
a gap of exactly 2 or more bars, beats after the last downbeat, and a lead-in longer than one bar. Beats are
detected on the full mix, at 50 frames/s, so expect about ±20 ms of jitter.

### `sections`

The clip is cut at bar starts into the fewest sections that each follow a constant or a straight line in
bar-level features: loudness, the level of each stem, kick/snare/hat hits per beat, and brightness.
Sections are usually at least 4 bars long, unless a change is dramatic, and there are at most 10.

| Field | Type | Description |
|-------|------|-------------|
| `index` | int | 0-based position |
| `start`, `end` | number | seconds; contiguous; the first starts at 0.0 and the last ends at `source.duration`; inner boundaries are bar starts |
| `bars` | `[first, last]` | inclusive bar range |
| `character` | string | what the music is doing (see the vocabulary below), **not** verse/chorus |
| `energy` | number | mean loudness, 0..1: 0 is 30 LU below the loudest bar, 1 is the loudest bar |
| `loudness_lufs` | number | mean momentary loudness over the section |
| `trend` | string | `rising` / `falling` / `flat`: the linear-fit loudness change across the section, with ±3 LU counting as flat |
| `change_lu` | number | that change, in LU |
| `active_stems` | array | stems active for at least 2 s or 25% of the section, whichever is shorter |
| `hits_per_beat` | object | `{kick, snare, hat, all}`: average drum hits per beat |
| `description` | string | a short generated phrase |

The **character** is the first rule that matches. "Loud" is relative to the clip's loudest bar:

| Character | Rule |
|-----------|------|
| `silence` | mean loudness more than 40 LU below the loudest bar |
| `build` | loudness rises at least 5 LU across the section, or rises at least 3 LU while drum hits per beat at least double |
| `outro` | the last section, falling at least 4 LU across it (a fade or wind-down) |
| `full` | within 3 LU of the loudest section, with a beat (kick/snare at least 0.5 per beat) or at least three stems active |
| `breakdown` | at least 4 LU quieter than the previous section, which was full, groove or build, and the beat has mostly gone |
| `groove` | a steady beat (kick/snare at least 0.5 per beat), but not full |
| `sparse` | anything else: quiet, few elements, no steady beat |

### `moments`

| Field | Type | Description |
|-------|------|-------------|
| `t`, `bar`, `beat` | | when it happens |
| `type` | string | see the vocabulary below |
| `description` | string | a short generated phrase, e.g. "full section hits after the build; kick and bass come in" |
| `strength` | number | 0..1, how pronounced the moment is |
| `end` | number, optional | `stop` and `silence` only: when the gap ends |

| Type | Meaning |
|------|---------|
| `drop` | start of a full section that follows a build, a stop or a silence, or that is at least 6 LU louder than the bar before; snapped to a kick within 80 ms |
| `build_peak` | loudest point in the last bar of a build, before any stop |
| `stop` | sudden gap: the mix falls at least 12 dB for 0.25–4 s, then comes back |
| `silence` | mix more than 50 dB below its loudest for at least 1 s (including at the start or end) |
| `breakdown` | start of a breakdown section |
| `loudest` | peak of the 1 s loudness curve (one per clip) |
| `vocal_entry` | vocals become active after at least 3 s without them |
| `drums_in`, `drums_out`, `bass_in`, `bass_out` | the stem starts or stops, with at least 2 s of inactivity on the other side |
| `kick_in` | first kick after at least 4 s without kicks |

Entries and exits within 0.3 s of a drop or breakdown are folded into that moment's description, not listed
separately.

### `hits`

| Field | Type | Description |
|-------|------|-------------|
| `about` | string | how the hits were estimated |
| `columns` | array | `["t", "bar", "beat", "strength"]`: the layout of each row below |
| `counts` | object | `{kick, snare, hat}` counts |
| `kick`, `snare`, `hat` | array of rows | `[t, bar, beat, strength]`; strength is 0..1 within that type in this clip |
| `patterns_about` | string | how to read `patterns` |
| `patterns` | array of `{bars: [first, last], kick, snare, hat}` | one character per 16th note (`x` = hit, `.` = none), so a 4/4 bar is 16 characters. Consecutive bars with identical patterns are merged into one entry. Bars without any hits are omitted. |

Rows are used instead of objects to keep the file compact: a busy 80 s clip can have more than 600 hits.

### `stems`

| Field | Type | Description |
|-------|------|-------------|
| `about` | string | how to read the block |
| `mix.per_beat` | array of numbers | mix energy 0..1 for each beat in `grid.beats` (same length and order), power-averaged from that beat to the next |
| `<stem>.per_beat` | array of numbers | the same, for `vocals`, `drums`, `bass` and `other` |
| `<stem>.active` | array of `{start, end, bar, beat, end_bar, end_beat}` | regions where the stem is clearly audible: within 18 dB of the local mix level and above −60 dBFS, gaps under 0.75 s merged, blips under 0.4 s dropped |
| `<stem>.active_fraction` | number | share of the clip the stem is active (0..1) |

`other` is Demucs' catch-all: pads, keys, guitars, synths, effects.

A stem can be **active while its `per_beat` energy is near 0**. Energy is relative to the stem's own
loudest moment, so soft hats in an intro read as about 0 when the same drum stem later carries loud kicks
(the test clip shows exactly this). For such parts, use `hits` and the `drums_high`/`drums_mid` curves,
which are normalised per band.

### `notes`

| Field | Type | Description |
|-------|------|-------------|
| `about` | string | how the notes were produced |
| `backend` | string | `"coreml"` or `"onnx"` (Basic Pitch runtime actually used) |
| `columns` | array | `["t", "bar", "beat", "dur", "midi", "name", "velocity"]` |
| `bass`, `vocals` | array of rows | `[t, bar, beat, duration_s, midi, name, velocity]`. `name` uses ASCII sharps (`"C#3"`). `velocity` is 1..127, from Basic Pitch's amplitude. |
| `vocals_note` | string, optional | why vocal notes are missing (for example, no vocals) |
| `per_bar_about` | string | how to read `per_bar` |
| `per_bar` | array of `{bar, bass?, vocals?, bass_root?}` | note names that start in each bar, in order with repeats collapsed (`"A2 C3"`); `bass_root` is the bass pitch class held longest in that bar |

Notes are gated to the stem's active regions, which removes bleed. Bass notes are filtered for harmonic
ghosts: a note an octave, twelfth, two octaves, and so on above a stronger, overlapping lower note is
dropped. Note `dur` tends to run on to the next onset, so sync visuals to `t`. The raw Basic Pitch output is
also written to `cache/bass.mid` and `cache/vocals.mid`.

### `lyrics`

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"ok"`, `"skipped"` (no vocals) or `"empty"` (vocals present, but nothing reliable was transcribed) |
| `reason` | string, optional | why it is not `ok` |
| `language` | string | ISO code |
| `model` | string | Whisper model repo |
| `lines` | array | `{start, end, bar, beat, text, words}` |
| `lines[].words` | array | `{w, start, end, bar, beat}`. `w` keeps Whisper's punctuation (`"night,"`). `bar`/`beat` are at the word's start. |

Only the regions where the vocal stem is active are transcribed. Segments that look like Whisper
hallucinations are dropped: known filler phrases, a high compression ratio, likely no speech, or low
log-probability. The full record, including `dropped` entries, per-word probabilities, per-line `avg_logprob`
and `no_speech_prob`, and the detected language probability, is in `cache/lyrics.json`.

### `curves`

`{file: "curves.json", hz, keys, about}`: a pointer to the dense envelopes, with the list of curve names.

### `meta`

| Field | Type | Description |
|-------|------|-------------|
| `generator` | string | `"audio4llm <version>"` |
| `models` | object | the models used by `stems`, `rhythm`, `lyrics` and `notes` |
| `devices` | object | the torch device actually used by `stems` and `rhythm` (`"mps"` or `"cpu"`, after any fallback) |
| `warnings` | array of strings | missing or failed optional parts |

## curves.json

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | `"1.0"` |
| `hz` | number | samples per second (`--curves-hz`, default 20) |
| `n` | int | samples per curve; sample `i` is at `t = i / hz` |
| `duration` | number | clip duration in seconds |
| `about`, `curves_info` | string, object | descriptions |
| `curves` | object of arrays | every array has length `n` |

Curves, all aligned to `t = i / hz` and each summarising the window `t ± 0.5/hz`:

| Name | Range | Description |
|------|-------|-------------|
| `mix`, `vocals`, `drums`, `bass`, `other` | 0..1 | energy (power-averaged over the window); follows every transient, so the mix pumps with the beat |
| `mix_smooth` | 0..1 | mix energy smoothed over 1 s; use it for overall intensity (loud versus quiet passages) |
| `loudness_lufs` | about −70..0 | momentary loudness of the mix (400 ms window, BS.1770 K-weighting) |
| `brightness` | 0..1 | spectral centroid of the mix on a log scale, from 200 Hz (0) to 8 kHz (1) |
| `onset` | 0..1 | transient strength of the mix (window maximum) |
| `drums_low`, `drums_mid`, `drums_high` | 0..1 | drum-stem band envelopes (window maximum): roughly kick, snare and hats. Good for pumping on hits without triggering on discrete events. |

To sample at an arbitrary time `t`: `i = round(t * hz)`, then interpolate linearly between `i` and `i + 1`.

## Example

A trimmed real `analysis.json` from the synthetic test clip (`python scripts/make_test_clip.py && audio4llm
test_data/test_clip.wav`). Long arrays are cut to their first items, followed by a `"… N more"` string, and
the absolute paths in `source` are shortened. Everything else is verbatim. Regenerate it with
`python scripts/update_schema_example.py runs/test_clip`.

<!-- example:analysis:start -->
```json
{
  "schema_version": "1.0",
  "conventions": {
    "time": "seconds from the start of input.wav (the audio to play/mux), 3 decimals",
    "bar": "1-based, counted from the first downbeat; 0 (or below) = before it",
    "beat": "1-based position within the bar, fractional: 1.0 = downbeat, 2.5 = halfway between beats 2 and 3",
    "energy": "0..1 within this clip, linear in dB over the top 30 dB of each stem (1 = that stem's loudest)",
    "estimates": "key, drum hit types, notes, section characters and moments are heuristic estimates",
    "dense_curves": "curves.json has fixed-rate envelopes for code; this file is the readable summary"
  },
  "source": {
    "file": "test_data/test_clip.wav",
    "input_wav": "runs/test_clip/input.wav",
    "start_offset": 0.0,
    "duration": 80.0,
    "sample_rate": 44100,
    "channels": 2
  },
  "global": {
    "tempo_bpm": 119.97,
    "tempo_std_bpm": 0.66,
    "time_signature": "4/4",
    "beats_per_bar": 4,
    "beat_count": 160,
    "bar_count": 40,
    "key": {"label": "C major", "confidence": 0.68, "runner_up": "A minor", "estimate": true},
    "loudness": {
      "integrated_lufs": -17.3,
      "loudness_range_lu": 18.7,
      "peak_dbfs": -1.0,
      "rms_dbfs": -20.0,
      "max_momentary_lufs": -12.2
    },
    "has_vocals": true,
    "language": "en"
  },
  "grid": {
    "beats": [
      {"t": 0.04, "bar": 1, "beat": 1, "is_downbeat": true},
      {"t": 0.504, "bar": 1, "beat": 2, "is_downbeat": false},
      {"t": 1.008, "bar": 1, "beat": 3, "is_downbeat": false},
      {"t": 1.507, "bar": 1, "beat": 4, "is_downbeat": false},
      {"t": 2.005, "bar": 2, "beat": 1, "is_downbeat": true},
      {"t": 2.504, "bar": 2, "beat": 2, "is_downbeat": false},
      "… 154 more"
    ],
    "bars": [
      {"bar": 1, "start": 0.04, "end": 2.005},
      {"bar": 2, "start": 2.005, "end": 4.006},
      {"bar": 3, "start": 4.006, "end": 6.006},
      "… 37 more"
    ]
  },
  "sections": [
    {
      "index": 0,
      "start": 0.0,
      "end": 16.004,
      "bars": [1, 8],
      "character": "sparse",
      "energy": 0.35,
      "loudness_lufs": -33.1,
      "trend": "flat",
      "change_lu": -0.2,
      "active_stems": ["drums", "other"],
      "hits_per_beat": {"kick": 0.0, "snare": 0.0, "hat": 2.0, "all": 2.0},
      "description": "sparse; drums, other instruments; hat 2.0 per beat"
    },
    {
      "index": 1,
      "start": 16.004,
      "end": 32.002,
      "bars": [9, 16],
      "character": "build",
      "energy": 0.76,
      "loudness_lufs": -20.6,
      "trend": "rising",
      "change_lu": 16.1,
      "active_stems": ["drums", "other"],
      "hits_per_beat": {"kick": 0.0, "snare": 1.88, "hat": 1.16, "all": 3.03},
      "description": "builds +16 LU; drums, other instruments; snare 1.9 / hat 1.2 per beat"
    },
    {
      "index": 2,
      "start": 32.002,
      "end": 56.02,
      "bars": [17, 28],
      "character": "full",
      "energy": 0.95,
      "loudness_lufs": -14.9,
      "trend": "flat",
      "change_lu": -0.8,
      "active_stems": ["vocals", "drums", "bass", "other"],
      "hits_per_beat": {"kick": 1.0, "snare": 0.5, "hat": 2.0, "all": 3.5},
      "description": "full energy; vocals, drums, bass, other instruments; kick 1.0 / snare 0.5 / hat 2.0 per beat"
    },
    {
      "index": 3,
      "start": 56.02,
      "end": 72.006,
      "bars": [29, 36],
      "character": "breakdown",
      "energy": 0.57,
      "loudness_lufs": -26.4,
      "trend": "flat",
      "change_lu": -0.2,
      "active_stems": ["vocals", "other"],
      "hits_per_beat": {"kick": 0.0, "snare": 0.0, "hat": 0.0, "all": 0.0},
      "description": "breakdown, 11 LU down; vocals, other instruments"
    },
    {
      "index": 4,
      "start": 72.006,
      "end": 80.0,
      "bars": [37, 40],
      "character": "outro",
      "energy": 0.85,
      "loudness_lufs": -18.0,
      "trend": "falling",
      "change_lu": -9.3,
      "active_stems": ["drums", "bass", "other"],
      "hits_per_beat": {"kick": 1.0, "snare": 0.5, "hat": 2.0, "all": 3.5},
      "description": "winds down -9 LU; drums, bass, other instruments; kick 1.0 / snare 0.5 / hat 2.0 per beat"
    }
  ],
  "moments": [
    {
      "t": 31.2,
      "bar": 16,
      "beat": 3.39,
      "type": "build_peak",
      "description": "top of the build (+16 LU over 16 s)",
      "strength": 1.0
    },
    {
      "t": 31.52,
      "bar": 16,
      "beat": 4.0,
      "type": "stop",
      "description": "music cuts out (25 dB down) for 0.44 s",
      "strength": 0.83,
      "end": 31.96
    },
    {
      "t": 32.0,
      "bar": 17,
      "beat": 1.0,
      "type": "drop",
      "description": "full section hits after the build; kick and bass come in",
      "strength": 0.97
    },
    {"t": 36.03, "bar": 19, "beat": 1.0, "type": "vocal_entry", "description": "vocals come in", "strength": 0.81},
    {
      "t": 37.31,
      "bar": 19,
      "beat": 3.61,
      "type": "loudest",
      "description": "loudest point (-12.2 LUFS momentary)",
      "strength": 1.0
    },
    {
      "t": 56.02,
      "bar": 29,
      "beat": 1.0,
      "type": "breakdown",
      "description": "breakdown: 11 LU quieter; drums and bass drop out",
      "strength": 0.64
    },
    {"t": 60.0, "bar": 30, "beat": 4.89, "type": "vocal_entry", "description": "vocals return", "strength": 0.88},
    {
      "t": 71.98,
      "bar": 37,
      "beat": 1.0,
      "type": "drums_in",
      "description": "drums come in, with kick",
      "strength": 0.57
    },
    {"t": 71.99, "bar": 37, "beat": 1.0, "type": "bass_in", "description": "bass comes in", "strength": 0.81}
  ],
  "hits": {
    "about": "estimate: drum-stem STFT split into low 30-150 Hz (kick) / mid 1-5 kHz (snare) / high 6-16 kHz (hat) bands; per-band onset peaks gated by level, with rules against bass bleed, kick-click and snare-noise leakage. Rows are [t, bar, beat, strength 0..1].",
    "columns": ["t", "bar", "beat", "strength"],
    "counts": {"kick": 64, "snare": 92, "hat": 229},
    "kick": [[32.0, 17, 1.0, 1.0], [32.5, 17, 2.0, 1.0], [33.0, 17, 3.0, 1.0], [33.5, 17, 4.0, 1.0], "… 60 more"],
    "snare": [[16.0, 9, 1.0, 0.63], [16.5, 9, 2.0, 0.63], [17.0, 9, 3.0, 0.63], "… 89 more"],
    "hat": [[0.01, 0, 4.94, 0.51], [0.25, 1, 1.5, 0.4], [0.5, 1, 2.0, 0.51], "… 226 more"],
    "patterns_about": "One character per 16th note (x = hit), 16 per bar; consecutive bars with the same pattern are merged into one entry.",
    "patterns": [
      {"bars": [1, 1], "kick": "................", "snare": "................", "hat": "..x.x.x.x.x.x.x."},
      {"bars": [2, 8], "kick": "................", "snare": "................", "hat": "x.x.x.x.x.x.x.x."},
      {"bars": [9, 9], "kick": "................", "snare": "x...x...x...x...", "hat": ".xxxxxxxxxxxxxxx"},
      "… 7 more"
    ]
  },
  "stems": {
    "about": "per_beat[i] is the stem's energy (0..1) during grid.beats[i] (until the next beat). active lists the regions where the stem is clearly audible.",
    "mix": {"per_beat": [0.13, 0.2, 0.22, 0.13, 0.13, 0.22, 0.2, 0.14, "… 152 more"]},
    "vocals": {
      "per_beat": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "… 152 more"],
      "active": [
        {"start": 36.03, "end": 38.45, "bar": 19, "beat": 1.0, "end_bar": 20, "end_beat": 1.9},
        {"start": 59.98, "end": 62.49, "bar": 30, "beat": 4.85, "end_bar": 32, "end_beat": 2.0}
      ],
      "active_fraction": 0.06
    },
    "drums": {
      "per_beat": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "… 152 more"],
      "active": [
        {"start": 0.0, "end": 55.88, "bar": 0, "beat": 4.92, "end_bar": 28, "end_beat": 4.75},
        {"start": 71.96, "end": 79.88, "bar": 36, "beat": 4.91, "end_bar": 40, "end_beat": 4.75}
      ],
      "active_fraction": 0.8
    },
    "bass": {
      "per_beat": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "… 152 more"],
      "active": [
        {"start": 31.97, "end": 55.93, "bar": 16, "beat": 4.94, "end_bar": 28, "end_beat": 4.83},
        {"start": 71.97, "end": 79.93, "bar": 36, "beat": 4.93, "end_bar": 40, "end_beat": 4.85}
      ],
      "active_fraction": 0.4
    },
    "other": {
      "per_beat": [0.59, 0.66, 0.68, 0.59, 0.58, 0.68, 0.66, 0.6, "… 152 more"],
      "active": [{"start": 0.0, "end": 79.94, "bar": 0, "beat": 4.92, "end_bar": 40, "end_beat": 4.87}],
      "active_fraction": 1.0
    }
  },
  "notes": {
    "about": "Basic Pitch transcription of the bass and vocal stems (estimates). Rows are [t, bar, beat, duration, midi, name, velocity 1..127].",
    "backend": "coreml",
    "columns": ["t", "bar", "beat", "dur", "midi", "name", "velocity"],
    "bass": [
      [31.994, 17, 1.0, 0.5, 45, "A2", 95],
      [32.494, 17, 2.0, 0.51, 45, "A2", 89],
      [33.004, 17, 3.0, 0.5, 45, "A2", 90],
      [33.504, 17, 4.0, 0.442, 45, "A2", 90],
      "… 60 more"
    ],
    "vocals": [[35.968, 18, 4.93, 0.116, 56, "G#3", 81], [36.095, 19, 1.18, 0.163, 57, "A3", 71], "… 18 more"],
    "per_bar_about": "Notes starting in each bar, in order (repeats collapsed); bass_root = bass pitch class held longest.",
    "per_bar": [
      {"bar": 17, "bass": "A2", "bass_root": "A"},
      {"bar": 18, "bass": "F2", "bass_root": "F", "vocals": "G#3"},
      {"bar": 19, "bass": "C3", "bass_root": "C", "vocals": "A3 F#3 G#3 F#3"},
      "… 15 more"
    ]
  },
  "lyrics": {
    "status": "ok",
    "language": "en",
    "model": "mlx-community/whisper-large-v3-turbo",
    "lines": [
      {
        "start": 36.0,
        "end": 36.91,
        "bar": 19,
        "beat": 1.0,
        "text": "Hold on to the night.",
        "words": [
          {"w": "Hold", "start": 36.0, "end": 36.25, "bar": 19, "beat": 1.0},
          {"w": "on", "start": 36.25, "end": 36.39, "bar": 19, "beat": 1.5},
          {"w": "to", "start": 36.39, "end": 36.53, "bar": 19, "beat": 1.75},
          "… 2 more"
        ]
      },
      {
        "start": 37.11,
        "end": 38.41,
        "bar": 19,
        "beat": 3.25,
        "text": "We are dancing in the light.",
        "words": [
          {"w": "We", "start": 37.11, "end": 37.31, "bar": 19, "beat": 3.25},
          {"w": "are", "start": 37.31, "end": 37.47, "bar": 19, "beat": 3.61},
          {"w": "dancing", "start": 37.47, "end": 37.85, "bar": 19, "beat": 3.93},
          "… 3 more"
        ]
      },
      "… 2 more"
    ]
  },
  "curves": {
    "file": "curves.json",
    "hz": 20.0,
    "keys": ["mix", "vocals", "drums", "bass", "other", "mix_smooth", "loudness_lufs", "brightness", "onset", "drums_low", "drums_mid", "drums_high"],
    "about": "dense fixed-rate envelopes for code to sample at render time (not meant to be read)"
  },
  "meta": {
    "generator": "audio4llm 0.1.0",
    "models": {
      "stems": "htdemucs",
      "rhythm": "beat_this final0",
      "lyrics": "mlx-community/whisper-large-v3-turbo",
      "notes": "basic-pitch (coreml)"
    },
    "devices": {"stems": "mps", "rhythm": "mps"},
    "warnings": []
  }
}
```
<!-- example:analysis:end -->

A trimmed `curves.json` from the same run, showing 8 samples (t = 32.0–32.35 s, around the drop) of each curve:

<!-- example:curves:start -->
```json
{
  "schema_version": "1.0",
  "hz": 20.0,
  "n": 1601,
  "duration": 80.0,
  "about": "Sample i is at t = i / hz seconds from the start of input.wav. Each value summarises the window t +/- 0.5/hz. Energies are 0..1 (see analysis.json conventions).",
  "curves_info": {
    "mix|vocals|drums|bass|other": "energy 0..1 (power-averaged per window, linear in dB over the top 30 dB)",
    "mix_smooth": "mix energy smoothed over 1 s: overall intensity without the per-beat pumping",
    "loudness_lufs": "momentary loudness of the mix (400 ms, BS.1770), LUFS, floor -70",
    "brightness": "spectral centroid of the mix, 0..1 on a log scale from 200 Hz to 8 kHz",
    "onset": "transient strength of the mix, 0..1 (window max)",
    "drums_low|drums_mid|drums_high": "drum-stem band envelopes 0..1 (window max): ~kick / ~snare / ~hats"
  },
  "curves": {
    "mix": [0.879, 0.988, 0.914, 0.866, 0.83, 0.814, 0.801, 0.782, "… 1593 more"],
    "vocals": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "… 1593 more"],
    "drums": [0.893, 0.987, 0.888, 0.802, 0.706, 0.634, 0.541, 0.348, "… 1593 more"],
    "bass": [0.785, 0.938, 0.948, 0.948, 0.944, 0.944, 0.942, 0.933, "… 1593 more"],
    "other": [0.256, 0.193, 0.407, 0.584, 0.714, 0.782, 0.828, 0.877, "… 1593 more"],
    "mix_smooth": [0.749, 0.768, 0.796, 0.809, 0.82, 0.828, 0.835, 0.842, "… 1593 more"],
    "loudness_lufs": [-15.6, -15.1, -14.6, -14.3, -14.1, -15.4, -16.5, -15.4, "… 1593 more"],
    "brightness": [0.999, 0.735, 0.103, 0.1, 0.131, 0.708, 0.696, 0.319, "… 1593 more"],
    "onset": [1.0, 0.794, 0.04, 0.054, 0.055, 0.994, 0.105, 0.106, "… 1593 more"],
    "drums_low": [0.976, 1.0, 0.91, 0.835, 0.722, 0.647, 0.539, 0.453, "… 1593 more"],
    "drums_mid": [0.495, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "… 1593 more"],
    "drums_high": [0.829, 0.621, 0.0, 0.0, 0.0, 0.644, 0.363, 0.0, "… 1593 more"]
  }
}
```
<!-- example:curves:end -->

## Changelog

- **1.0**: first documented version.
