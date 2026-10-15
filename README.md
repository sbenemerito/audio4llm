# audio4llm

**audio4llm** turns a short music clip (up to ~80 s; a song or an instrumental) into structured data that an
LLM can read in order to write code for a **beat-synced visualization** of that clip. The output tells the
LLM, and the code it writes, where the beats and bars are, when drums hit, which notes the bass and vocals
play, which passages are loud or quiet, how the clip is structured, and when each lyric word is sung, all on
one shared timeline.

Everything runs locally on an Apple Silicon Mac. There are no API keys, and model weights download
automatically on the first run.

```
audio4llm song.mp3
# -> runs/song/analysis.json   what an LLM reads (beats, bars, sections, moments, hits, stems, notes, lyrics)
#    runs/song/curves.json     dense 20 Hz envelopes for the generated code to sample
#    runs/song/summary.md      a short plain-language description
#    runs/song/overview.png    a plot for checking the analysis at a glance
#    runs/song/input.wav       the exact audio every timestamp refers to (mux this into your video)
```

To test the outputs with an LLM-written visualizer, see [docs/VISUALIZER_TEST.md](docs/VISUALIZER_TEST.md).
The field-by-field reference for the outputs is [docs/OUTPUT_SCHEMA.md](docs/OUTPUT_SCHEMA.md).

---

## How it works

Each stage reads from and writes to the run folder `runs/<clip>/`. A stage is skipped when its outputs
already exist and nothing it depends on has changed (see [Caching](#caching-and-forced-reruns)).

```
  clip.mp3 / .m4a / .wav / .flac   (--start / --duration trim here)
              │
              ▼
     ┌──────────────────┐
     │ 0 ingest  ffmpeg │──► input.wav  44.1 kHz stereo: the one timeline everything is measured on
     └────────┬─────────┘
              ├──────────────────────────────────┐
              ▼                                  ▼
     ┌──────────────────┐               ┌──────────────────┐
     │ 1 stems  Demucs  │               │ 2 rhythm         │
     │ vocals · drums · │               │ beat_this: beats │
     │ bass · other     │               │ + downbeats →    │
     │ (MPS, CPU backup)│               │ bars, meter,     │
     └────────┬─────────┘               │ bar.beat, tempo  │
              ▼                         └────────┬─────────┘
     ┌──────────────────┐                        │
     │ 3 energy         │ RMS + LUFS curves,     │
     │                  │ stem activity,         │
     │                  │ kick/snare/hat hits    │
     └────────┬─────────┘                        │
      ┌───────┴────────┬───────────────┐         │
      ▼                ▼               ▼         ▼
 ┌──────────┐   ┌─────────────┐   ┌──────────────────────┐
 │ 5 lyrics │   │ 6 notes     │   │ 4 sections + moments │
 │ mlx-     │   │ Basic Pitch │   │ snapped to bar starts│
 │ whisper  │   │ + key       │   │ (needs rhythm too)   │
 └────┬─────┘   └──────┬──────┘   └──────────┬───────────┘
      └────────────────┼─────────────────────┘
                       ▼
             ┌──────────────────┐
             │ 7 assemble       │  reads cache/ (+ input.wav for the plot), ~1 s, runs every time
             └────────┬─────────┘
                      ▼
   analysis.json · curves.json · summary.md · overview.png
```

| # | Stage | What it does | Main library | Writes |
|---|-------|--------------|--------------|--------|
| 0 | `ingest` | decodes and trims the input to 44.1 kHz stereo WAV | ffmpeg | `input.wav`, `cache/ingest.json` |
| 1 | `stems` | separates vocals, drums, bass and other | Demucs `htdemucs` (or `htdemucs_ft`) | `stems/*.wav`, `cache/stems.json` |
| 2 | `rhythm` | finds beats, downbeats and tempo, then bars, meter and `bar.beat` | beat_this `final0` | `cache/rhythm.json` |
| 3 | `energy` | computes RMS/LUFS curves, stem activity and drum hits | numpy, librosa, pyloudnorm | `cache/energy.{json,npz}`, `cache/hits.json` |
| 4 | `sections` | segments the clip at bar starts, labels each segment's character, finds notable moments | numpy | `cache/sections.json` |
| 5 | `lyrics` | transcribes the active vocal regions with word timestamps | mlx-whisper `large-v3-turbo` | `cache/lyrics.json` |
| 6 | `notes` | transcribes notes on the bass and vocal stems, and estimates the key | Basic Pitch (CoreML), librosa | `cache/notes.json`, `cache/*.mid` |
| 7 | `assemble` | merges the caches into the final files | matplotlib | `analysis.json`, `curves.json`, `summary.md`, `overview.png` |

`sections`, `lyrics` and `notes` are *optional* stages. If one fails, the pipeline logs the error, carries on,
and `assemble` writes `null` for that part with a note in `meta.warnings`. `lyrics` skips itself cleanly
when the vocal stem is silent (instrumentals).

**`--no-vocals`** tells audio4llm the clip is instrumental, whatever the vocal stem contains:

- no vocal activity regions are cut, so there is no `vocal_entry` moment and sections don't list vocals
- Whisper is never loaded, so there is no language detection and no lyrics
- vocal note transcription is skipped
- `global.has_vocals` is false and `global.vocals_disabled` is true

Demucs still writes `stems/vocals.wav`, because it separates all four stems in one pass and skipping one
would save nothing. That stem's energy stays in `curves.json`: on an instrumental it is usually a lead synth
or bleed. Use the flag when Whisper hallucinates on an instrumental, or to save the lyrics stage's time.
`--vocals` switches back to auto-detection.

---

## Setup

Requirements:

- An Apple Silicon Mac. This was developed on an M1 Pro with 16 GB and macOS 15. `mlx-whisper` needs Apple Silicon and macOS 14 or newer.
- Python **3.12 or newer**. It is tested on **3.14.2**; see [Python version notes](#python-version-notes).
- `ffmpeg` on your `PATH` (`brew install ffmpeg`).

Install into an existing virtualenv (the project uses a pyenv virtualenv called `audio4llm`):

```bash
pyenv activate audio4llm            # or any Python >= 3.12 environment
pip install -e ".[dev]"             # audio4llm + all pinned dependencies + pytest

# Basic Pitch must be installed WITHOUT its dependencies. Its metadata requires tensorflow-macos on
# Python > 3.11, which has no build for 3.14, and we don't want TensorFlow anyway. Its real runtime
# dependencies (coremltools, onnxruntime, pretty_midi, mir_eval, ...) are already pinned in pyproject.toml.
pip install --no-deps "basic-pitch @ git+https://github.com/spotify/basic-pitch@fa5997af0a8210982619003269994a1be25eddf3"

audio4llm --help
```

### First run downloads

Model weights download on first use and are cached under `~/.cache`:

| Model | Used by | Size | Cache location | Notes |
|-------|---------|------|----------------|-------|
| `htdemucs` | stems | 84 MB | `~/.cache/huggingface/hub/models--adefossez--HTDemucs` | about 10 s |
| `htdemucs_ft` (optional) | stems | 4 × 84 MB | `~/.cache/huggingface/hub/models--adefossez--HTDemucs-ft` | only with `--demucs-model htdemucs_ft` |
| beat_this `final0` | rhythm | 81 MB | `~/.cache/torch/hub/checkpoints/beat_this-final0.ckpt` | the JKU server is slow and sometimes stalls (1–5 min); audio4llm downloads it with resume and retries and checks its sha256 |
| whisper `large-v3-turbo` (MLX) | lyrics | 1.6 GB | `~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo` | about 1 min (the first lyrics run took 58 s in total) |
| Basic Pitch | notes | <1 MB | ships inside the `basic-pitch` package | no download |

Expect the first run to take a few minutes, mostly downloads plus one-time numba/matplotlib cache
building. After that, a new 80 s clip takes **about 30 s** on an M1 Pro. Re-running on an analysed clip only
re-assembles the outputs, which takes about 1 s.

Measured stage times for the 80 s test clip (M1 Pro, warm model caches, default settings):

| ingest | stems | rhythm | energy | sections | lyrics | notes | assemble | total |
|---|---|---|---|---|---|---|---|---|
| 0.2 s | 11.1 s (MPS) | 3.0 s | 1.8 s | <0.1 s | 6.5 s | 3.6 s | 1.6 s | **27.8 s** |

On CPU, `stems` takes about 40–50 s instead of 11–13 s. `--demucs-model htdemucs_ft` runs four models, so expect
roughly 4× the stems time. Most of the lyrics time is loading the 1.6 GB model.

---

## Usage

```bash
audio4llm path/to/clip.mp3                          # whole file -> runs/clip/
python -m audio4llm path/to/clip.mp3                # same thing
audio4llm song.m4a --start 62 --duration 30         # analyse 1:02-1:32 only (trimmed into input.wav)
audio4llm song.mp3 --name song-chorus               # choose the run folder name
audio4llm song.mp3 --demucs-model htdemucs_ft       # better stems, ~4x slower
audio4llm song.mp3 --language es                    # skip language detection for lyrics
audio4llm song.mp3 --no-vocals                      # instrumental: skip vocal regions, Whisper and vocal notes
audio4llm song.mp3 --whisper-model mlx-community/whisper-small-mlx   # lighter lyrics model (481 MB)
audio4llm song.mp3 --curves-hz 30                   # curves.json at 30 Hz (only re-runs assemble)
audio4llm song.mp3 --force-stage lyrics             # redo one stage (plus anything downstream of it)
audio4llm song.mp3 --force                          # redo everything
audio4llm runs/song                                 # re-run a run folder with the settings it was made with
audio4llm runs/song --curves-hz 30                  # ...changing just one setting (only assemble re-runs)
audio4llm song.mp3 --device cpu --force-stage stems --force-stage rhythm   # redo the torch stages on CPU
audio4llm song.mp3 --no-plot                        # skip overview.png (--plot turns it back on)
```

At the end the CLI prints a table showing which stages ran or came from cache, and how long each took:

```
stage     status    seconds  note
ingest    cached          -
stems     ran          17.0  demucs_model: 'htdemucs_ft' -> 'htdemucs'
...
```

Notes on the run folder and settings:

- **Folder name.** The run folder is named after the input file (`song.mp3` → `runs/song/`, lower-cased and slugified). `song.mp3` and `song.m4a` therefore share a folder: analysing one after the other re-runs everything. Use `--name` to keep them apart.
- **Settings with a file input.** The settings are what the flags say, with defaults for any flag you leave out. Changing a setting re-runs only the stages that depend on it.
- **Settings with a run folder** (`audio4llm runs/song`). The settings saved by its last run (`cache/config.json`) are re-used, and only the flags you pass change them. `--start` and `--duration` are ignored, because `input.wav` is already trimmed.
- **`--device`** only affects stages that actually run. It is not part of the cache key, because it doesn't change the results, so add `--force-stage` to redo stages that are already cached.
- **Minimum length.** Clips must be at least 2 s long. Inputs over 90 s are analysed, with a warning.

Exit codes: `0` on success, `1` when a required stage failed, `2` when an optional stage failed (the outputs
are still written).

### Using the output with an LLM

Give the LLM `analysis.json` (or `summary.md` plus `analysis.json`) and ask for a visualizer that loads
`curves.json` at runtime and plays `input.wav`. Every timestamp is relative to `input.wav`, so playback and
analysis line up exactly, including when you used `--start`. A worked example is in
[docs/VISUALIZER_TEST.md](docs/VISUALIZER_TEST.md).

---

## Output files

| File | For | What's in it |
|------|-----|--------------|
| `analysis.json` | the LLM | Compact and readable, about 20–60 KB. `global` (tempo, meter, key, loudness, vocals, language), `grid` (every beat with `t`, `bar`, `beat` and `is_downbeat`, plus bars), `sections` (character, energy, active stems), `moments` (drop, stop, vocal entry, ...), `hits` (kick/snare/hat rows, every drum onset, and 16th-note pattern strings), `stems` (per-beat energy arrays and active regions), `notes` (bass/vocal note rows plus a per-bar summary), `lyrics` (lines and words), and a pointer to `curves.json`. Every event carries `t`, `bar` and `beat`. |
| `curves.json` | the generated code | Fixed-rate envelopes (default 20 Hz): energy for the mix and each stem, a smoothed mix, momentary LUFS, brightness, onset strength, and drum band envelopes. Not meant to be read by an LLM. |
| `summary.md` | humans and LLMs | A templated description: tempo, key, a section timeline, when stems come in, the loudest and quietest bars, drum patterns, bass roots and the first lyric lines. |
| `overview.png` | you | Waveform with sections and moments, energy lanes per stem, drum hits, a piano roll and lyric lines, with downbeats drawn through every panel. Use it to sanity-check a run. |
| `input.wav` | the video | The canonical audio. Mux it into the rendered video so the visuals and sound share a timeline. |
| `stems/*.wav` | debugging and reuse | Demucs stems (float32 WAV). |
| `cache/` | the pipeline | Per-stage intermediates plus `manifest.json` (what ran, with which settings, and how long it took). Also `bass.mid` and `vocals.mid`. |

Conventions used throughout (also embedded in `analysis.json` under `conventions`):

- Times are seconds from the start of `input.wav`, with 3 decimals.
- `bar` is 1-based from the first downbeat. Beats before it (a pickup) are bar 0.
- `beat` is 1-based and fractional: `1.0` is the downbeat and `2.5` is the "and" after beat 2.
- Energies are 0–1 *within the clip*. They are linear in dB over the top 30 dB of each stem, and 1 is that stem's loudest moment.

The full field reference, with a trimmed real example, is in [docs/OUTPUT_SCHEMA.md](docs/OUTPUT_SCHEMA.md).
`schema_version` (currently `1.0`) is bumped on any breaking change to the layout.

---

## Codebase tour

```
audio4llm/
  __init__.py          version and SCHEMA_VERSION
  __main__.py          `python -m audio4llm`
  cli.py               argparse front end: builds a Config, calls pipeline.run, prints the timing table
  config.py            Config dataclass (every setting that can change results) and constants (44.1 kHz, stem names)
  context.py           RunContext: run-folder paths, cache read/write helpers, torch device choice
  pipeline.py          Stage record, the caching rules (manifest.json), the run loop, timing, and run() for Python callers
  grid.py              BeatGrid: bar/beat numbering, 4/4-unless-clear meter inference, time -> (bar, beat) lookup
  audio.py             soundfile load/save and soxr resampling (no torchaudio I/O anywhere)
  jsonfmt.py           compact JSON writer (small objects one per line, number arrays inline, NaN -> null)
  drums.py             kick/snare/hat estimation by band-splitting the drum stem (called by the energy stage)
  key.py               Krumhansl-Kessler key estimate from chroma
  stages/
    __init__.py        STAGES: the ordered stage list
    ingest.py          0: ffmpeg decode/trim -> input.wav; warns on inputs over 90 s
    stems.py           1: Demucs get_model/apply_model called directly; MPS with CPU fallback
    rhythm.py          2: beat_this Audio2Beats on the mix, verified checkpoint download, grid -> rhythm.json
    energy.py          3: dense 100 Hz curves, LUFS and loudness range, stem activity detection, calls drums.py
    sections.py        4: bar-snapped segmentation, section character labels, moments
    lyrics.py          5: mlx-whisper on active vocal regions, hallucination filters, line splitting
    notes.py           6: Basic Pitch (CoreML, falls back to ONNX) on the bass/vocal stems, ghost-note filter, key
    assemble.py        7: writes the four final files from cache/
  report/
    __init__.py        (package marker)
    analysis.py        builds analysis.json and curves.json; puts bar/beat on every event; drum patterns
    summary.py         summary.md templates
    overview.py        overview.png (matplotlib)
scripts/
  make_test_clip.py    synthesizes the 80 s, 120 BPM test clip and ground_truth.json (uses macOS `say`)
  update_schema_example.py   refreshes the trimmed real example in docs/OUTPUT_SCHEMA.md from a run folder
  viz_prompt.py        copies a visualizer prompt (test, or --creative) filled in for a run folder to the clipboard
tests/
  conftest.py          fixtures: build the test clip, run the pipeline once per session
  test_end_to_end.py   the full pipeline against ground truth (marked slow)
  test_assemble.py     assemble on hand-made caches: no beat grid, no hits, failed stages, --no-plot
  test_cli.py          flag validation and run-folder settings
  test_grid.py, test_pipeline_cache.py, test_jsonfmt.py, test_energy.py, test_drums.py,
  test_sections.py, test_lyrics.py, test_notes.py, test_key.py   other fast unit tests (no models)
docs/
  OUTPUT_SCHEMA.md     field-by-field reference for analysis.json and curves.json
  VISUALIZER_TEST.md   (yours) how to test the output with an LLM-written visualizer
  VISUALIZER_CREATIVE.md   same foundations, but a prompt that gives the agent artistic freedom
```

To add a stage: write `stages/<name>.py` exposing `STAGE = Stage(name=..., run=..., outputs=..., deps=...,
config_keys=...)`, add it to `STAGES` in `stages/__init__.py`, and read its cache file in `report/analysis.py`.
The CLI's `--force-stage` choices come from `STAGES`. If the stage needs a new setting, add it to `Config`
and to `CONFIG_FLAGS` in `cli.py`.

---

## Caching and forced reruns

Every stage writes its outputs into the run folder and records an entry in `cache/manifest.json`. The entry
holds the stage version, the settings it used, the run tokens of its upstream stages, the time it finished
and its duration. On the next run a stage is **skipped** when:

1. all of its output files exist, and
2. the settings it depends on are unchanged (for example, `lyrics` depends on `--whisper-model`, `--language` and `--no-vocals`; `ingest` depends on the input file's sha1 plus `--start` and `--duration`), and
3. every upstream stage still has the same run token as when this stage ran.

So changing a setting, or re-running an upstream stage, automatically re-runs exactly the stages
downstream of it. Changing `--whisper-model` re-runs only `lyrics`. Changing `--demucs-model` re-runs
`stems`, `energy`, `sections`, `lyrics` and `notes`, but not `rhythm`, which works on the mix.

- `--force` re-runs everything.
- `--force-stage NAME` (repeatable) re-runs that stage, and therefore everything downstream of it.
- `assemble` always runs. It only reads `cache/` (plus `input.wav` for the plot's waveform), so it takes about a second. That's the part to iterate on when you change the output format: edit `report/*.py` and run `audio4llm runs/<clip>`.
- Deleting a file under `cache/` or `stems/` re-runs the stage that produced it.
- Passing a run folder (`audio4llm runs/song`) re-uses its `input.wav` and the settings saved in `cache/config.json`. `ingest` can't re-run without the original file.

Each stage's duration is logged as it finishes and summarised in the final table.

---

## Testing

```bash
python scripts/make_test_clip.py          # writes test_data/test_clip.wav + ground_truth.json (optional; tests do it)
python -m pytest -m "not slow"            # 160 fast unit tests, a few seconds
python -m pytest                          # everything, including the end-to-end run (first time: minutes)
```

The test clip is synthetic: 80 s at 120 BPM in A minor, with a quiet intro (hats and a pad), a build with a
snare roll and riser, a stop, a full section (kick, snare, hats, a bass line with known notes, chords), a
breakdown, and two phrases spoken by macOS `say`. The end-to-end tests run the real pipeline on it and check
against `ground_truth.json`:

- tempo within ±1 BPM
- at least 90% of beats and downbeats within 70 ms
- at least 80% kick recall in the full section, with at least 70% precision
- a clear energy contrast between the breakdown and the full section, and correct stem activity
- section boundaries within a bar of the truth, with the right characters
- the drop and the vocal entry found as moments
- at least 60% of the spoken words recovered, with timestamps inside the spoken phrases
- the bass root matching the chord root in at least 75% of bars
- a key estimate of A minor (or its relative, C major)

Where the tests are looser than ideal, it's for these reasons:

- **Key:** the clip's chords are Am | F | C | G with equal weight. A minor and C major contain exactly the same notes, so a chroma-profile estimator can't separate them well. It reports C major (runner-up A minor), and the test accepts either.
- **Hats are not tested.** Kick and snare are found perfectly on this clip, and hats are found perfectly where the groove plays. During the build, though, Demucs puts the noise riser into the drum stem, and only about 30% of those build hats are found.
- **Lyrics are spoken TTS**, not singing. Whisper recovers 20 of 20 words here, but the 60% threshold leaves room for the real gap between speech and sung vocals.
- **Beats:** after onset refinement, beats are a median 5 ms from the true onsets. In the percussion-less breakdown there's nothing to refine to, and beat_this's grid can sit about 50 ms off. That's why the tests use a 70 ms window.

---

## Known limitations and accuracy caveats

audio4llm has only been validated on the synthetic test clip. Real recordings are messier. Treat everything
labelled *estimate* as a strong hint, not ground truth, and check `overview.png`.

- **Stems (Demucs):**
  - Separation bleeds. Reverb tails, backing vocals and synths can leak into the wrong stem.
  - `other` is a catch-all.
  - A stem counts as *active* when it is within 18 dB of the local mix level, so quiet parts under a loud mix can be missed, and loud bleed can count as activity.
- **Beats and meter:**
  - beat_this runs on the full mix. The meter is assumed to be 4/4 unless the downbeat spacing clearly says otherwise. 6/8 and compound meters come out as their beat-level equivalent, with a denominator always of 4.
  - Swing and triplets are not detected; 16th-note snapping of `beat` positions deliberately leaves triplet positions alone.
  - `tempo_bpm` is one number. Use the beat times for rubato or tempo changes.
  - Half-time and double-time readings are possible on ambiguous material.
  - With no steady pulse at all (drones, ambient, silence) the grid is empty, every event is placed in bar 1, and `summary.md` says so.
  - Beats are pulled onto nearby onsets (±40 ms), so in sparse passages without transients they inherit the tracker's ±20–50 ms error.
- **Drum hits:**
  - These are heuristic band splits of the drum stem: kick below 150 Hz, snare 1–5 kHz, hats above 6 kHz. Toms, claps, rides and percussion land in whichever band fits.
  - Hats under risers or snare rolls are missed.
  - Bass bleed can occasionally produce kicks.
  - `strength` is relative within each type in this clip.
- **Sections and moments:**
  - These are rule-based labels on bar-level loudness, stem and hit features.
  - "full" means within 3 LU of the loudest section, so a pop verse may come out as `groove`.
  - The thresholds were tuned on the test clip and on simulated data, not on a corpus of real songs.
- **Lyrics (Whisper):**
  - Sung vocals, ad-libs, heavy effects and non-English mixes are harder than the clean speech in the tests.
  - Hallucination filters drop known filler phrases ("Thank you.", "Thanks for watching"), so a real lyric that is only "thank you" gets dropped too; it's listed under `dropped` in `cache/lyrics.json`.
  - Word timings are typically within 0.1–0.2 s, and the last word of a line can end early.
- **Notes (Basic Pitch):**
  - Note ends tend to run on to the next onset, so sync visuals to `t`, not `t + dur`.
  - Spoken or rapped vocals produce intonation "notes".
  - Polyphony in the vocal stem (harmonies) is transcribed, but crudely.
  - Basic Pitch's CoreML path is hard-coded to CPU inside basic-pitch, so it's no faster than ONNX.
- **Key:**
  - One estimate for the whole clip, so modulations aren't tracked.
  - The relative major and minor are easily confused (see `runner_up`).
  - Names use sharps only.
- **Loudness:** integrated LUFS follows BS.1770 (pyloudnorm). The loudness range is a close approximation of EBU R128 LRA.
- **Platform:**
  - Lyrics need Apple Silicon (MLX).
  - The pins target Python 3.14 on macOS arm64. On other platforms the torch, Demucs, beat_this, Basic Pitch (ONNX) and DSP parts should work, but they are untested there.

---

## Troubleshooting

**`ffmpeg not found on PATH`**: install it with `brew install ffmpeg` and check `ffmpeg -version`. ffmpeg
decodes every input format. `ffprobe`, which is installed alongside it, reads the input's duration.

**MPS problems.** Demucs and beat_this try Apple's GPU (MPS) first. If MPS raises an error or produces
non-finite output, the stage logs `falling back to CPU` and re-runs on the CPU. `meta.devices` in
analysis.json records what was actually used. To skip MPS entirely, pass `--device cpu`, plus
`--force-stage stems --force-stage rhythm` if those stages are already cached. Demucs then takes about 3–4×
longer (40–50 s instead of 13 s for 80 s of audio on an M1 Pro), with identical stems. `PYTORCH_ENABLE_MPS_FALLBACK` is not needed, because Demucs 4.1 already does its STFT
on the CPU.

**Model downloads.**

- **Slow or stalled beat_this download.** The JKU server is slow. The rhythm stage resumes and retries, and checks the file's sha256. If it still fails, it prints a `curl` command to fetch the file by hand:
  ```bash
  curl -L -C - --retry 8 -o ~/.cache/torch/hub/checkpoints/beat_this-final0.ckpt \
    https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt
  ```
- **"You are sending unauthenticated requests to the HF Hub."** Harmless. Setting `HF_TOKEN` gives faster downloads.
- **Working offline.** Once everything is downloaded, `HF_HUB_OFFLINE=1 audio4llm clip.mp3` avoids network calls.
- **Corrupted Hugging Face download.** Delete its folder under `~/.cache/huggingface/hub/` and run again.

**`notes FAILED ... No module named 'basic_pitch'`**: Basic Pitch must be installed separately with
`--no-deps` (see [Setup](#setup)). Because `notes` is an optional stage, the rest of the analysis is still
written. `pip check` will complain about basic-pitch's declared tensorflow/resampy requirements; that is
expected.

**Version warnings from coremltools** ("scikit-learn 1.9.1 is not supported", "Torch version 2.14.1 has not
been tested") only concern coremltools' model converters, which aren't used here. The notes stage captures
them and logs them at debug level.

**Lyrics are `skipped`** when the vocal stem never rises above the activity threshold, as in instrumentals,
or when `--no-vocals` is set. That is intended. If a quiet vocal is missed, look at the vocals lane in
`overview.png`. If Whisper invents lyrics on an instrumental, because a lead synth landed in the vocal stem,
re-run with `--no-vocals`.

**The analysis looks stale** after you changed code in a stage: the cache only knows about settings, not
code. Run `--force-stage <stage>` (or `--force`), or bump that stage's `version=` in its `STAGE` record.

**Inputs longer than 90 s** work but log a warning. analysis.json grows and every stage gets slower. Use
`--start/--duration` to pick an excerpt.

---

## Python version notes

The brief targeted Python 3.11, but the `audio4llm` environment is **3.14.2**, so everything here is pinned
and tested on 3.14. These are the incompatibilities found and how each was handled:

| Issue on 3.14 | Resolution |
|---|---|
| `basic-pitch` 0.4.0 requires `tensorflow-macos` on Python > 3.11, which has no 3.14 build, so `pip install basic-pitch` fails to resolve | install it `--no-deps` from a pinned commit (`fa5997a`, which also fixes the end-of-audio cut-off in 0.4.0); its real runtime deps are pinned in `pyproject.toml`; TensorFlow is never installed |
| `coremltools` 9.0 (stable) has wheels only up to 3.13 | pinned `coremltools==9.1.dev1`, a pre-release with a 3.14 arm64 wheel; the notes stage falls back to ONNX Runtime automatically if CoreML fails, and both give identical notes |
| `resampy<0.4.3` (basic-pitch's pin) imports `pkg_resources`, which is gone from current setuptools | pinned `resampy==0.4.3`, which works with basic-pitch |
| Demucs 4.0.1 on PyPI was archived and used torchaudio I/O | `demucs==4.1.0` from the maintained fork (adefossez/demucs): pure Python, safetensors weights, no torchaudio in inference. audio4llm calls `get_model`/`apply_model` directly and uses soundfile for I/O anyway. |
| torchaudio stopped being released in lockstep with torch | `torchaudio==2.11.0` (the last release, built on torch's stable ABI) with `torch==2.14.1`; it is only used for beat_this's mel spectrogram |
| librosa 1.0 needs Python ≥ 3.12 | `requires-python = ">=3.12"`. These exact pins won't install on 3.11 (older librosa and numpy would be needed there). |

---

## Next steps

- **Validate on real songs:**
  - Build a small set of hand-annotated clips (beats, sections, hits, lyrics) across genres.
  - Re-tune the activity, section and hit thresholds against it.
  - Turn it into a regression suite alongside the synthetic clip.
- **Harmony:** add per-bar chord estimates (chroma templates or a small chord model), with the key estimate per section instead of per clip.
- **Rhythm detail:**
  - swing and triplet detection
  - compound meters (6/8)
  - a tempo map for rubato
  - optional DBN-free beat smoothing for very sparse intros
- **Drums:** an optional learned drum transcriber (more classes: toms, cymbals, claps), with the band-split estimates kept as the fallback.
- **Lyrics:**
  - optional forced alignment to user-supplied lyrics (much better than ASR on sung vocals)
  - syllable- or phoneme-level timing for lip-sync-style effects
- **Output:** an optional `--preview` that renders a quick MP4 with beat and hit flashes over `input.wav`, to check sync by eye and ear.
- **Visualizer loop:** close the loop described in [docs/VISUALIZER_TEST.md](docs/VISUALIZER_TEST.md). Feed `analysis.json` to an LLM, render the result, and use failures to refine the schema (bump `schema_version` when it changes).
