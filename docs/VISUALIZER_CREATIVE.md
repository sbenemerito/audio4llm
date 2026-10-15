# Creative visualizer: a music video, not a meter

[VISUALIZER_TEST.md](VISUALIZER_TEST.md) checks *coverage*: its prompt requires every beat, drum hit, stem, note
and lyric to drive something visible, which is the right shape for testing the data but tends to produce a
literal "audio visualizer". This prompt keeps the same foundations (inputs, deterministic renderer, toolset,
sync checks, deliverables) and hands the creative decisions to the agent: it reads the analysis as a score,
picks an idea for this specific song, and syncs only what serves that idea.

Setup, permissions and troubleshooting are the same as sections 1, 3 and 6 of VISUALIZER_TEST.md.

## Running it

```bash
python scripts/viz_prompt.py --creative runs/<clip>   # copies this prompt, filled in, to the clipboard
claude                                                 # from the repo root, then paste
```

The copied prompt ends with a "Special instructions" heading, so you can paste and keep typing, for example
"only 2D, flat vector, for a lyric video".

## The prompt

`scripts/viz_prompt.py --creative` copies the block below with `<clip>` filled in.

````text
Make a music video for the audio clip analyzed in `runs/<clip>/`. This is not a test of data coverage; it's a piece of art. Treat audio4llm's analysis the way an animator treats an exposure sheet: it tells you exactly when things happen and how the music is shaped, and you decide what the piece is about and how it looks. Work fully autonomously from start to finish: make every creative decision yourself, don't ask me for approval, and deliver a finished MP4.

## Inputs (read these first)
- `README.md` and `docs/OUTPUT_SCHEMA.md`, to understand the data
- `runs/<clip>/analysis.json` (main data), `runs/<clip>/curves.json` (dense envelopes), `runs/<clip>/summary.md`, and `runs/<clip>/overview.png` (look at it)
- Audio for the final video: the `input.wav` referenced in `analysis.json` (`source`). Use exactly that file, so the video shares the analysis timeline.

Do not modify audio4llm's code or the files in `runs/`. If the analysis looks wrong or is missing something you needed, work around it and record that in the report.

## Step 1: Find the idea
Before writing any code, write `CONCEPT.md`:
- **What you hear.** Genre, mood, energy arc, what the lyrics are about (read `lyrics` in full, if present), and what's distinctive about this track. Ground each claim in the data: sections, moments, loudness, key, tempo, which stems carry the track.
- **Three different concepts**, a sentence or two each (for example a narrative, an abstract or generative piece, a typographic piece, one evolving image, a camera journey), then which one you're choosing and why it fits this song better than the others.
- **The look.** Medium (2D, 3D, shader, mixed), palette, texture, typography, camera language, and references described in words.
- **The arc.** What happens visually in each section, and the 3–6 moments that must land exactly on the music (a drop, the first vocal, a stop, the last hit). These moments are the sync contract.
- **What you're deliberately not syncing.** You don't have to visualize every kick, stem or note. Use the data that serves the idea and ignore or reinterpret the rest. Restraint is allowed: an image held through a quiet section can hit harder than a pulse on every beat.

Avoid the default audio-visualizer vocabulary (spectrum bars, pulsing circles, waveform lines, particles that just scale with loudness) unless you subvert it on purpose. The goal is something that looks like a person made it for this specific song.

It must still feel locked to the music. When something changes on screen, it should change on a musical boundary (a beat, bar, phrase, section, moment or sung word), not at an arbitrary time, and the pace of motion should feel tied to the tempo even where it isn't flashing on the beat.

## Step 2: Build a deterministic renderer
The core rule is that every frame must be a pure function of time `t`. Never use real-time playback, `requestAnimationFrame` timing, `setTimeout`, or the wall clock to drive animation. Use a seeded PRNG instead of `Math.random`. Real-time screen capture drifts; deterministic stepping stays in sync.

Toolset: anything that renders locally and deterministically. Install libraries with npm and load them from disk, so renders don't depend on the network.
- Browser: three.js (including post-processing and custom GLSL shaders), 2D canvas, SVG, p5.js in instance mode with `noLoop()` and manual `redraw()`, d3 for geometry and layout, anime.js or GSAP timelines driven only by `seek(t)`.
- Fonts: bundle font files locally (for example an `@fontsource/*` package) and wait for them to load before the first frame.
- Blender (`blender -b -P script.py`, Eevee or Cycles) as the main renderer if the concept is genuinely 3D or physical and benefits from it. Keyframe it from the JSON on the same time base, render frames to disk, mux with ffmpeg, then delete the frames.
- Python in the audio4llm env (numpy, matplotlib) for preprocessing the data or generating textures.
- No paid APIs, accounts or image-generation services. Every pixel comes from code you write.

Browser render contract:
- The page exposes `window.renderFrame(t)` (async, resolves when the frame is fully drawn).
- Create any WebGL renderer with `preserveDrawingBuffer: true`. Wait for fonts and textures to load before the first frame.
- A Node script uses Playwright (headless Chromium) to load the page at a fixed viewport. For frame `n` it calls `renderFrame(n / fps)`, grabs the frame (canvas `toDataURL`/`toBlob`, or `page.screenshot`), and pipes it into ffmpeg (`-f image2pipe`). The final mux uses the input.wav with `-c:v libx264 -pix_fmt yuv420p -crf 18 -c:a aac -b:a 192k -shortest`.
- If headless WebGL fails (the context is null or frames are black), retry with the flags `--use-angle=swiftshader --enable-unsafe-swiftshader`, or use headed mode.
- Default output is 1920×1080 at 30 fps. Total frames = ceil(duration × fps).

## Step 3: Check that the key moments land
1. Render `debug_sync.mp4` at 640×360 with an overlay showing `t`, `bar.beat`, the current section, and the current lyric word.
2. With ffprobe, confirm the video and audio durations match within one frame.
3. Use ffmpeg `-ss` to extract the frame just before and the frame at each sync-contract moment from `CONCEPT.md`, plus each section boundary and, if you use lyrics on screen, 3 word onsets. Look at each pair and confirm the change happens on the right frame. Fix any offset, off-by-one-frame error, or easing that lags, and re-check.

## Step 4: Final render and critique
1. Render `final.mp4` at 1920×1080 / 30fps.
2. Make `contact_sheet.png` (for example, `ffmpeg -i final.mp4 -vf "fps=1,scale=320:-1,tile=8x10" -frames:v 1 contact_sheet.png`) and look at it. Does the piece have an arc? Would a still from it be recognizable as this piece rather than a generic visualizer?
3. Spawn a fresh review subagent that hasn't seen your code. Give it `CONCEPT.md`, `summary.md`, `analysis.json`, the contact sheet and the check frames. Ask it to critique like a music video director: does it serve the song, is it distinctive and coherent, do the key moments land, and which section is weakest. Fix whatever is worth fixing, re-render, and stop after at most 3 rounds. Keep the idea and fix the execution; don't let the review sand it down into something safe.

## Using subagents
Use subagents where they help:
- research (check the current Playwright, three.js, shader or ffmpeg APIs if unsure)
- parallel implementation of separate scenes or layers against the shared `renderFrame(t)` contract
- the independent critique in step 4

## Deliverables in `viz-runs/<clip>/<YYYYMMDD-HHMM>-creative/`
- `final.mp4`
- `debug_sync.mp4`, `contact_sheet.png`, `checks/` (the extracted verification frames)
- `CONCEPT.md`
- `src/` (all page and render code, plus a one-line command to re-render)
- `REPORT.md`, covering: the concept in a paragraph; what the data drives and what you chose to ignore; what was verified and how; render time; known issues; and **feedback on audio4llm's output**, especially anything that would have helped you make creative decisions (mood, phrasing, lyric meaning, energy descriptors) that the analysis lacks

No git commits. When done, run `open final.mp4` and give me a 5-line summary.
````

## Judging the result

Watch `final.mp4` with sound.

- [ ] It has an idea: you could describe it in a sentence, and it isn't "bars that bounce"
- [ ] The sync-contract moments in `CONCEPT.md` land on the music
- [ ] Visual changes happen on musical boundaries, not at random times
- [ ] The arc follows the song: quiet parts breathe, the peak feels like a peak
- [ ] It looks made for this track; running a different clip should give a different piece

Running the same clip through both prompts is a useful comparison: the test prompt shows whether the data is
complete, and this one shows whether it carries enough meaning to make choices from.
