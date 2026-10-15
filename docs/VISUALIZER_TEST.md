# Visualizer test: does audio4llm's output actually work?

This is a **prompt-driven test harness**. You point Claude Code at an `analysis.json` that audio4llm produced, paste the prompt below, and it builds a beat-synced animation, checks its own sync, and hands back a finished MP4 with no check-ins. If the video lands on the beats, kicks, quiet parts and lyrics, the analysis is good enough for an LLM to work from. If it doesn't, the report tells you where the data let it down.

Put this file at `docs/VISUALIZER_TEST.md` in the audio4llm repo (the README links there).

---

## 1. One-time setup

| What | Why | Install / check |
|---|---|---|
| Node.js 20+ | Runs the render page and the Playwright capture script | `brew install node` → `node -v` |
| ffmpeg | Encodes frames and muxes the audio | already installed → `ffmpeg -version` |
| Playwright + Chromium | Headless browser that renders three.js / canvas frame by frame | the agent runs `npm i -D playwright && npx playwright install chromium` itself (~150MB) |
| Blender 4.x *(optional)* | Only needed if you want a Blender-rendered 3D variant | `brew install --cask blender`, then `ln -s /Applications/Blender.app/Contents/MacOS/Blender /opt/homebrew/bin/blender` → `blender -v` |
| audio4llm env | Python helpers (the agent may use it for frame checks or plots) | `pyenv activate audio4llm` |

No API keys or accounts are needed. Everything renders locally.

Free disk space: about 2GB is plenty for the browser route, because frames are piped straight into ffmpeg. A Blender render writes frames to disk, which can take several GB at 1080p, so the prompt tells the agent to clean those up.

## 2. Running a test

```bash
cd ~/path/to/audio4llm
pyenv activate audio4llm
audio4llm path/to/clip.mp3                 # produces runs/<clip>/analysis.json etc.
claude                                     # launch from the repo root
```

Then paste the prompt from section 4, filling in the `<clip>` name. `python scripts/viz_prompt.py runs/<clip>` copies it to the clipboard already filled in, ending with a "Special instructions" heading you can type after. Output goes to `viz-runs/<clip>/<run-id>/`, which is gitignored.

Start with the synthetic test clip (`runs/test_clip/`). Its ground truth is known, so if the video is out of sync there, the bug is in the visualizer, not the analysis.

## 3. Letting it run hands-off

The agent will run `npm`, `npx`, `node`, `ffmpeg`, `ffprobe`, `python`, and possibly `blender`, many times over. To stop it waiting on approvals, pick one:

- **Allowlist (recommended).** Add `.claude/settings.local.json` to the repo:
  ```json
  {
    "permissions": {
      "allow": [
        "Bash(npm:*)", "Bash(npx:*)", "Bash(node:*)",
        "Bash(ffmpeg:*)", "Bash(ffprobe:*)",
        "Bash(python:*)", "Bash(blender:*)",
        "Bash(mkdir:*)", "Bash(ls:*)", "Bash(open:*)"
      ]
    }
  }
  ```
  Then launch with `claude --permission-mode acceptEdits` so file writes don't prompt either. If a rule doesn't seem to match, check `/permissions` inside Claude Code; the rule syntax has changed between versions.
- **Fully unattended.** `claude --dangerously-skip-permissions`. This is fine in a repo you don't mind it touching, but it skips every guard, so don't use it in directories holding things you care about.

## 4. The prompt

Copy everything inside the block and replace `<clip>`.

````text
Build a test visualization for the audio clip analyzed in `runs/<clip>/`. The point is to test whether audio4llm's output gives an LLM enough to make a convincingly music-synced animation. Work fully autonomously from start to finish: make every visual decision yourself, don't ask me for approval, and deliver a finished MP4.

## Inputs (read these first)
- `README.md` and `docs/OUTPUT_SCHEMA.md`, to understand the data
- `runs/<clip>/analysis.json` (main data), `runs/<clip>/curves.json` (dense envelopes), `runs/<clip>/summary.md`, and `runs/<clip>/overview.png` (look at it)
- Audio for the final video: the `input.wav` referenced in `analysis.json` (`source`). Use exactly that file, so the video shares the analysis timeline.

Do not modify audio4llm's code or the files in `runs/`. If the analysis looks wrong or is missing something you needed, work around it and record that in the report. That feedback is part of what this test is for.

## Step 1: Write a visual score
Before writing any code, write `VISUAL_SCORE.md`. It should set a concept and palette, then map musical features to visual behavior, section by section. Every one of these must drive something visible:
- beats → a pulse; downbeats and bar starts → a stronger accent (a cut, a color shift, a camera move)
- drum hits → distinct, recognizable responses for kick, snare and hats, scaled by hit strength
- per-stem energy (`curves.json`) → continuous motion and intensity of the elements tied to that stem
- loud vs quiet sections and `moments` (drops, first vocal, stops, builds) → clear changes in density, brightness, scale and camera; quiet parts must look calm
- notes → melodic elements (bass notes and vocal or lead notes mapped to height, hue or position by pitch)
- lyrics, if present → kinetic typography timed per word, appearing as each word is sung
Make 2D, 3D, or both, whichever serves the music best. If there are no vocals or no notes, adapt: don't fake them.

## Step 2: Build a deterministic renderer
The core rule is that every frame must be a pure function of time `t`. Never use real-time playback, `requestAnimationFrame` timing, `setTimeout`, or the wall clock to drive animation. Use a seeded PRNG instead of `Math.random`. Real-time screen capture drifts; deterministic stepping stays in sync.

Default stack (browser):
- One HTML page using three.js for 3D and/or a 2D canvas layer. anime.js is fine for easing and timelines only if you drive them with `timeline.seek(t*1000)`.
- The page exposes `window.renderFrame(t)` (async, resolves when the frame is fully drawn).
- Create the WebGL renderer with `preserveDrawingBuffer: true`. Wait for fonts and textures to load before the first frame.
- A Node script uses Playwright (headless Chromium) to load the page at a fixed viewport. For frame `n` it calls `renderFrame(n / fps)`, grabs the frame (canvas `toDataURL`/`toBlob`, or `page.screenshot`), and pipes it into ffmpeg (`-f image2pipe`). The final mux uses the input.wav with `-c:v libx264 -pix_fmt yuv420p -crf 18 -c:a aac -b:a 192k -shortest`.
- If headless WebGL fails (the context is null or frames are black), retry with the flags `--use-angle=swiftshader --enable-unsafe-swiftshader`, or use headed mode.
- Default output is 1920×1080 at 30 fps. Total frames = ceil(duration × fps).

Blender (`blender -b -P script.py`, Eevee) is allowed only for an optional extra 3D variant once the browser version is done and verified. Keyframe it from the JSON on the same time base, render frames to disk, mux with ffmpeg, then delete the frames.

## Step 3: Verify sync before the final render
1. Render `debug_sync.mp4` at 640×360 with an overlay showing `t`, `bar.beat`, the current section, a flash per hit type, and the current lyric word.
2. With ffprobe, confirm the video and audio durations match within one frame.
3. Use ffmpeg `-ss` to extract single frames at about 6 kick-hit times, 4 downbeats, 2 points in the quietest section, 2 in the loudest, and 3 lyric word onsets, plus a few off-beat times as controls. Look at each one and confirm the visual state matches what the data says should be happening at that moment. Fix any mismatch, whether it's an off-by-one-frame error, a timing offset, or easing that lags the hit, and re-check.

## Step 4: Final render and self-review
1. Render `final.mp4` at 1920×1080 / 30fps.
2. Make `contact_sheet.png` (for example, `ffmpeg -i final.mp4 -vf "fps=1,scale=320:-1,tile=8x10" -frames:v 1 contact_sheet.png`) and look at it. Sections should be visibly distinct, quiet parts calmer, and the drop or climax unmistakable.
3. Spawn a fresh review subagent that hasn't seen your code. Give it `VISUAL_SCORE.md`, `analysis.json`, the contact sheet and the extracted check frames. Ask it whether the video follows the music and the score, and where it doesn't. Fix whatever is worth fixing, re-render, and stop after at most 3 rounds.

## Using subagents
Use subagents where they help:
- research (check the current Playwright, three.js and ffmpeg piping APIs if unsure)
- parallel implementation of separate visual layers against the shared `renderFrame(t)` contract
- the independent review in step 4

## Deliverables in `viz-runs/<clip>/<YYYYMMDD-HHMM>/`
- `final.mp4` (and `final_blender.mp4` if you did the optional variant)
- `debug_sync.mp4`, `contact_sheet.png`, `checks/` (the extracted verification frames)
- `VISUAL_SCORE.md`
- `src/` (all page and render code, plus a one-line command to re-render)
- `REPORT.md`, covering: what drives what; what was verified and how; render time; known issues; and **feedback on audio4llm's output**: wrong or missing data, fields you didn't use, and fields you wished existed

No git commits. When done, run `open final.mp4` and give me a 5-line summary.
````

## 5. Judging the result

Watch `final.mp4` with sound, ideally twice: once looking at the whole frame, and once watching only the drum reactions.

- [ ] Kicks and snares land on the hit, not a few frames late
- [ ] Bar and downbeat accents feel musical, not random
- [ ] Quiet parts look quiet, and the loudest section or drop clearly peaks
- [ ] Section changes coincide with visual changes
- [ ] Lyrics appear as the words are sung (if any)
- [ ] Bass and melody notes visibly move something that follows the pitch

Then read `REPORT.md`. Its "feedback on audio4llm" section is the main output of the test: feed it back into the analyzer, for example missing fields, mislabeled hits, or noisy sections.

## 6. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Everything is consistently early or late | The muxed audio isn't the analyzed `input.wav`, or there's a frame-index off-by-one |
| Sync drifts over time | Animation is being driven by real time, or the fps is mismatched between render and encode |
| Black or empty frames | Headless WebGL failed; use the swiftshader flags or headed mode, and check `preserveDrawingBuffer` |
| The render is very slow | Normal-ish: expect 3–10 min for 80s at 1080p30. Use 720p while iterating |
| Hits are visible but feel mushy | Attack easing is too slow; hits need near-instant attack and a longer decay |
| `blender: command not found` | Add the symlink from section 1 |

## 7. Variations to try after the first pass

Append one of these to the prompt:
- "Make it purely 2D, flat vector, as if for a lyric video."
- "Make it a 3D scene with a moving camera; the camera path should follow the sections."
- "Do the Blender variant too."
- "Render a vertical 1080×1920 version for phones."

Running the same clip with two different styles is a good test of whether the data carries enough meaning on its own, rather than only enough for one look.

For a version that keeps this toolset but gives the agent artistic freedom instead of a coverage checklist, see [VISUALIZER_CREATIVE.md](VISUALIZER_CREATIVE.md) (`python scripts/viz_prompt.py --creative runs/<clip>`).
