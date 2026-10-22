# Context-aware robot — editable production package

Saved 20 October 2026. **Production is on hold at the user's request.** Completed generation outputs are cached and must be reused. No completed source asset needs regeneration. No plan upgrade, credit purchase, video-generation POST or lip-sync-generation POST was made.

## Saved assets

- Four continuity reference PNGs: worker, robot, corridor and blank monitor close-up in `refs/`; generation prompts in `reference_prompts.md`.
- Six ElevenLabs dialogue WAVs, their original MP3s, character alignment and cached responses in `dialogue/`.
- Selected stock voice names, IDs and settings in `voice_selection.json`; four existing stock voice previews in `voices/`. Voice IDs were also configured locally; the API key is excluded from this package.
- Corridor ambience and rubber-wheel source WAVs/MP3s in `soundscape/`.
- A **60.000-second audio edit**, `audio/context_aware_robot_60s_mix.wav`, plus `audio/soundbed_60s.wav`. Local mixing preserved all source audio. Measured mix: −16.06 LUFS integrated, −1.50 dBTP, stereo 48 kHz / 24-bit PCM. Listening review remains pending.
- Two padded, sample-aligned audio inputs for future worker lip-sync in `dialogue/lipsync_inputs/`; no lip-sync video exists.
- Editable `timeline.json`: 60 seconds at 24 fps; twelve planned shots, each no longer than eight seconds. Nine subtitle cues in `subtitles.srt` use the saved speech alignments. `subtitles_provisional.srt` is an archived earlier timing draft.
- `film_script.md`, `shot_prompts.md`, source-backed `implementation_notes.md`, `review_checklist.md` and reproducible editing/generation scripts.
- Monitor design previews in `review/`, clearly labeled as graphics/stills rather than footage. The early robot composite proof uses the former tall UI; the current compositor adapts to wide screens. The cursor is now drawn geometrically to avoid missing font glyphs.

**No source motion clips or final MP4s exist.** `source_clips/` is intentionally empty. The requested `context_aware_robot_60s.mp4` and `context_aware_robot_60s_subtitled.mp4` remain pending actual cinematic footage, measured screen corners, accurate worker lip-sync and full film review. No slideshow or still-image motion substitute was produced.

## API failure and website eligibility

The exact failing read-only permission probes were:

```text
GET https://api.elevenlabs.io/v1/flows/video?page_size=1
HTTP 402
{"status":"paid_plan_required","message":"This endpoint requires a Pro plan or above."}

GET https://api.elevenlabs.io/v1/flows/image?page_size=1
HTTP 402
{"status":"paid_plan_required","message":"This endpoint requires a Pro plan or above."}
```

These are listing checks, not failed generation submissions. Credentials and request headers are excluded. Raw structured evidence is in `research/api_access_evidence.json`.

**Creator has published website video eligibility.** ElevenLabs distinguishes a paid-plan website requirement from a Pro-or-above API requirement. Its pricing documentation says products use one shared monthly credit pool. The verified Creator account can therefore use its existing credits for website video, subject to the selected model's actual quote and availability. The API rejection does not establish a website video restriction. See the [official capability overview](https://elevenlabs.io/docs/overview/capabilities/image-video) and [pricing](https://elevenlabs.io/pricing).

The final read-only balance check reported **129,762 included credits remaining**, with usage-based overage disabled. Consumption changed from 805 before production to 1,238 afterward, a net account delta of 433 credits; the deliberately conservative audio reservation was 1,375 within a 1,500-credit local cap. This is not a video budget or a guarantee that the full film fits the balance. Exact video/lip-sync cost depends on settings and retries.

A signed-in ElevenLabs website session was unavailable to these tools. No website Generate button was pressed, no account-specific model quote was inspected, and no successful website generation is claimed. The published entitlement is verified; the exact workspace/model state is unobserved. Website evidence and links are saved in `research/website_video_eligibility.md`. No purchase is needed merely to inspect a quote in the existing [Image & Video web app](https://elevenlabs.io/app/image-video).

## Resume the edit

Use Python 3.12 or later. From the repository root, the existing `video_demo/.venv/` is ready locally. For a fresh environment, install `requirements-production.txt` into a virtual environment. The archive excludes the machine-specific virtual environment and redundant working renders; all completed sources and the finished audio edit are included.

```sh
python3 -m venv video_demo/.venv
video_demo/.venv/bin/python -m pip install -r video_demo/requirements-production.txt
video_demo/.venv/bin/python video_demo/build_film.py --validate
```

Validation currently refuses export because footage, monitor corners and lip-sync review are absent. After actual shots are available, place them at the paths in `timeline.json`, measure stable screen corners on normalized 1920×1080 frames, and verify both speaking shots against the padded inputs. Moving screens need real tracking; fixed guessed overlays are unsupported. Review exact displayed video prices before any future generation. Existing audio/image sources must be reused.

```sh
video_demo/.venv/bin/python video_demo/build_film.py --write-srt
video_demo/.venv/bin/python video_demo/build_film.py --validate
video_demo/.venv/bin/python video_demo/build_film.py --render
```

`prepare_audio_package.py` performs local assembly without APIs and reuses existing outputs. `generate_assets.py` and `generate_soundscape.py` are preserved for reproducibility/recovery; **do not invoke their generation commands to resume this paused package**. Their caches/ledger prevent repeated paid requests. `collect` can recover known cached jobs without buying anything. Read-only capability checks are separate.

The output film is designed for 1920×1080, 16:9, constant 24 fps, H.264/AAC, clean and English-subtitled delivery. A two-second internal moving test-pattern encode verified monitor composition, subtitles, H.264/AAC format and complete decoding. That test was not film footage and was removed; its metadata report remains in `review/`.

## Technical distinction and remaining review

The repository implements synthetic longitudinal car-following with five physical and five context observation values, context-conditioned PPO and context-dependent reward shaping. It does not implement the proposed camera, facial/behavioral cue perception, live LLM, uncertainty schema, speech interaction or hospital robot. Comfort/efficiency preference features are policy inputs, not direct reward multipliers. Changing a fixed policy's reward settings alone does not change its action. Independent motion safeguards remain outside the proposed LLM's discretion. The hospital behavior must remain explicitly illustrative.

Source controller/training files were preserved. No model checkpoint, real robot behavior, stress-recognition accuracy or hospital deployment was validated.

Auditory review was unavailable in this runtime; saved voice previews and final speech were checked as media/timing/level assets, not certified by listening. Natural delivery, unwanted content in generated ambience, character consistency in motion, physical wheel/gap behavior, lip-sync and full visual/audio polish remain unreviewed. Reference images are 1672×941; final footage should request native 1080p where supported. A 720p lip-sync source would require upscaling and delivery-scale review. The final review checklist is deliberately incomplete.

End captions remain:

```text
Proposed perception–LLM–PPO architecture
Robot behavior illustrated; real-world validation required
```
