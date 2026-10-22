# Context-aware assistance — production script

This is a 60-second concept film about a **proposed hospital application** of simulation work. The repository's existing context-conditioned PPO implementation concerns synthetic longitudinal car-following. Camera perception, a live LLM, uncertainty estimates, spoken confirmation, hospital motion and hardware integration are proposed. See [implementation_notes.md](implementation_notes.md) for source evidence. The LLM supplies context summaries, never direct motor commands or hidden chain-of-thought.

**Production status:** speech is generated and measured. The 60-second timeline below accommodates the actual six ElevenLabs WAVs. `subtitles.srt` uses returned character alignment for phrase boundaries; its visual integration awaits footage. `subtitles_provisional.srt` is a superseded historical draft, not a delivery file. All source shots fit within an 8-second generation limit, with no frozen filler. Do not create delivery MP4s without real generated/camera footage and reviewed worker lip-sync.

| Target | Picture and deterministic graphics | ElevenLabs speech |
|---|---|---|
| 00–06 | Tracking wide, teal-scrubbed worker walking down a bright hospital logistics corridor; white-and-teal supply robot follows. Small opening label: **Concept demonstration**. Title: **Context-aware assistance**. | Robot VO: “What if a robot could adapt to signs that someone needs a gentler pace?” |
| 06–09 | Worker slows, exhales, briefly rubs a temple. Gentle brow furrow and heavy eyelids remain ambiguous cues. No spectacle or diagnosis. | Natural pause; quiet corridor/wheels. |
| 09–14 | Five-second stable robot-monitor close-up. Persistent top-left **PERCEPTION**. Words appear beneath: “Visible cues: tired eyes, furrowed brow” / “Possible fatigue or stress” / “Estimate uncertain”. Completed copy holds before the cut. | No speech. |
| 14–21 | Stable monitor, with actual worker/corridor behind it. **ANALYSING**: “Checking motion and surrounding space” / “Person has slowed” / “More space is available”. | No speech. |
| 21–26.5 | Hold robot display for 5.5 seconds. **REASONING**: “LLM context interpretation” / “Possible need: less intrusive assistance” / “Proposed response: smoother motion, more space”. All words finish before a readable hold and worker cut. | Robot at 21.100–23.979: “Would you prefer a little more space and a gentler pace?” |
| 26.5–29 | Medium worker response, accurately lip-synced to the selected worker WAV. The worker calmly confirms a preference. | Worker at 26.625–28.761: “Yes, please. It's been a long shift.” |
| 29–37 | Eight-second stable monitor: **DECIDED**. “Context confirmed by user” / “Send context to control module” / “Prioritise smoothness and comfortable spacing” / “Safety limits remain active”. Compact architecture strip and annotations: **Context-conditioned control**; **Training: context-dependent reward shaping**; **Hospital application proposed**. | Robot at 30.000–33.019: “Understood. I'll give you more room and move more gently.” |
| 37–40 | Stable monitor cut: **WORKING**. Compact words appear: “Proposed motion” / “Smooth motion” / “Comfortable following distance”. Let the completed copy hold before the motion shot. | Quiet corridor; no speech. |
| 40–47 | Seven-second continuous side view. Worker resumes at an agreed pace. Robot eases off gradually, separation increases, then stabilizes; wheels stay planted. The gap change remains visible in one coherent actual motion shot. | Quiet wheels and corridor. |
| 47–50 | Worker response in the same corridor and lighting; subtle smile, accurate lip-sync. | Worker at 47.200–48.872: “Thank you. That's much better.” |
| 50–56 | Continue real motion footage. Stable screen composition can reprise the completed **WORKING** summary while the worker and robot continue coherently. No false live dashboard. | Robot VO at 50.400–55.137: “Human context informs the controller's priorities, while safety limits remain active.” |
| 56–60 | Wide continuation. **Perceive context. Adapt assistance.** Smaller: **Proposed perception–LLM–PPO architecture** and **Robot behavior illustrated; real-world validation required**. | Let ambience breathe; no extra dialogue. |

Every monitor close-up retains **Illustrative system summary** at its bottom. “Context confirmed” refers to the person's requested assistance preference; no clinical or emotional diagnosis is confirmed. Independent physical safeguards are proposed outside the LLM's discretion. Context-dependent reward shaping is a learning mechanism; changing reward weights alone does not alter a fixed policy's inference action. The immediate behavior in this concept uses the proposed extension of a policy trained with context inputs. No arbitrary context-to-motion guarantee is claimed.

The complete architecture for explanatory materials is:

```text
PROPOSED HOSPITAL EXTENSION
Camera observations → Visible-cue perception → LLM context interpretation
→ Structured context + uncertainty → Context-conditioned PPO control
                                      ↓
                            Independent motion safeguards

SIMULATION ANALOGUE IN THIS REPOSITORY
Synthetic context → Five-dimensional context in policy observation
Training: context-dependent reward shaping
```

The actual opening narration plays at 00.400–04.487, leaving a short natural picture/ambience hold before the first worker cue. Source WAV durations are 4.087, 2.879, 2.136, 3.019, 1.672 and 4.737 seconds in script order. Worker lip-sync must use these exact final files and their timeline offsets.

Use only the licensed stock voices selected from ElevenLabs, recorded separately in the production voice metadata. The robot voice is calm and warm; the worker is conversational and understated. Maintain the measured pauses between speakers. Remove the video model's audio and dialogue; the assembly script discards source audio and mixes only supplied tracks.
