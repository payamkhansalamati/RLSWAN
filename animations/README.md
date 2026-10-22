# RLSWAN presentation animations

The presentation renderer uses the saved trajectory CSVs and
`animations/data/animation_results.json`. Rendering does **not** train a policy,
run a simulation, or change the recorded outcomes. It writes exactly two new
presentation MP4s and leaves the original MP4s and trajectory data untouched.

## Render the presentation versions

From the repository root, using the existing Python 3.11 environment:

```bash
/tmp/rlswan-venv/bin/python scripts/create_animations.py render
```

Install the project's dependencies and the animation add-ons first if needed:

```bash
uv python install 3.11
uv venv /tmp/rlswan-venv --python 3.11
uv pip install --python /tmp/rlswan-venv/bin/python \
  -r requirements.txt -r requirements-animations.txt
```

Outputs:

- `animations/rendered/animation1_five_profiles_presentation.mp4`
- `animations/rendered/animation2_matched_comparison_presentation.mp4`

Both render at 1920×1080, 30 fps. Per-render settings and recorded event timing
are saved in `animations/data/animation1_presentation_render.json` and
`animations/data/animation2_presentation_render.json`.

## Animation 1 — five profiles

The video is exactly 50 seconds and 1,500 frames. Each scene is 10 seconds and
uses uniform 5× playback to cover the complete 0–50 simulation-second recorded
episode. In configured order the profiles are Constant, Sinusoidal, Braking,
Stop-and-go, and Smooth random. The simulation clock resets for each episode;
the video clock and progress strip identify the scene and overall presentation
time.

The prominent speed plot shows the recorded ego and lead speeds. The road uses
a shared linear longitudinal scale in metres; the displayed gap comes directly
from each trajectory. The renderer does not cap or compress the gap. The
simulation itself caps its gap state at 100 m, which is labeled when reached.
The car drawings are schematic, not to physical scale.

Live TTC uses the project's proxy and threshold. The recorded `999` sentinel
for non-closing traffic is shown as “TTC: Not closing.” Live headway is shown
as “— (ego stopped)” when ego speed is at or below the configured 0.1 m/s
headway speed floor; recorded headway-violation counts are retained in each
profile's outcome line. Context is shown by its recorded regime label.

## Animation 2 — matched policy comparison

The video compares the saved, matched Context-aware PPO and Hybrid Context-PID
trajectories with synchronized simulation clocks and a shared road scale. The
right controller is the same `context_ppo` checkpoint with the project's
physical-risk PID supervision and safety filtering; it is not a separately
trained PPO checkpoint. The recorded right-side `hybrid_pid_active` and
`hybrid_filter_active` fields drive the live intervention labels.

The first recorded Context-aware PPO TTC-proxy sample below 3.0 seconds is at
15.6 simulation seconds. The graph labels the measured 3-second threshold and
the event, and emphasizes the affected panel at the event. Recorded outcomes
for this pair are:

- Context-aware PPO: 25 TTC<3 s steps, minimum TTC proxy 1.64 s, minimum gap
  2.48 m, zero collisions, and 103 steps below the 1.5 s headway threshold.
- Hybrid Context-PID: zero TTC<3 s steps, zero collisions, and zero steps below
  the 1.5 s headway threshold.
- Recorded episode progress: 276.4 m for Context-aware PPO and 334.7 m for
  Hybrid Context-PID.

The 12-second chart ceiling is a display limit, not a measurement cap: finite
TTC values above it receive overflow markers, while non-closing sentinel values
are omitted from the curve. Live status still reports the measured current
value or “TTC: Not closing.” Current headway and its independent 1.5-second
threshold remain visible; no-TTC-violation wording is not a claim that all
possible safety criteria are satisfied.

## Provenance and limitations

Trajectories are the existing `animations/data/animation1_*.csv` and
`animation2_*.csv` recordings, described in `animation_results.json`. The
matched comparison uses the same initial state, constant lead profile, calm
context, seed 700, and recorded lead-speed/context traces. Its conclusion is
illustrative for this one episode, not a general performance guarantee.

The phrase “Conservative response: larger gap, reduced progress” is not used as
a comparison claim: in this matched pair, the risk-governed controller's
recorded progress is greater, and its final gap is smaller than the PPO
episode's 100 m simulation limit. Animation 1 visibly preserves the recorded
stops and large gaps instead. These are synthetic simulations and do not
establish real-world driving safety.
