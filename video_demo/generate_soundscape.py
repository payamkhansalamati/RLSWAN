#!/usr/bin/env python3
"""Prepare two restrained SFX stems; never regenerate an uncertain paid request.

  python generate_soundscape.py plan --max-credits 1500
  python generate_soundscape.py generate --max-credits 1500
  python generate_soundscape.py collect

Official duration pricing: https://elevenlabs.io/docs/overview/capabilities/sound-effects
API schema: https://elevenlabs.io/docs/api-reference/text-to-sound-effects/convert
Specified durations cost 40 credits/second: 20s room tone + 6s wheels = 1040 credits.
Plan and collect make no paid requests. Generated content still needs listening review.
"""
import argparse
import json
import math
import ssl
import subprocess
import sys
import urllib.error
import urllib.request

from elevenlabs_client import ROOT, load_environment
from generate_assets import (UNCERTAIN, budget_guard, existing_job, fail, ffmpeg_binary,
                             fingerprint, local_path, locked_ledger, now, persist,
                             prepare_post, read_json, relative, safe_subscription)

ENDPOINT = "/v1/sound-generation?output_format=mp3_44100_128"
RATE = 40
SOURCE = "https://elevenlabs.io/docs/overview/capabilities/sound-effects"
STEMS = [
    {"name": "corridor", "output": "soundscape/corridor_loop.wav", "duration": 20,
     "loop": True, "recommended_gain_db": -34,
     "text": "Quiet clean modern corridor: steady soft ventilation air and distant building room tone. "
             "Sparse natural stereo ambience, even dynamics, seamless loop. "
             "No speech, voices, footsteps, patients, alarms, beeps, music or dramatic effects."},
    {"name": "wheels", "output": "soundscape/rubber_wheels.wav", "duration": 6,
     "loop": False, "recommended_gain_db": -28,
     "text": "Isolated quiet rubber wheels of a small service robot rolling slowly over smooth vinyl floor. "
             "Soft low swish and gentle bearing sound, steady pace, subtle natural interior reverb. "
             "No voices, speech, footsteps, alarms, beeps, squeaks, music or impacts."},
]


def payload_for(stem):
    return {"text": stem["text"], "model_id": "eleven_text_to_sound_v2", "loop": stem["loop"],
            "duration_seconds": stem["duration"], "prompt_influence": 0.65}


def finish(ledger, job):
    target = local_path(job["output"])
    mp3 = target.with_suffix(".mp3")
    if not mp3.is_file() or not mp3.stat().st_size:
        fail("Missing cached SFX MP3. Inspect account history; do not resubmit automatically.")
    temporary = target.with_name(target.stem + ".part.wav")
    result = subprocess.run([ffmpeg_binary(), "-nostdin", "-y", "-v", "error", "-i", str(mp3),
                             "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le", str(temporary)],
                            capture_output=True, text=True)
    if result.returncode:
        fail("SFX WAV conversion failed. Recover the cached MP3 with collect; no new generation is needed.")
    temporary.replace(target)
    job.update(status="completed", completed_at=now(), mp3=relative(mp3), listening_review="pending")
    persist(ledger)
    print("completed", job["output"], "— listening review pending")


def post_audio_once(ledger, job, payload):
    """The shared JSON client cannot decode this endpoint's binary response."""
    key = load_environment()
    req = urllib.request.Request("https://api.elevenlabs.io" + ENDPOINT,
                                 data=json.dumps(payload).encode(),
                                 headers={"xi-api-key": key, "Content-Type": "application/json",
                                          "Accept": "audio/mpeg"})
    target = local_path(job["output"])
    mp3 = target.with_suffix(".mp3")
    temporary = mp3.with_name(mp3.name + ".part")
    try:
        context = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
        with urllib.request.urlopen(req, timeout=120, context=context) as response:
            if response.headers.get_content_type() not in {"audio/mpeg", "audio/mp3", "application/octet-stream"}:
                fail("Unexpected SFX response content type.")
            audio = response.read(20 * 1024 * 1024 + 1)
            if not audio or len(audio) > 20 * 1024 * 1024:
                fail("Unexpected SFX audio size.")
            raw_cost = response.headers.get("character-cost")
        temporary.write_bytes(audio)
        temporary.replace(mp3)
    except Exception as exc:
        job.update(status="ambiguous", error_type=type(exc).__name__)
        if isinstance(exc, urllib.error.HTTPError):
            job["http_status"] = exc.code
        persist(ledger)
        fail("SFX POST did not finish with usable audio; account-history review is required before retrying.")
    job.update(status="response_cached", mp3=relative(mp3))
    if raw_cost is not None:
        try:
            actual_cost = float(raw_cost)
            if math.isfinite(actual_cost) and actual_cost >= 0:
                job["response_character_cost"] = actual_cost
                job["estimated_credits"] = max(job["estimated_credits"], actual_cost)
        except ValueError:
            pass
    persist(ledger)


def generate(ledger, cap):
    ffmpeg_binary()
    prepared = []
    for stem in STEMS:
        target = local_path(stem["output"])
        payload = payload_for(stem)
        job = existing_job(ledger, target, fingerprint(ENDPOINT, payload))
        prepared.append((stem, target, payload, job))
    total_new = sum(stem["duration"] * RATE for stem, _, _, job in prepared if job is None)
    budget_guard(ledger, cap, total_new)
    print("New SFX estimate:", total_new, "credits; includes explicit-duration rate", RATE, "/second")
    for stem, target, payload, job in prepared:
        if job:
            if job["status"] == "completed" and target.is_file():
                print("cached", job["output"])
            else:
                finish(ledger, job)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        estimate = stem["duration"] * RATE
        job = prepare_post(ledger, target, ENDPOINT, payload, estimate, cap, "sfx")
        snapshot = read_json(ROOT / "research" / "production_budget.json")
        before = snapshot["character_count"]
        job.update(model_id=payload["model_id"], duration_seconds=stem["duration"], loop=stem["loop"],
                   estimate_source=SOURCE, credits_per_second=RATE,
                   recommended_gain_db=stem["recommended_gain_db"], prompt=stem["text"])
        persist(ledger)
        post_audio_once(ledger, job, payload)
        finish(ledger, job)
        after = safe_subscription()["character_count"]
        delta = max(0, after - before)
        job["observed_account_credit_delta"] = delta
        job["estimated_credits"] = max(job["estimated_credits"], delta)
        persist(ledger)
        if delta > estimate:
            fail("Observed usage exceeded the documented estimate; remaining SFX submissions are stopped.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "generate"):
        command = commands.add_parser(name)
        command.add_argument("--max-credits", required=True, type=float)
    commands.add_parser("collect")
    args = parser.parse_args()
    with locked_ledger() as ledger:
        if args.command == "plan":
            new = sum(stem["duration"] * RATE for stem in STEMS if stem["output"] not in ledger["jobs"])
            subscription, reserved = budget_guard(ledger, args.max_credits, new)
            print(json.dumps({"source": SOURCE, "credits_per_second": RATE,
                              "new_sfx_estimate": new, "already_reserved": reserved,
                              "project_cap": args.max_credits,
                              "included_remaining": subscription["included_remaining"],
                              "stems": STEMS}, indent=2))
        elif args.command == "generate":
            generate(ledger, args.max_credits)
        else:
            for job in ledger["jobs"].values():
                if job["kind"] == "sfx":
                    if job["status"] in UNCERTAIN:
                        print("requires account-history review", job["output"])
                    elif job["status"] == "response_cached":
                        finish(ledger, job)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        print(f"Stopped: {exc}" if isinstance(exc, RuntimeError) else f"Stopped: {type(exc).__name__}", file=sys.stderr)
        sys.exit(1)
