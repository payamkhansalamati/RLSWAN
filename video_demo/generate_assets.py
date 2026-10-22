#!/usr/bin/env python3
"""Generate and resume ElevenLabs assets using included credits only.

Examples:
  python generate_assets.py preflight
  python generate_assets.py check-budget --max-credits 1500
  python generate_assets.py speech --max-credits 1500
  python generate_assets.py submit-shot --payload shot.json --output sources/shot.mp4 --max-credits 1500
  python generate_assets.py collect --wait-seconds 600

Shot JSON: {"request": {...}, "estimated_credits": 123,
            "estimate_source": "https://elevenlabs.io/app/image-video",
            "estimate_note": "Exact app quote for these settings, YYYY-MM-DD"}
No operation changes billing, buys credits, or retries a paid POST.
"""
import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from elevenlabs_client import ROOT, load_environment, request, save

LEDGER = ROOT / "asset_jobs.json"
TTS_MODEL = "eleven_multilingual_v2"
TTS_LIMIT = 1500
UNCERTAIN = {"submitting", "ambiguous"}


def fail(message):
    raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def local_path(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        fail("Asset output must stay inside video_demo.")
    return path


def relative(path):
    return str(path.relative_to(ROOT))


def number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        fail(f"{label} must be a finite number.")
    if value < 0 or (positive and value == 0):
        fail(f"{label} must be {'positive' if positive else 'non-negative'}.")
    return value


@contextmanager
def locked_ledger():
    with (ROOT / ".asset_jobs.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("Another asset production command is running.")
        ledger = read_json(LEDGER) if LEDGER.exists() else {"schema_version": 1, "jobs": {}}
        if ledger.get("schema_version") != 1 or not isinstance(ledger.get("jobs"), dict):
            fail("Unrecognized asset ledger; inspect it before proceeding.")
        yield ledger


def persist(ledger):
    write_json(LEDGER, ledger)


def safe_subscription():
    data = request("/v1/user/subscription")
    fields = ("tier", "status", "character_count", "character_limit",
              "max_credit_limit_extension", "can_extend_character_limit", "current_overage")
    result = {key: data.get(key) for key in fields}
    used = number(result["character_count"], "Subscription consumption")
    limit = number(result["character_limit"], "Subscription allowance")
    result["included_remaining"] = max(0, limit - used)
    save("production_budget", result)
    return result


def budget_guard(ledger, cap, estimate=0):
    """Re-read account state before every paid POST; reserve uncertain jobs too."""
    number(cap, "Project credit cap", positive=True)
    number(estimate, "Generation estimate")
    subscription = safe_subscription()  # Missing scope or unknown balance blocks paid work.
    extension = subscription["max_credit_limit_extension"]
    if type(extension) is not int or extension != 0:
        fail("Included-credit-only generation requires max_credit_limit_extension exactly 0.")
    if subscription["included_remaining"] < estimate:
        fail("The generation estimate exceeds the verified included credit balance.")
    original_cap = ledger.get("project_max_credits", cap)
    number(original_cap, "Recorded project credit cap", positive=True)
    if cap > original_cap:
        fail("The requested cap exceeds asset_jobs.json's existing project cap; review that file explicitly.")
    reserved = sum(number(job.get("estimated_credits"), "Recorded job estimate")
                   for job in ledger["jobs"].values())
    if reserved + estimate > cap:
        fail(f"Project credit cap exceeded: {reserved:g} reserved + {estimate:g} estimated > {cap:g}.")
    ledger["project_max_credits"] = cap
    return subscription, reserved


def speech_payload(utterance, voices, model):
    voice = voices.get(utterance["speaker"], {}).get("voice_id")
    if not isinstance(voice, str) or not voice:
        fail(f"Missing selected voice for {utterance['speaker']}.")
    text = utterance.get("text")
    if not isinstance(text, str) or not text.strip():
        fail(f"Missing speech text for {utterance['id']}.")
    settings = {"stability": 0.5, "similarity_boost": 0.75,
                "style": 0, "use_speaker_boost": True, "speed": 1}
    settings.update(voices.get(utterance["speaker"], {}).get("settings", {}))
    return voice, {"text": text, "model_id": model, "voice_settings": settings}


def speech_rate(model):
    models = request("/v1/models")
    selected = next((item for item in models if item.get("model_id") == model), None)
    if not selected or not selected.get("can_do_text_to_speech"):
        fail(f"Speech model {model} is unavailable.")
    rates = selected.get("model_rates") or {}
    multiplier = number(rates.get("character_cost_multiplier"), "Character cost multiplier", True)
    discount = number(rates.get("cost_discount_multiplier", 1), "Cost discount multiplier", True)
    return multiplier * discount


def fingerprint(endpoint, payload):
    encoded = json.dumps({"endpoint": endpoint, "request": payload}, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def existing_job(ledger, target, digest):
    job = ledger["jobs"].get(relative(target))
    if job and job.get("request_sha256") != digest:
        fail(f"{relative(target)} already has a different request; choose a new output path.")
    if not job and target.exists():
        fail(f"{relative(target)} exists without a production record; inspect it before generation.")
    if job and job.get("status") in UNCERTAIN:
        fail(f"{relative(target)} has an uncertain POST outcome. Inspect account history; do not resubmit.")
    return job


def prepare_post(ledger, target, endpoint, payload, estimate, cap, kind):
    budget_guard(ledger, cap, estimate)
    job = {"kind": kind, "output": relative(target), "endpoint": endpoint,
           "request_sha256": fingerprint(endpoint, payload), "estimated_credits": estimate,
           "created_at": now(), "status": "submitting"}
    ledger["jobs"][relative(target)] = job
    persist(ledger)  # Persist before the paid POST, including crash/timeout uncertainty.
    return job


def post_once(ledger, job, payload):
    try:
        return request(job["endpoint"], payload)
    except Exception as exc:
        job["status"] = "ambiguous"
        job["error_type"] = type(exc).__name__  # No raw response, headers, URL or credentials.
        persist(ledger)
        fail("Paid POST did not return a usable response. Outcome is uncertain; automatic retry is disabled.")


def ffmpeg_binary():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        fail("Install imageio-ffmpeg or provide ffmpeg on PATH before generating speech.")


def finish_speech(ledger, job):
    target = local_path(job["output"])
    response_path = target.with_suffix(".tts_response.json")
    if not response_path.exists():
        fail(f"No cached speech response for {job['output']}; do not resubmit automatically.")
    response = read_json(response_path)
    audio = base64.b64decode(response["audio_base64"], validate=True)
    if not audio:
        fail("The cached speech response contains no audio.")
    mp3 = target.with_suffix(".mp3")
    mp3.write_bytes(audio)
    write_json(target.with_suffix(".alignment.json"), {
        "alignment": response.get("alignment"),
        "normalized_alignment": response.get("normalized_alignment")})
    temporary = target.with_name(target.stem + ".part.wav")
    result = subprocess.run([ffmpeg_binary(), "-nostdin", "-y", "-v", "error", "-i", str(mp3),
                             "-ar", "48000", "-ac", "1", "-c:a", "pcm_s24le", str(temporary)],
                            capture_output=True, text=True)
    if result.returncode:
        fail("Speech audio conversion failed; cached MP3/response can be collected again without a paid call.")
    temporary.replace(target)
    job.update(status="completed", completed_at=now(), mp3=relative(mp3),
               alignment=relative(target.with_suffix(".alignment.json")))
    persist(ledger)
    print("completed", job["output"])


def generate_speech(ledger, args):
    ffmpeg_binary()  # Resolve local conversion dependency before any paid call.
    timeline = read_json(ROOT / "timeline.json")
    voices = read_json(ROOT / "voice_selection.json")
    rate = speech_rate(args.model)
    prepared = []
    for utterance in timeline["speech"]:
        if utterance.get("provider") != "elevenlabs":
            fail("Every timeline speech provider must be elevenlabs.")
        target = local_path(utterance["path"])
        if target.suffix.lower() != ".wav":
            fail("Timeline speech output must be WAV.")
        voice, payload = speech_payload(utterance, voices, args.model)
        endpoint = f"/v1/text-to-speech/{urllib.parse.quote(voice, safe='')}/with-timestamps?output_format=mp3_44100_128"
        digest = fingerprint(endpoint, payload)
        job = existing_job(ledger, target, digest)
        prepared.append((utterance, target, endpoint, payload, job, math.ceil(len(payload["text"]) * rate)))
    new_estimate = sum(item[5] for item in prepared if not item[4])
    total_tts = sum(job["estimated_credits"] for job in ledger["jobs"].values() if job["kind"] == "speech")
    if total_tts + new_estimate >= TTS_LIMIT:
        fail(f"Estimated total speech usage must stay below {TTS_LIMIT} credits.")
    budget_guard(ledger, args.max_credits, new_estimate)
    print(f"Speech estimate: {new_estimate:g} new credits; rate {rate:g} per character.")
    for utterance, target, endpoint, payload, job, estimate in prepared:
        if job:
            if job["status"] == "completed" and target.exists():
                print("cached", job["output"])
                continue
            finish_speech(ledger, job)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        job = prepare_post(ledger, target, endpoint, payload, estimate, args.max_credits, "speech")
        job.update(speech_id=utterance["id"], speaker=utterance["speaker"], model_id=args.model,
                   input_characters=len(payload["text"]), credits_per_character=rate)
        persist(ledger)
        response = post_once(ledger, job, payload)
        if not isinstance(response, dict) or not isinstance(response.get("audio_base64"), str):
            job["status"] = "ambiguous"
            persist(ledger)
            fail("Unrecognized speech response; automatic retry is disabled.")
        cached = {key: response.get(key) for key in ("audio_base64", "alignment", "normalized_alignment")}
        write_json(target.with_suffix(".tts_response.json"), cached)
        job["status"] = "response_cached"
        persist(ledger)
        finish_speech(ledger, job)


def download(url, target):
    if not isinstance(url, str) or urllib.parse.urlsplit(url).scheme != "https":
        fail("The generation did not return an HTTPS download URL.")
    temporary = target.with_name(target.name + ".part")
    context = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    # Download with a new unauthenticated request: never forward the ElevenLabs API key.
    with urllib.request.urlopen(urllib.request.Request(url), timeout=120, context=context) as source:
        with temporary.open("wb") as output:
            shutil.copyfileobj(source, output)
    if not temporary.stat().st_size:
        fail("The generation download was empty.")
    temporary.replace(target)


def collect_video(ledger, job):
    generation_id = job.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        fail("Missing generation ID; inspect account history before attempting recovery.")
    result = request("/v1/flows/video/" + urllib.parse.quote(generation_id, safe=""))
    if not isinstance(result, dict):
        fail("Unrecognized generation response; no new generation was submitted.")
    status = result.get("status")
    if status not in {"pending", "generating", "completed", "failed"}:
        fail("Unrecognized generation status; no new generation was submitted.")
    if status == "completed":
        if result.get("content_mime_type") != "video/mp4":
            fail("Unexpected video content type; inspect generation before downloading.")
        target = local_path(job["output"])
        target.parent.mkdir(parents=True, exist_ok=True)
        download(result.get("content_url"), target)
        job.update(status="completed", completed_at=now())
    elif status == "failed":
        job.update(status="failed", failure_reason=result.get("failure_reason", "unknown"))
    else:
        job["status"] = status
    persist(ledger)  # Signed content_url is deliberately never persisted.
    print(job["status"], job["output"])


def submit_shot(ledger, args):
    envelope = read_json(args.payload)
    payload = envelope.get("request")
    if not isinstance(payload, dict) or not isinstance(payload.get("model_id"), str):
        fail("Shot JSON must contain request with a model_id.")
    estimate = number(envelope.get("estimated_credits"), "Documented shot estimate", True)
    source = envelope.get("estimate_source", "")
    parsed = urllib.parse.urlsplit(source)
    if parsed.scheme != "https" or parsed.hostname != "elevenlabs.io" or not envelope.get("estimate_note"):
        fail("Shot estimate requires an official ElevenLabs source URL and exact quote/settings note.")
    if "webhook" in payload:
        fail("Omit webhook: this script collects results through documented polling.")
    target = local_path(args.output)
    if target.suffix.lower() != ".mp4":
        fail("Shot output must be an MP4 path.")
    endpoint = "/v1/flows/video"
    digest = fingerprint(endpoint, payload)
    job = existing_job(ledger, target, digest)
    if job:
        print("recorded", job["output"], job["status"], "— use collect to resume")
        return
    request("/v1/flows/video?page_size=1")  # Capability check cannot incur a generation charge.
    job = prepare_post(ledger, target, endpoint, payload, estimate, args.max_credits, "video")
    job.update(model_id=payload["model_id"], estimate_source=source, estimate_note=envelope["estimate_note"])
    persist(ledger)
    result = post_once(ledger, job, payload)
    if (not isinstance(result, dict) or result.get("status") != "pending"
            or not isinstance(result.get("id"), str) or not result["id"]):
        job["status"] = "ambiguous"
        persist(ledger)
        fail("Unrecognized submission response; automatic retry is disabled.")
    job.update(status="pending", generation_id=result["id"])
    persist(ledger)
    print("submitted", job["output"], "generation", job["generation_id"])


def collect(ledger, wait_seconds):
    deadline = time.monotonic() + wait_seconds
    interval = 10
    while True:
        for job in ledger["jobs"].values():
            if job["status"] in UNCERTAIN:
                print("requires manual account-history review", job["output"])
            elif job["kind"] == "speech" and job["status"] == "response_cached":
                finish_speech(ledger, job)
            elif job["kind"] == "video" and job["status"] in {"pending", "generating"}:
                collect_video(ledger, job)
        pending = any(job["status"] in {"pending", "generating"} for job in ledger["jobs"].values())
        if not pending or time.monotonic() + interval > deadline:
            return
        time.sleep(interval)
        interval = min(60, interval * 2)


def preflight():
    for name, path in (("subscription", "/v1/user/subscription"),
                       ("video", "/v1/flows/video?page_size=1"),
                       ("image", "/v1/flows/image?page_size=1")):
        try:
            result = safe_subscription() if name == "subscription" else request(path)
            print(name, json.dumps(result) if name == "subscription" else "read-only listing accessible")
        except Exception as exc:
            # Client sanitizes API errors. Never print an arbitrary transport exception.
            print(name, str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="Read-only capability and subscription checks")
    check = commands.add_parser("check-budget", help="Verify no overage and estimate final timeline speech")
    check.add_argument("--max-credits", required=True, type=float)
    check.add_argument("--model", default=TTS_MODEL)
    speech = commands.add_parser("speech", help="Generate missing speech or recover cached responses")
    speech.add_argument("--max-credits", required=True, type=float)
    speech.add_argument("--model", default=TTS_MODEL)
    shot = commands.add_parser("submit-shot", help="Submit one explicitly quoted video request")
    shot.add_argument("--payload", required=True, type=Path)
    shot.add_argument("--output", required=True)
    shot.add_argument("--max-credits", required=True, type=float)
    gather = commands.add_parser("collect", help="Collect known jobs without new paid requests")
    gather.add_argument("--wait-seconds", type=float, default=0)
    args = parser.parse_args()
    load_environment()
    if args.command == "preflight":
        preflight()
        return
    with locked_ledger() as ledger:
        if args.command == "check-budget":
            subscription, reserved = budget_guard(ledger, args.max_credits)
            timeline = read_json(ROOT / "timeline.json")
            rate = speech_rate(args.model)
            chars = sum(len(item["text"]) for item in timeline["speech"])
            print(json.dumps({"subscription": subscription, "project_cap": args.max_credits,
                              "reserved_estimates": reserved, "speech_characters": chars,
                              "speech_rate": rate, "speech_estimated_credits": math.ceil(chars * rate)}, indent=2))
        elif args.command == "speech":
            generate_speech(ledger, args)
        elif args.command == "submit-shot":
            submit_shot(ledger, args)
        else:
            number(args.wait_seconds, "Wait seconds")
            collect(ledger, args.wait_seconds)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        # Avoid arbitrary exception detail: transports may include credential-bearing URLs.
        print(f"Stopped: {exc}" if isinstance(exc, RuntimeError) else f"Stopped: {type(exc).__name__}", file=sys.stderr)
        sys.exit(1)
