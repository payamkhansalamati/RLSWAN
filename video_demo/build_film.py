#!/usr/bin/env python3
"""Assemble real cinematic footage, ElevenLabs WAVs, and deterministic overlays.

Install the two optional rendering dependencies in a virtual environment:
    python -m pip install Pillow imageio-ffmpeg

    python video_demo/build_film.py --validate
    python video_demo/build_film.py --render

All paths in timeline.json are relative to that file. Missing footage is an
error. This program never turns reference images into substitute video shots.
The blank monitor corners in the production template MUST be measured from
the actual, stable source shots; invented coordinates are not used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import wave


ROOT = Path(__file__).resolve().parent
STATUSES = ["PERCEPTION", "ANALYSING", "REASONING", "DECIDED", "WORKING"]
# The reference robot has a wide physical display. Actual footage may override
# its aspect ratio per clip; this default is only for the graphic contact sheet.
SCREEN_SIZE = (1040, 425)
PIL = None


def fail(message):
    raise RuntimeError(message)


def ffmpeg_binary():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        fail("ffmpeg is unavailable. Install imageio-ffmpeg in a virtual environment, or provide ffmpeg on PATH.")


def run(command, log=None):
    result = subprocess.run([str(x) for x in command], capture_output=True, text=True)
    if log:
        Path(log).write_text(result.stderr)
    if result.returncode:
        tail = "\n".join(result.stderr.splitlines()[-18:])
        fail(f"Media processing failed (exit {result.returncode}):\n{tail}")
    return result


def load_timeline(path):
    data = json.loads(path.read_text())
    data["_root"] = path.resolve().parent
    return data


def asset(data, relative):
    path = (data["_root"] / relative).resolve()
    if not path.is_relative_to(data["_root"].resolve()):
        fail(f"Production asset must remain inside video_demo/: {relative}")
    return path


def probe(ffmpeg, path):
    """Prefer ffprobe; the bundled imageio ffmpeg can probe without ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path])
        info = json.loads(result.stdout)
        duration = float(info.get("format", {}).get("duration", 0))
        videos = [s for s in info["streams"] if s.get("codec_type") == "video"]
        audios = [s for s in info["streams"] if s.get("codec_type") == "audio"]
        return {"duration": duration, "video": videos, "audio": audios}
    result = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    duration = sum(float(n) * factor for n, factor in zip(match.groups(), (3600, 60, 1))) if match else 0
    video = re.search(r"Stream .*Video: (.*)", result.stderr)
    audio = re.search(r"Stream .*Audio: (.*)", result.stderr)
    videos = []
    if video:
        dimensions = re.search(r"(\d{2,5})x(\d{2,5})", video.group(1))
        fps = re.search(r"([\d.]+) fps", video.group(1))
        videos.append({"width": int(dimensions[1]) if dimensions else 0,
                       "height": int(dimensions[2]) if dimensions else 0,
                       "avg_frame_rate": fps[1] if fps else "0"})
    return {"duration": duration, "video": videos, "audio": [{}] if audio else []}


def wav_duration(path):
    with wave.open(str(path), "rb") as stream:
        return stream.getnframes() / stream.getframerate()


def quad_bounds(quad):
    return min(p[0] for p in quad), min(p[1] for p in quad), max(p[0] for p in quad), max(p[1] for p in quad)


def rectangles_intersect(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def subtitle_bounds(position, width=1920, height=1080):
    y = 70 if position == "top" else height - 190
    return (180, y, width - 180, y + 134)


def validate(data, ffmpeg):
    errors, warnings = [], []
    project = data["project"]
    width, height, fps, duration = (project[k] for k in ("width", "height", "fps", "duration"))
    if width < 1920 or height < 1080 or abs(width / height - 16 / 9) > .001:
        errors.append("Delivery must be landscape 16:9 and at least 1920×1080.")
    if fps != 24:
        errors.append("This production timeline is designed for consistent 24 fps.")
    if abs(duration - 60) > 1:
        errors.append("The film duration must be 60 seconds within one second.")
    if not data.get("implementation_audit_complete"):
        errors.append("Complete the implementation audit and replace pending technical wording before export.")
    states = data["monitor_states"]
    if [s["status"] for s in states] != STATUSES:
        errors.append("Monitor states must progress PERCEPTION → ANALYSING → REASONING → DECIDED → WORKING.")
    for state in states:
        if state.get("footer") != "Illustrative system summary":
            errors.append(f"{state['status']}: illustrative-summary footer is required.")
        if len(state["lines"]) > 4:
            errors.append(f"{state['status']}: keep at most four short summary lines.")
    cursor = 0.0
    clips = sorted(data["clips"], key=lambda item: item["start"])
    for clip in clips:
        name = clip["id"]
        if abs(clip["start"] - cursor) > .5 / fps:
            errors.append(f"{name}: source shots must tile the timeline without gaps or overlaps.")
        cursor = clip["start"] + clip["duration"]
        if abs(clip["duration"] * fps - round(clip["duration"] * fps)) > .001:
            errors.append(f"{name}: shot length must align to a 24 fps frame boundary.")
        path = asset(data, clip["path"])
        if clip.get("source_kind") not in ("generated_video", "live_action"):
            errors.append(f"{name}: source_kind must identify real video footage.")
        if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg"):
            errors.append(f"{name}: still-image sources are prohibited.")
        if not path.is_file():
            errors.append(f"{name}: missing source footage: {clip['path']}")
        else:
            try:
                info = probe(ffmpeg, path)
                if not info["video"]:
                    errors.append(f"{name}: source has no video stream.")
                elif str(info["video"][0].get("nb_frames", "")).isdigit() and int(info["video"][0]["nb_frames"]) < 2:
                    errors.append(f"{name}: source must contain multiple video frames.")
                elif min(info["video"][0].get("width", 0), info["video"][0].get("height", 0)) < 720:
                    warnings.append(f"{name}: source resolution is below 720p; inspect delivery quality after scaling.")
                if info["duration"] + 1 / fps < clip.get("source_in", 0) + clip["duration"]:
                    errors.append(f"{name}: source footage is too short; do not freeze or loop frames to fill the shot.")
            except (RuntimeError, ValueError) as exc:
                errors.append(f"{name}: cannot inspect video: {exc}")
        quad = clip.get("monitor_quad")
        if clip.get("needs_monitor_quad") and not quad:
            errors.append(f"{name}: measure the stable monitor's four corners in the actual normalized shot.")
        if quad:
            if len(quad) != 4 or any(len(p) != 2 for p in quad):
                errors.append(f"{name}: monitor_quad needs TL, TR, BR, BL coordinate pairs.")
            elif any(not 0 <= p[0] <= width or not 0 <= p[1] <= height for p in quad):
                errors.append(f"{name}: monitor corners are outside the normalized output frame.")
            else:
                turns = []
                for i in range(4):
                    a, b, c = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
                    turns.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
                if min(turns) <= 0:
                    errors.append(f"{name}: monitor corners must describe a convex clockwise quad starting at top-left.")
                bounds = quad_bounds(quad)
                if bounds[2] - bounds[0] < 650:
                    warnings.append(f"{name}: monitor is under 650 pixels wide; check small text at delivery size.")
        if clip.get("requires_lipsync") and not clip.get("lipsync_verified"):
            errors.append(f"{name}: worker dialogue lip-sync has not been reviewed and verified.")
    if abs(cursor - duration) > .5 / fps:
        errors.append(f"Source-shot timeline ends at {cursor:.3f}s, expected {duration:.3f}s.")
    for state in states:
        visible = [clip for clip in clips if clip.get("monitor_quad") and max(clip["start"], state["start"]) < min(clip["start"] + clip["duration"], state["end"])]
        if not visible:
            errors.append(f"{state['status']}: no measured monitor shot is present in this phase.")
    speech = sorted(data["speech"], key=lambda item: item["start"])
    speech_end = 0.0
    for utterance in speech:
        path = asset(data, utterance["path"])
        if utterance.get("provider") != "elevenlabs":
            errors.append(f"{utterance['id']}: all speech must use ElevenLabs.")
        if not path.is_file():
            errors.append(f"{utterance['id']}: missing dialogue WAV: {utterance['path']}")
            continue
        try:
            length = wav_duration(path)
        except (wave.Error, EOFError):
            errors.append(f"{utterance['id']}: dialogue must be an uncompressed PCM WAV.")
            continue
        utterance["_duration"] = length
        end = utterance["start"] + length
        if utterance["start"] < speech_end + .08 and speech_end:
            errors.append(f"{utterance['id']}: speech overlaps or lacks a natural turn-taking pause.")
        speech_end = end
        if end > duration:
            errors.append(f"{utterance['id']}: speech extends beyond the final frame.")
        position = utterance.get("subtitle_position", "bottom")
        if position not in ("top", "bottom"):
            errors.append(f"{utterance['id']}: subtitle_position must be top or bottom.")
        box = subtitle_bounds(position, width, height)
        for clip in clips:
            if clip.get("monitor_quad") and max(utterance["start"], clip["start"]) < min(end, clip["start"] + clip["duration"]):
                if rectangles_intersect(box, quad_bounds(clip["monitor_quad"])):
                    errors.append(f"{utterance['id']}: subtitle safe area overlaps {clip['id']} monitor; move subtitles or reframe shot.")
    for label in ("ambience", "music"):
        track = data.get(label)
        if track and not asset(data, track["path"]).is_file():
            errors.append(f"Missing {label} track: {track['path']}")
    return errors, warnings


def srt_time(seconds):
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"


def subtitles(data):
    cues = []
    for utterance in data["speech"]:
        duration = utterance["_duration"]
        segments = utterance.get("subtitle_segments") or [{"start": 0, "end": duration, "text": utterance["text"]}]
        for segment in segments:
            if not 0 <= segment["start"] < segment["end"] <= duration + .02:
                fail(f"{utterance['id']}: subtitle segment lies outside its dialogue WAV.")
            cues.append({"start": utterance["start"] + segment["start"],
                         "end": utterance["start"] + segment["end"], "text": segment["text"],
                         "position": utterance.get("subtitle_position", "bottom")})
    return sorted(cues, key=lambda item: item["start"])


def write_srt(data, cues):
    text = "\n\n".join(f"{i}\n{srt_time(c['start'])} --> {srt_time(c['end'])}\n{c['text']}" for i, c in enumerate(cues, 1)) + "\n"
    (data["_root"] / "subtitles.srt").write_text(text)


def font_path(bold=False):
    requested = os.environ.get("VIDEO_DEMO_FONT_BOLD" if bold else "VIDEO_DEMO_FONT")
    choices = [requested,
               "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/Library/Fonts/Arial.ttf"]
    for choice in choices:
        if choice and Path(choice).is_file():
            return choice
    fail("No suitable TrueType font found. Set VIDEO_DEMO_FONT and VIDEO_DEMO_FONT_BOLD locally.")


def font(size, bold=False):
    return PIL[2].truetype(font_path(bold), size)


def wrap(draw, text, face, max_width):
    result, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=face) <= max_width:
            current = candidate
        else:
            if current:
                result.append(current)
            current = word
    if current:
        result.append(current)
    return result


def monitor_content(state, elapsed):
    """Return only user-visible summaries; there is no model-thought logging."""
    remaining = max(0, elapsed)
    shown = []
    for line in state["lines"]:
        words = line.split()
        count = min(len(words), max(0, int(remaining * state.get("words_per_second", 3.4)) + 1))
        shown.append(" ".join(words[:count]))
        remaining -= len(words) / state.get("words_per_second", 3.4) + state.get("line_pause", .4)
        if remaining < 0:
            break
    return shown


def monitor_size(quad, aspect_ratio=None):
    if aspect_ratio is None:
        # Appropriate only for the requested near-frontal stable close-ups.
        # Known physical screen proportions can be supplied per source clip.
        top = math.dist(quad[0], quad[1])
        bottom = math.dist(quad[2], quad[3])
        left = math.dist(quad[0], quad[3])
        right = math.dist(quad[1], quad[2])
        aspect_ratio = (top + bottom) / (left + right)
    if not .9 <= aspect_ratio <= 3:
        fail("Monitor aspect ratio is outside the supported stable-close-up layout range.")
    return 1040, round(1040 / aspect_ratio)


def monitor_image(state, shown, cursor_on, size=SCREEN_SIZE):
    Image, ImageDraw, _ = PIL
    screen = Image.new("RGBA", size, (8, 21, 27, 255))
    draw = ImageDraw.Draw(screen)
    unit = size[1] / 650
    text_scale = math.sqrt(unit)
    x = 38
    chip_y = round(34 * unit)
    chip_h = round(73 * unit)
    chip_font_size = max(27, round(39 * text_scale))
    chip_font = font(chip_font_size, True)
    chip_width = int(draw.textlength(state["status"], font=chip_font)) + 70
    draw.rounded_rectangle((x, chip_y, x + chip_width, chip_y + chip_h), radius=round(16 * text_scale), fill=(24, 74, 78), outline=(74, 188, 179), width=2)
    dot_y = chip_y + chip_h // 2
    draw.ellipse((59, dot_y - 5, 69, dot_y + 5), fill=(117, 231, 211))
    draw.text((85, chip_y + max(0, (chip_h - chip_font_size) // 2 - 2)), state["status"], font=chip_font, fill=(222, 251, 246))
    draw.line((39, round(131 * unit), 1000, round(131 * unit)), fill=(35, 66, 72), width=2)
    body_size = max(25, round(36 * text_scale) - (1 if size[1] < 424 else 0))
    face = font(body_size)
    line_height = max(round(body_size * 1.2), round(47 * unit))
    line_gap = max(5, round(13 * unit))
    y = round(164 * unit)
    for number, line in enumerate(shown):
        if not line:
            continue
        fragments = wrap(draw, line, face, 944)
        for fragment_number, fragment in enumerate(fragments):
            draw.text((42, y), fragment, font=face, fill=(233, 243, 243))
            if cursor_on and number == len(shown) - 1 and fragment_number == len(fragments) - 1:
                # Draw the cursor directly: stock fonts need not contain the
                # Unicode narrow-bar glyph, which can become a missing box.
                cursor_x = round(42 + draw.textlength(fragment, font=face) + 7)
                draw.rectangle((cursor_x, y + 4, cursor_x + 2, y + body_size), fill=(117, 231, 211))
            y += line_height
        y += line_gap
    annotation = state.get("annotation", [])
    annotation_size = max(18, round(25 * text_scale))
    annotation_height = max(round(annotation_size * 1.2), round(33 * unit))
    footer_line_y = round(603 * unit)
    annotation_y = footer_line_y - 2 - annotation_height * len(annotation) if annotation else footer_line_y
    if annotation and annotation_y < y + 5:
        fail(f"{state['status']}: display too wide for all summary lines; use shorter copy or separate timed pages.")
    if state.get("architecture"):
        architecture_y = max(y + 5, round(430 * unit))
        architecture_height = max(annotation_size + 9, round(45 * unit))
        if architecture_y + architecture_height > annotation_y - 5:
            fail(f"{state['status']}: shorten summary to leave room for the architecture graphic.")
        draw.rounded_rectangle((38, architecture_y, 1002, architecture_y + architecture_height), radius=8, fill=(17, 45, 50))
        architecture_font = font(annotation_size, True)
        draw.text((51, architecture_y + 3), state["architecture"], font=architecture_font, fill=(184, 235, 224))
    if annotation:
        if annotation_y + annotation_height * len(annotation) > footer_line_y - 2:
            fail(f"{state['status']}: monitor annotation overflows; shorten the summary.")
        for line in annotation:
            face_small = font(annotation_size)
            if draw.textlength(line, font=face_small) > 956:
                fail(f"{state['status']}: annotation is too wide for the monitor.")
            draw.text((42, annotation_y), line, font=face_small, fill=(125, 210, 200))
            annotation_y += annotation_height
    elif y > footer_line_y - 5:
        fail(f"{state['status']}: monitor summary overflows; shorten the lines.")
    draw.line((39, footer_line_y, 1000, footer_line_y), fill=(35, 66, 72), width=1)
    draw.text((42, round(615 * unit)), state["footer"], font=font(max(17, round(23 * text_scale))), fill=(151, 175, 179))
    return screen


def solve(matrix, values):
    """Small Gaussian elimination keeps perspective projection dependency-free."""
    rows = [list(map(float, row)) + [float(value)] for row, value in zip(matrix, values)]
    for col in range(len(values)):
        pivot = max(range(col, len(values)), key=lambda row: abs(rows[row][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        if abs(rows[col][col]) < 1e-10:
            fail("Degenerate monitor perspective corners.")
        scale = rows[col][col]
        rows[col] = [value / scale for value in rows[col]]
        for row in range(len(values)):
            if row != col:
                scale = rows[row][col]
                rows[row] = [a - scale * b for a, b in zip(rows[row], rows[col])]
    return [row[-1] for row in rows]


def project_screen(screen, quad, size):
    Image = PIL[0]
    source = [(0, 0), (screen.width, 0), screen.size, (0, screen.height)]
    matrix, values = [], []
    for (x, y), (u, v) in zip(quad, source):
        matrix.extend(([x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]))
        values.extend((u, v))
    coefficients = solve(matrix, values)
    return screen.transform(size, Image.Transform.PERSPECTIVE, coefficients, resample=Image.Resampling.BICUBIC)


def centered_text(canvas, text, center_y, size, color=(247, 250, 250, 255), bold=False):
    draw = PIL[1].Draw(canvas)
    face = font(size, bold)
    width = draw.textlength(text, font=face)
    x = (canvas.width - width) / 2
    draw.text((x + 2, center_y + 3), text, font=face, fill=(0, 0, 0, 185), stroke_width=2)
    draw.text((x, center_y), text, font=face, fill=color)


def overlay_signature(data, cues, t, subtitle_only=False):
    signature = {"subtitles": []}
    for cue in cues:
        if cue["start"] <= t < cue["end"]:
            signature["subtitles"].append((cue["text"], cue["position"]))
    if not subtitle_only:
        signature["titles"] = [title for title in data["titles"] if title["start"] <= t < title["end"]]
        clip = next((c for c in data["clips"] if c["start"] <= t < c["start"] + c["duration"]), None)
        state = next((s for s in data["monitor_states"] if s["start"] <= t < s["end"]), None)
        if clip and state and clip.get("monitor_quad"):
            # Reveal begins when a phase's monitor first becomes visible, so the
            # PERCEPTION words do not complete while the film shows the worker.
            first_visible = min(c["start"] for c in data["clips"] if c.get("monitor_quad") and c["start"] < state["end"] and c["start"] + c["duration"] > state["start"])
            elapsed = t - max(state["start"], first_visible)
            signature["monitor"] = {"state": state, "shown": monitor_content(state, elapsed),
                                      "cursor": int(t * 2) % 2 == 0, "quad": clip["monitor_quad"],
                                      "logical_size": monitor_size(clip["monitor_quad"], clip.get("monitor_aspect_ratio"))}
    return signature


def render_overlay(signature, size):
    Image, ImageDraw, _ = PIL
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    monitor = signature.get("monitor")
    if monitor:
        screen = monitor_image(monitor["state"], monitor["shown"], monitor["cursor"], monitor["logical_size"])
        canvas.alpha_composite(project_screen(screen, monitor["quad"], size))
    for title in signature.get("titles", []):
        if title.get("style") == "closing":
            shade = Image.new("RGBA", size, (3, 13, 18, 105))
            canvas.alpha_composite(shade)
            for index, line in enumerate(title["lines"]):
                centered_text(canvas, line, 370 + index * 103, 62 if index == 0 else 32, bold=index == 0)
        else:
            y = {"top": 69, "upper": 145, "center": 400, "bottom": 820}.get(title["position"], 145)
            for index, line in enumerate(title["lines"]):
                centered_text(canvas, line, y + index * 74, 30 if title.get("style") == "small" else 65, bold=title.get("style") != "small")
    draw = ImageDraw.Draw(canvas)
    for text, position in signature["subtitles"]:
        face = font(43)
        lines = wrap(draw, text, face, size[0] - 420)
        if len(lines) > 2:
            fail("Subtitle exceeds two readable lines; supply shorter timed subtitle_segments.")
        bounds = subtitle_bounds(position, *size)
        box_height = len(lines) * 54 + 28
        y = bounds[1] + (134 - box_height) / 2
        text_width = max(draw.textlength(line, font=face) for line in lines)
        x = (size[0] - text_width) / 2
        draw.rounded_rectangle((x - 22, y, x + text_width + 22, y + box_height), radius=13, fill=(3, 12, 17, 210))
        for index, line in enumerate(lines):
            width = draw.textlength(line, font=face)
            draw.text(((size[0] - width) / 2, y + 9 + index * 54), line, font=face, fill=(255, 255, 255))
    return canvas


def normalize_clips(data, ffmpeg, work):
    project = data["project"]
    width, height, fps = (project[k] for k in ("width", "height", "fps"))
    files = []
    for clip in sorted(data["clips"], key=lambda c: c["start"]):
        path = asset(data, clip["path"])
        output = work / f"{clip['id']}_normalized.mp4"
        key = hashlib.sha256(json.dumps({"clip": clip, "size": [width, height], "fps": fps,
                                         "mtime": path.stat().st_mtime_ns, "bytes": path.stat().st_size}, sort_keys=True).encode()).hexdigest()
        stamp = output.with_suffix(".sha256")
        if not output.is_file() or not stamp.is_file() or stamp.read_text() != key:
            print(f"Normalize real footage: {clip['id']}", flush=True)
            vf = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}"
            run([ffmpeg, "-y", "-v", "warning", "-ss", clip.get("source_in", 0), "-i", path,
                 "-map", "0:v:0", "-an", "-vf", vf, "-frames:v", round(clip["duration"] * fps),
                 "-c:v", "libx264", "-crf", "14", "-preset", "fast", "-pix_fmt", "yuv420p", output], work / f"{clip['id']}_normalize.log")
            stamp.write_text(key)
        files.append(output)
    manifest = work / "footage.ffconcat"
    manifest.write_text("ffconcat version 1.0\n" + "".join("file '" + str(path).replace("'", "'\\''") + "'\n" for path in files))
    base = work / "footage_only.mp4"
    run([ffmpeg, "-y", "-v", "warning", "-f", "concat", "-safe", "0", "-i", manifest,
         "-map", "0:v:0", "-an", "-c:v", "copy", "-movflags", "+faststart", base], work / "concatenate.log")
    return base


def mix_audio(data, ffmpeg, work):
    duration = data["project"]["duration"]
    command = [ffmpeg, "-y", "-v", "info", "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={duration}"]
    filters, sources = [], ["[0:a]"]
    index = 1
    for speech in data["speech"]:
        command += ["-i", asset(data, speech["path"])]
        delay = round(speech["start"] * 1000)
        label = f"a{index}"
        filters.append(f"[{index}:a]aresample=48000,volume={speech.get('gain_db', 0)}dB,adelay={delay}:all=1[{label}]")
        sources.append(f"[{label}]")
        index += 1
    for category, default_gain in (("ambience", -26), ("music", -32)):
        track = data.get(category)
        if track:
            command += ["-stream_loop", "-1", "-i", asset(data, track["path"])]
            label = f"a{index}"
            filters.append(f"[{index}:a]aresample=48000,volume={track.get('gain_db', default_gain)}dB,atrim=duration={duration},afade=t=in:d=1,afade=t=out:st={duration - 2}:d=2[{label}]")
            sources.append(f"[{label}]")
            index += 1
    assembly = data["assembly"]
    target = f"I={assembly['loudness_i']}:TP={assembly['loudness_tp']}:LRA={assembly['loudness_lra']}"
    mix = "".join(sources) + f"amix=inputs={len(sources)}:duration=first:normalize=0,atrim=duration={duration}"
    preliminary = work / "dialogue_mix_premaster.wav"
    filters.append(mix + "[mix]")
    run(command + ["-filter_complex", ";".join(filters), "-map", "[mix]", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le", preliminary], work / "mix.log")
    measurement = run([ffmpeg, "-v", "info", "-i", preliminary, "-af", f"loudnorm={target}:print_format=json", "-f", "null", "-"], work / "loudness_analysis.log")
    matches = re.findall(r'\{\s*"input_i".*?\}', measurement.stderr, re.S)
    if not matches:
        fail("Unable to measure final audio loudness.")
    values = json.loads(matches[-1])
    if not math.isfinite(float(values["input_i"])):
        fail("Dialogue mix is silent; no final video will be produced.")
    measured = f":measured_I={values['input_i']}:measured_TP={values['input_tp']}:measured_LRA={values['input_lra']}:measured_thresh={values['input_thresh']}:offset={values['target_offset']}:linear=true"
    output = work / "final_mix.wav"
    run([ffmpeg, "-y", "-v", "warning", "-i", preliminary, "-af", f"loudnorm={target}{measured}", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le", output], work / "master.log")
    (work / "audio_measurements.json").write_text(json.dumps({"premaster": values, "target": target}, indent=2))
    return output


def export(data, ffmpeg, footage, mixed_audio, cues, output, work, subtitle_only=False):
    project, assembly = data["project"], data["assembly"]
    size = project["width"], project["height"]
    frames = round(project["duration"] * project["fps"])
    command = [ffmpeg, "-y", "-v", "warning", "-i", footage,
               "-f", "rawvideo", "-pixel_format", "rgba", "-video_size", f"{size[0]}x{size[1]}",
               "-framerate", project["fps"], "-i", "pipe:0", "-i", mixed_audio,
               "-filter_complex", "[0:v][1:v]overlay=0:0:shortest=1:format=auto,format=yuv420p[v]",
               "-map", "[v]", "-map", "2:a:0", "-frames:v", frames, "-t", project["duration"],
               "-c:v", "libx264", "-preset", assembly["preset"], "-crf", assembly["crf"],
               "-c:a", "aac", "-b:a", assembly["audio_bitrate"], "-ar", "48000",
               "-movflags", "+faststart", "-metadata", "title=Context-aware assistance — concept demonstration", output]
    previous_signature, previous_bytes = None, None
    with (work / f"{output.stem}_encode.log").open("w") as log:
        process = subprocess.Popen([str(x) for x in command], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for frame in range(frames):
                t = frame / project["fps"]
                signature = overlay_signature(data, cues, t, subtitle_only)
                if signature != previous_signature:
                    previous_bytes = render_overlay(signature, size).tobytes()
                    previous_signature = signature
                process.stdin.write(previous_bytes)
                if frame % (project["fps"] * 5) == 0:
                    print(f"{output.name}: {t:.0f}/{project['duration']:.0f}s", flush=True)
            process.stdin.close()
            result = process.wait()
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            output.unlink(missing_ok=True)
            raise
    if result:
        output.unlink(missing_ok=True)
        fail(f"Export failed. Review {work / (output.stem + '_encode.log')}")


def verify_export(data, ffmpeg, output):
    info = probe(ffmpeg, output)
    video = info["video"][0] if info["video"] else {}
    project = data["project"]
    if (video.get("width"), video.get("height")) != (project["width"], project["height"]):
        fail(f"Output resolution check failed: {output.name}")
    if abs(info["duration"] - project["duration"]) > .15 or not info["audio"]:
        fail(f"Output duration/audio check failed: {output.name}")
    rate = video.get("avg_frame_rate", "0")
    numerator, _, denominator = rate.partition("/")
    rate = float(numerator) / float(denominator or 1)
    if abs(rate - project["fps"]) > .001:
        fail(f"Output frame-rate check failed: {output.name}")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timeline", type=Path, default=ROOT / "timeline.json")
    parser.add_argument("--validate", action="store_true", help="Inspect required inputs without producing substitute output.")
    parser.add_argument("--render", action="store_true", help="Build both final exports after all validation passes.")
    parser.add_argument("--monitor-preview", type=Path, help="Write an explicitly labeled contact sheet of interface states, not a film.")
    parser.add_argument("--write-provisional-srt", action="store_true", help="Write subtitles_provisional.srt from explicitly estimated timings; never a delivery subtitle file.")
    parser.add_argument("--write-srt", action="store_true", help="Write measured subtitles.srt from final WAVs/aligned segments independently of footage availability.")
    args = parser.parse_args()
    data = load_timeline(args.timeline)
    if args.write_srt:
        previous_end = 0
        for utterance in sorted(data["speech"], key=lambda u: u["start"]):
            if utterance.get("provider") != "elevenlabs":
                fail(f"{utterance['id']}: all speech must use ElevenLabs.")
            utterance["_duration"] = wav_duration(asset(data, utterance["path"]))
            if utterance["start"] < previous_end:
                fail(f"{utterance['id']}: dialogue overlaps; adjust actual speech timing first.")
            previous_end = utterance["start"] + utterance["_duration"]
            if previous_end > data["project"]["duration"]:
                fail(f"{utterance['id']}: dialogue exceeds film duration.")
        write_srt(data, subtitles(data))
        print(f"Saved measured subtitle timing: {data['_root'] / 'subtitles.srt'}")
        return 0
    if args.write_provisional_srt:
        cues = [{"start": u["start"], "end": u["start"] + u.get("estimated_duration", u.get("measured_duration", 0)), "text": u["text"]} for u in data["speech"]]
        text = "\n\n".join(f"{i}\n{srt_time(c['start'])} --> {srt_time(c['end'])}\n{c['text']}" for i, c in enumerate(cues, 1)) + "\n"
        path = data["_root"] / "subtitles_provisional.srt"
        path.write_text(text)
        print(f"Saved PROVISIONAL estimated subtitles, not delivery timing: {path}")
        return 0
    if args.monitor_preview:
        global PIL
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            fail("Install Pillow in a virtual environment to preview monitor graphics.")
        PIL = Image, ImageDraw, ImageFont
        contact = Image.new("RGB", (SCREEN_SIZE[0] * 2, SCREEN_SIZE[1] * 3 + 100), (24, 29, 34))
        draw = ImageDraw.Draw(contact)
        draw.text((30, 25), "MONITOR DESIGN PREVIEW — NOT GENERATED FOOTAGE", font=font(28, True), fill="white")
        for index, state in enumerate(data["monitor_states"]):
            image = monitor_image(state, state["lines"], False)
            contact.paste(image, ((index % 2) * SCREEN_SIZE[0], 100 + (index // 2) * SCREEN_SIZE[1]))
        path = args.monitor_preview.resolve()
        if not path.is_relative_to(data["_root"]):
            fail("Monitor preview must be inside video_demo/.")
        path.parent.mkdir(parents=True, exist_ok=True)
        contact.save(path)
        print(f"Saved interface preview: {path}")
        return
    ffmpeg = ffmpeg_binary()
    errors, warnings = validate(data, ffmpeg)
    for warning in warnings:
        print(f"REVIEW: {warning}")
    if errors:
        print("Production inputs are incomplete; no final MP4 has been generated.")
        for error in errors:
            print(f"- {error}")
        return 2
    print("Input validation passed.")
    if not args.render:
        return 0
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        fail("Install Pillow in a virtual environment before rendering.")
    PIL = Image, ImageDraw, ImageFont
    font_path()
    font_path(True)
    work = data["_root"] / "work"
    work.mkdir(exist_ok=True)
    cues = subtitles(data)
    write_srt(data, cues)
    footage = normalize_clips(data, ffmpeg, work)
    audio = mix_audio(data, ffmpeg, work)
    clean = data["_root"] / "context_aware_robot_60s.mp4"
    subtitled = data["_root"] / "context_aware_robot_60s_subtitled.mp4"
    export(data, ffmpeg, footage, audio, [], clean, work)
    # Both deliveries start from the same normalized sources, avoiding another
    # lossy generation on the subtitled version.
    export(data, ffmpeg, footage, audio, cues, subtitled, work)
    report = {"clean": verify_export(data, ffmpeg, clean), "subtitled": verify_export(data, ffmpeg, subtitled),
              "automated_checks": "Resolution, duration, audio presence and consistent frame rate passed.",
              "human_review_required": "Watch both films with sound and complete review_checklist.md; metadata checks cannot certify lip-sync, realism, continuity or scientific validity."}
    (data["_root"] / "export_verification.json").write_text(json.dumps(report, indent=2))
    print("Both delivery files and subtitles.srt are ready for the complete visual/audio review.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, KeyError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
