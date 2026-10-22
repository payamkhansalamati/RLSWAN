#!/usr/bin/env python3
"""Assemble saved audio locally. No API calls or speech regeneration.

Completed dialogue and sound-effect source files remain unchanged. Existing
assembled outputs are reused. To revise an edit, use a versioned output name.
"""
import json
import math
from pathlib import Path
import re
import shutil

import build_film as editor

ROOT = Path(__file__).resolve().parent


def measure(ffmpeg, path):
    result = editor.run([ffmpeg, '-hide_banner', '-i', path, '-af',
                         'loudnorm=I=-19:TP=-2:LRA=8:print_format=json', '-f', 'null', '-'])
    matches = re.findall(r'\{\s*"input_i".*?\}', result.stderr, re.S)
    if not matches:
        raise RuntimeError('Cannot measure saved audio')
    return json.loads(matches[-1])


def prepare():
    ffmpeg = editor.ffmpeg_binary()
    data = editor.load_timeline(ROOT / 'timeline.json')
    work = ROOT / 'audio' / 'assembly'
    work.mkdir(parents=True, exist_ok=True)
    measurements = {}
    for utterance in data['speech']:
        path = editor.asset(data, utterance['path'])
        reading = measure(ffmpeg, path)
        integrated = float(reading['input_i'])
        if not math.isfinite(integrated):
            raise RuntimeError(f"Silent speech file: {utterance['id']}")
        utterance['gain_db'] = round(max(-8, min(8, -19 - integrated)), 2)
        utterance['measured_duration'] = editor.wav_duration(path)
        utterance['_duration'] = utterance['measured_duration']
        measurements[utterance['id']] = reading

    soundbed = ROOT / 'audio' / 'soundbed_60s.wav'
    if not soundbed.exists():
        # Only known moving shots receive wheel sound. All stems are cached.
        moving = [(c['start'], c['duration']) for c in data['clips']
                  if c['id'] in ('establish', 'working', 'close')]
        graph = ['[0:a]atrim=duration=60,volume=-34dB,afade=t=in:d=1,afade=t=out:st=58:d=2[room]',
                 '[1:a]asplit=' + str(len(moving)) + ''.join(f'[w{i}]' for i in range(len(moving)))]
        labels = ['[room]']
        for index, (start, duration) in enumerate(moving):
            label = f'roll{index}'
            graph.append(f'[w{index}]atrim=duration={duration},asetpts=PTS-STARTPTS,'
                         f'volume=-28dB,afade=t=in:d=0.4,afade=t=out:st={duration-.6}:d=0.6,'
                         f'adelay={round(start*1000)}:all=1[{label}]')
            labels.append(f'[{label}]')
        graph.append(''.join(labels) + f'amix=inputs={len(labels)}:duration=first:normalize=0[out]')
        editor.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
                    '-stream_loop', '-1', '-i', ROOT / 'soundscape/corridor_loop.wav',
                    '-stream_loop', '-1', '-i', ROOT / 'soundscape/rubber_wheels.wav',
                    '-filter_complex', ';'.join(graph), '-map', '[out]', '-t', '60',
                    '-ar', '48000', '-ac', '2', '-c:a', 'pcm_s24le', soundbed])
    data['ambience'] = {'path': 'audio/soundbed_60s.wav', 'gain_db': 0,
                        'note': 'Room tone and wheels mixed locally from saved ElevenLabs stems; listening review pending.'}
    data['project']['notes'] = ('60-second editorial timing fitted to saved ElevenLabs speech. '
                                'Picture generation and visual/lip-sync review remain pending. '
                                'All sources preserved; no still-image substitutes.')

    # Create padded, sample-aligned inputs for future lip-sync jobs. These are
    # new editing derivatives, not regenerated speech.
    lip_manifest = []
    mapping = [('confirmation', 'worker_confirmation'), ('thanks', 'worker_thanks')]
    for speech_id, clip_id in mapping:
        speech = next(s for s in data['speech'] if s['id'] == speech_id)
        clip = next(c for c in data['clips'] if c['id'] == clip_id)
        lead = speech['start'] - clip['start']
        tail = clip['duration'] - lead - speech['measured_duration']
        if lead < 0 or tail < 0:
            raise RuntimeError(f'{speech_id} does not fit its planned speaking shot')
        target = ROOT / 'dialogue' / 'lipsync_inputs' / (speech_id + '_padded.wav')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            editor.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
                        '-i', editor.asset(data, speech['path']), '-af',
                        f'adelay={round(lead*1000)}:all=1,apad,atrim=duration={clip["duration"]}',
                        '-ar', '48000', '-ac', '1', '-c:a', 'pcm_s24le', target])
        lip_manifest.append({'speech_id': speech_id, 'clip_id': clip_id,
                             'path': str(target.relative_to(ROOT)), 'duration': clip['duration'],
                             'leading_silence': lead, 'trailing_silence': tail,
                             'status': 'Audio input prepared; no lip-sync video generated'})

    # Preserve the pre-mix editorial data and write the measured edit.
    backup = ROOT / 'audio' / 'timeline_before_mix.json'
    if not backup.exists():
        shutil.copy2(ROOT / 'timeline.json', backup)
    serializable = {k: v for k, v in data.items() if not k.startswith('_')}
    for speech in serializable['speech']:
        speech.pop('_duration', None)
    (ROOT / 'timeline.json').write_text(json.dumps(serializable, indent=2) + '\n')
    (ROOT / 'dialogue/lipsync_inputs/manifest.json').write_text(json.dumps(lip_manifest, indent=2) + '\n')
    mix_path = ROOT / 'audio' / 'context_aware_robot_60s_mix.wav'
    if not mix_path.exists():
        mixed = editor.mix_audio(data, ffmpeg, work)
        shutil.copy2(mixed, mix_path)
    measurements['mixed_audio'] = measure(ffmpeg, mix_path)
    measurements['mixed_audio']['duration_seconds'] = editor.wav_duration(mix_path)
    measurements['review'] = 'Automated level measurements only. Auditory review pending; runtime cannot receive audio input.'
    (ROOT / 'audio/measurements.json').write_text(json.dumps(measurements, indent=2) + '\n')
    print('Saved 60-second mix and lip-sync audio inputs. No API calls; source audio unchanged.')


if __name__ == '__main__':
    prepare()
