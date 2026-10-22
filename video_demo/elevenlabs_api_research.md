# ElevenLabs API research

Verified against official ElevenLabs documentation on 2026-10-20. Research only: no credentials were read and no generation requests were made.

## Availability and authentication

The Image & Video API is publicly documented and requires **Pro or above**, with **Image & Video or Flows permission** on the API key. Lower tiers return `402 paid_plan_required`. ByteDance models require separate approval and otherwise return `model_access_denied`. All calls below use base URL `https://api.elevenlabs.io` and header `xi-api-key: <secret>`.

Sources: [quickstart](https://elevenlabs.io/docs/eleven-api/guides/cookbooks/image-and-video), [launch announcement](https://elevenlabs.io/blog/introducing-the-image-video-and-templates-apis).

## Endpoints and practical payloads

| Action | Endpoint |
|---|---|
| Submit video or lip sync | `POST /v1/flows/video` |
| Fetch video status/result | `GET /v1/flows/video/{generation_id}` |
| List API video generations | `GET /v1/flows/video` |
| Submit image | `POST /v1/flows/image` |
| Fetch image status/result | `GET /v1/flows/image/{generation_id}` |
| Upload reusable media | `POST /v1/assets` |
| Fetch uploaded media | `GET /v1/assets/{asset_id}` |
| List available speech voices | `GET /v2/voices?page_size=100` |
| List speech models and rates | `GET /v1/models` |
| Generate speech + character timings | `POST /v1/text-to-speech/{voice_id}/with-timestamps` |
| Check subscription / consumption | `GET /v1/user/subscription` |

Sources: [video create](https://elevenlabs.io/docs/api-reference/flows/video/create), [video get](https://elevenlabs.io/docs/api-reference/flows/video/get), [image create](https://elevenlabs.io/docs/api-reference/flows/image/create), [asset upload](https://elevenlabs.io/docs/api-reference/assets/create), [voices](https://elevenlabs.io/docs/api-reference/voices/search), [speech models](https://elevenlabs.io/docs/api-reference/models/list).

For read-only permission checks, call `GET /v1/flows/video?page_size=1` and `GET /v1/flows/image?page_size=1`. These list previous API generations without starting one. `GET /v1/user` also returns a nested `subscription`, though it may require the same user-read permission as the subscription endpoint. Sources: [video listing](https://elevenlabs.io/docs/api-reference/flows/video/list), [image listing](https://elevenlabs.io/docs/api-reference/flows/image/list), [user](https://elevenlabs.io/docs/api-reference/user/get).

Veo example, intended for a single planned shot:

```json
{
  "model_id": "veo-3.1-fast-generate-001",
  "prompt": "<precise shot description>",
  "duration_secs": 8,
  "aspect_ratio": "16:9",
  "resolution": "1080p",
  "generate_audio": false,
  "enhance_prompt": false,
  "start_frame": {"type": "asset", "asset_id": "<uploaded frame>"}
}
```

Both Veo 3.1 models support durations 4/6/8 seconds, 16:9 or 9:16, 720p/1080p/4K, optional `negative_prompt` and `seed`. `end_frame` requires `start_frame`. Alternatively use up to three references: `"images": [{"image": {"type":"asset", "asset_id":"..."}, "role":"subject"}]`; `role` is `subject` or `style`, duration must be eight seconds, and `images` cannot be combined with start/end frames. Source: [video request schema](https://elevenlabs.io/docs/api-reference/flows/video/create).

Image example; `images` is optional and uses plain media references:

```json
{
  "model_id": "gemini-3.1-flash-image",
  "prompt": "<composition and character continuity>",
  "aspect_ratio": "16:9",
  "resolution": "1K",
  "images": [{"type": "asset", "asset_id": "<reference>"}]
}
```

This image model accepts up to 14 references and resolutions `512`, `1K`, `2K`, `4K`. Other documented API image models include Gemini 2.5 Flash, Gemini 3 Pro, GPT Image 1/1.5/2/2.5 Sunburst/2.5 Flare, and approval-gated Seedream 5 Lite/Pro. Parameters vary by model; unknown fields are rejected. Source: [image schema](https://elevenlabs.io/docs/api-reference/flows/image/create).

Creatify Aurora lip sync example:

```json
{
  "model_id": "creatify-aurora",
  "image": {"type": "asset", "asset_id": "<character portrait>"},
  "audio": {"type": "asset", "asset_id": "<speech mp3>"},
  "resolution": "720p"
}
```

Aurora requires `image` and `audio`. Optional fields are `guidance_scale`, `audio_guidance_scale`, `resolution` (480p/720p), and `webhook`. It accepts no `prompt`, `duration_secs`, or aspect ratio. Source: [video schema](https://elevenlabs.io/docs/api-reference/flows/video/create).

## References, uploads and result collection

Upload with multipart form fields **`asset`** (file) and **`name`** (display name); preserve the returned `asset_id`. References can be `{type:"asset",asset_id:"..."}`, `{type:"generation",generation_id:"..."}`, or `{type:"inline_base64",content_base64:"...",mime_type:"..."}`. Inline files have a 25 MB decoded limit. Audio supports MP3/WAV; image supports JPEG/PNG/WebP/HEIC/HEIF; video supports MP4/QuickTime/WebM. Reusable inputs should use assets, since inline media is ephemeral. Sources: [uploads](https://elevenlabs.io/docs/api-reference/assets/create), [references](https://elevenlabs.io/docs/eleven-api/guides/how-to/image-and-video/references).

Creation returns `{id,status:"pending"}`. Poll images at intervals of at least two seconds and videos at least ten seconds, with backoff and a timeout. States are `pending`, `generating`, `completed`, `failed`. Completed responses include `content_url` and `content_mime_type`; download the file promptly because URLs expire in about one hour. Fetch status again to obtain a fresh URL. Save the generation ID before waiting so a retry resumes polling instead of submitting another paid generation. Source: [quickstart](https://elevenlabs.io/docs/eleven-api/guides/cookbooks/image-and-video).

## Speech, voices and subtitle timing

Read `GET /v2/voices`, paginate with `next_page_token` and `has_more`, and choose a returned voice ID. Results include descriptions/labels and preview URLs. `GET /v1/models` exposes TTS capability flags, request length limits, and `model_rates.character_cost_multiplier` / `cost_discount_multiplier`; it is a **speech model catalog**, not the image/video catalog. Sources: [voices](https://elevenlabs.io/docs/api-reference/voices/search), [models](https://elevenlabs.io/docs/api-reference/models/list).

```json
{
  "text": "<final narration>",
  "model_id": "eleven_multilingual_v2",
  "voice_settings": {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0,
    "use_speaker_boost": true,
    "speed": 1
  }
}
```

The timestamp endpoint returns `audio_base64`, `alignment`, and `normalized_alignment`. Alignment arrays contain characters and start/end times in seconds; join characters into words/caption cues locally. Default output is `mp3_44100_128`; request `?output_format=mp3_44100_192` on Creator+, or WAV 44.1 kHz on Pro+. Store audio and timing JSON together. Source: [speech with timing](https://elevenlabs.io/docs/api-reference/text-to-speech/convert-with-timestamps).

For generation chaining, asynchronous `POST /v1/flows/text-to-speech` accepts `model_id`, `text`, and **`voice`** (different from the synchronous path parameter); its returned generation ID can feed Aurora's audio reference. This route has no documented character-timing response, so the synchronous timestamp route plus upload is more direct when subtitles are required. Source: [Flows speech generation](https://elevenlabs.io/docs/api-reference/flows/text-to-speech/create).

## Credit safeguards and pricing limits

`GET /v1/user/subscription` provides `tier`, `character_count`, `character_limit`, `current_overage`, and **`max_credit_limit_extension`**. The last field is authoritative for the extension cap: **0 disables usage-based billing**, a positive integer permits that many additional credits, and `"unlimited"` has no cap. Do not infer disabled overage from a zero current overage. Source: [subscription schema](https://elevenlabs.io/docs/api-reference/user/subscription/get).

All products share the monthly credit pool. Multilingual V2 is approximately one credit per input character; Flash/Turbo API rates range from 0.5 to one credit per character, depending on model/plan. Obtain current account model rates before budgeting. Source: [official pricing](https://elevenlabs.io/pricing).

Image/video pricing depends on model, parameters, and inputs. Official docs say API and app pricing match, with the selected credit cost visible in the app. No public image/video price-estimation endpoint or numeric rate table was found in the documented endpoint set. Avoid inventing a numeric estimate. Source: [generation pricing](https://elevenlabs.io/docs/eleven-api/guides/cookbooks/image-and-video#pricing).

For a strict existing-credit-only workflow, verify overage is disabled (or a separately verified key quota prevents exceeding included balance), impose a local project cap, serialize paid submissions, record every generation ID, and re-read consumption before each next submission. Never retry an ambiguous POST blindly. Failed image/video generations are documented as uncharged. API keys support custom credit quotas in the dashboard. Sources: [API key controls](https://elevenlabs.io/docs/overview/administration/workspaces/api-keys), [generation failures](https://elevenlabs.io/docs/eleven-api/guides/cookbooks/image-and-video#handling-failures).

A reviewable budget can therefore be expressed as `sum(app-quoted shot costs) + sum(app-quoted still costs) + app-quoted lip-sync costs + narration character count × account model rate`, with a separately reserved retry allowance. A 60-second narration containing 750 characters is about 750 Multilingual V2 credits before any unusual voice-specific pricing; image/video costs must be filled from verified selected-setting quotes. This is an estimate derived from the documented rate, not an account-specific quote.
