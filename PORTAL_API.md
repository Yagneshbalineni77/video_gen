# Dextora Creator — Portal Integration API (v1)

API reference for the **education portal ↔ video pipeline** integration.

The portal authors the scripts (existing curriculum API). **Dextora Creator** generates
the videos and exposes them here, **keyed by the portal's own script `id`** — so the
portal fetches results with the same `id`s it already has. No mapping, no webhook: the
portal **polls**.

---

## Base URL
```
https://videogen.dextora.org
```

## Authentication
Optional. If the server sets `PORTAL_API_KEY`, every request must include the key —
**either** way:
- **Header** (for API calls): `X-API-Key: <key>`
- **Query param** (for embeddable media): `?key=<key>` on the URL

The **media URLs** returned in responses (`video_url`, `audio_url`, `thumbnail_url`)
**already include `?key=...`** when a key is set — so you can drop them straight into a
`<video>` / `<img>` tag (browsers can't send the header on those). If `PORTAL_API_KEY`
is unset, everything is open.

## Content type
All responses are `application/json` (except the video file stream, which is `video/mp4`).

---

## 1. Generate a video (push)  ⭐ start here
The portal sends a full script; we render the video and store it keyed by **your own `id`**.

```
POST /api/v1/generate
```
**Body**
| Field | Type | Notes |
|---|---|---|
| `id` | int | **required** — your script id; all results are keyed by it |
| `audio_script` | string | **required** — the narration text |
| `title` | string | optional |
| `visual_directions` | string | optional — timestamped cues (`0:00-0:05 → …`) become scene prompts |
| `language` | string | `English` (default) or `Hindi` |
| `subject` | string | sets visual style (e.g. `Chemistry`, `History`) |
| `duration_seconds` | int | optional target length |
| `class_id` / `subject_id` / `chapter_id` | int | optional — stored for fetch-by-chapter |
| `script_type` | string | `VIDEO` (default) or `REEL` |

**curl**
```bash
curl -X POST "https://videogen.dextora.org/api/v1/generate" \
  -H "X-API-Key: <key>" -H "Content-Type: application/json" \
  -d '{"id":12345,"title":"Mole Concept","audio_script":"Today we explore the mole...","visual_directions":"0:00-0:10 → 3D molecules","language":"English","subject":"Chemistry","duration_seconds":240,"class_id":11,"subject_id":49,"chapter_id":509}'
```
**Response** `{ "id": 12345, "status": "queued" }`

→ then **poll #2** (`GET /api/v1/videos?ids=12345`) for status + video/audio/thumbnail URLs.

**Codes:** `200` queued · `400` missing `audio_script` · `401` bad key

---

## 2. List / fetch videos  ⭐ poll for results
Returns generation status + the playable URLs for one or more scripts.

```
GET /api/v1/videos
```

**Query (use either form):**
| Param | Type | Example | Notes |
|---|---|---|---|
| `ids` | string (csv) | `881,882,883` | fetch specific script ids |
| `class_id` | int | `11` | filter by class |
| `subject_id` | int | `49` | filter by subject |
| `chapter_id` | int | `509` | filter by chapter |

You can combine `class_id`+`subject_id`+`chapter_id`, **or** pass `ids`.

**curl**
```bash
curl "https://videogen.dextora.org/api/v1/videos?ids=881,882" -H "X-API-Key: <key>"
curl "https://videogen.dextora.org/api/v1/videos?class_id=11&subject_id=49&chapter_id=509" -H "X-API-Key: <key>"
```

**200 Response**
```json
{
  "count": 1,
  "videos": [
    {
      "id": 881,
      "atom_id": 2247,
      "chapter_id": 509,
      "subject_id": 49,
      "language": "English",
      "title": "Ancient Indian Atomic Theory: Paramãnu & Acharya Kanda",
      "status": "done",
      "duration": 240,
      "video_url": "https://videogen.dextora.org/api/v1/videos/881/file",
      "audio_url": "https://videogen.dextora.org/api/v1/videos/881/audio",
      "thumbnail_url": "https://videogen.dextora.org/api/v1/videos/881/thumbnail"
    }
  ]
}
```
- `audio_url` = narration-only MP3 (for a "Listen" feature).
- `thumbnail_url` = JPG poster image for the video.
- All three URLs are set when `status=done`.

**`status` values:** `queued` → `generating` → `done` → (`error`)
- `video_url` is **non-null only when `status == "done"`**.
- Scripts not yet submitted for generation simply won't appear in `videos`.

**Codes:** `200` OK · `401` invalid/missing `X-API-Key`

---

## 2. Stream / download a video file
The actual MP4 — this is exactly what `video_url` points to.

```
GET /api/v1/videos/{script_id}/file
```
**curl**
```bash
curl -L "https://videogen.dextora.org/api/v1/videos/881/file" -H "X-API-Key: <key>" -o 881.mp4
```
Returns the video (`Content-Type: video/mp4`). Embed `video_url` directly in an HTML
`<video>` tag or your player.

**Codes:** `200` OK · `401` bad key · `404` video not ready / file missing

---

## 2a. Narration audio (MP3)
The narration track on its own — for the portal's "Listen" feature. Extracted from
the final video on first request, then cached.

```
GET /api/v1/videos/{script_id}/audio
```
**curl**
```bash
curl -L "https://videogen.dextora.org/api/v1/videos/881/audio" -H "X-API-Key: <key>" -o 881.mp3
```
Returns `audio/mpeg`. This is what `audio_url` points to.

**Codes:** `200` OK · `401` bad key · `404` not ready

---

## 2c. Thumbnail (JPG)
A poster image for the video — the clean scene-1 keyframe (or a frame from the video).
Generated on first request, then cached.

```
GET /api/v1/videos/{script_id}/thumbnail
```
**curl**
```bash
curl -L "https://videogen.dextora.org/api/v1/videos/881/thumbnail" -H "X-API-Key: <key>" -o 881.jpg
```
Returns `image/jpeg` (1280px wide). This is what `thumbnail_url` points to.

**Codes:** `200` OK · `401` bad key · `404` not ready

---

## 2b. Per-run backend log + progress
Watch a single render's live progress and pipeline log — no host shell needed.

```
GET /api/v1/videos/{script_id}/log?lines=80
```
**curl**
```bash
curl "https://videogen.dextora.org/api/v1/videos/1531/log?lines=60" -H "X-API-Key: <key>"
```
**200 Response**
```json
{
  "script_id": 1531,
  "phase": "Generating visuals",
  "percent": 45,
  "scenes_done": 5,
  "scenes_total": 11,
  "log_tail": "...last N lines of the pipeline log..."
}
```
- `phase` ∈ Writing script → Generating visuals → Assembling clips → Finalizing.
- `lines` (optional, default 80, max 1000) = how many log lines to return.

**Codes:** `200` OK · `401` bad key · `404` no log for this script yet

---

## 3. Trigger generation for a chapter  (optional · admin)
Kick off video generation for a whole chapter. Runs in the background; poll **#1** for
results. *(Optional — Dextora can also start generation on its side.)*

```
POST /api/v1/bridge/run
Authorization: Bearer <admin-token>
Content-Type: application/json
```

**Body**
| Field | Type | Default | Notes |
|---|---|---|---|
| `class_id` | int | — | required |
| `subject_id` | int | — | required |
| `chapter_id` | int | — | required |
| `subject` | string | `""` | selects visual style (e.g. `"Chemistry"`) |
| `languages` | string[] | `["English"]` | `["English","Hindi"]` |
| `types` | string[] | `["VIDEO"]` | `["VIDEO","REEL"]` |
| `limit` | int\|null | `null` | cap how many to generate |
| `workers` | int | `2` | videos generated in parallel |

**curl**
```bash
curl -X POST "https://videogen.dextora.org/api/v1/bridge/run" \
  -H "Authorization: Bearer <admin-token>" -H "Content-Type: application/json" \
  -d '{"class_id":11,"subject_id":49,"chapter_id":509,"subject":"Chemistry","languages":["English"],"types":["VIDEO"],"limit":5}'
```
**202/200 Response** `{ "started": true, "class_id": 11, ... }`

**Codes:** `200` started · `401` not logged in · `403` not an admin

---

## Typical integration flow
1. Generation is triggered (by us, or via **#3**).
2. Portal **polls #1** (`GET /api/v1/videos?...`) and watches `status`.
3. When `status == "done"`, take `video_url` and embed/play it (served by **#2**).

## Notes
- **Keyed by your `id`** — fetch with the same script ids you already store.
- **No webhook** — poll #1; it's cheap.
- **CDN later:** `video_url` streams from our host today; we can repoint it to a
  bucket/CDN with **no change to these endpoints** (same response shape).
- **Languages:** `English → indian_english`, `Hindi → hindi` voices applied automatically.
