"""
Phase 5 – YouTube Publisher
Uploads final_render.mp4 + sets thumbnail using YouTube Data API v3.

OAuth2 flow:
  - First local run: opens browser for consent → saves token to credentials/youtube_token.json
  - Subsequent / CI runs: refreshes token silently (token must be pre-committed to secrets)

Upload pattern from ChaituRajSagar/gemini-youtube-automation uploader.py:
  - Resumable chunked upload (robust for large files)
  - thumbnails.set after video is processed
"""
from __future__ import annotations

import logging
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from schemas.models import PublishResult, RenderResult, VideoScript
from utils.ffmpeg_helpers import generate_thumbnail

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB


class YouTubePublisher:
    def __init__(self) -> None:
        self._youtube = None if settings.DRY_RUN else self._authenticate()
        self._thumb_dir = settings.OUTPUT_DIR / "thumbnails"
        self._thumb_dir.mkdir(parents=True, exist_ok=True)

    # ── public ──────────────────────────────────────────────────────────────

    def run(self, render: RenderResult, script: VideoScript) -> PublishResult:
        if settings.DRY_RUN:
            log.warning("DRY_RUN=true – skipping YouTube upload")
            return PublishResult(
                youtube_video_id="dry_run_id",
                youtube_url="https://youtube.com/watch?v=dry_run_id",
                title=script.metadata.title,
            )

        log.info("Phase 5 – uploading to YouTube: '%s'", script.metadata.title)
        video_id = self._upload_video(render.final_video_path, script)
        log.info("  [upload] video_id: %s", video_id)

        thumb_path = self._make_thumbnail(script)
        self._upload_thumbnail(video_id, thumb_path)
        log.info("  [upload] thumbnail set")

        url = f"https://www.youtube.com/watch?v={video_id}"
        log.info("Phase 5 done → %s", url)

        return PublishResult(
            youtube_video_id=video_id,
            youtube_url=url,
            title=script.metadata.title,
        )

    # ── private ─────────────────────────────────────────────────────────────

    def _authenticate(self):
        creds: Credentials | None = None
        token_path = settings.YOUTUBE_TOKEN_PATH

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not settings.YOUTUBE_CLIENT_SECRETS_PATH.exists():
                    raise FileNotFoundError(
                        f"YouTube client_secrets.json not found at "
                        f"{settings.YOUTUBE_CLIENT_SECRETS_PATH}. "
                        "Download it from Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(settings.YOUTUBE_CLIENT_SECRETS_PATH), SCOPES
                )
                creds = flow.run_local_server(port=0)

            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json(), encoding="utf-8")
            log.info("YouTube token saved → %s", token_path)

        return build("youtube", "v3", credentials=creds)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=60))
    def _upload_video(self, video_path: Path, script: VideoScript) -> str:
        meta = script.metadata
        body = {
            "snippet": {
                "title": meta.title,
                "description": meta.description,
                "tags": meta.tags,
                "categoryId": settings.YT_CATEGORY_ID,
                "defaultLanguage": settings.YT_DEFAULT_LANGUAGE,
            },
            "status": {
                "privacyStatus": settings.YT_PRIVACY_STATUS,
                "selfDeclaredMadeForKids": False,
            },
        }

        media = MediaFileUpload(
            str(video_path),
            mimetype="video/mp4",
            resumable=True,
            chunksize=CHUNK_SIZE,
        )
        request = self._youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=media,
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                log.info("  [upload] %d%%", pct)

        return response["id"]

    def _make_thumbnail(self, script: VideoScript) -> Path:
        dest = self._thumb_dir / "thumbnail.jpg"
        generate_thumbnail(
            concept=script.metadata.thumbnail_concept,
            title=script.metadata.title,
            dest=dest,
        )
        return dest

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=30))
    def _upload_thumbnail(self, video_id: str, thumb_path: Path) -> None:
        media = MediaFileUpload(str(thumb_path), mimetype="image/jpeg")
        self._youtube.thumbnails().set(
            videoId=video_id,
            media_body=media,
        ).execute()
