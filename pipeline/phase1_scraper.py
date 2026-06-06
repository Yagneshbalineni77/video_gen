"""
Phase 1 – Trend Scraper
Primary: YouTube Data API v3 search + statistics.
Fallback: Gemini generates realistic trending topic ideas when the YouTube
          API key is missing or blocked (common with AI Studio keys).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from schemas.models import TrendCandidate, TrendReport

log = logging.getLogger(__name__)

_GEMINI_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title":         {"type": "string"},
            "channel_title": {"type": "string"},
            "views":         {"type": "integer"},
        },
        "required": ["title", "channel_title", "views"],
    },
}


class TrendScraper:
    def __init__(self) -> None:
        self._yt = None
        if settings.YOUTUBE_API_KEY:
            try:
                from googleapiclient.discovery import build
                self._yt = build("youtube", "v3", developerKey=settings.YOUTUBE_API_KEY)
            except Exception as e:
                log.warning("YouTube client init failed (%s) – will use Gemini fallback", e)

    # ── public ──────────────────────────────────────────────────────────────

    def run(self) -> TrendReport:
        log.info("Phase 1 – scraping trends for keywords: %s", settings.NICHE_KEYWORDS)
        candidates: list[TrendCandidate] = []

        for keyword in settings.NICHE_KEYWORDS:
            results = self._fetch_keyword(keyword)
            candidates.extend(results)
            log.info("  '%s' -> %d candidates", keyword, len(results))

        if not candidates:
            raise RuntimeError("Phase 1: no candidates found.")

        seen: set[str] = set()
        unique: list[TrendCandidate] = []
        for c in candidates:
            if c.video_id not in seen:
                seen.add(c.video_id)
                unique.append(c)

        ranked = sorted(unique, key=lambda c: c.view_velocity, reverse=True)
        max_vel = ranked[0].view_velocity if ranked else 1.0
        for c in ranked:
            c.velocity_score = round(c.view_velocity / max_vel, 4)

        top = ranked[: settings.TOP_TRENDS_COUNT]
        report = TrendReport(candidates=ranked, top_concepts=top)
        self._save_report(report)
        log.info("Phase 1 done – top trend: '%s' (velocity=%.0f v/day)", top[0].title, top[0].view_velocity)
        return report

    # ── private ─────────────────────────────────────────────────────────────

    def _fetch_keyword(self, keyword: str) -> list[TrendCandidate]:
        if self._yt:
            try:
                return self._search_yt(keyword)
            except Exception as e:
                log.warning("YouTube API failed for '%s' (%s) – falling back to Gemini", keyword, e)
        return self._gemini_topics(keyword)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
    def _search_yt(self, keyword: str) -> list[TrendCandidate]:
        assert self._yt is not None
        published_after = (datetime.now(timezone.utc) - timedelta(days=7)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        search_resp = (
            self._yt.search()
            .list(
                q=keyword,
                part="id,snippet",
                type="video",
                publishedAfter=published_after,
                order="viewCount",
                maxResults=25,
                relevanceLanguage="en",
                regionCode="US",
            )
            .execute()
        )
        video_ids = [
            item["id"]["videoId"]
            for item in search_resp.get("items", [])
            if item["id"].get("videoId")
        ]
        if not video_ids:
            return []

        stats_resp = (
            self._yt.videos()
            .list(part="statistics,snippet", id=",".join(video_ids))
            .execute()
        )
        candidates: list[TrendCandidate] = []
        now = datetime.now(timezone.utc)
        for item in stats_resp.get("items", []):
            views = int(item["statistics"].get("viewCount", 0))
            if views < 50_000:
                continue
            published_str = item["snippet"]["publishedAt"]
            published_dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            days_old = max((now - published_dt).total_seconds() / 86400, 0.1)
            candidates.append(TrendCandidate(
                video_id=item["id"],
                title=item["snippet"]["title"],
                channel_title=item["snippet"]["channelTitle"],
                views=views,
                published_at=published_dt,
                days_since_upload=round(days_old, 2),
                view_velocity=round(views / days_old, 2),
                niche_keyword=keyword,
            ))
        return candidates

    def _gemini_topics(self, keyword: str) -> list[TrendCandidate]:
        log.info("  [fallback] asking Gemini for trending '%s' topics…", keyword)
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        prompt = (
            f"Generate 5 trending YouTube video titles and channel names for the niche: '{keyword}'.\n"
            "These should be realistic high-performing titles (500k–2M views) from the past week.\n"
            "Focus on shocking historical facts, untold stories, dark secrets, or mysteries.\n"
            "For views, use realistic numbers between 400000 and 2000000."
        )
        response = client.models.generate_content(
            model=settings.GEMINI_SCRIPT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_GEMINI_SCHEMA,
                temperature=1.0,
            ),
        )
        items = json.loads(response.text or "[]")
        now = datetime.now(timezone.utc)
        candidates = []
        for i, item in enumerate(items):
            views = int(item.get("views", 700_000))
            days_old = 3.0 + i * 0.5
            candidates.append(TrendCandidate(
                video_id=f"gemini_{keyword[:8]}_{i}",
                title=item["title"],
                channel_title=item.get("channel_title", "AI Generated"),
                views=views,
                published_at=now - timedelta(days=days_old),
                days_since_upload=days_old,
                view_velocity=round(views / days_old, 2),
                niche_keyword=keyword,
            ))
        log.info("  [fallback] Gemini returned %d topic ideas", len(candidates))
        return candidates

    def _save_report(self, report: TrendReport) -> None:
        out = settings.OUTPUT_DIR / "trends.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        log.info("Trend report saved -> %s", out)
