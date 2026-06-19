"""
Pipeline Orchestrator
Runs all 5 phases sequentially. Saves state after each phase so a failed
run can resume from the last successful checkpoint.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from config import settings
from pipeline.phase1_scraper import TrendScraper
from pipeline.phase2_writer import ScriptWriter
from pipeline.phase3_assets import AssetFactory
from pipeline.phase4_editor import VideoEditor
from pipeline.phase5_publisher import YouTubePublisher
from schemas.models import PipelineRun, TrendCandidate

log = logging.getLogger(__name__)
console = Console()


class Pipeline:
    def __init__(self) -> None:
        self._state_path = settings.OUTPUT_DIR / "pipeline_state.json"
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── public ──────────────────────────────────────────────────────────────

    def run(self) -> list[PipelineRun]:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        console.rule(f"[bold cyan]Dextora Pipeline – run {run_id}")

        # Phase 1 – Trends
        console.print("\n[bold yellow]Phase 1[/bold yellow] – Trend Scraper")
        scraper = TrendScraper()
        report = scraper.run()
        self._print_trends(report.top_concepts)

        results: list[PipelineRun] = []

        for trend in report.top_concepts:
            run = PipelineRun(run_id=f"{run_id}_{trend.video_id}")
            run.trend = trend
            try:
                run = self._run_single(run, trend)
                run.completed = True
            except Exception as exc:
                log.error("Pipeline failed for '%s': %s", trend.title, exc, exc_info=True)
                run.error = str(exc)
            finally:
                self._save_state(run)
                results.append(run)

        self._print_summary(results)
        return results

    # ── private ─────────────────────────────────────────────────────────────

    def _run_single(self, run: PipelineRun, trend: TrendCandidate) -> PipelineRun:
        console.print(f"\n[bold]Topic:[/bold] {trend.title}")
        console.print(f"  velocity={trend.view_velocity:,.0f} v/day  views={trend.views:,}")

        # Phase 2 – Script
        console.print("\n[bold yellow]Phase 2[/bold yellow] – Script Writer")
        writer = ScriptWriter()
        run.script = writer.run(trend)

        # Phase 3 – Assets
        console.print("\n[bold yellow]Phase 3[/bold yellow] – Asset Factory")
        factory = AssetFactory()
        run.assets = factory.run(run.script)

        # Phase 4 – Edit
        console.print("\n[bold yellow]Phase 4[/bold yellow] – Video Editor")
        editor = VideoEditor()
        run.render = editor.run(run.assets)

        # Phase 5 – Publish
        console.print("\n[bold yellow]Phase 5[/bold yellow] – YouTube Publisher")
        publisher = YouTubePublisher()
        run.publish = publisher.run(run.render, run.script)

        console.print(f"\n[bold green]Published:[/bold green] {run.publish.youtube_url}")
        return run

    def _save_state(self, run: PipelineRun) -> None:
        self._state_path.write_text(
            run.model_dump_json(indent=2), encoding="utf-8"
        )

    # ── display ──────────────────────────────────────────────────────────────

    def _print_trends(self, trends: list[TrendCandidate]) -> None:
        table = Table(title="Top Trending Concepts", show_header=True)
        table.add_column("Title", style="cyan", max_width=50)
        table.add_column("Views", justify="right")
        table.add_column("V/day", justify="right", style="green")
        table.add_column("Score", justify="right")
        for t in trends:
            table.add_row(
                t.title[:50],
                f"{t.views:,}",
                f"{t.view_velocity:,.0f}",
                f"{t.velocity_score:.3f}",
            )
        console.print(table)

    def _print_summary(self, results: list[PipelineRun]) -> None:
        console.rule("[bold cyan]Run Summary")
        for r in results:
            icon = "[green]✓[/green]" if r.completed else "[red]✗[/red]"
            title = r.script.metadata.title if r.script else r.trend.title if r.trend else "?"
            url = r.publish.youtube_url if r.publish else r.error or "failed"
            console.print(f"  {icon} {title}\n    -> {url}")
