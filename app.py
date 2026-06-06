"""
Faceless AI Studio — demo frontend
Run: streamlit run app.py
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
OUTPUT = ROOT / "output"

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Faceless AI Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── theme ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* --- base --- */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d0d14; color: #e8e0d0; }

/* --- sidebar --- */
section[data-testid="stSidebar"] {
    background: #10101a;
    border-right: 1px solid #1e1e30;
}
section[data-testid="stSidebar"] * { color: #c8c0b8 !important; }

/* --- cards --- */
.card {
    background: #161622;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-title {
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #666688;
    margin-bottom: 8px;
}

/* --- phase stepper --- */
.stepper { display: flex; gap: 0; margin-bottom: 28px; }
.step {
    flex: 1;
    text-align: center;
    padding: 10px 4px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    border-bottom: 3px solid #1e1e30;
    color: #444466;
    transition: all 0.3s;
}
.step.done   { border-bottom-color: #4ade80; color: #4ade80; }
.step.active { border-bottom-color: #f5a623; color: #f5a623; }
.step.error  { border-bottom-color: #f87171; color: #f87171; }

/* --- log terminal --- */
.log-box {
    background: #06060c;
    border: 1px solid #1e1e30;
    border-radius: 8px;
    padding: 14px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #8888aa;
    max-height: 320px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.6;
}

/* --- scene card --- */
.scene {
    background: #0f0f1c;
    border: 1px solid #1e1e30;
    border-left: 3px solid #f5a623;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.scene-id {
    font-size: 10px;
    letter-spacing: 2px;
    color: #f5a623;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.narration { font-size: 15px; color: #e8e0d0; line-height: 1.7; }
.prompt    { font-size: 12px; color: #555577; margin-top: 8px; font-style: italic; }

/* --- tags --- */
.tag {
    display: inline-block;
    background: #1e1e30;
    color: #8888aa;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    margin: 2px;
}

/* --- stats row --- */
.stats { display: flex; gap: 24px; flex-wrap: wrap; }
.stat { text-align: center; }
.stat-val { font-size: 28px; font-weight: 700; color: #f5a623; }
.stat-lbl { font-size: 11px; color: #555577; letter-spacing: 1px; text-transform: uppercase; }

/* --- generate button --- */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #f5a623, #e8891a) !important;
    color: #0d0d14 !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #ffc044, #f5a623) !important;
    transform: translateY(-1px);
}

/* --- headings --- */
h1 { color: #e8e0d0 !important; font-size: 28px !important; }
h2, h3 { color: #c8c0b8 !important; }

/* --- misc --- */
.stTextInput input, .stSelectbox select, .stTextArea textarea {
    background: #10101a !important;
    border-color: #1e1e30 !important;
    color: #e8e0d0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── session state ─────────────────────────────────────────────────────────────

def _init():
    defaults = {
        "phase_status": {1: "idle", 2: "idle", 3: "idle", 4: "idle", 5: "idle"},
        "logs": [],
        "running": False,
        "script": None,
        "error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── helpers ───────────────────────────────────────────────────────────────────

PHASE_LABELS = {
    1: "01 Trends",
    2: "02 Script",
    3: "03 Assets",
    4: "04 Edit",
    5: "05 Publish",
}

VOICES = ["Charon", "Fenrir", "Kore", "Aoede", "Puck", "Leda"]


def _stepper_html(status: dict[int, str]) -> str:
    parts = []
    for pid, label in PHASE_LABELS.items():
        cls = status.get(pid, "idle")
        icon = {"done": "✓ ", "active": "◉ ", "error": "✗ "}.get(cls, "○ ")
        parts.append(f'<div class="step {cls}">{icon}{label}</div>')
    return '<div class="stepper">' + "".join(parts) + "</div>"


def _log_html(lines: list[str]) -> str:
    colored = []
    for ln in lines[-80:]:
        if "[ERROR]" in ln or "Error" in ln or "failed" in ln.lower():
            colored.append(f'<span style="color:#f87171">{ln}</span>')
        elif "[INFO]" in ln:
            colored.append(f'<span style="color:#6868aa">{ln}</span>')
        elif "OK -" in ln or "done" in ln.lower() or "saved" in ln.lower():
            colored.append(f'<span style="color:#4ade80">{ln}</span>')
        else:
            colored.append(ln)
    return '<div class="log-box">' + "<br>".join(colored) + "</div>"


def _run_phase(phase: int, extra_env: dict) -> bool:
    env = {**os.environ, **extra_env, "PYTHONUTF8": "1"}
    ffmpeg_bin = (
        r"C:\Users\kotes\AppData\Local\Microsoft\WinGet\Packages"
        r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        r"\ffmpeg-8.1.1-full_build\bin"
    )
    env["PATH"] = env.get("PATH", "") + os.pathsep + ffmpeg_bin

    cmd = [sys.executable, str(ROOT / "main.py"), "--phase", str(phase)]
    st.session_state["phase_status"][phase] = "active"
    st.session_state["logs"].append(f"▶ Starting phase {phase}…")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        env=env,
    )
    for line in proc.stdout:
        st.session_state["logs"].append(line.rstrip())
    proc.wait()

    if proc.returncode == 0:
        st.session_state["phase_status"][phase] = "done"
        return True
    else:
        st.session_state["phase_status"][phase] = "error"
        st.session_state["logs"].append(f"✗ Phase {phase} failed (exit {proc.returncode})")
        return False


def _load_script() -> dict | None:
    f = OUTPUT / "script.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _render_script(script: dict):
    meta = script.get("metadata", {})
    scenes = script.get("scenes", [])

    st.markdown(f"""
    <div class="card">
        <div class="card-title">Video Title</div>
        <div style="font-size:20px;font-weight:700;color:#f5a623">{meta.get('title','')}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <div class="card-title">Stats</div>
        <div class="stats">
            <div class="stat">
                <div class="stat-val">{len(scenes)}</div>
                <div class="stat-lbl">Scenes</div>
            </div>
            <div class="stat">
                <div class="stat-val">{sum(s.get('duration_ms') or 0 for s in scenes)//1000}s</div>
                <div class="stat-lbl">Duration</div>
            </div>
            <div class="stat">
                <div class="stat-val">{len(meta.get('tags',[]))}</div>
                <div class="stat-lbl">Tags</div>
            </div>
        </div>
        <div style="margin-top:12px">
            {"".join(f'<span class="tag">#{t}</span>' for t in meta.get('tags', [])[:12])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card-title" style="margin:20px 0 12px">Scenes</div>', unsafe_allow_html=True)
    for scene in scenes:
        st.markdown(f"""
        <div class="scene">
            <div class="scene-id">Scene {scene['scene_id']}</div>
            <div class="narration">{scene['narration']}</div>
            <div class="prompt">🎬 {scene.get('video_prompt','')[:120]}…</div>
        </div>
        """, unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎬 Faceless AI Studio")
    st.markdown("---")

    st.markdown("### Settings")
    topic_override = st.text_area(
        "Topic override",
        placeholder="Leave blank to auto-detect trending topic…",
        height=80,
    )
    niche = st.text_input("Niche keywords", value="dark history,untold history")
    voice = st.selectbox("Narrator voice", VOICES, index=0)
    duration = st.slider("Target duration (s)", 60, 180, 90, step=15)
    dry_run = st.toggle("Dry run (skip upload)", value=True)

    st.markdown("---")
    st.markdown("### Run phases")
    run_p1 = st.checkbox("Phase 1 – Trends", value=not bool(topic_override))
    run_p2 = st.checkbox("Phase 2 – Script", value=True)
    run_p3 = st.checkbox("Phase 3 – Assets (TTS + Veo)", value=False)
    run_p4 = st.checkbox("Phase 4 – Edit (FFmpeg)", value=False)
    run_p5 = st.checkbox("Phase 5 – Publish", value=False)
    st.markdown("---")

    generate = st.button("⚡ Generate Video", use_container_width=True)


# ── main ──────────────────────────────────────────────────────────────────────

st.markdown("# 🎬 Faceless AI Studio")
st.markdown(
    '<p style="color:#555577;margin-top:-12px;margin-bottom:24px">'
    "Automated dark-history YouTube channel · Powered by Gemini + Veo 3.0</p>",
    unsafe_allow_html=True,
)

stepper_slot = st.empty()
stepper_slot.markdown(_stepper_html(st.session_state["phase_status"]), unsafe_allow_html=True)

log_slot = st.empty()

# ── run pipeline ──────────────────────────────────────────────────────────────

if generate and not st.session_state["running"]:
    st.session_state["running"] = True
    st.session_state["logs"] = []
    st.session_state["error"] = None
    st.session_state["phase_status"] = {p: "idle" for p in range(1, 6)}

    extra_env: dict[str, str] = {
        "NICHE_KEYWORDS": niche,
        "GEMINI_TTS_VOICE": voice,
        "TARGET_VIDEO_DURATION": str(duration),
        "DRY_RUN": "true" if dry_run else "false",
    }

    phases_to_run = []
    if run_p1 and not topic_override:
        phases_to_run.append(1)
    if run_p2:
        phases_to_run.append(2)
    if run_p3:
        phases_to_run.append(3)
    if run_p4:
        phases_to_run.append(4)
    if run_p5:
        phases_to_run.append(5)

    # If topic override, write a mock trend file then skip phase 1
    if topic_override and run_p2:
        import datetime as _dt
        OUTPUT.mkdir(exist_ok=True)
        mock_trend = {
            "scraped_at": _dt.datetime.utcnow().isoformat(),
            "candidates": [],
            "top_concepts": [{
                "video_id": "manual",
                "title": topic_override,
                "channel_title": "Manual",
                "views": 500000,
                "published_at": _dt.datetime.utcnow().isoformat(),
                "days_since_upload": 2.0,
                "view_velocity": 250000.0,
                "niche_keyword": niche.split(",")[0].strip(),
                "velocity_score": 1.0,
            }]
        }
        (OUTPUT / "trends.json").write_text(json.dumps(mock_trend), encoding="utf-8")
        st.session_state["logs"].append("Topic override saved to trends.json")

    ok = True
    for phase in phases_to_run:
        stepper_slot.markdown(_stepper_html(st.session_state["phase_status"]), unsafe_allow_html=True)
        log_slot.markdown(_log_html(st.session_state["logs"]), unsafe_allow_html=True)
        ok = _run_phase(phase, extra_env)
        stepper_slot.markdown(_stepper_html(st.session_state["phase_status"]), unsafe_allow_html=True)
        log_slot.markdown(_log_html(st.session_state["logs"]), unsafe_allow_html=True)
        if not ok:
            st.session_state["error"] = f"Phase {phase} failed — check logs above"
            break

    if ok:
        st.session_state["script"] = _load_script()

    st.session_state["running"] = False


# ── always render logs if any ─────────────────────────────────────────────────

if st.session_state["logs"]:
    log_slot.markdown(_log_html(st.session_state["logs"]), unsafe_allow_html=True)

if st.session_state.get("error"):
    st.error(st.session_state["error"])

# ── results ───────────────────────────────────────────────────────────────────

script = st.session_state.get("script") or _load_script()

if script:
    tab_script, tab_audio, tab_video = st.tabs(["📄 Script", "🔊 Audio", "🎬 Video"])

    with tab_script:
        _render_script(script)

    with tab_audio:
        audio_dir = OUTPUT / "audio"
        wavs = sorted(audio_dir.glob("scene_*.wav")) if audio_dir.exists() else []
        if wavs:
            for wav in wavs:
                sid = wav.stem.replace("scene_", "")
                st.markdown(f'<div class="card-title">Scene {sid}</div>', unsafe_allow_html=True)
                st.audio(str(wav))
        else:
            st.markdown(
                '<div class="card" style="color:#555577">No audio files yet — run Phase 3 to generate TTS narration.</div>',
                unsafe_allow_html=True,
            )

    with tab_video:
        render = OUTPUT / "renders" / "final_render.mp4"
        if render.exists():
            st.video(str(render))
        else:
            st.markdown(
                '<div class="card" style="color:#555577">No render yet — run Phases 3 + 4 to assemble the final video.</div>',
                unsafe_allow_html=True,
            )

elif not st.session_state["running"]:
    st.markdown("""
    <div class="card" style="text-align:center;padding:48px 24px">
        <div style="font-size:48px;margin-bottom:16px">🎬</div>
        <div style="font-size:20px;font-weight:600;color:#c8c0b8;margin-bottom:8px">Ready to generate</div>
        <div style="color:#444466;font-size:14px">
            Configure your settings in the sidebar, then hit <strong style="color:#f5a623">Generate Video</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
