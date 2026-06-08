"""
Auth for the studio — email+password accounts, bcrypt hashing, JWT sessions,
SQLite store. Sized for a small trusted team (tens of users); zero extra infra
(one SQLite file under output/). Sign-up requires a shared invite code.

Env:
  JWT_SECRET           signing key (auto-generated + persisted if unset)
  SIGNUP_INVITE_CODE   shared code required to register (default "dextora-invite")
  ADMIN_EMAILS         comma-separated emails granted the admin role on signup
"""
from __future__ import annotations

import os
import re
import secrets
import sqlite3
import time
from pathlib import Path

import bcrypt
import jwt

from config import settings

_DB = settings.OUTPUT_DIR / "studio.db"
_TOKEN_TTL = 60 * 60 * 24 * 7  # 7 days
INVITE_CODE = os.getenv("SIGNUP_INVITE_CODE", "dextora-invite")
_ADMIN_EMAILS = {e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()}
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _secret() -> str:
    """Stable JWT secret: env if set, else generate once and persist beside the DB."""
    if os.getenv("JWT_SECRET"):
        return os.environ["JWT_SECRET"]
    f = settings.OUTPUT_DIR / ".jwt_secret"
    if f.exists():
        return f.read_text().strip()
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    s = secrets.token_urlsafe(48)
    f.write_text(s)
    return s


def _conn() -> sqlite3.Connection:
    _DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB))
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS users (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   email TEXT UNIQUE NOT NULL,
                   password_hash TEXT NOT NULL,
                   role TEXT NOT NULL DEFAULT 'user',
                   created_at REAL NOT NULL
               )"""
        )


# ── password + token ─────────────────────────────────────────────────────────

def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def verify_password(pw: str, pw_hash: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), pw_hash.encode())
    except Exception:
        return False


def make_token(user: dict) -> str:
    now = int(time.time())
    payload = {"sub": str(user["id"]), "email": user["email"], "role": user["role"],
               "iat": now, "exp": now + _TOKEN_TTL}
    return jwt.encode(payload, _secret(), algorithm="HS256")


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, _secret(), algorithms=["HS256"])
    except Exception:
        return None


# ── user store ────────────────────────────────────────────────────────────────

def _row_to_user(r: sqlite3.Row) -> dict:
    return {"id": r["id"], "email": r["email"], "role": r["role"], "created_at": r["created_at"]}


def get_user_by_email(email: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM users WHERE email=?", (email.lower(),)).fetchone()
    return r  # raw row (has password_hash) — caller decides


def get_user_by_id(uid: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return _row_to_user(r) if r else None


def create_user(email: str, password: str) -> dict:
    """Create a user. Raises ValueError on bad input / duplicate."""
    email = (email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        raise ValueError("invalid email")
    if len(password or "") < 6:
        raise ValueError("password must be at least 6 characters")
    if get_user_by_email(email):
        raise ValueError("an account with this email already exists")
    role = "admin" if email in _ADMIN_EMAILS else "user"
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO users (email, password_hash, role, created_at) VALUES (?,?,?,?)",
            (email, hash_password(password), role, time.time()),
        )
        uid = cur.lastrowid
    return {"id": uid, "email": email, "role": role, "created_at": time.time()}


def authenticate(email: str, password: str) -> dict | None:
    r = get_user_by_email(email)
    if r and verify_password(password, r["password_hash"]):
        return _row_to_user(r)
    return None
