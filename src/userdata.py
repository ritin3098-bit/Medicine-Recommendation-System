"""Per-user saved profile, recommendation history and feedback + aggregate analytics."""
from __future__ import annotations

import json
import time

import pandas as pd

from .db import connect

PROFILE_DEFAULTS = {"age": 35, "gender": "female", "pregnant": False, "bp": 1, "chol": 1,
                    "allergies": [], "conditions": [], "city": "Ludhiana"}


# ----------------------------------------------------------------------------- profile
def get_profile(user_id: int) -> dict:
    with connect() as c:
        row = c.execute("SELECT data FROM profiles WHERE user_id=?", (user_id,)).fetchone()
    return {**PROFILE_DEFAULTS, **(json.loads(row["data"]) if row else {})}


def save_profile(user_id: int, data: dict) -> None:
    clean = {k: data[k] for k in PROFILE_DEFAULTS if k in data}
    with connect() as c:
        c.execute("INSERT INTO profiles(user_id,data,updated_at) VALUES(?,?,?) "
                  "ON CONFLICT(user_id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at",
                  (user_id, json.dumps(clean), time.time()))


# ----------------------------------------------------------------------------- history
def save_history(user_id: int, snapshot: dict) -> int:
    with connect() as c:
        cur = c.execute("INSERT INTO history(user_id,created_at,disease,mode,risk_level,intensity,payload) VALUES(?,?,?,?,?,?,?)",
                        (user_id, time.time(), snapshot["disease"], snapshot.get("mode"), snapshot.get("risk_level"),
                         snapshot.get("intensity"), json.dumps(snapshot)))
        return cur.lastrowid


def list_history(user_id: int) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT id, created_at, disease, mode, risk_level, intensity FROM history "
                         "WHERE user_id=? ORDER BY created_at DESC, id DESC", (user_id,)).fetchall()
    return [dict(r) for r in rows]


def get_history_item(user_id: int, history_id: int) -> dict | None:
    """Only returns the item if it belongs to user_id (prevents reading other users' records)."""
    with connect() as c:
        row = c.execute("SELECT payload FROM history WHERE id=? AND user_id=?", (history_id, user_id)).fetchone()
    return json.loads(row["payload"]) if row else None


def delete_history(user_id: int, history_id: int) -> bool:
    with connect() as c:
        return c.execute("DELETE FROM history WHERE id=? AND user_id=?", (history_id, user_id)).rowcount > 0


# ----------------------------------------------------------------------------- feedback
def add_feedback(user_id: int, disease: str, rating: int, comment: str = "") -> None:
    if not 1 <= int(rating) <= 5:
        raise ValueError("rating must be 1-5")
    with connect() as c:
        c.execute("INSERT INTO feedback(user_id,created_at,disease,rating,comment) VALUES(?,?,?,?,?)",
                  (user_id, time.time(), disease, int(rating), comment.strip()[:500]))


# ----------------------------------------------------------------------------- aggregate analytics (no user identities)
def usage_stats(include_comments: bool = False) -> dict[str, pd.DataFrame]:
    with connect() as c:
        hist = pd.read_sql_query("SELECT created_at, disease, mode, risk_level, intensity FROM history", c)
        fb = pd.read_sql_query("SELECT created_at, disease, rating" + (", comment" if include_comments else "") + " FROM feedback", c)
        users = pd.read_sql_query("SELECT role, created_at FROM users", c)
    for df in (hist, fb, users):
        if not df.empty:
            df["date"] = pd.to_datetime(df["created_at"], unit="s").dt.date
    return {"history": hist, "feedback": fb, "users": users}
