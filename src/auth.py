"""Authentication, registration and role-based account management.

Security notes
* Passwords and recovery codes are hashed with scrypt + a per-secret random salt (stdlib, no extra dependency).
* Login errors are generic, a dummy hash is checked for unknown users (timing), and 5 failures lock the account 10 min.
* There is no email service, so password recovery uses a one-time recovery code shown at registration.
* Roles: user (default) / analyst / admin. Registration can only create 'user' accounts.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time

import pandas as pd

from .db import connect

ROLES = ("user", "analyst", "admin")
MAX_FAILS, LOCK_SECONDS = 5, 600
GENERIC_LOGIN_ERROR = "Invalid username/email or password."
_N, _R, _P = 2**14, 8, 1
USERNAME_RE = re.compile(r"^[a-z0-9_]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")
COMMON_PASSWORDS = {"password", "password1", "password123", "12345678", "123456789", "qwertyui", "qwerty123", "iloveyou",
                    "admin123", "letmein1", "welcome1", "abc12345", "11111111", "1q2w3e4r", "passw0rd", "changeme1"}
_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


class AuthError(Exception):
    """A user-presentable authentication / authorisation problem."""


class ValidationError(AuthError):
    def __init__(self, errors: list[str]):
        super().__init__(" ".join(errors))
        self.errors = errors


# ----------------------------------------------------------------------------- hashing
def hash_secret(secret: str) -> str:
    salt = os.urandom(16)
    h = hashlib.scrypt(secret.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${h.hex()}"


def verify_secret(secret: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        h = hashlib.scrypt(secret.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(h.hex(), digest)
    except Exception:
        return False


_DUMMY_HASH = hash_secret("timing-equaliser")


def new_recovery_code() -> str:
    raw = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(16))
    return "-".join(raw[i:i + 4] for i in range(0, 16, 4))


def _norm_code(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", code.upper())


# ----------------------------------------------------------------------------- validation
def password_problems(password: str, username: str = "") -> list[str]:
    errs = []
    if len(password) < 8:
        errs.append("Password must be at least 8 characters.")
    if len(password) > 128:
        errs.append("Password must be at most 128 characters.")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        errs.append("Password must contain at least one letter and one number.")
    if password.lower() in COMMON_PASSWORDS:
        errs.append("That password is too common.")
    if len(username) >= 3 and username.lower() in password.lower():
        errs.append("Password must not contain your username.")
    return errs


def _public(row) -> dict:
    return {"id": row["id"], "username": row["username"], "email": row["email"], "full_name": row["full_name"],
            "role": row["role"], "must_change_password": bool(row["must_change_password"])}


# ----------------------------------------------------------------------------- registration / login
def _create_user(username, email, full_name, password, confirm, consent, role) -> tuple[dict, str]:
    u, e, n = username.strip().lower(), email.strip().lower(), " ".join(full_name.split())
    errs = []
    if not USERNAME_RE.match(u):
        errs.append("Username must be 3-30 characters: letters, numbers or underscore.")
    if not EMAIL_RE.match(e) or len(e) > 254:
        errs.append("Enter a valid email address.")
    if not 2 <= len(n) <= 80:
        errs.append("Full name must be 2-80 characters.")
    errs += password_problems(password, u)
    if password != confirm:
        errs.append("Passwords do not match.")
    if not consent:
        errs.append("Please accept the terms to register.")
    if errs:
        raise ValidationError(errs)
    code = new_recovery_code()
    try:
        with connect() as c:
            cur = c.execute(
                "INSERT INTO users(username,email,full_name,password_hash,recovery_hash,role,created_at) VALUES(?,?,?,?,?,?,?)",
                (u, e, n, hash_secret(password), hash_secret(_norm_code(code)), role, time.time()))
            uid = cur.lastrowid
    except sqlite3.IntegrityError as ex:
        raise ValidationError(["That username is already taken." if "username" in str(ex).lower()
                               else "An account with that email already exists."]) from None
    return {"id": uid, "username": u, "email": e, "full_name": n, "role": role, "must_change_password": False}, code


def register(username: str, email: str, full_name: str, password: str, confirm: str, consent: bool) -> tuple[dict, str]:
    """Create a normal 'user' account. Returns (user, one-time recovery code)."""
    return _create_user(username, email, full_name, password, confirm, consent, "user")


def authenticate(identifier: str, password: str) -> dict:
    """Return the public user dict or raise AuthError. Tracks failures and locks the account temporarily."""
    ident = identifier.strip().lower()
    error, user = None, None
    with connect() as c:
        row = c.execute("SELECT * FROM users WHERE username=? OR email=?", (ident, ident)).fetchone()
        if row is None:
            verify_secret(password, _DUMMY_HASH)
            error = GENERIC_LOGIN_ERROR
        elif row["locked_until"] > time.time():
            error = f"Too many failed attempts. Try again in {int((row['locked_until'] - time.time()) // 60) + 1} minute(s)."
        elif not verify_secret(password, row["password_hash"]):
            fails = row["failed_attempts"] + 1
            if fails >= MAX_FAILS:
                c.execute("UPDATE users SET failed_attempts=0, locked_until=? WHERE id=?", (time.time() + LOCK_SECONDS, row["id"]))
            else:
                c.execute("UPDATE users SET failed_attempts=? WHERE id=?", (fails, row["id"]))
            error = GENERIC_LOGIN_ERROR
        elif not row["is_active"]:
            error = "This account is disabled. Please contact an administrator."
        else:
            c.execute("UPDATE users SET failed_attempts=0, locked_until=0, last_login=? WHERE id=?", (time.time(), row["id"]))
            user = _public(row)
    if error:
        raise AuthError(error)
    return user


# ----------------------------------------------------------------------------- self-service account actions
def get_user(user_id: int) -> dict | None:
    with connect() as c:
        row = c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return _public(row) if row else None


def get_active_user(user_id: int) -> dict | None:
    """Fresh copy of a user if the account still exists and is active (used to re-check every session run)."""
    with connect() as c:
        row = c.execute("SELECT * FROM users WHERE id=? AND is_active=1", (user_id,)).fetchone()
    return _public(row) if row else None


def _check_password(c, user_id: int, password: str) -> bool:
    row = c.execute("SELECT password_hash FROM users WHERE id=?", (user_id,)).fetchone()
    return bool(row) and verify_secret(password, row["password_hash"])


def change_password(user_id: int, current: str, new: str, confirm: str) -> None:
    with connect() as c:
        row = c.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
        if not row or not _check_password(c, user_id, current):
            raise AuthError("Current password is incorrect.")
        errs = password_problems(new, row["username"]) + (["Passwords do not match."] if new != confirm else [])
        if current == new:
            errs.append("New password must differ from the current one.")
        if errs:
            raise ValidationError(errs)
        c.execute("UPDATE users SET password_hash=?, must_change_password=0 WHERE id=?", (hash_secret(new), user_id))


def update_account(user_id: int, full_name: str, email: str) -> dict:
    n, e = " ".join(full_name.split()), email.strip().lower()
    errs = []
    if not 2 <= len(n) <= 80:
        errs.append("Full name must be 2-80 characters.")
    if not EMAIL_RE.match(e):
        errs.append("Enter a valid email address.")
    if errs:
        raise ValidationError(errs)
    try:
        with connect() as c:
            c.execute("UPDATE users SET full_name=?, email=? WHERE id=?", (n, e, user_id))
    except sqlite3.IntegrityError:
        raise ValidationError(["An account with that email already exists."]) from None
    return get_user(user_id)


def regenerate_recovery_code(user_id: int, password: str) -> str:
    code = new_recovery_code()
    with connect() as c:
        if not _check_password(c, user_id, password):
            raise AuthError("Password is incorrect.")
        c.execute("UPDATE users SET recovery_hash=? WHERE id=?", (hash_secret(_norm_code(code)), user_id))
    return code


def reset_password_with_code(username: str, code: str, new: str, confirm: str) -> str:
    """Forgot-password flow. Returns a NEW recovery code (the old one is consumed)."""
    u = username.strip().lower()
    error, new_code = None, None
    with connect() as c:
        row = c.execute("SELECT * FROM users WHERE username=? OR email=?", (u, u)).fetchone()
        if row is None:
            verify_secret(code, _DUMMY_HASH)
            error = "Username or recovery code is incorrect."
        elif row["locked_until"] > time.time():
            error = "Too many failed attempts. Try again later."
        elif not verify_secret(_norm_code(code), row["recovery_hash"]):
            fails = row["failed_attempts"] + 1
            if fails >= MAX_FAILS:
                c.execute("UPDATE users SET failed_attempts=0, locked_until=? WHERE id=?", (time.time() + LOCK_SECONDS, row["id"]))
            else:
                c.execute("UPDATE users SET failed_attempts=? WHERE id=?", (fails, row["id"]))
            error = "Username or recovery code is incorrect."
        else:
            errs = password_problems(new, row["username"]) + (["Passwords do not match."] if new != confirm else [])
            if errs:
                error = " ".join(errs)
            else:
                new_code = new_recovery_code()
                c.execute("UPDATE users SET password_hash=?, recovery_hash=?, failed_attempts=0, locked_until=0, "
                          "must_change_password=0 WHERE id=?", (hash_secret(new), hash_secret(_norm_code(new_code)), row["id"]))
    if error:
        raise AuthError(error)
    return new_code


def delete_account(user_id: int, password: str) -> None:
    with connect() as c:
        row = c.execute("SELECT role FROM users WHERE id=?", (user_id,)).fetchone()
        if not row or not _check_password(c, user_id, password):
            raise AuthError("Password is incorrect.")
        if row["role"] == "admin" and c.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND is_active=1").fetchone()[0] <= 1:
            raise AuthError("You are the only active administrator - promote someone else before deleting this account.")
        c.execute("DELETE FROM users WHERE id=?", (user_id,))


# ----------------------------------------------------------------------------- admin actions
def _require_admin(c, actor_id: int):
    row = c.execute("SELECT role, is_active FROM users WHERE id=?", (actor_id,)).fetchone()
    if not row or row["role"] != "admin" or not row["is_active"]:
        raise AuthError("Administrator access required.")


def list_users(actor_id: int) -> pd.DataFrame:
    with connect() as c:
        _require_admin(c, actor_id)
        rows = c.execute("SELECT id, username, email, full_name, role, is_active, must_change_password, locked_until, "
                         "created_at, last_login FROM users ORDER BY id").fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if not df.empty:
        df["locked"] = df["locked_until"] > time.time()
        for col in ("created_at", "last_login"):
            df[col] = pd.to_datetime(df[col], unit="s").dt.strftime("%Y-%m-%d %H:%M")
        df = df.drop(columns="locked_until")
    return df


def _admin_target(c, actor_id: int, target_id: int, action: str):
    _require_admin(c, actor_id)
    if actor_id == target_id:
        raise AuthError(f"You cannot {action} your own account.")
    if not c.execute("SELECT id FROM users WHERE id=?", (target_id,)).fetchone():
        raise AuthError("User not found.")


def admin_set_role(actor_id: int, target_id: int, role: str) -> None:
    if role not in ROLES:
        raise AuthError("Unknown role.")
    with connect() as c:
        _admin_target(c, actor_id, target_id, "change the role of")
        c.execute("UPDATE users SET role=? WHERE id=?", (role, target_id))


def admin_set_active(actor_id: int, target_id: int, active: bool) -> None:
    with connect() as c:
        _admin_target(c, actor_id, target_id, "disable")
        c.execute("UPDATE users SET is_active=? WHERE id=?", (int(active), target_id))


def admin_unlock(actor_id: int, target_id: int) -> None:
    with connect() as c:
        _require_admin(c, actor_id)
        c.execute("UPDATE users SET failed_attempts=0, locked_until=0 WHERE id=?", (target_id,))


def admin_reset_password(actor_id: int, target_id: int) -> str:
    """Set a temporary password (shown once); the user must change it at next login."""
    temp = secrets.token_urlsafe(9) + "7"
    with connect() as c:
        _admin_target(c, actor_id, target_id, "reset the password of")
        c.execute("UPDATE users SET password_hash=?, must_change_password=1, failed_attempts=0, locked_until=0 WHERE id=?",
                  (hash_secret(temp), target_id))
    return temp


def admin_delete_user(actor_id: int, target_id: int) -> None:
    with connect() as c:
        _admin_target(c, actor_id, target_id, "delete")
        c.execute("DELETE FROM users WHERE id=?", (target_id,))


def bootstrap_admin_from_env() -> bool:
    """Create the first admin from MEDREC_ADMIN_USER / MEDREC_ADMIN_EMAIL / MEDREC_ADMIN_PASSWORD if no admin exists.

    Raises ValidationError when the configured values break the registration rules (so misconfiguration is visible)."""
    user, email, pw = (os.environ.get(k) for k in ("MEDREC_ADMIN_USER", "MEDREC_ADMIN_EMAIL", "MEDREC_ADMIN_PASSWORD"))
    if not (user and email and pw):
        return False
    with connect() as c:
        if c.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]:
            return False
    _create_user(user, email, "Administrator", pw, pw, True, "admin")   # raises ValidationError if the env values are invalid
    return True
