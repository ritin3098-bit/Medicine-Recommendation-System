import pytest


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Fresh, isolated SQLite database per test."""
    monkeypatch.setenv("MEDREC_DB", str(tmp_path / "test.db"))
    return tmp_path / "test.db"


@pytest.fixture()
def logged_in(db):
    """Factory: create an account with the given role, inject it into the session and run the app."""
    import time

    from streamlit.testing.v1 import AppTest

    from src import auth
    from src.config import ROOT

    def make(role="admin", username=None, run=True):
        username = username or f"{role}_user"
        user, _ = auth._create_user(username, f"{username}@example.com", "Test Person", "Str0ngPass99", "Str0ngPass99", True, role)
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        at.session_state["user"] = user
        at.session_state["last_active"] = time.time()
        at.user = user
        return at.run() if run else at

    return make
