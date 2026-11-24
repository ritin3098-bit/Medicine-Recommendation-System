import pytest

from src import auth, userdata
from src.db import connect


def make(username="alice", email="alice@example.com", pw="Secret123", role="user"):
    if role == "user":
        return auth.register(username, email, "Alice Test", pw, pw, True)
    return auth._create_user(username, email, "Admin Test", pw, pw, True, role)


def test_register_and_login(db):
    user, code = make()
    assert user["role"] == "user" and len(code) == 19
    assert auth.authenticate("ALICE", "Secret123")["id"] == user["id"]        # case-insensitive username
    assert auth.authenticate("alice@example.com", "Secret123")["id"] == user["id"]  # email login


def test_passwords_are_hashed_not_stored(db):
    make()
    with connect() as c:
        row = c.execute("SELECT password_hash, recovery_hash FROM users").fetchone()
    assert row["password_hash"].startswith("scrypt$") and "Secret123" not in row["password_hash"]
    assert auth.verify_secret("Secret123", row["password_hash"]) and not auth.verify_secret("secret123", row["password_hash"])


@pytest.mark.parametrize("kwargs,msg", [
    (dict(pw="short1"), "at least 8"),
    (dict(pw="onlyletters"), "letter and one number"),
    (dict(pw="password123"), "too common"),
    (dict(username="bob_x", pw="bob_x12345"), "username"),
    (dict(username="a!", email="a@b.co"), "Username"),
    (dict(email="not-an-email"), "valid email"),
])
def test_registration_validation(db, kwargs, msg):
    with pytest.raises(auth.ValidationError) as e:
        make(**kwargs)
    assert msg.lower() in " ".join(e.value.errors).lower()


def test_consent_and_confirm_required_and_duplicates_rejected(db):
    with pytest.raises(auth.ValidationError):
        auth.register("alice", "a@x.io", "Alice Test", "Secret123", "Secret123", False)
    with pytest.raises(auth.ValidationError):
        auth.register("alice", "a@x.io", "Alice Test", "Secret123", "Different1", True)
    make()
    with pytest.raises(auth.ValidationError, match="username is already taken"):
        make(email="other@example.com")
    with pytest.raises(auth.ValidationError, match="email already exists"):
        make(username="alice2")


def test_wrong_password_is_generic_and_unknown_user_same_message(db):
    make()
    for ident in ("alice", "nobody"):
        with pytest.raises(auth.AuthError) as e:
            auth.authenticate(ident, "WrongPass1")
        assert str(e.value) == auth.GENERIC_LOGIN_ERROR


def test_lockout_after_repeated_failures_even_with_correct_password(db):
    make()
    for _ in range(auth.MAX_FAILS):
        with pytest.raises(auth.AuthError):
            auth.authenticate("alice", "WrongPass1")
    with pytest.raises(auth.AuthError, match="Too many failed attempts"):
        auth.authenticate("alice", "Secret123")


def test_successful_login_resets_failed_counter(db):
    make()
    for _ in range(auth.MAX_FAILS - 1):
        with pytest.raises(auth.AuthError):
            auth.authenticate("alice", "WrongPass1")
    auth.authenticate("alice", "Secret123")
    for _ in range(auth.MAX_FAILS - 1):   # would lock if the counter had not been reset
        with pytest.raises(auth.AuthError, match="Invalid"):
            auth.authenticate("alice", "WrongPass1")


def test_change_password(db):
    user, _ = make()
    with pytest.raises(auth.AuthError, match="Current password"):
        auth.change_password(user["id"], "nope", "NewSecret456", "NewSecret456")
    with pytest.raises(auth.ValidationError):
        auth.change_password(user["id"], "Secret123", "weak", "weak")
    auth.change_password(user["id"], "Secret123", "NewSecret456", "NewSecret456")
    with pytest.raises(auth.AuthError):
        auth.authenticate("alice", "Secret123")
    assert auth.authenticate("alice", "NewSecret456")


def test_recovery_code_resets_password_and_is_single_use(db):
    _, code = make()
    with pytest.raises(auth.AuthError):
        auth.reset_password_with_code("alice", "AAAA-BBBB-CCCC-DDDD", "NewSecret456", "NewSecret456")
    new_code = auth.reset_password_with_code("alice", code.lower().replace("-", " "), "NewSecret456", "NewSecret456")
    assert new_code != code and auth.authenticate("alice", "NewSecret456")
    with pytest.raises(auth.AuthError):   # old code is consumed
        auth.reset_password_with_code("alice", code, "Another789", "Another789")


def test_regenerate_recovery_code_requires_password(db):
    user, code = make()
    with pytest.raises(auth.AuthError):
        auth.regenerate_recovery_code(user["id"], "wrong")
    new = auth.regenerate_recovery_code(user["id"], "Secret123")
    with pytest.raises(auth.AuthError):
        auth.reset_password_with_code("alice", code, "NewSecret456", "NewSecret456")
    assert auth.reset_password_with_code("alice", new, "NewSecret456", "NewSecret456")


def test_admin_permissions_and_safeguards(db):
    admin, _ = make("root", "root@example.com", role="admin")
    user, _ = make()
    # non-admins cannot use admin actions
    with pytest.raises(auth.AuthError, match="Administrator"):
        auth.list_users(user["id"])
    with pytest.raises(auth.AuthError, match="Administrator"):
        auth.admin_set_role(user["id"], admin["id"], "user")
    # admin can manage others but not themselves
    auth.admin_set_role(admin["id"], user["id"], "analyst")
    assert auth.get_user(user["id"])["role"] == "analyst"
    with pytest.raises(auth.AuthError, match="your own"):
        auth.admin_set_role(admin["id"], admin["id"], "user")
    with pytest.raises(auth.AuthError, match="your own"):
        auth.admin_delete_user(admin["id"], admin["id"])
    with pytest.raises(auth.AuthError, match="Unknown role"):
        auth.admin_set_role(admin["id"], user["id"], "superuser")
    df = auth.list_users(admin["id"])
    assert "password_hash" not in df.columns and len(df) == 2


def test_disabled_account_cannot_login_and_admin_reset_forces_change(db):
    admin, _ = make("root", "root@example.com", role="admin")
    user, _ = make()
    auth.admin_set_active(admin["id"], user["id"], False)
    with pytest.raises(auth.AuthError, match="disabled"):
        auth.authenticate("alice", "Secret123")
    auth.admin_set_active(admin["id"], user["id"], True)
    temp = auth.admin_reset_password(admin["id"], user["id"])
    logged = auth.authenticate("alice", temp)
    assert logged["must_change_password"] is True
    auth.change_password(user["id"], temp, "BrandNew123", "BrandNew123")
    assert auth.authenticate("alice", "BrandNew123")["must_change_password"] is False


def test_admin_unlock(db):
    admin, _ = make("root", "root@example.com", role="admin")
    user, _ = make()
    for _ in range(auth.MAX_FAILS):
        with pytest.raises(auth.AuthError):
            auth.authenticate("alice", "WrongPass1")
    auth.admin_unlock(admin["id"], user["id"])
    assert auth.authenticate("alice", "Secret123")


def test_delete_account_cascades_all_user_data(db):
    user, _ = make()
    userdata.save_profile(user["id"], {"age": 40})
    userdata.save_history(user["id"], {"disease": "Asthma", "risk_level": "Low", "intensity": 2})
    userdata.add_feedback(user["id"], "Asthma", 5, "good")
    with pytest.raises(auth.AuthError):
        auth.delete_account(user["id"], "wrong")
    auth.delete_account(user["id"], "Secret123")
    with connect() as c:
        counts = [c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("users", "profiles", "history", "feedback")]
    assert counts == [0, 0, 0, 0]


def test_last_admin_cannot_delete_own_account(db):
    admin, _ = make("root", "root@example.com", role="admin")
    with pytest.raises(auth.AuthError, match="only active administrator"):
        auth.delete_account(admin["id"], "Secret123")


def test_env_bootstrap_creates_single_admin(db, monkeypatch):
    monkeypatch.setenv("MEDREC_ADMIN_USER", "boss")
    monkeypatch.setenv("MEDREC_ADMIN_EMAIL", "boss@example.com")
    monkeypatch.setenv("MEDREC_ADMIN_PASSWORD", "weak")
    with pytest.raises(auth.ValidationError):              # misconfiguration is reported, not swallowed
        auth.bootstrap_admin_from_env()
    monkeypatch.setenv("MEDREC_ADMIN_PASSWORD", "Str0ngAdminPass")
    assert auth.bootstrap_admin_from_env() is True
    assert auth.bootstrap_admin_from_env() is False       # idempotent
    assert auth.authenticate("boss", "Str0ngAdminPass")["role"] == "admin"


def test_profile_history_and_isolation_between_users(db):
    a, _ = make()
    b, _ = make("bob", "bob@example.com")
    assert userdata.get_profile(a["id"])["age"] == 35                    # defaults
    userdata.save_profile(a["id"], {"age": 52, "allergies": ["Sulfa drugs"], "city": "Delhi", "evil": "x"})
    p = userdata.get_profile(a["id"])
    assert p["age"] == 52 and p["allergies"] == ["Sulfa drugs"] and "evil" not in p
    hid = userdata.save_history(a["id"], {"disease": "Asthma", "mode": "m", "risk_level": "Low", "intensity": 2, "x": 1})
    assert userdata.get_history_item(a["id"], hid)["x"] == 1
    assert userdata.get_history_item(b["id"], hid) is None              # other users cannot read it
    assert userdata.delete_history(b["id"], hid) is False               # ... or delete it
    assert len(userdata.list_history(a["id"])) == 1
    assert userdata.delete_history(a["id"], hid) is True


def test_feedback_validation_and_usage_stats(db):
    a, _ = make()
    userdata.save_history(a["id"], {"disease": "Asthma", "risk_level": "Low", "intensity": 2})
    userdata.add_feedback(a["id"], "Asthma", 4, "ok")
    with pytest.raises(ValueError):
        userdata.add_feedback(a["id"], "Asthma", 9)
    s = userdata.usage_stats()
    assert len(s["history"]) == 1 and s["feedback"]["rating"].tolist() == [4]
    assert "comment" not in s["feedback"].columns and "user_id" not in s["history"].columns   # no identities
    assert "comment" in userdata.usage_stats(include_comments=True)["feedback"].columns
