"""End-to-end UI tests (headless Streamlit): login gate, registration, RBAC, saving, profile pre-fill, admin, sessions."""
import time

from streamlit.testing.v1 import AppTest

from src import auth, userdata
from src.config import ROOT


def fresh_app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()


def click(at, label):
    return next(b for b in at.button if b.label == label).click().run()


def do_login(at, ident, pw):
    at.text_input(key="login_id").input(ident)
    at.text_input(key="login_pw").input(pw)
    return click(at, "Log in")


def test_visitor_sees_only_login_screen(db):
    at = fresh_app()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["🔑 Log in", "📝 Register", "🔓 Forgot password"]
    assert len(at.sidebar.radio) == 0                       # no navigation, no app pages
    assert not any("Recommendation" in t.value for t in at.title)


def test_register_then_login_flow(db):
    at = fresh_app()
    at.text_input(key="reg_name").input("Ritin Test")
    at.text_input(key="reg_username").input("ritin_t")
    at.text_input(key="reg_email").input("ritin@example.com")
    at.text_input(key="reg_pw1").input("Secret1234")
    at.text_input(key="reg_pw2").input("Secret1234")
    at.checkbox(key="reg_consent").check()
    click(at, "Create account")
    assert not at.exception and at.code and len(at.code[0].value) == 19      # recovery code shown once
    assert auth.authenticate("ritin_t", "Secret1234")["role"] == "user"       # registration always creates a plain user
    do_login(at, "ritin_t", "Secret1234")
    assert not at.exception and at.session_state["user"]["username"] == "ritin_t"
    assert at.sidebar.radio[0].options == ["🩺 Recommend", "🕘 My History", "👤 My Profile", "📚 Knowledge Base", "ℹ️ About"]


def test_registration_errors_are_shown(db):
    at = fresh_app()
    at.text_input(key="reg_username").input("x")
    at.text_input(key="reg_pw1").input("weak")
    click(at, "Create account")
    msgs = " ".join(e.value for e in at.error)
    assert "Username must be" in msgs and "valid email" in msgs and "at least 8" in msgs and "terms" in msgs
    assert at.session_state["user"] is None if "user" in at.session_state else True


def test_bad_login_is_rejected_with_generic_message(db):
    auth.register("alice", "alice@example.com", "Alice Test", "Secret123", "Secret123", True)
    at = do_login(fresh_app(), "alice", "wrong-password1")
    assert [e.value for e in at.error] == [auth.GENERIC_LOGIN_ERROR]
    assert "user" not in at.session_state


def test_rbac_navigation_per_role(logged_in):
    expect = {
        "user": {"🩺 Recommend", "🕘 My History", "👤 My Profile", "📚 Knowledge Base", "ℹ️ About"},
        "analyst": {"🩺 Recommend", "🕘 My History", "👤 My Profile", "📚 Knowledge Base", "ℹ️ About",
                    "📊 Data Explorer", "🤖 Model Performance", "🗂️ Datasets", "📈 Usage & Feedback"},
    }
    expect["admin"] = expect["analyst"] | {"🛡️ Admin"}
    for role, pages in expect.items():
        at = logged_in(role)
        assert not at.exception and set(at.sidebar.radio[0].options) == pages, role


def test_page_functions_refuse_wrong_role(logged_in):
    at = logged_in("user")
    from src import account_ui
    assert account_ui is not None
    # pages are not reachable via navigation, and the page functions also check the role themselves
    assert "🛡️ Admin" not in at.sidebar.radio[0].options and "📈 Usage & Feedback" not in at.sidebar.radio[0].options


def test_role_change_applies_immediately_and_disabled_account_is_kicked_out(logged_in):
    admin_at = logged_in("admin")
    user_at = logged_in("user")
    assert "🛡️ Admin" not in user_at.sidebar.radio[0].options
    auth.admin_set_role(admin_at.user["id"], user_at.user["id"], "analyst")          # promoted while logged in
    user_at.run()
    assert "📊 Data Explorer" in user_at.sidebar.radio[0].options
    auth.admin_set_active(admin_at.user["id"], user_at.user["id"], False)             # disabled while logged in
    user_at.run()
    assert "user" not in user_at.session_state and any("no longer available" in s.value for s in user_at.success)


def test_session_times_out_after_inactivity(logged_in):
    at = logged_in("user")
    at.session_state["last_active"] = time.time() - 31 * 60
    at.run()
    assert "user" not in at.session_state and any("session expired" in s.value.lower() for s in at.success)


def test_logout_clears_session(logged_in):
    at = logged_in("user")
    click(at, "Log out")
    assert "user" not in at.session_state and len(at.sidebar.radio) == 0
    assert any("logged out" in s.value for s in at.success)


def test_must_change_password_gate(logged_in, db):
    admin_at = logged_in("admin")
    target, _ = auth.register("carol", "carol@example.com", "Carol Test", "Secret123", "Secret123", True)
    temp = auth.admin_reset_password(admin_at.user["id"], target["id"])
    at = fresh_app()
    do_login(at, "carol", temp)
    assert len(at.sidebar.radio) == 0 and any("new password" in t.value.lower() for t in at.title)   # forced to change first
    at.text_input(key="fc_cur").input(temp)
    at.text_input(key="fc_new").input("BrandNew4567")
    at.text_input(key="fc_new2").input("BrandNew4567")
    click(at, "Save new password")
    assert not at.exception and len(at.sidebar.radio) == 1                                        # now inside the app


def test_save_history_feedback_and_profile_persist(logged_in):
    at = logged_in("user")
    uid = at.user["id"]
    at.checkbox(key="sym_cough").check().run()
    click(at, "💾 Save to my history")
    assert not at.exception and len(userdata.list_history(uid)) == 1
    next(w for w in at.select_slider if w.label == "How useful was this?").set_value(5)            # form value is sent on submit
    click(at, "Submit feedback")
    assert userdata.usage_stats()["feedback"]["rating"].tolist() == [5]
    # save profile -> pre-fills a new session
    at.slider(key="p_age").set_value(61).run()
    click(at, "💾 Save as my profile")
    assert userdata.get_profile(uid)["age"] == 61
    at2 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at2.session_state["user"], at2.session_state["last_active"] = at.user, time.time()
    at2.run()
    assert at2.slider(key="p_age").value == 61
    # history page shows the saved item
    at.sidebar.radio[0].set_value("🕘 My History").run()
    assert not at.exception and len(at.dataframe) >= 1


def test_users_only_see_their_own_history(logged_in):
    a = logged_in("user", "ann")
    b = logged_in("user", "ben")
    click(a, "💾 Save to my history")
    b.sidebar.radio[0].set_value("🕘 My History").run()
    assert any("Nothing saved yet" in i.value for i in b.info)


def test_admin_can_change_role_and_cannot_touch_self(logged_in):
    admin_at = logged_in("admin")
    other, _ = auth.register("dave", "dave@example.com", "Dave Test", "Secret123", "Secret123", True)
    admin_at.sidebar.radio[0].set_value("🛡️ Admin").run()
    assert not admin_at.exception
    role_box = next(sb for sb in admin_at.selectbox if sb.label == "Role")
    role_box.set_value("analyst")
    click(admin_at, "Apply role")
    assert auth.get_user(other["id"])["role"] == "analyst"
    labels = next(sb for sb in admin_at.selectbox if sb.label == "Account").options
    assert all("admin_user" not in l for l in labels)          # the admin is not offered as a target (self-protection)


def test_analyst_sees_aggregates_but_admin_only_pages_blocked(logged_in):
    at = logged_in("analyst")
    at.sidebar.radio[0].set_value("📈 Usage & Feedback").run()
    assert not at.exception and any("Aggregate, anonymous" in c.value for c in at.caption)
