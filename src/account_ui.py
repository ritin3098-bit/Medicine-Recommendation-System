"""Streamlit pages for accounts: login / register / recovery, history, profile, admin and usage analytics."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from . import auth, userdata
from .config import LEVELS
from .doctors import CITY_COORDS
from .knowledge_base import specialties_for
from .recommender import ALLERGY_OPTIONS, CONDITION_OPTIONS, INTENSITY_NAME

TERMS = ("I understand this is an educational tool, not medical advice, and I agree to my account and saved "
         "recommendations being stored in this app's database.")
STATUS_ICON = {"ok": "✅", "caution": "⚠️", "avoid": "⛔"}


# ----------------------------------------------------------------------------- session helpers
def logout(message: str | None = None) -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    if message:
        st.session_state["flash"] = message


def set_user(user: dict) -> None:
    st.session_state["user"] = user
    st.session_state["last_active"] = time.time()


def seed_profile_state(user: dict) -> None:
    """Load the saved health profile into the Recommend widgets once per login (or after the profile is saved)."""
    if st.session_state.get("_seeded_for") == user["id"]:
        return
    p = userdata.get_profile(user["id"])
    st.session_state.update(
        p_age=int(p["age"]), p_gender=p["gender"], p_pregnant=bool(p["pregnant"]), p_bp=int(p["bp"]), p_chol=int(p["chol"]),
        p_allergies=[a for a in p["allergies"] if a in ALLERGY_OPTIONS],
        p_conditions=[c for c in p["conditions"] if c in CONDITION_OPTIONS],
        doc_city=p["city"] if p["city"] in CITY_COORDS else "Ludhiana")
    st.session_state["_seeded_for"] = user["id"]


def _show_recovery_code(code: str, key: str) -> None:
    st.warning("**Save your recovery code now.** It is the only way to reset a forgotten password and it will not be shown again.")
    st.code(code, language=None)
    st.download_button("⬇️ Download recovery code", f"Medicine Recommender recovery code\n{code}\n", "recovery_code.txt", key=f"dl_{key}")


# ----------------------------------------------------------------------------- login / register / forgot password
def page_auth() -> None:
    _, mid, _ = st.columns([1, 1.7, 1])
    with mid:
        st.title("💊 Medicine & Exercise Recommender")
        st.caption("Log in to get safety-checked medicine options, an exercise plan scaled to your risk, and doctor suggestions - "
                   "and to keep your own history.")
        if st.session_state.get("_bootstrap_error"):
            st.error("Admin bootstrap failed: " + st.session_state["_bootstrap_error"])
        if msg := st.session_state.pop("flash", None):
            st.success(msg)
        if code := st.session_state.get("new_recovery"):
            _show_recovery_code(code, "new")
            if st.button("I have saved my recovery code", key="ack_recovery"):
                st.session_state.pop("new_recovery")
                st.rerun()

        tab_login, tab_register, tab_forgot = st.tabs(["🔑 Log in", "📝 Register", "🔓 Forgot password"])

        with tab_login:
            with st.form("login_form"):
                ident = st.text_input("Username or email", key="login_id")
                pw = st.text_input("Password", type="password", key="login_pw")
                submitted = st.form_submit_button("Log in", type="primary", width="stretch")
            if submitted:
                try:
                    user = auth.authenticate(ident, pw)
                except auth.AuthError as e:
                    st.error(str(e))
                else:
                    set_user(user)
                    st.rerun()

        with tab_register:
            with st.form("register_form", clear_on_submit=False):
                name = st.text_input("Full name", key="reg_name")
                c1, c2 = st.columns(2)
                username = c1.text_input("Username", key="reg_username", help="3-30 characters: letters, numbers, underscore")
                email = c2.text_input("Email", key="reg_email")
                pw1 = c1.text_input("Password", type="password", key="reg_pw1")
                pw2 = c2.text_input("Confirm password", type="password", key="reg_pw2")
                st.caption("At least 8 characters with a letter and a number; not a common password or your username.")
                consent = st.checkbox(TERMS, key="reg_consent")
                submitted = st.form_submit_button("Create account", type="primary", width="stretch")
            if submitted:
                try:
                    _, recovery = auth.register(username, email, name, pw1, pw2, consent)
                except auth.ValidationError as e:
                    for err in e.errors:
                        st.error(err)
                else:
                    st.session_state["new_recovery"] = recovery
                    st.session_state["flash"] = "Account created - you can log in now."
                    st.rerun()

        with tab_forgot:
            st.caption("Enter the recovery code you saved at registration. (This app has no email service.) "
                       "If you lost it, ask an administrator for a temporary password.")
            with st.form("forgot_form", clear_on_submit=True):
                ident = st.text_input("Username or email", key="fg_user")
                rc = st.text_input("Recovery code", key="fg_code", placeholder="XXXX-XXXX-XXXX-XXXX")
                n1 = st.text_input("New password", type="password", key="fg_pw1")
                n2 = st.text_input("Confirm new password", type="password", key="fg_pw2")
                submitted = st.form_submit_button("Reset password", type="primary", width="stretch")
            if submitted:
                try:
                    new_code = auth.reset_password_with_code(ident, rc, n1, n2)
                except auth.AuthError as e:
                    st.error(str(e))
                else:
                    st.session_state["new_recovery"] = new_code
                    st.session_state["flash"] = "Password reset - log in with your new password. Your recovery code has been replaced."
                    st.rerun()


def page_force_change(user: dict) -> None:
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        st.title("🔐 Choose a new password")
        st.info("An administrator issued you a temporary password. Please set your own before continuing.")
        with st.form("force_change"):
            cur = st.text_input("Temporary password", type="password", key="fc_cur")
            n1 = st.text_input("New password", type="password", key="fc_new")
            n2 = st.text_input("Confirm new password", type="password", key="fc_new2")
            ok = st.form_submit_button("Save new password", type="primary", width="stretch")
        if ok:
            try:
                auth.change_password(user["id"], cur, n1, n2)
            except auth.AuthError as e:
                st.error(str(e))
            else:
                set_user(auth.get_active_user(user["id"]))
                st.rerun()
        if st.button("Log out", key="fc_logout"):
            logout()
            st.rerun()


# ----------------------------------------------------------------------------- save / feedback widget (Recommend page)
def make_snapshot(mode: str, disease: str, profile, rep: dict) -> dict:
    return {
        "disease": disease, "mode": mode, "category": rep["info"]["category"], "severity": rep["info"]["severity"],
        "profile": {"age": profile.age, "gender": profile.gender, "blood_pressure": LEVELS[profile.blood_pressure],
                    "cholesterol": LEVELS[profile.cholesterol_level], "pregnant": profile.pregnant,
                    "symptoms": [s for s in ("fever", "cough", "fatigue", "difficulty_breathing") if getattr(profile, s)],
                    "allergies": list(profile.allergies), "conditions": list(profile.conditions)},
        "risk_level": rep["risk_level"], "risk": rep["risk"], "intensity": rep["cap"], "cap_reasons": rep["cap_reasons"],
        "medicines": [{k: m[k] for k in ("name", "status", "rx", "use", "reasons")} for m in rep["medicines"]],
        "exercises": [{k: e[k] for k in ("name", "minutes", "intensity", "benefit")} for e in rep["exercises"]],
        "plan": json.loads(rep["plan"].to_json(orient="records")),
        "specialties": specialties_for(disease), "note": rep["info"]["note"],
    }


def snapshot_markdown(snap: dict, when: float | None = None) -> str:
    d = datetime.fromtimestamp(when or time.time()).strftime("%Y-%m-%d %H:%M")
    L = [f"# Recommendation summary - {snap['disease']}", f"*Generated {d}*", "",
         "> Educational tool - not medical advice. Consult a doctor or pharmacist before taking any medicine.", "",
         f"- **Category:** {snap['category']}  |  **ML risk level:** {snap['risk_level']}  |  **Exercise intensity allowed:** {INTENSITY_NAME[snap['intensity']]}"]
    if snap.get("note"):
        L += ["", f"**Note:** {snap['note']}"]
    L += ["", "## Medicines"]
    L += [f"- {STATUS_ICON[m['status']]} **{m['name']}** ({'prescription' if m['rx'] else 'OTC'}) - {m['use']}"
          + (f"  \n  _{'; '.join(m['reasons'])}_" if m["reasons"] else "") for m in snap["medicines"]] or ["- none suggested"]
    L += ["", "## Exercise", *[f"- **{e['name']}** - {e['minutes']} min ({INTENSITY_NAME[e['intensity']]})" for e in snap["exercises"]]]
    L += ["", "### Weekly plan", *[f"- {p['Day']}: {p['Activity']} ({p['Minutes']} min)" for p in snap["plan"]]]
    L += ["", "## Who to see", " -> ".join(snap["specialties"])]
    return "\n".join(L)


def save_and_feedback(user: dict, disease: str, mode: str, profile, rep: dict) -> None:
    snap = make_snapshot(mode, disease, profile, rep)
    slug = re.sub(r"[^a-z0-9]+", "_", disease.lower()).strip("_")
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("💾 Save to my history", key="save_history_btn"):
        userdata.save_history(user["id"], snap)
        c3.success("Saved - find it under 🕘 My History.")
    c2.download_button("⬇️ Download summary", snapshot_markdown(snap), f"recommendation_{slug}.md", "text/markdown", key="dl_summary")
    with st.expander("⭐ Rate these recommendations"):
        with st.form("feedback_form", clear_on_submit=True):
            rating = st.select_slider("How useful was this?", options=[1, 2, 3, 4, 5], value=4)
            comment = st.text_input("Comment (optional - please don't include personal details)", max_chars=300)
            if st.form_submit_button("Submit feedback"):
                userdata.add_feedback(user["id"], disease, rating, comment)
                st.success("Thanks - your feedback helps improve the recommendations.")


# ----------------------------------------------------------------------------- history
def page_history(user: dict) -> None:
    st.title("🕘 My History")
    items = userdata.list_history(user["id"])
    if not items:
        st.info("Nothing saved yet. Open **Recommend** and press *Save to my history*.")
        return
    df = pd.DataFrame(items)
    df["When"] = pd.to_datetime(df["created_at"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
    df["Intensity"] = df["intensity"].map(lambda i: INTENSITY_NAME.get(i, "-"))
    table = df.rename(columns={"disease": "Condition", "risk_level": "ML risk"})[["When", "Condition", "ML risk", "Intensity"]]
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("⬇️ Export list (CSV)", table.to_csv(index=False), "my_history.csv", "text/csv")

    label = {r["id"]: f"{w} - {r['disease']}" for r, w in zip(items, df["When"])}
    sel = st.selectbox("Open a saved recommendation", list(label), format_func=label.get)
    snap = userdata.get_history_item(user["id"], sel)
    if snap is None:
        st.error("Record not found.")
        return
    st.subheader(snap["disease"])
    c1, c2, c3 = st.columns(3)
    c1.metric("ML risk level", snap["risk_level"])
    c2.metric("Exercise intensity", INTENSITY_NAME[snap["intensity"]])
    c3.metric("Medicines listed", len(snap["medicines"]))
    if snap.get("note"):
        st.info(snap["note"])
    t1, t2, t3 = st.tabs(["💊 Medicines", "🏃 Exercise", "👩‍⚕️ Who to see"])
    with t1:
        if not snap["medicines"]:
            st.info("No self-directed medicines were suggested.")
        for m in snap["medicines"]:
            st.markdown(f"{STATUS_ICON[m['status']]} **{m['name']}** - {m['use']} (`{'Rx' if m['rx'] else 'OTC'}`)"
                        + (f"  \n:small[{'; '.join(m['reasons'])}]" if m["reasons"] else ""))
    with t2:
        plan = pd.DataFrame(snap["plan"])
        st.plotly_chart(px.bar(plan, x="Day", y="Minutes", color="Intensity", hover_data=["Activity"],
                               category_orders={"Day": list(plan["Day"])}).update_layout(height=300), width="stretch")
        st.dataframe(plan, hide_index=True, width="stretch")
    with t3:
        st.markdown(" → ".join(snap["specialties"]))
    st.download_button("⬇️ Download summary", snapshot_markdown(snap, next(r["created_at"] for r in items if r["id"] == sel)),
                       f"recommendation_{sel}.md", "text/markdown", key="hist_dl")
    if st.button("🗑️ Delete this record", key="hist_del"):
        userdata.delete_history(user["id"], sel)
        st.rerun()


# ----------------------------------------------------------------------------- profile / security
def page_profile(user: dict) -> None:
    st.title("👤 My Profile")
    t_acc, t_health, t_sec, t_danger = st.tabs(["Account", "Health profile", "Security", "Danger zone"])

    with t_acc:
        st.caption(f"Username **@{user['username']}** · role **{user['role']}**")
        with st.form("acct_form"):
            n = st.text_input("Full name", user["full_name"])
            e = st.text_input("Email", user["email"])
            if st.form_submit_button("Save account details", type="primary"):
                try:
                    set_user(auth.update_account(user["id"], n, e))
                    st.success("Account updated.")
                except auth.ValidationError as ex:
                    for err in ex.errors:
                        st.error(err)

    with t_health:
        st.caption("Saved defaults that pre-fill the Recommend page. Stored in this app's database; you can delete them at any time.")
        p = userdata.get_profile(user["id"])
        with st.form("health_form"):
            c1, c2 = st.columns(2)
            age = c1.number_input("Age", 1, 100, int(p["age"]))
            gender = c2.radio("Sex", ["female", "male"], index=["female", "male"].index(p["gender"]), horizontal=True)
            pregnant = st.checkbox("Pregnant / planning pregnancy", bool(p["pregnant"]))
            c3, c4 = st.columns(2)
            bp = c3.select_slider("Blood pressure", [0, 1, 2], int(p["bp"]), format_func=LEVELS.get)
            chol = c4.select_slider("Cholesterol", [0, 1, 2], int(p["chol"]), format_func=LEVELS.get)
            allergies = st.multiselect("Known drug allergies", list(ALLERGY_OPTIONS), [a for a in p["allergies"] if a in ALLERGY_OPTIONS])
            conditions = st.multiselect("Other health conditions", list(CONDITION_OPTIONS), [c for c in p["conditions"] if c in CONDITION_OPTIONS])
            cities = list(CITY_COORDS)
            city = st.selectbox("Default city (doctor search)", cities, cities.index(p["city"]) if p["city"] in cities else 0)
            if st.form_submit_button("Save health profile", type="primary"):
                userdata.save_profile(user["id"], dict(age=int(age), gender=gender, pregnant=bool(pregnant and gender == "female"),
                                                       bp=bp, chol=chol, allergies=allergies, conditions=conditions, city=city))
                st.session_state.pop("_seeded_for", None)
                st.success("Health profile saved - it will pre-fill the Recommend page.")

    with t_sec:
        st.subheader("Change password")
        with st.form("pw_form", clear_on_submit=True):
            cur = st.text_input("Current password", type="password")
            n1 = st.text_input("New password", type="password")
            n2 = st.text_input("Confirm new password", type="password")
            if st.form_submit_button("Change password", type="primary"):
                try:
                    auth.change_password(user["id"], cur, n1, n2)
                    st.success("Password changed.")
                except auth.AuthError as ex:
                    st.error(str(ex))
        st.subheader("Recovery code")
        st.caption("Generating a new code invalidates the old one.")
        with st.form("rc_form", clear_on_submit=True):
            pw = st.text_input("Confirm your password", type="password")
            if st.form_submit_button("Generate new recovery code"):
                try:
                    st.session_state["profile_recovery"] = auth.regenerate_recovery_code(user["id"], pw)
                except auth.AuthError as ex:
                    st.error(str(ex))
        if code := st.session_state.get("profile_recovery"):
            _show_recovery_code(code, "profile")

    with t_danger:
        st.error("Deleting your account permanently removes your profile, saved history and feedback.")
        with st.form("del_form"):
            pw = st.text_input("Password", type="password", key="del_pw")
            sure = st.checkbox("I understand this cannot be undone", key="del_sure")
            if st.form_submit_button("Delete my account"):
                if not sure:
                    st.error("Please tick the confirmation box.")
                else:
                    try:
                        auth.delete_account(user["id"], pw)
                    except auth.AuthError as ex:
                        st.error(str(ex))
                    else:
                        logout("Your account and data have been deleted.")
                        st.rerun()


# ----------------------------------------------------------------------------- admin
def page_admin(user: dict) -> None:
    st.title("🛡️ Administration")
    if user["role"] != "admin":
        st.error("Administrator access required.")
        return
    try:
        users = auth.list_users(user["id"])
    except auth.AuthError as e:
        st.error(str(e))
        return
    if msg := st.session_state.pop("flash_admin", None):
        st.success(msg)
    stats = userdata.usage_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accounts", len(users))
    c2.metric("Admins / analysts", f"{(users.role == 'admin').sum()} / {(users.role == 'analyst').sum()}")
    c3.metric("Saved recommendations", len(stats["history"]))
    c4.metric("Feedback entries", len(stats["feedback"]))
    st.caption("Admins manage accounts only - saved health histories are private to each user.")
    st.dataframe(users.drop(columns=["must_change_password"]), hide_index=True, width="stretch")

    others = users[users["id"] != user["id"]]
    if others.empty:
        st.info("No other accounts yet.")
        return
    st.subheader("Manage an account")
    labels = {int(r.id): f"@{r.username} ({r.role}{'' if r.is_active else ', disabled'})" for r in others.itertuples()}
    tid = st.selectbox("Account", list(labels), format_func=labels.get)
    row = others[others["id"] == tid].iloc[0]
    a, b = st.columns(2)
    with a:
        role = st.selectbox("Role", list(auth.ROLES), list(auth.ROLES).index(row["role"]), key="adm_role")
        if st.button("Apply role", key="adm_apply_role"):
            _admin_do(lambda: auth.admin_set_role(user["id"], tid, role), f"Role set to {role}.")
        if st.button("Enable account" if not row["is_active"] else "Disable account", key="adm_toggle"):
            _admin_do(lambda: auth.admin_set_active(user["id"], tid, not bool(row["is_active"])), "Account status updated.")
    with b:
        if st.button("Unlock account", key="adm_unlock"):
            _admin_do(lambda: auth.admin_unlock(user["id"], tid), "Account unlocked.")
        if st.button("Issue temporary password", key="adm_reset"):
            try:
                st.session_state["temp_pw"] = (tid, auth.admin_reset_password(user["id"], tid))
            except auth.AuthError as e:
                st.error(str(e))
        if (tp := st.session_state.get("temp_pw")) and tp[0] == tid:
            st.warning("Share this temporary password securely - it is shown only now. The user must change it at next login.")
            st.code(tp[1], language=None)
        sure = st.checkbox("Confirm permanent deletion of this account and its data", key="adm_del_ok")
        if st.button("Delete account", key="adm_delete", disabled=not sure):
            _admin_do(lambda: auth.admin_delete_user(user["id"], tid), "Account deleted.")


def _admin_do(action, ok_msg: str) -> None:
    try:
        action()
    except auth.AuthError as e:
        st.error(str(e))
    else:
        st.session_state["flash_admin"] = ok_msg
        st.rerun()


# ----------------------------------------------------------------------------- analytics (analyst / admin)
def page_analytics(user: dict) -> None:
    st.title("📈 Usage & Feedback")
    if user["role"] not in ("analyst", "admin"):
        st.error("Analyst or administrator access required.")
        return
    st.caption("Aggregate, anonymous statistics - no usernames or personal profiles are shown here.")
    s = userdata.usage_stats(include_comments=user["role"] == "admin")
    hist, fb, users = s["history"], s["feedback"], s["users"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accounts", len(users))
    c2.metric("Saved recommendations", len(hist))
    c3.metric("Feedback entries", len(fb))
    c4.metric("Average rating", f"{fb['rating'].mean():.2f} / 5" if len(fb) else "-")
    if hist.empty and fb.empty:
        st.info("No usage data yet - it appears here once users save recommendations or leave feedback.")
        return
    a, b = st.columns(2)
    if not hist.empty:
        a.plotly_chart(px.bar(hist["disease"].value_counts().head(10).iloc[::-1], orientation="h", title="Most-requested conditions",
                              labels={"value": "recommendations", "index": ""}).update_layout(showlegend=False, height=340), width="stretch")
        b.plotly_chart(px.bar(hist["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).dropna(), title="ML risk levels served",
                              labels={"value": "recommendations", "index": ""}).update_layout(showlegend=False, height=340), width="stretch")
        per_day = hist.groupby("date").size().rename("recommendations").reset_index()
        st.plotly_chart(px.bar(per_day, x="date", y="recommendations", title="Recommendations per day").update_layout(height=300), width="stretch")
    if not fb.empty:
        a, b = st.columns(2)
        a.plotly_chart(px.bar(fb["rating"].value_counts().sort_index(), title="Rating distribution",
                              labels={"value": "responses", "index": "rating"}).update_layout(showlegend=False, height=300), width="stretch")
        by = fb.groupby("disease")["rating"].agg(["mean", "count"]).round(2).sort_values("count", ascending=False).head(10)
        b.markdown("**Average rating by condition (top 10 by volume)**")
        b.dataframe(by.rename(columns={"mean": "avg rating", "count": "responses"}), width="stretch")
        if "comment" in fb.columns:
            with st.expander("Recent comments (admin only)"):
                for _, r in fb[fb["comment"].astype(str).str.len() > 0].tail(20).iterrows():
                    st.markdown(f"- **{r['disease']}** ({r['rating']}★): {r['comment']}")
