"""Personalized Healthcare - Medicine & Exercise Recommendation System (Streamlit dashboard).

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src import account_ui, auth, userdata
from src.config import LEVELS, MODEL_DIR, RISK_ORDER, SYMPTOM_LABELS, SYMPTOMS
from src.doctors import CITY_COORDS, WEIGHTS, emergency_advice, load_doctors, rank_doctors
from src.external import dataset_overview, get_external
from src.knowledge_base import SEVERITY_TEXT, all_diseases, get_info, specialties_for
from src.symptom_model import load_training
from src.recommender import ALLERGY_OPTIONS, CONDITION_OPTIONS, INTENSITY_NAME, Profile, Recommender

st.set_page_config(page_title="Medicine & Exercise Recommender", page_icon="💊", layout="wide")

RISK_COLORS = {"Low": "#2e9e6b", "Medium": "#e0a526", "High": "#d64545"}
INTENSITY_COLORS = {"Rest": "#b8c2cc", "Light": "#7bc6a4", "Moderate": "#f0b429", "Vigorous": "#e5484d"}
DISCLAIMER = ("⚠️ **Educational project - not medical advice.** This tool does not diagnose or prescribe. "
              "Medicine names are shown as commonly used options; doses are intentionally omitted. "
              "Always consult a qualified doctor or pharmacist before taking any medicine or starting exercise.")


# ----------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading models...")
def get_engine() -> Recommender:
    return Recommender()


@st.cache_data
def get_metrics() -> dict:
    return json.loads((MODEL_DIR / "metrics.json").read_text())


@st.cache_data
def get_directory():
    return load_doctors()


@st.cache_data
def get_training():
    return load_training()


def banner(kind: str, text: str):
    {"info": st.info, "warning": st.warning, "error": st.error}[kind](text)


# ----------------------------------------------------------------------------- page 1
def page_recommend(engine: Recommender, user: dict):
    st.title("🩺 Personalized Medicine & Exercise Recommendation")
    st.caption("Choose a diagnosis (or let the ML model suggest one from symptoms) and get a safety-checked medicine list, "
               "an exercise plan scaled to your risk, and similar conditions.")
    st.warning(DISCLAIMER)

    account_ui.seed_profile_state(user)
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Patient profile")
        age = st.slider("Age", 1, 100, key="p_age")
        gender = st.radio("Sex", ["female", "male"], horizontal=True, key="p_gender")
        pregnant = st.checkbox("Pregnant / planning pregnancy", disabled=gender != "female", key="p_pregnant")
        bp = st.select_slider("Blood pressure", options=[0, 1, 2], format_func=LEVELS.get, key="p_bp")
        chol = st.select_slider("Cholesterol", options=[0, 1, 2], format_func=LEVELS.get, key="p_chol")
        st.markdown("**Current symptoms**")
        sym = {s: st.checkbox(SYMPTOM_LABELS[s], key=f"sym_{s}") for s in SYMPTOMS}
        allergies = st.multiselect("Known drug allergies", list(ALLERGY_OPTIONS), key="p_allergies")
        conditions = st.multiselect("Other health conditions", list(CONDITION_OPTIONS), key="p_conditions")
        if st.button("💾 Save as my profile", key="save_profile_btn", help="Pre-fills these fields next time you log in"):
            userdata.save_profile(user["id"], dict(age=age, gender=gender, pregnant=bool(pregnant and gender == "female"), bp=bp, chol=chol,
                                                   allergies=allergies, conditions=conditions, city=st.session_state.get("doc_city", "Ludhiana")))
            st.success("Profile saved.")

    profile = Profile(age=age, gender=gender, blood_pressure=bp, cholesterol_level=chol, pregnant=pregnant,
                      allergies=tuple(ALLERGY_OPTIONS[a] for a in allergies),
                      conditions=tuple(CONDITION_OPTIONS[c] for c in conditions), **sym)

    with right:
        mode = st.radio("How do you want to pick the disease?",
                        ["I know my diagnosis", "Predict: detailed symptoms (ML, 41 diseases)", "Predict: quick check (4 symptoms, 131 diseases)"])
        diseases = all_diseases()
        if mode.startswith("I know"):
            disease = st.selectbox("Diagnosed condition", diseases, index=diseases.index("Influenza"))
        else:
            if mode.startswith("Predict: detailed"):
                opts = engine.symptom_options
                if not opts:
                    st.error("The detailed model is not trained yet - run `python -m src.train`.")
                    return
                picked = st.multiselect("Select every symptom you have (type to search)", list(opts), format_func=opts.get)
                if not picked:
                    st.info("Pick at least one symptom to get a prediction.")
                    return
                pred = engine.predict_from_symptoms(picked, k=5)
                st.caption(f"Symptom burden score: **{engine.symptom_burden(picked):.0f}** (sum of severity weights from the open dataset - a rough guide only).")
                note = ("Trained on a templated open dataset (41 diseases): scores look high but real-world accuracy will be lower - see Model Performance. "
                        "Not a diagnosis.")
            else:
                pred = engine.predict_disease(profile, k=5)
                note = "Quick check uses only 4 symptoms + vitals, so confidence is limited - treat it as a starting suggestion, not a diagnosis."
            fig = px.bar(pred.iloc[::-1], x="probability", y="disease", orientation="h", text=pred.iloc[::-1]["probability"].map("{:.0%}".format),
                         title="ML: most likely conditions", color_discrete_sequence=["#3b82f6"])
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=40, b=0), xaxis_tickformat=".0%", yaxis_title=None)
            st.plotly_chart(fig, width="stretch")
            st.caption(note)
            disease = st.selectbox("Use this condition for recommendations", list(pred["disease"]) + [d for d in diseases if d not in set(pred["disease"])])

        rep = engine.full_report(disease, profile)
        info = rep["info"]
        kind, text = SEVERITY_TEXT[info["severity"]]
        st.subheader(disease)
        banner(kind, f"**{info['category']}** - {text}")
        if info["note"]:
            st.info(info["note"])
        ext = get_external(disease)
        if ext:
            with st.expander("📖 About this condition, precautions & diet (open dataset)"):
                st.write(ext["description"])
                if ext["precautions"]:
                    st.markdown("**Precautions**\n" + "\n".join(f"- {p}" for p in ext["precautions"]))
                if ext["diet"]:
                    st.markdown("**Foods / diet reference:** " + ", ".join(ext["diet"]))
                if ext["tips"]:
                    st.markdown("**Lifestyle tips:** " + "; ".join(ext["tips"][:8]))
                st.caption("General reference text from an open dataset - not a personalised plan.")

        c1, c2, c3 = st.columns(3)
        c1.metric("ML risk level", rep["risk_level"], f"{rep['risk'][rep['risk_level']]:.0%} confidence", delta_color="off")
        c2.metric("Exercise intensity allowed", INTENSITY_NAME[rep["cap"]])
        usable = sum(m["status"] != "avoid" for m in rep["medicines"])
        c3.metric("Suitable medicine options", f"{usable} / {len(rep['medicines'])}")

        t_med, t_ex, t_doc, t_sim, t_why = st.tabs(["💊 Medicines", "🏃 Exercise plan", "👩‍⚕️ Find a doctor", "🔗 Similar conditions", "🧠 Why these results"])

        with t_med:
            if not rep["medicines"]:
                st.info("No self-directed medicines are suggested for this condition. Please see the specialist named above.")
            for m in rep["medicines"]:
                icon = {"ok": "✅", "caution": "⚠️", "avoid": "⛔"}[m["status"]]
                badge = "Prescription only" if m["rx"] else "Over the counter"
                with st.container(border=True):
                    a, b = st.columns([3, 1])
                    a.markdown(f"**{icon} {m['name']}**  \n{m['use']}  \n*{m['drug_class']}*")
                    b.markdown(f"`{badge}`")
                    if m["status"] == "avoid":
                        st.error("Avoid for you: " + "; ".join(m["reasons"]))
                    elif m["status"] == "caution":
                        st.warning("Discuss with a doctor/pharmacist first: " + "; ".join(m["reasons"]))
            st.caption("Dose, timing and duration must come from your doctor or the product label.")

        with t_ex:
            if rep["cap"] == 0:
                st.error("No exercise is recommended right now - rest and recover. Ask your doctor when it is safe to restart.")
            else:
                st.success(f"Plan capped at **{INTENSITY_NAME[rep['cap']].lower()}** intensity for your profile.")
            for e in rep["exercises"]:
                st.markdown(f"- **{e['name']}** - {e['minutes']} min · *{INTENSITY_NAME[e['intensity']]}* - {e['benefit']}" if e["minutes"] else f"- **{e['name']}** - {e['benefit']}")
            plan = rep["plan"]
            fig = px.bar(plan, x="Day", y="Minutes", color="Intensity", hover_data=["Activity"], color_discrete_map=INTENSITY_COLORS,
                         category_orders={"Day": list(plan["Day"])}, title="Suggested weekly plan")
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(plan, hide_index=True, width="stretch")
            st.warning("**Safety:** " + info["avoid_exercise"] + " Stop and seek help for chest pain, severe breathlessness, fainting or new severe pain.")

        with t_doc:
            emerg = emergency_advice(disease)
            if emerg:
                st.error(emerg)
            st.markdown("**Who to see:** " + " → ".join(specialties_for(disease)))
            docs, is_sample = get_directory()
            if is_sample:
                st.warning("Demo directory: every doctor below is a **synthetic placeholder** (not a real person). "
                           "Import a verified directory with `python scripts/import_doctors.py` to use real data.")
            c1, c2, c3, c4 = st.columns(4)
            city = c1.selectbox("Your city", list(CITY_COORDS), key="doc_city")
            fee = c2.slider("Max fee (₹)", 300, 3000, 3000, step=100)
            langs = sorted({x for v in docs["languages"].dropna() for x in v.split(";")})
            lang = c3.selectbox("Language", ["Any"] + langs)
            tele = c4.checkbox("Video consult only")
            ranked = rank_doctors(docs, disease, city, max_fee=fee, language=None if lang == "Any" else lang, teleconsult_only=tele)
            if ranked.empty:
                st.info("No doctors match these filters - try raising the fee limit or another city.")
            else:
                show = ranked.assign(distance_km=ranked["distance_km"].round(0), score=ranked["score"].round(2))[
                    ["name", "specialty", "hospital", "city", "distance_km", "experience_years", "rating", "reviews", "fee_inr", "teleconsult", "languages", "score"]]
                st.dataframe(show, hide_index=True, width="stretch")
                comp = ranked.head(5).copy()
                long = pd.concat([pd.DataFrame({"doctor": comp["name"], "component": c, "points": comp[c] * w}) for c, w in WEIGHTS.items()])
                fig = px.bar(long, x="doctor", y="points", color="component", title="Why these doctors rank on top (weighted score)")
                fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, width="stretch")
                st.caption("Score = 40% specialty match + 25% location + 20% rating (shrunk when few reviews) + 10% experience + 5% availability. "
                           "Always verify a doctor's registration before booking.")

        with t_sim:
            sim = rep["similar"]
            st.caption("Hybrid content-based filtering: 60% TF-IDF similarity of treatment/exercise profiles + 40% symptom/vitals profile similarity.")
            fig = px.bar(sim.iloc[::-1], x="similarity", y="condition", color="category", orientation="h", title=f"Conditions most similar to {disease}")
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0), yaxis_title=None)
            st.plotly_chart(fig, width="stretch")

        with t_why:
            st.markdown("**Exercise intensity logic** (the lowest applicable limit wins)")
            for w in rep["cap_reasons"]:
                st.markdown(f"- {w}")
            risk_df = pd.DataFrame({"Risk": RISK_ORDER, "Probability": [rep["risk"].get(r, 0) for r in RISK_ORDER]})
            fig = px.bar(risk_df, x="Risk", y="Probability", color="Risk", color_discrete_map=RISK_COLORS, title="ML risk-level probabilities")
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=40, b=0), yaxis_tickformat=".0%", showlegend=False)
            st.plotly_chart(fig, width="stretch")
            tags = sorted(profile.active_tags())
            st.markdown("**Safety tags checked against each medicine:** " + (", ".join(f"`{t}`" for t in tags) if tags else "none"))

        st.divider()
        account_ui.save_and_feedback(user, disease, mode, profile, rep)


# ----------------------------------------------------------------------------- page 2
def page_explore(engine: Recommender):
    st.title("📊 Data Explorer")
    df = engine.df
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patient records", len(df))
    c2.metric("Diseases", df["disease"].nunique())
    c3.metric("Median age", int(df["age"].median()))
    c4.metric("High-risk share", f"{(df['risk_level'] == 'High').mean():.0%}")

    cats = st.multiselect("Filter by category", sorted(df["category"].unique()))
    d = df[df["category"].isin(cats)] if cats else df

    a, b = st.columns(2)
    top = d["disease"].value_counts().head(12).iloc[::-1]
    a.plotly_chart(px.bar(top, orientation="h", title="Most frequent diseases", labels={"value": "records", "index": ""}).update_layout(showlegend=False, height=380), width="stretch")
    b.plotly_chart(px.histogram(d, x="age", color="risk_level", nbins=20, category_orders={"risk_level": RISK_ORDER},
                                color_discrete_map=RISK_COLORS, title="Age distribution by risk level").update_layout(height=380), width="stretch")

    a, b = st.columns(2)
    sym = d.groupby("category")[SYMPTOMS].mean().rename(columns=SYMPTOM_LABELS)
    a.plotly_chart(px.imshow(sym, text_auto=".0%", aspect="auto", color_continuous_scale="Blues", title="Symptom prevalence by category").update_layout(height=420), width="stretch")
    risk_cat = pd.crosstab(d["category"], d["risk_level"], normalize="index")[[r for r in RISK_ORDER if r in d["risk_level"].unique()]]
    b.plotly_chart(px.bar(risk_cat, barmode="stack", color_discrete_map=RISK_COLORS, title="Risk mix by category", labels={"value": "share"}).update_layout(height=420), width="stretch")

    st.subheader("Patient segmentation (K-Means + PCA)")
    k = st.slider("Number of segments", 2, 6, 4)
    feats = ["age", "male", "blood_pressure", "cholesterol_level", "symptom_count"]
    Z = StandardScaler().fit_transform(df[feats])
    seg = KMeans(k, n_init=10, random_state=42).fit_predict(Z)
    xy = PCA(2, random_state=42).fit_transform(Z)
    plot = pd.DataFrame({"PC1": xy[:, 0], "PC2": xy[:, 1], "Segment": [f"Segment {s + 1}" for s in seg], "Disease": df["disease"], "Risk": df["risk_level"]})
    st.plotly_chart(px.scatter(plot, x="PC1", y="PC2", color="Segment", hover_data=["Disease", "Risk"], title="Patient segments (2-D projection)").update_layout(height=420), width="stretch")
    prof = df.assign(Segment=[f"Segment {s + 1}" for s in seg]).groupby("Segment").agg(
        patients=("age", "size"), mean_age=("age", "mean"), male_share=("male", "mean"), mean_symptoms=("symptom_count", "mean"),
        high_risk_share=("risk_level", lambda s: (s == "High").mean())).round(2)
    st.dataframe(prof, width="stretch")


# ----------------------------------------------------------------------------- page 3
def page_models():
    st.title("🤖 Model Performance")
    m = get_metrics()
    st.caption(f"{m['n_rows']} records · {m['n_diseases']} diseases · {m['cv_folds']}-fold cross-validation (small data, so no single hold-out split).")

    st.subheader("1. Disease classifier (symptoms + vitals → disease)")
    sc = pd.DataFrame(m["disease"]["scores"]).T.rename(columns={"top1": "Top-1", "top3": "Top-3", "top5": "Top-5"})
    long = sc.reset_index().melt(id_vars="index", var_name="Metric", value_name="Accuracy")
    fig = px.bar(long, x="index", y="Accuracy", color="Metric", barmode="group", labels={"index": "Model"}, title="Cross-validated accuracy")
    fig.add_hline(y=m["disease"]["baseline_top1"], line_dash="dot", annotation_text="majority-class baseline")
    fig.update_layout(yaxis_tickformat=".0%", height=360)
    st.plotly_chart(fig, width="stretch")
    st.success(f"Selected: **{m['disease']['best']}** (best top-3 accuracy).")
    st.warning("Accuracy is modest by design of the data: only 4 yes/no symptoms for 116 diseases, with 1-5 records per disease. "
               "That is why the app's main path is *diagnosis → recommendation*, and prediction is only a suggestion.")
    imp = pd.Series(m["disease"]["importance"]).sort_values()
    st.plotly_chart(px.bar(imp, orientation="h", title="Feature importance (Random Forest)", labels={"value": "importance", "index": ""}).update_layout(showlegend=False, height=300), width="stretch")

    st.subheader("2. Risk classifier (profile + disease category → Low / Medium / High)")
    rs = pd.DataFrame(m["risk"]["scores"]).T.rename(columns={"accuracy": "Accuracy", "macro_f1": "Macro-F1"})
    st.dataframe(rs.style.format("{:.1%}", subset=["Accuracy"]).format("{:.2f}", subset=["Macro-F1"]), width="stretch")
    st.success(f"Selected: **{m['risk']['best']}** · accuracy vs. majority-class baseline {m['risk']['baseline_accuracy']:.1%}.")
    a, b = st.columns(2)
    cm = pd.DataFrame(m["risk"]["confusion_matrix"], index=m["risk"]["labels"], columns=m["risk"]["labels"])
    a.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual"), title="Cross-validated confusion matrix").update_layout(height=340), width="stretch")
    imp_r = pd.Series(m["risk"]["importance"]).sort_values()
    b.plotly_chart(px.bar(imp_r, orientation="h", title="Top risk drivers (Random Forest)", labels={"value": "importance", "index": ""}).update_layout(showlegend=False, height=340), width="stretch")
    st.caption("The risk label in the source data tracks the type of disease strongly, which is why adding the disease category lifts accuracy well above baseline.")

    sm = m.get("symptom")
    if sm:
        st.subheader("3. Detailed symptom model (132 symptoms → 41 diseases)")
        st.caption(f"{sm['info']['unique_rows']} unique rows after removing {sm['info']['raw_rows'] - sm['info']['unique_rows']:,} duplicates · "
                   f"3×3 repeated stratified CV · 'forgotten' = {sm['dropout']:.0%} of the reported symptoms randomly removed to mimic incomplete input.")
        rows = []
        for n, v in sm["scores"].items():
            rows += [(n, "Top-1", v["top1"]), (n, "Top-3", v["top3"]), (n, "Top-1 (30% forgotten)", v["top1_dropout"]), (n, "Top-3 (30% forgotten)", v["top3_dropout"])]
        d = pd.DataFrame(rows, columns=["Model", "Metric", "Accuracy"])
        fig = px.bar(d, x="Model", y="Accuracy", color="Metric", barmode="group", title="Cross-validated accuracy")
        fig.update_layout(yaxis_tickformat=".0%", height=380, yaxis_range=[0, 1.05])
        st.plotly_chart(fig, width="stretch")
        st.success(f"Selected: **{sm['best']}** (best accuracy with forgotten symptoms).")
        st.warning("Near-perfect scores reflect the dataset, not clinical skill: each disease has a fixed, templated symptom pattern. "
                   "Real patients are noisier, so treat this as a demo of the method.")


# ----------------------------------------------------------------------------- page: datasets
def page_datasets():
    st.title("🗂️ Datasets")
    st.dataframe(dataset_overview(), hide_index=True, width="stretch")
    sm = get_metrics().get("symptom")
    if sm:
        info = sm["info"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Training.csv raw rows", f"{info['raw_rows']:,}")
        c2.metric("Unique rows", f"{info['unique_rows']:,}", f"-{1 - info['unique_rows'] / info['raw_rows']:.0%} duplicates", delta_color="off")
        c3.metric("Symptoms / diseases", f"{info['n_symptoms']} / {sm['n_diseases']}")
        st.warning("94% of the rows in Training.csv are exact duplicates. Evaluating on them would let the same row sit in both train and test "
                   "and report fake ~100% accuracy, so the project deduplicates first and also tests with 30% of symptoms forgotten.")
        X, y, _ = get_training()
        freq = (X.mean().sort_values(ascending=False).head(15).iloc[::-1])
        freq.index = [i.replace("_", " ") for i in freq.index]
        a, b = st.columns(2)
        a.plotly_chart(px.bar(freq, orientation="h", title="Most common symptoms (unique rows)", labels={"value": "share of rows", "index": ""}).update_layout(showlegend=False, height=380), width="stretch")
        b.plotly_chart(px.histogram(X.sum(axis=1), nbins=15, title="Symptoms per record", labels={"value": "symptoms"}).update_layout(showlegend=False, height=380), width="stretch")

    st.subheader("Doctor directory")
    docs, is_sample = get_directory()
    (st.warning if is_sample else st.success)("Synthetic demo directory (placeholder names)." if is_sample else "Verified directory loaded from data/doctors.csv.")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(docs["specialty"].value_counts().iloc[::-1], orientation="h", title="Doctors per specialty", labels={"value": "doctors", "index": ""}).update_layout(showlegend=False, height=460), width="stretch")
    b.plotly_chart(px.bar(docs["city"].value_counts().iloc[::-1], orientation="h", title="Doctors per city", labels={"value": "doctors", "index": ""}).update_layout(showlegend=False, height=460), width="stretch")
    st.markdown("""
**Adding more data**
- **Symptoms/diseases:** replace `data/external/Training.csv` (binary symptom columns + `prognosis`) and run `python -m src.train`; add any new disease names to `src/knowledge_base.py`.
- **Medicines:** extend `_M` in `src/knowledge_base.py` using official labels (openFDA, DailyMed, WHO Essential Medicines); keep the risk tags.
- **Doctors:** prepare a CSV with provenance (e.g. an official medical-council register export or hospital directory you are allowed to use) and run `python scripts/import_doctors.py file.csv`.
""")


# ----------------------------------------------------------------------------- page 4
def page_kb():
    st.title("📚 Knowledge Base")
    rows = []
    for d in all_diseases():
        i = get_info(d)
        rows.append({"Disease": d, "Category": i["category"], "Severity": i["severity"].replace("_", " "),
                     "Medicines": ", ".join(m["name"] for m in i["medicines"]) or "specialist-led",
                     "Max exercise": INTENSITY_NAME[i["max_intensity"]]})
    kb = pd.DataFrame(rows)
    q = st.text_input("Search disease, medicine or category")
    if q:
        kb = kb[kb.apply(lambda r: q.lower() in " ".join(r.astype(str)).lower(), axis=1)]
    st.dataframe(kb, hide_index=True, width="stretch", height=520)
    st.download_button("⬇️ Download as CSV", kb.to_csv(index=False), "knowledge_base.csv", "text/csv")


# ----------------------------------------------------------------------------- page 5
def page_about():
    st.title("ℹ️ About")
    st.markdown(f"""
{DISCLAIMER}

### How it works
1. **Disease** - picked by you, or predicted by ML: a detailed model (132 symptoms, 41 diseases) or a quick model (4 symptoms + vitals, 131 diseases).
2. **Risk level** - a second ML model (Low / Medium / High) using the patient profile and the disease category.
3. **Medicines** - from a curated knowledge base, then filtered by rules for allergies, pregnancy, age and conditions such as kidney/liver disease.
4. **Exercise** - activities for the condition, capped by the lowest of: condition limit, ML risk, age, blood pressure, breathing difficulty and fever.
5. **Similar conditions** - hybrid content-based filtering (TF-IDF + symptom-profile similarity).\n6. **Accounts** - register / log in; saved profile, history and feedback per user; roles: user, analyst, admin.\n7. **Doctors** - disease → specialties, then ranked by specialty match, location, rating, experience and availability.

### Limits
- Training data: a 349-row dataset plus a templated 132-symptom dataset - good for a demo, **not** for clinical use.\n- The bundled doctor directory is synthetic placeholder data.
- The medicine/exercise knowledge base is hand-curated for education and has no numeric doses.
- Serious conditions (emergencies, cancers, genetic syndromes) intentionally return no self-directed medicines.
""")


# ----------------------------------------------------------------------------- main
SESSION_TIMEOUT = 30 * 60   # seconds of inactivity before automatic logout
ALL = ("user", "analyst", "admin")
STAFF = ("analyst", "admin")
PAGES = {  # label -> (key, roles allowed)
    "🩺 Recommend": ("recommend", ALL), "🕘 My History": ("history", ALL), "👤 My Profile": ("profile", ALL),
    "📊 Data Explorer": ("explore", STAFF), "🤖 Model Performance": ("models", STAFF), "🗂️ Datasets": ("datasets", STAFF),
    "📈 Usage & Feedback": ("analytics", STAFF), "📚 Knowledge Base": ("kb", ALL),
    "🛡️ Admin": ("admin", ("admin",)), "ℹ️ About": ("about", ALL),
}


def main():
    try:
        auth.bootstrap_admin_from_env()
    except auth.ValidationError as e:
        st.session_state["_bootstrap_error"] = str(e)

    user = st.session_state.get("user")
    if user:
        idle = time.time() - st.session_state.get("last_active", time.time())
        fresh = auth.get_active_user(user["id"])   # re-check the database so role changes / disabled accounts apply at once
        if idle > SESSION_TIMEOUT:
            account_ui.logout("Your session expired. Please log in again.")
            user = None
        elif fresh is None:
            account_ui.logout("Your account is no longer available.")
            user = None
        else:
            user = st.session_state["user"] = fresh
            st.session_state["last_active"] = time.time()

    if not user:
        account_ui.page_auth()
        return
    if user["must_change_password"]:
        account_ui.page_force_change(user)
        return

    allowed = {label: key for label, (key, roles) in PAGES.items() if user["role"] in roles}
    if st.session_state.get("nav") not in allowed:
        st.session_state.pop("nav", None)
    st.sidebar.markdown(f"### 👋 {user['full_name']}")
    st.sidebar.caption(f"@{user['username']} · {user['role']}")
    if st.sidebar.button("Log out", key="logout_btn", width="stretch"):
        account_ui.logout("You have been logged out.")
        st.rerun()
    choice = st.sidebar.radio("Navigate", list(allowed), key="nav")
    st.sidebar.caption("Medicine & Exercise Recommender · educational demo")

    engine = get_engine()
    {"recommend": lambda: page_recommend(engine, user), "history": lambda: account_ui.page_history(user),
     "profile": lambda: account_ui.page_profile(user), "explore": lambda: page_explore(engine), "models": page_models,
     "datasets": page_datasets, "analytics": lambda: account_ui.page_analytics(user), "kb": page_kb,
     "admin": lambda: account_ui.page_admin(user), "about": page_about}[allowed[choice]]()
    st.divider()
    st.caption(DISCLAIMER)


main()
