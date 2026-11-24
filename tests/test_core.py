import pytest
from streamlit.testing.v1 import AppTest

from src.config import ROOT

from src.data import NAME_FIXES, load_clean
from src.knowledge_base import all_diseases, get_info
from src.recommender import Profile, Recommender, assess_medicine


@pytest.fixture(scope="module")
def engine():
    return Recommender()


def test_every_dataset_disease_is_in_knowledge_base():
    df = load_clean()  # raises KeyError if a disease has no KB entry
    assert set(df["disease"]) <= set(all_diseases())
    assert not any(d in df["disease"].values for d in NAME_FIXES)


def test_emergencies_have_no_medicines_and_no_exercise(engine):
    for d in ["Stroke", "Sepsis", "Appendicitis", "Rabies"]:
        rep = engine.full_report(d, Profile())
        assert rep["medicines"] == [] and rep["cap"] == 0
        assert (rep["plan"]["Minutes"] == 0).all()


def test_allergy_pregnancy_and_age_filters():
    ibu = next(m for m in get_info("Migraine")["medicines"] if m["key"] == "ibuprofen")
    assert assess_medicine(ibu, Profile(allergies=("allergy:nsaid",)))[0] == "avoid"
    assert assess_medicine(ibu, Profile(gender="female", pregnant=True))[0] == "avoid"
    assert assess_medicine(ibu, Profile(conditions=("kidney",)))[0] == "caution"
    assert assess_medicine(ibu, Profile())[0] == "ok"


def test_fever_forces_rest_and_high_risk_lowers_intensity(engine):
    assert engine.full_report("Common Cold", Profile(fever=True))["cap"] == 0
    assert engine.full_report("Diabetes", Profile(age=80))["cap"] <= 1


def test_prediction_and_similarity_shapes(engine):
    p = engine.predict_disease(Profile(cough=True, difficulty_breathing=True), k=5)
    assert len(p) == 5 and abs(p["probability"].sum()) <= 1.0 + 1e-6
    s = engine.similar_conditions("Asthma", k=4)
    assert len(s) == 4 and "Asthma" not in set(s["condition"])


@pytest.mark.parametrize("page", ["🩺 Recommend", "🕘 My History", "👤 My Profile", "📊 Data Explorer", "🤖 Model Performance", "🗂️ Datasets", "📈 Usage & Feedback", "📚 Knowledge Base", "🛡️ Admin", "ℹ️ About"])
def test_dashboard_pages_render(page, logged_in):
    at = logged_in("admin")
    assert not at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_external_dataset_maps_to_knowledge_base():
    from src.external import get_external, load_external
    from src.symptom_model import load_training
    X, y, info = load_training()
    assert info["unique_rows"] < info["raw_rows"] and info["conflicting_rows"] == 0   # duplicates were removed
    assert set(y) <= set(all_diseases())
    assert get_external("Asthma")["description"] and len(load_external()["precautions"]) == 41


def test_detailed_symptom_model(engine):
    opts = engine.symptom_options
    assert len(opts) == 132
    pred = engine.predict_from_symptoms(["itching", "skin_rash", "nodal_skin_eruptions"], k=3)
    assert pred.iloc[0]["disease"] == "Fungal Infection" and pred["probability"].iloc[0] > 0.3
    assert engine.symptom_burden(["itching"]) > 0


def test_doctor_ranking_and_safety():
    from src.doctors import emergency_advice, load_doctors, rank_doctors
    from src.knowledge_base import SPECIALTIES, specialties_for
    df, is_sample = load_doctors()
    assert is_sample and df["name"].str.startswith("Sample Doctor").all() and set(df["specialty"]) <= set(SPECIALTIES)
    assert all(set(specialties_for(d)) <= set(SPECIALTIES) for d in all_diseases())
    r = rank_doctors(df, "Asthma", "Ludhiana")
    assert len(r) > 0 and r["specialty"].isin(specialties_for("Asthma")).all()
    assert r["score"].is_monotonic_decreasing and r["score"].between(0, 1).all()
    assert emergency_advice("Stroke") and emergency_advice("Common Cold") is None
    assert rank_doctors(df, "Asthma", "Ludhiana", teleconsult_only=True)["teleconsult"].all()


def test_recommend_page_detailed_mode_and_doctor_tab(logged_in):
    at = logged_in("user")
    at.radio[1].set_value("Predict: detailed symptoms (ML, 41 diseases)").run()
    assert not at.exception and any("Pick at least one symptom" in i.value for i in at.info)
    sym = next(m for m in at.multiselect if m.label.startswith("Select every symptom"))
    sym.set_value(["itching", "skin_rash", "nodal_skin_eruptions"]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox[0].options[0] == "Fungal Infection"          # ML top-1 is offered first
    assert any("synthetic placeholder" in w.value for w in at.warning)  # demo-directory banner is shown
    assert any(e.label.startswith("📖") for e in at.expander)           # open-dataset info panel
    assert len(at.dataframe) >= 1                                      # ranked doctor table
    # an emergency condition shows emergency advice instead of a self-care plan
    at.selectbox[0].select("Stroke").run()
    assert not at.exception and any("emergency" in e.value.lower() for e in at.error)
